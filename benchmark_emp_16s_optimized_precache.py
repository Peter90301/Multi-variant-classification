#!/usr/bin/env python3
"""Benchmark an optimized memory-mapped EMP GPU-HDC prediction cache."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cuml.svm import LinearSVC as CuLinearSVC
from scipy import sparse
from sklearn.metrics import accuracy_score

from benchmark_emp_16s_full_gpu_cuml import load_precache, log


SOURCE_CACHE = Path("/home/tsl012/Multi_variant classification/emp_16s_gpu_precache_4096")
OPTIMIZED_CACHE = Path("/home/tsl012/Multi_variant classification/emp_16s_gpu_precache_4096_optimized")
OUTDIR = Path("/home/tsl012/Multi_variant classification/emp_16s_optimized_precache_benchmark")
LEVELS = ["empo_1", "empo_2", "empo_3"]
RF_PIPELINE_SECONDS = {"empo_1": 4.472511, "empo_2": 4.444939, "empo_3": 4.432314}

CSR_ACCUM_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__ void emp_hdc_accumulate_csr_kernel(
    int n_samples,
    int active_dims,
    int dim,
    const int *indptr,
    const int *obs_idx,
    const float *values,
    const unsigned short *obs_cols,
    const signed char *obs_signs,
    float *matrix
) {
    int sample = blockIdx.x;
    if (sample >= n_samples) return;
    int start = indptr[sample];
    int end = indptr[sample + 1];
    for (int nz = start + threadIdx.x; nz < end; nz += blockDim.x) {
        int obs = obs_idx[nz];
        float abundance = values[nz];
        for (int a = 0; a < active_dims; ++a) {
            int offset = obs * active_dims + a;
            int col = (int)obs_cols[offset];
            float sign = (float)obs_signs[offset];
            atomicAdd(matrix + sample * dim + col, abundance * sign);
        }
    }
}
""",
    "emp_hdc_accumulate_csr_kernel",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, default=SOURCE_CACHE)
    parser.add_argument("--optimized-cache", type=Path, default=OPTIMIZED_CACHE)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--hdc-dim", type=int, default=4096)
    parser.add_argument("--active-dims-per-sequence", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-sample-sum", type=float, default=1000.0)
    return parser.parse_args()


def optimized_paths(cache_dir):
    return {
        "manifest": cache_dir / "manifest.json",
        "indptr": cache_dir / "indptr.npy",
        "obs_idx": cache_dir / "obs_idx.npy",
        "values": cache_dir / "values.npy",
        "obs_cols": cache_dir / "obs_cols.npy",
        "obs_signs": cache_dir / "obs_signs.npy",
    }


def create_optimized_cache(args):
    paths = optimized_paths(args.optimized_cache)
    if all(path.exists() for path in paths.values()):
        return
    args.optimized_cache.mkdir(parents=True, exist_ok=True)
    log("Creating compact uncompressed HDC-only cache")
    x_counts = sparse.load_npz(args.source_cache / "x_counts.npz").tocsr()
    with np.load(args.source_cache / "hdc_arrays.npz") as data:
        np.save(paths["indptr"], x_counts.indptr.astype(np.int32, copy=False), allow_pickle=False)
        np.save(paths["obs_idx"], data["obs_idx"].astype(np.int32, copy=False), allow_pickle=False)
        np.save(paths["values"], data["values"].astype(np.float32, copy=False), allow_pickle=False)
        np.save(paths["obs_cols"], data["obs_cols"].astype(np.uint16), allow_pickle=False)
        np.save(paths["obs_signs"], data["obs_signs"].astype(np.int8), allow_pickle=False)
    manifest = {
        "samples": int(x_counts.shape[0]),
        "observations": int(x_counts.shape[1]),
        "nnz": int(x_counts.nnz),
        "hdc_dim": args.hdc_dim,
        "active_dims_per_sequence": args.active_dims_per_sequence,
        "format": "uncompressed-npy-mmap-csr-no-sample-idx",
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True))


def open_mmaps(cache_dir):
    paths = optimized_paths(cache_dir)
    started = time.perf_counter()
    manifest = json.loads(paths["manifest"].read_text())
    arrays = {
        key: np.load(paths[key], mmap_mode="r", allow_pickle=False)
        for key in ["indptr", "obs_idx", "values", "obs_cols", "obs_signs"]
    }
    return manifest, arrays, time.perf_counter() - started


def build_gpu_matrix(args, manifest, arrays):
    timings = {}
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    device_arrays = {key: cp.asarray(value) for key, value in arrays.items()}
    matrix = cp.zeros((manifest["samples"], args.hdc_dim), dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    timings["h2d_alloc_sec"] = time.perf_counter() - started

    start_event, stop_event = cp.cuda.Event(), cp.cuda.Event()
    start_event.record()
    CSR_ACCUM_KERNEL(
        (manifest["samples"],),
        (256,),
        (
            np.int32(manifest["samples"]),
            np.int32(args.active_dims_per_sequence),
            np.int32(args.hdc_dim),
            device_arrays["indptr"],
            device_arrays["obs_idx"],
            device_arrays["values"],
            device_arrays["obs_cols"],
            device_arrays["obs_signs"],
            matrix,
        ),
    )
    stop_event.record()
    stop_event.synchronize()
    timings["gpu_hdc_sec"] = cp.cuda.get_elapsed_time(start_event, stop_event) / 1000.0

    started = time.perf_counter()
    norms = cp.sqrt(cp.sum(matrix * matrix, axis=1, keepdims=True))
    matrix = matrix / cp.maximum(norms, cp.float32(1e-12))
    cp.cuda.Stream.null.synchronize()
    timings["gpu_normalize_sec"] = time.perf_counter() - started
    return matrix, device_arrays, timings


def load_labels(source_cache):
    with np.load(source_cache / "labels_and_splits.npz") as data:
        return {key: data[key] for key in data.files}


def prepare_models(args, labels_and_splits):
    manifest, arrays, _ = open_mmaps(args.optimized_cache)
    matrix, device_arrays, _ = build_gpu_matrix(args, manifest, arrays)
    models = {}
    truth = {}
    test_indices = {}
    for level in LEVELS:
        labels = labels_and_splits[f"{level}_labels"]
        valid_idx = labels_and_splits[f"{level}_valid_idx"]
        train_rel = labels_and_splits[f"{level}_train_rel"]
        test_rel = labels_and_splits[f"{level}_test_rel"]
        _, encoded = np.unique(labels, return_inverse=True)
        encoded = encoded.astype(np.int32)
        train_abs = valid_idx[train_rel].astype(np.int32)
        test_abs = valid_idx[test_rel].astype(np.int32)
        model = CuLinearSVC(
            C=1.0,
            max_iter=20000,
            tol=1e-6,
            linesearch_max_iter=200,
            fit_intercept=True,
            verbose=0,
        )
        model.fit(matrix[cp.asarray(train_abs)], cp.asarray(encoded[train_rel]))
        cp.cuda.runtime.deviceSynchronize()
        models[level] = model
        truth[level] = encoded[test_rel]
        test_indices[level] = test_abs
    del matrix, device_arrays
    cp.get_default_memory_pool().free_all_blocks()
    return models, truth, test_indices


def run_pipeline_once(args, model, test_indices):
    total_started = time.perf_counter()
    manifest, arrays, cache_open_sec = open_mmaps(args.optimized_cache)
    matrix, device_arrays, timings = build_gpu_matrix(args, manifest, arrays)
    test_gpu = cp.asarray(test_indices)
    cp.cuda.runtime.deviceSynchronize()
    predict_started = time.perf_counter()
    prediction = model.predict(matrix[test_gpu])
    cp.cuda.runtime.deviceSynchronize()
    predict_sec = time.perf_counter() - predict_started
    prediction_cpu = cp.asnumpy(prediction).astype(np.int32)
    timings.update({
        "cache_open_sec": cache_open_sec,
        "predict_sec": predict_sec,
        "total_sec": time.perf_counter() - total_started,
    })
    del matrix, device_arrays, test_gpu, prediction
    return prediction_cpu, timings


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cp.cuda.Device(args.device).use()
    create_optimized_cache(args)
    labels_and_splits = load_labels(args.source_cache)
    log("Training deployment models once; training is excluded from timing")
    models, truth, test_indices = prepare_models(args, labels_and_splits)

    rows = []
    stage_rows = []
    for level in LEVELS:
        log(f"Warm-up optimized prediction pipeline for {level}")
        run_pipeline_once(args, models[level], test_indices[level])
        runs = []
        prediction = None
        for run in range(1, args.repeats + 1):
            prediction, timings = run_pipeline_once(args, models[level], test_indices[level])
            timings["run"] = run
            runs.append(timings)
        accuracy = accuracy_score(truth[level], prediction)
        total_times = np.asarray([run["total_sec"] for run in runs])
        row = {
            "method": "GPU-HDC-CPU-precache-optimized",
            "empo_level": level.upper().replace("_", ""),
            "accuracy": accuracy,
            "warmup_runs": 1,
            "timed_runs": args.repeats,
            "pipeline_mean_sec": float(total_times.mean()),
            "pipeline_median_sec": float(np.median(total_times)),
            "pipeline_std_sec": float(total_times.std()),
            "pipeline_min_sec": float(total_times.min()),
            "pipeline_max_sec": float(total_times.max()),
            "rf_pipeline_sec": RF_PIPELINE_SECONDS[level],
            "speedup_vs_random_forest": RF_PIPELINE_SECONDS[level] / float(total_times.mean()),
        }
        rows.append(row)
        for run in runs:
            stage_rows.append({"empo_level": row["empo_level"], **run})
        log(
            f"{row['empo_level']}: accuracy={accuracy:.4f}, "
            f"pipeline={row['pipeline_mean_sec']:.4f}s, speedup={row['speedup_vs_random_forest']:.2f}x"
        )

    with (args.outdir / "optimized_precache_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.outdir / "optimized_precache_stage_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)
    cache_bytes = sum(path.stat().st_size for key, path in optimized_paths(args.optimized_cache).items() if key != "manifest")
    (args.outdir / "optimized_cache_stats.json").write_text(
        json.dumps({"cache_bytes": cache_bytes, "cache_mib": cache_bytes / 1024**2, **vars(args)}, default=str, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
