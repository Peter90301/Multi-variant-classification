#!/usr/bin/env python3
"""Benchmark five pretrained-model prediction paths on the 168K gut compendium."""

from __future__ import annotations

import argparse
import ctypes
import csv
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

cuda_root = Path(sys.executable).parent.parent
os.environ.setdefault("CUDA_PATH", str(cuda_root))
nvrtc_library = cuda_root / "lib" / "libnvrtc.so.12"
if nvrtc_library.exists():
    ctypes.CDLL(str(nvrtc_library), mode=ctypes.RTLD_GLOBAL)

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from biom import load_table
from joblib import dump
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Compatibility for the older Matplotlib build in the project environment.
if not hasattr(np, "Inf"):
    np.Inf = np.inf

DATA_DIR = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/Multi_variant_classification/"
    "168000_sample_human_gut_biom"
)
OUTDIR = Path(
    "/home/tsl012/Multi_variant classification/human_gut_168k_five_methods_opt"
)
METHODS = [
    "Explicit-Vocab",
    "HDC-Hash",
    "Random Forest",
    "GPU-HDC Hash",
    "GPU-HDC-CPU-precache_opt",
]

CSR_ACCUM_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__ void human_gut_hdc_accumulate_csr_kernel(
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
    "human_gut_hdc_accumulate_csr_kernel",
    backend="nvrtc",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--target", default="iso", choices=["iso", "region", "geo_loc_name"])
    parser.add_argument("--min-sample-sum", type=float, default=1000.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hdc-dim", type=int, default=4096)
    parser.add_argument("--active-dims-per-sequence", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def extract_qza_member(qza_path: Path, suffix: str, output_path: Path) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with zipfile.ZipFile(qza_path) as archive:
        member = next(name for name in archive.namelist() if name.endswith(suffix))
        with archive.open(member) as source, temporary.open("wb") as destination:
            while chunk := source.read(8 * 1024 * 1024):
                destination.write(chunk)
    temporary.replace(output_path)


def extract_sequences(qza_path: Path, observation_ids: list[str], cache_path: Path):
    if cache_path.exists():
        sequences = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if len(sequences) == len(observation_ids):
            return sequences

    wanted = set(observation_ids)
    found: dict[str, list[str]] = {}

    def target_id(header: str):
        if header in wanted:
            return header
        match = re.fullmatch(r"(.+)_\d+", header)
        if match and match.group(1) in wanted:
            return match.group(1)
        return None

    with zipfile.ZipFile(qza_path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("/data/dna-sequences.fasta"))
        with archive.open(member) as handle:
            target = None
            chunks: list[str] = []
            for raw_line in handle:
                line = raw_line.decode("ascii").strip()
                if line.startswith(">"):
                    if target is not None and chunks:
                        found.setdefault(target, []).append("".join(chunks))
                    target = target_id(line[1:].split()[0])
                    chunks = []
                elif target is not None:
                    chunks.append(line)
            if target is not None and chunks:
                found.setdefault(target, []).append("".join(chunks))

    missing = [observation for observation in observation_ids if observation not in found]
    if missing:
        raise ValueError(f"Missing {len(missing)} reference sequences; first IDs: {missing[:5]}")
    sequences = np.asarray(["N".join(sorted(found[obs])) for obs in observation_ids], dtype=str)
    np.save(cache_path, sequences, allow_pickle=False)
    return sequences


def sequence_hypervectors(sequences, dim: int, active_dims: int, seed: int):
    columns = np.empty((len(sequences), active_dims), dtype=np.uint16)
    signs = np.empty((len(sequences), active_dims), dtype=np.int8)
    sign_values = np.asarray([-1, 1], dtype=np.int8)
    for index, sequence in enumerate(sequences):
        digest = hashlib.blake2b(
            f"{seed}|asv_sequence:{sequence}".encode(), digest_size=16
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        columns[index] = rng.choice(dim, size=active_dims, replace=False).astype(np.uint16)
        signs[index] = rng.choice(sign_values, size=active_dims)
    return columns.ravel(), signs.ravel()


def build_gpu_matrix(args, manifest, arrays):
    timings = {}
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    device_arrays = {key: cp.asarray(value) for key, value in arrays.items()}
    matrix = cp.zeros((manifest["samples"], args.hdc_dim), dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    timings["h2d_alloc_sec"] = time.perf_counter() - started

    started = time.perf_counter()
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
    cp.cuda.Stream.null.synchronize()
    timings["gpu_hdc_sec"] = time.perf_counter() - started

    started = time.perf_counter()
    norms = cp.sqrt(cp.sum(matrix * matrix, axis=1, keepdims=True))
    matrix = matrix / cp.maximum(norms, cp.float32(1e-12))
    cp.cuda.Stream.null.synchronize()
    timings["gpu_normalize_sec"] = time.perf_counter() - started
    return matrix, device_arrays, timings


def load_dataset(args, biom_path: Path):
    table = load_table(str(biom_path))
    sample_ids = np.asarray(list(map(str, table.ids(axis="sample"))))
    observation_ids = list(map(str, table.ids(axis="observation")))
    sums = np.asarray(table.sum(axis="sample"), dtype=np.float64)
    metadata = pd.read_csv(
        args.data_dir / "sample_metadata.tsv", sep="\t", dtype=str, low_memory=False
    ).set_index("srs")

    labels_by_sample = metadata[args.target].astype("string").str.strip()
    invalid = {"", "unknown", "na", "nan", "not available", "not applicable", "missing"}
    sample_position = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    valid_ids = [
        sample_id
        for sample_id, sample_sum in zip(sample_ids, sums, strict=True)
        if sample_sum >= args.min_sample_sum
        and sample_id in labels_by_sample.index
        and pd.notna(labels_by_sample.at[sample_id])
        and str(labels_by_sample.at[sample_id]).lower() not in invalid
    ]
    labels = labels_by_sample.loc[valid_ids].astype(str).to_numpy()
    counts_per_class = pd.Series(labels).value_counts()
    supported = set(counts_per_class[counts_per_class >= 2].index)
    keep = np.asarray([label in supported for label in labels])
    valid_ids = np.asarray(valid_ids)[keep]
    labels = labels[keep]
    positions = np.asarray([sample_position[sample_id] for sample_id in valid_ids], dtype=np.int32)
    counts = table.matrix_data.tocsc()[:, positions].T.tocsr().astype(np.float32)
    counts.sort_indices()
    return valid_ids, observation_ids, counts, labels


def cpu_predict_benchmark(model, matrix, repeats: int):
    model.predict(matrix)
    times = []
    prediction = None
    for _ in range(repeats):
        started = time.perf_counter()
        prediction = model.predict(matrix)
        times.append(time.perf_counter() - started)
    return np.asarray(prediction), np.asarray(times)


def gpu_linear_predict(matrix, coefficients, intercept):
    return cp.argmax(matrix @ coefficients.T + intercept, axis=1).astype(cp.int32)


def gpu_predict_benchmark(matrix, coefficients, intercept, repeats: int):
    gpu_linear_predict(matrix, coefficients, intercept)
    cp.cuda.Stream.null.synchronize()
    times = []
    prediction = None
    for _ in range(repeats):
        cp.cuda.Stream.null.synchronize()
        started = time.perf_counter()
        prediction = gpu_linear_predict(matrix, coefficients, intercept)
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - started)
    return cp.asnumpy(prediction).astype(np.int32), np.asarray(times)


def result_row(method, truth, prediction, times, train_count: int, test_count: int):
    return {
        "method": method,
        "accuracy": accuracy_score(truth, prediction),
        "train_samples": train_count,
        "test_samples": test_count,
        "warmup_runs": 1,
        "timed_runs": len(times),
        "predict_mean_sec": float(times.mean()),
        "predict_median_sec": float(np.median(times)),
        "predict_std_sec": float(times.std()),
        "predict_min_sec": float(times.min()),
        "predict_max_sec": float(times.max()),
    }


def plot_results(rows, outdir: Path) -> None:
    labels = [row["method"] for row in rows]
    colors = ["#4575b4", "#91bfdb", "#777777", "#fdae61", "#d73027"]
    for metric, ylabel, title, filename, fmt in [
        ("accuracy", "Accuracy", "Human Gut Country Prediction Accuracy", "human_gut_168k_accuracy_opt.svg", ".4f"),
        (
            "speedup_vs_random_forest",
            "Prediction Speedup vs Random Forest (x)",
            "Human Gut Classification-Only Prediction Speedup",
            "human_gut_168k_predict_speedup_vs_rf_opt.svg",
            ".1f",
        ),
    ]:
        values = [row[metric] for row in rows]
        fig, axis = plt.subplots(figsize=(12, 7))
        bars = axis.bar(labels, values, color=colors, width=0.68)
        axis.set_title(title, fontsize=20, pad=16)
        axis.set_xlabel("Method", fontsize=16)
        axis.set_ylabel(ylabel, fontsize=16)
        axis.tick_params(axis="both", labelsize=13)
        axis.tick_params(axis="x", rotation=15)
        axis.bar_label(bars, labels=[format(value, fmt) for value in values], padding=4, fontsize=12)
        axis.set_ylim(0, max(values) * 1.14)
        fig.tight_layout()
        fig.savefig(outdir / filename, format="svg")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.outdir / "cache"
    cache_dir.mkdir(exist_ok=True)
    cp.cuda.Device(args.device).use()

    biom_path = cache_dir / "feature-table.biom"
    extract_qza_member(
        args.data_dir / "gg2-2022.10-cref99.biom.qza",
        "/data/feature-table.biom",
        biom_path,
    )
    log("Loading merged Greengenes2 BIOM and filtering labeled samples")
    sample_ids, observation_ids, counts, labels = load_dataset(args, biom_path)
    classes, encoded = np.unique(labels, return_inverse=True)
    encoded = encoded.astype(np.int32)
    indices = np.arange(len(labels), dtype=np.int32)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=encoded,
    )
    y_train, y_test = encoded[train_idx], encoded[test_idx]
    log(
        f"Using {len(labels)} samples, {len(observation_ids)} features, "
        f"{len(classes)} {args.target} classes; train={len(train_idx)}, test={len(test_idx)}"
    )

    hypervector_cache = cache_dir / "obs_hypervectors.npz"
    if hypervector_cache.exists():
        with np.load(hypervector_cache) as data:
            obs_cols, obs_signs = data["obs_cols"], data["obs_signs"]
    else:
        log("Extracting representative DNA sequences and creating HDC hash table")
        sequences = extract_sequences(
            args.data_dir / "gg2-2022.10-cref99.reps.fna.qza",
            observation_ids,
            cache_dir / "selected_reference_sequences.npy",
        )
        obs_cols, obs_signs = sequence_hypervectors(
            sequences, args.hdc_dim, args.active_dims_per_sequence, args.random_state
        )
        np.savez(hypervector_cache, obs_cols=obs_cols, obs_signs=obs_signs)

    arrays = {
        "indptr": counts.indptr.astype(np.int32, copy=False),
        "obs_idx": counts.indices.astype(np.int32, copy=False),
        "values": counts.data.astype(np.float32, copy=False),
        "obs_cols": obs_cols,
        "obs_signs": obs_signs,
    }
    manifest = {"samples": counts.shape[0], "nnz": counts.nnz}
    log("Building normalized 4096-dimensional HDC matrix on GPU")
    hdc_gpu, device_arrays, hdc_build_times = build_gpu_matrix(args, manifest, arrays)
    hdc_cpu = sparse.csr_matrix(cp.asnumpy(hdc_gpu))
    rows = []

    log("Training Explicit-Vocab and benchmarking prediction")
    tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
    explicit_train = tfidf.fit_transform(counts[train_idx])
    explicit_test = tfidf.transform(counts[test_idx])
    explicit = LinearSVC(C=1.0, dual="auto", max_iter=10000, random_state=args.random_state)
    explicit.fit(explicit_train, y_train)
    prediction, times = cpu_predict_benchmark(explicit, explicit_test, args.repeats)
    rows.append(result_row("Explicit-Vocab", y_test, prediction, times, len(train_idx), len(test_idx)))
    del explicit_train

    log("Training CPU HDC readout and benchmarking HDC prediction paths")
    hdc_model = LinearSVC(C=1.0, dual="auto", max_iter=10000, random_state=args.random_state)
    hdc_model.fit(hdc_cpu[train_idx], y_train)
    hdc_test_cpu = hdc_cpu[test_idx]
    for method in ["HDC-Hash", "GPU-HDC Hash"]:
        prediction, times = cpu_predict_benchmark(hdc_model, hdc_test_cpu, args.repeats)
        rows.append(result_row(method, y_test, prediction, times, len(train_idx), len(test_idx)))

    log("Training Random Forest and benchmarking prediction")
    forest = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=args.random_state,
        n_jobs=-1,
    )
    forest.fit(counts[train_idx], y_train)
    prediction, times = cpu_predict_benchmark(forest, counts[test_idx], args.repeats)
    rows.append(result_row("Random Forest", y_test, prediction, times, len(train_idx), len(test_idx)))

    log("Benchmarking optimized precached GPU HDC readout")
    hdc_test_gpu = hdc_gpu[cp.asarray(test_idx)]
    coefficients = cp.asarray(hdc_model.coef_.astype(np.float32, copy=False))
    intercept = cp.asarray(hdc_model.intercept_.astype(np.float32, copy=False))
    prediction, times = gpu_predict_benchmark(
        hdc_test_gpu, coefficients, intercept, args.repeats
    )
    rows.append(
        result_row(
            "GPU-HDC-CPU-precache_opt",
            y_test,
            prediction,
            times,
            len(train_idx),
            len(test_idx),
        )
    )

    log("Saving pretrained models and fixed test-input caches")
    dump(
        {
            "tfidf": tfidf,
            "explicit": explicit,
            "hdc": hdc_model,
            "forest": forest,
            "classes": classes,
        },
        args.outdir / "pretrained_models.joblib",
        compress=3,
    )
    test_counts = counts[test_idx].tocsr()
    sparse.save_npz(cache_dir / "test_counts.npz", test_counts, compressed=True)
    np.savez(
        cache_dir / "test_labels_and_indices.npz",
        y_test=y_test,
        train_idx=train_idx,
        test_idx=test_idx,
        sample_ids=sample_ids,
    )
    optimized_cache = cache_dir / "pipeline_opt"
    optimized_cache.mkdir(exist_ok=True)
    np.save(optimized_cache / "indptr.npy", test_counts.indptr.astype(np.int32), allow_pickle=False)
    np.save(optimized_cache / "obs_idx.npy", test_counts.indices.astype(np.int32), allow_pickle=False)
    np.save(optimized_cache / "values.npy", test_counts.data.astype(np.float32), allow_pickle=False)
    np.save(optimized_cache / "obs_cols.npy", obs_cols.astype(np.uint16), allow_pickle=False)
    np.save(optimized_cache / "obs_signs.npy", obs_signs.astype(np.int8), allow_pickle=False)
    (optimized_cache / "manifest.json").write_text(
        json.dumps(
            {
                "samples": int(test_counts.shape[0]),
                "observations": int(test_counts.shape[1]),
                "nnz": int(test_counts.nnz),
                "hdc_dim": args.hdc_dim,
                "active_dims_per_sequence": args.active_dims_per_sequence,
            },
            indent=2,
            sort_keys=True,
        )
    )

    order = {method: index for index, method in enumerate(METHODS)}
    rows.sort(key=lambda row: order[row["method"]])
    rf_time = next(row["predict_mean_sec"] for row in rows if row["method"] == "Random Forest")
    for row in rows:
        row["speedup_vs_random_forest"] = rf_time / row["predict_mean_sec"]

    summary_path = args.outdir / "human_gut_168k_predict_summary_opt.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    settings = {
        **vars(args),
        "samples": len(labels),
        "features": len(observation_ids),
        "classes": len(classes),
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "source_table": "gg2-2022.10-cref99.biom.qza",
        "sequence_encoding": "whole Greengenes2 representative sequence -> 4096D HDC",
        "prediction_timing_scope": "pretrained classifier only; representation already resident",
        "gpu_hdc_build_times_sec": hdc_build_times,
    }
    (args.outdir / "benchmark_settings.json").write_text(
        json.dumps(settings, default=str, indent=2, sort_keys=True)
    )
    plot_results(rows, args.outdir)
    del device_arrays
    print(json.dumps(rows, indent=2), flush=True)
    log(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
