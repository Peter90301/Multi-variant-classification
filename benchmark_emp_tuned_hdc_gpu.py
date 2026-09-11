#!/usr/bin/env python3
"""Benchmark validation-selected EMP HDC encodings on a GPU-resident readout."""

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import os
import sys
import time
from pathlib import Path

CUDA_ROOT = Path(sys.executable).parent.parent
os.environ.setdefault("CUDA_PATH", str(CUDA_ROOT))
nvrtc_library = CUDA_ROOT / "lib" / "libnvrtc.so.12"
if nvrtc_library.exists():
    ctypes.CDLL(str(nvrtc_library), mode=ctypes.RTLD_GLOBAL)

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from benchmark_emp_16s_empo import (
    DEFAULT_BIOM,
    DEFAULT_METADATA,
    clean_label,
    load_emp_dataset,
    split_for_label,
)
from benchmark_human_gut_168k_five_methods import build_gpu_matrix, gpu_linear_predict


ROOT = Path("/home/tsl012/Multi_variant classification")
TUNING_DIR = ROOT / "emp_16s_hdc_encoding_tuning"
OUTDIR = ROOT / "emp_16s_tuned_hdc_gpu_benchmark"
LEVELS = ["empo_1", "empo_2", "empo_3"]

if not hasattr(np, "Inf"):
    np.Inf = np.inf


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biom", type=Path, default=DEFAULT_BIOM)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--tuning-dir", type=Path, default=TUNING_DIR)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--min-sample-sum", type=float, default=1000.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rf-estimators", type=int, default=300)
    return parser.parse_args()


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_cache(path, weighted_test, projection, dimension, active):
    path.mkdir(parents=True, exist_ok=True)
    row_lengths = np.diff(projection.indptr)
    if not np.all(row_lengths == active):
        raise ValueError("GPU fixed-active kernel does not match projection row lengths")
    arrays = {
        "indptr": weighted_test.indptr.astype(np.int32, copy=False),
        "obs_idx": weighted_test.indices.astype(np.int32, copy=False),
        "values": weighted_test.data.astype(np.float32, copy=False),
        "obs_cols": projection.indices.astype(np.uint16, copy=False),
        "obs_signs": projection.data.astype(np.int8, copy=False),
    }
    for key, value in arrays.items():
        np.save(path / f"{key}.npy", value, allow_pickle=False)
    manifest = {
        "samples": int(weighted_test.shape[0]),
        "observations": int(weighted_test.shape[1]),
        "nnz": int(weighted_test.nnz),
        "hdc_dim": int(dimension),
        "active_dims_per_sequence": int(active),
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def open_cache(path):
    started = time.perf_counter()
    manifest = json.loads((path / "manifest.json").read_text())
    arrays = {
        key: np.load(path / f"{key}.npy", mmap_mode="r", allow_pickle=False)
        for key in ["indptr", "obs_idx", "values", "obs_cols", "obs_signs"]
    }
    return manifest, arrays, time.perf_counter() - started


def timed_gpu_readout(matrix, coefficients, intercept, repeats):
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


def gpu_pipeline_once(cache_path, gpu_args, coefficients, intercept):
    total_started = time.perf_counter()
    manifest, arrays, cache_sec = open_cache(cache_path)
    matrix, device_arrays, build_times = build_gpu_matrix(gpu_args, manifest, arrays)
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    prediction_gpu = gpu_linear_predict(matrix, coefficients, intercept)
    cp.cuda.Stream.null.synchronize()
    prediction = cp.asnumpy(prediction_gpu).astype(np.int32)
    predict_sec = time.perf_counter() - started
    result = {
        "cache_load_sec": cache_sec,
        "h2d_alloc_sec": build_times["h2d_alloc_sec"],
        "gpu_hdc_sec": build_times["gpu_hdc_sec"],
        "gpu_normalize_sec": build_times["gpu_normalize_sec"],
        "predict_sec": predict_sec,
        "total_sec": time.perf_counter() - total_started,
    }
    del matrix, device_arrays, prediction_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return prediction, result


def timed_cpu_predict(model, matrix, repeats):
    model.predict(matrix)
    times = []
    prediction = None
    for _ in range(repeats):
        started = time.perf_counter()
        prediction = model.predict(matrix)
        times.append(time.perf_counter() - started)
    return np.asarray(prediction), np.asarray(times)


def rf_pipeline_once(cache_path, model):
    started = time.perf_counter()
    matrix = sparse.load_npz(cache_path).tocsr()
    cache_sec = time.perf_counter() - started
    predict_started = time.perf_counter()
    prediction = model.predict(matrix)
    predict_sec = time.perf_counter() - predict_started
    return prediction, {
        "cache_load_sec": cache_sec,
        "h2d_alloc_sec": 0.0,
        "gpu_hdc_sec": 0.0,
        "gpu_normalize_sec": 0.0,
        "predict_sec": predict_sec,
        "total_sec": time.perf_counter() - started,
    }


def plot_speedup(rows, key, title, output):
    values = [row[key] for row in rows]
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = axis.bar([row["level"].replace("_", "") for row in rows], values,
                    color=["#2563eb", "#f97316", "#16a34a"])
    axis.bar_label(bars, labels=[f"{value:.2f}x" for value in values], padding=5, fontsize=13)
    axis.set_title(title, fontsize=21)
    axis.set_xlabel("Classification level", fontsize=15)
    axis.set_ylabel("Speedup vs Random Forest (x)", fontsize=15)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#dbe3ef", linewidth=0.8)
    axis.set_axisbelow(True)
    fig.savefig(output)
    plt.close(fig)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cp.cuda.Device(args.device).use()
    selected = pd.read_csv(args.tuning_dir / "untouched_test_results.csv").set_index("level")
    log("Loading EMP data")
    sample_ids, _, counts, metadata, dataset_stats = load_emp_dataset(args)
    sample_ids = np.asarray(sample_ids)
    summary_rows = []
    stage_rows = []

    for level in LEVELS:
        selected_row = selected.loc[level.upper()]
        dimension = int(selected_row["dimension"])
        active = int(selected_row["active_sequence"])
        c_value = float(selected_row["c_value"])
        projection_name = f"whole_sequence_d{dimension}_as{active}.npz"
        projection = sparse.load_npz(
            args.tuning_dir / "projection_cache" / projection_name
        ).tocsr()

        valid_idx, labels = [], []
        for index, value in enumerate(metadata[level].tolist()):
            label = clean_label(value)
            if label:
                valid_idx.append(index)
                labels.append(label)
        valid_idx = np.asarray(valid_idx, dtype=np.int32)
        labels = np.asarray(labels)
        train_rel, test_rel = split_for_label(
            sample_ids[valid_idx].tolist(), labels.tolist(),
            args.test_size, args.random_state,
        )
        level_counts = counts[valid_idx]
        train_counts, test_counts = level_counts[train_rel], level_counts[test_rel]
        y_train, y_test = labels[train_rel], labels[test_rel]

        transformer = TfidfTransformer(norm=None, sublinear_tf=True)
        transformer.fit(train_counts)
        weighted_train = transformer.transform(train_counts).tocsr().astype(np.float32)
        weighted_test = transformer.transform(test_counts).tocsr().astype(np.float32)
        hdc_train = (weighted_train @ projection).tocsr()
        hdc_test = (weighted_test @ projection).tocsr()
        normalize(hdc_train, norm="l2", copy=False)
        normalize(hdc_test, norm="l2", copy=False)
        model = LinearSVC(C=c_value, dual="auto", max_iter=12000, random_state=42)
        model.fit(hdc_train, y_train)
        cpu_prediction = model.predict(hdc_test)
        cpu_accuracy = accuracy_score(y_test, cpu_prediction)

        forest = RandomForestClassifier(
            n_estimators=args.rf_estimators, max_features="sqrt",
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        )
        forest.fit(train_counts, y_train)
        rf_prediction, rf_class_times = timed_cpu_predict(forest, test_counts, args.repeats)

        level_dir = args.outdir / level
        cache_dir = level_dir / "gpu_precache"
        level_dir.mkdir(parents=True, exist_ok=True)
        save_cache(cache_dir, weighted_test, projection, dimension, active)
        rf_cache = level_dir / "rf_test_counts.npz"
        sparse.save_npz(rf_cache, test_counts, compressed=True)

        manifest, arrays, _ = open_cache(cache_dir)
        gpu_args = argparse.Namespace(
            hdc_dim=dimension, active_dims_per_sequence=active,
        )
        gpu_matrix, resident_arrays, _ = build_gpu_matrix(gpu_args, manifest, arrays)
        coefficients = cp.asarray(model.coef_.astype(np.float32, copy=False))
        intercept = cp.asarray(model.intercept_.astype(np.float32, copy=False))
        gpu_indices, gpu_class_times = timed_gpu_readout(
            gpu_matrix, coefficients, intercept, args.repeats
        )
        gpu_prediction = model.classes_[gpu_indices]
        gpu_accuracy = accuracy_score(y_test, gpu_prediction)
        prediction_match = float(np.mean(cpu_prediction == gpu_prediction))

        gpu_pipeline_once(cache_dir, gpu_args, coefficients, intercept)
        gpu_runs = []
        for run in range(1, args.repeats + 1):
            prediction_indices, timings = gpu_pipeline_once(
                cache_dir, gpu_args, coefficients, intercept
            )
            gpu_runs.append(timings)
            stage_rows.append({"level": level.upper(), "method": "GPU-HDC-tuned", "run": run, **timings})
        rf_pipeline_once(rf_cache, forest)
        rf_runs = []
        for run in range(1, args.repeats + 1):
            _, timings = rf_pipeline_once(rf_cache, forest)
            rf_runs.append(timings)
            stage_rows.append({"level": level.upper(), "method": "Random Forest", "run": run, **timings})

        gpu_total = np.asarray([run["total_sec"] for run in gpu_runs])
        rf_total = np.asarray([run["total_sec"] for run in rf_runs])
        row = {
            "level": level.upper(),
            "dimension": dimension,
            "active_dims": active,
            "c_value": c_value,
            "test_samples": len(test_rel),
            "cpu_accuracy": cpu_accuracy,
            "gpu_accuracy": gpu_accuracy,
            "cpu_gpu_prediction_match": prediction_match,
            "warmup_runs": 1,
            "timed_runs": args.repeats,
            "gpu_classification_mean_sec": gpu_class_times.mean(),
            "rf_classification_mean_sec": rf_class_times.mean(),
            "classification_speedup_vs_rf": rf_class_times.mean() / gpu_class_times.mean(),
            "gpu_pipeline_mean_sec": gpu_total.mean(),
            "gpu_pipeline_std_sec": gpu_total.std(),
            "rf_pipeline_mean_sec": rf_total.mean(),
            "full_pipeline_speedup_vs_rf": rf_total.mean() / gpu_total.mean(),
            "gpu_pipeline_samples_per_sec": len(test_rel) / gpu_total.mean(),
        }
        summary_rows.append(row)
        log(f"{level.upper()}: accuracy={gpu_accuracy:.4f}, parity={prediction_match:.6f}, "
            f"classification={row['classification_speedup_vs_rf']:.2f}x, "
            f"pipeline={row['full_pipeline_speedup_vs_rf']:.2f}x")
        pd.DataFrame({
            "truth": y_test,
            "cpu_prediction": cpu_prediction,
            "gpu_prediction": gpu_prediction,
        }).to_csv(level_dir / "prediction_parity.csv", index=False)
        del gpu_matrix, resident_arrays, coefficients, intercept
        cp.get_default_memory_pool().free_all_blocks()

    write_rows(args.outdir / "gpu_tuned_hdc_summary.csv", summary_rows)
    write_rows(args.outdir / "pipeline_stage_runs.csv", stage_rows)
    plot_speedup(
        summary_rows, "classification_speedup_vs_rf",
        "Tuned GPU-HDC Classification-Only Speedup",
        args.outdir / "classification_only_speedup.svg",
    )
    plot_speedup(
        summary_rows, "full_pipeline_speedup_vs_rf",
        "Tuned GPU-HDC Full Prediction Pipeline Speedup",
        args.outdir / "full_pipeline_speedup.svg",
    )
    (args.outdir / "settings.json").write_text(json.dumps({
        **vars(args), **dataset_stats,
        "weighting": "sublinear TF-IDF fit on training samples only",
        "classification_scope": "GPU-resident HDC matrix and linear readout",
        "pipeline_scope": "mmap cache open, H2D, HDC construction, normalization, and prediction",
        "rf_pipeline_scope": "compressed test CSR load and RF prediction",
    }, default=str, indent=2, sort_keys=True))
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
