#!/usr/bin/env python3
"""GPU HDC feature-build benchmark for EMP 16S EMPO classification."""

from __future__ import annotations

import argparse
import ctypes
import csv
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from benchmark_emp_16s_empo import DEFAULT_BIOM, DEFAULT_METADATA, EMPO_COLUMNS, clean_label, load_emp_dataset, log


DEFAULT_OUTDIR = Path("/home/tsl012/Multi_variant classification/emp_16s_gpu_hdc")
CUDA_SOURCE = Path("/home/tsl012/Multi_variant classification/cuda_emp_hdc_build.cu")


def compile_cuda(source: Path, output: Path) -> None:
    if output.exists() and output.stat().st_mtime >= source.stat().st_mtime:
        return
    subprocess.run(
        ["nvcc", "-O3", "-Xcompiler", "-fPIC", "--shared", str(source), "-o", str(output)],
        check=True,
    )


def load_library(path: Path):
    lib = ctypes.CDLL(str(path))
    int_p = ctypes.POINTER(ctypes.c_int)
    float_p = ctypes.POINTER(ctypes.c_float)
    lib.build_emp_hdc_matrix.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        int_p,
        int_p,
        float_p,
        int_p,
        float_p,
        float_p,
        float_p,
    ]
    lib.build_emp_hdc_matrix.restype = ctypes.c_int
    return lib


def obs_hv(sequence: str, dim: int, active_dims: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    digest = hashlib.blake2b(f"{seed}|asv_sequence:{sequence}".encode(), digest_size=16).digest()
    token_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(token_seed)
    cols = rng.choice(dim, size=active_dims, replace=False).astype(np.int32)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=active_dims)
    return cols, signs


def build_obs_hvs(obs_ids: list[str], dim: int, active_dims: int, seed: int):
    cols = np.empty((len(obs_ids), active_dims), dtype=np.int32)
    signs = np.empty((len(obs_ids), active_dims), dtype=np.float32)
    for i, obs in enumerate(obs_ids):
        c, s = obs_hv(obs, dim=dim, active_dims=active_dims, seed=seed)
        cols[i] = c
        signs[i] = s
    return cols.ravel(), signs.ravel()


def csr_nnz_arrays(x_counts: sparse.csr_matrix):
    counts = np.diff(x_counts.indptr).astype(np.int64)
    sample_idx = np.repeat(np.arange(x_counts.shape[0], dtype=np.int32), counts)
    obs_idx = x_counts.indices.astype(np.int32, copy=False)
    values = x_counts.data.astype(np.float32, copy=False)
    return sample_idx, obs_idx, values


def gpu_build_hdc(lib, args, x_counts: sparse.csr_matrix, obs_ids: list[str]):
    started = time.perf_counter()
    obs_cols, obs_signs = build_obs_hvs(
        obs_ids,
        dim=args.hdc_dim,
        active_dims=args.active_dims_per_sequence,
        seed=args.random_state,
    )
    obs_hv_sec = time.perf_counter() - started

    started = time.perf_counter()
    sample_idx, obs_idx, values = csr_nnz_arrays(x_counts)
    nnz_array_sec = time.perf_counter() - started

    matrix = np.zeros((x_counts.shape[0], args.hdc_dim), dtype=np.float32)
    elapsed_ms = ctypes.c_float(0.0)
    started = time.perf_counter()
    err = lib.build_emp_hdc_matrix(
        ctypes.c_int(args.device),
        ctypes.c_int(x_counts.shape[0]),
        ctypes.c_int(args.hdc_dim),
        ctypes.c_int(x_counts.nnz),
        ctypes.c_int(len(obs_ids)),
        ctypes.c_int(args.active_dims_per_sequence),
        sample_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        obs_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        obs_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        obs_signs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(elapsed_ms),
    )
    gpu_call_wall_sec = time.perf_counter() - started
    if err != 0:
        raise RuntimeError(f"CUDA build_emp_hdc_matrix failed with error code {err}")

    started = time.perf_counter()
    x_hdc = sparse.csr_matrix(matrix)
    dense_to_sparse_sec = time.perf_counter() - started
    return x_hdc, {
        "obs_hv_cpu_prep_sec": obs_hv_sec,
        "csr_nnz_array_cpu_prep_sec": nnz_array_sec,
        "gpu_call_wall_sec_including_transfers": gpu_call_wall_sec,
        "gpu_cuda_event_ms_including_transfers": float(elapsed_ms.value),
        "dense_to_sparse_sec": dense_to_sparse_sec,
    }


def split_for_label(labels: list[str], test_size: float, random_state: int):
    counts = Counter(labels)
    stratify = labels if min(counts.values()) >= 2 else None
    idx = np.arange(len(labels))
    return train_test_split(idx, test_size=test_size, random_state=random_state, stratify=stratify)


def train_eval_hdc(x_hdc, labels, train_idx, test_idx, random_state: int):
    model = Pipeline(
        [
            ("normalize", Normalizer(norm="l2")),
            ("svc", LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=random_state)),
        ]
    )
    started = time.perf_counter()
    model.fit(x_hdc[train_idx], [labels[i] for i in train_idx])
    train_sec = time.perf_counter() - started
    started = time.perf_counter()
    pred = model.predict(x_hdc[test_idx])
    predict_sec = time.perf_counter() - started
    y_true = [labels[i] for i in test_idx]
    return y_true, pred, train_sec, predict_sec


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biom", type=Path, default=DEFAULT_BIOM)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--min-sample-sum", type=float, default=1000.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hdc-dim", type=int, default=4096)
    parser.add_argument("--active-dims-per-sequence", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    so_path = args.outdir / "cuda_emp_hdc_build.so"

    log("Compiling/loading CUDA EMP HDC kernel")
    compile_cuda(CUDA_SOURCE, so_path)
    lib = load_library(so_path)

    log("Loading and filtering EMP BIOM + metadata")
    sample_ids, obs_ids, x_counts, metadata, stats = load_emp_dataset(args)
    log(f"Loaded filtered matrix {x_counts.shape[0]} samples x {x_counts.shape[1]} ASVs")

    log(f"Building GPU HDC matrix dim={args.hdc_dim}, active_dims={args.active_dims_per_sequence}")
    x_hdc, build_stats = gpu_build_hdc(lib, args, x_counts, obs_ids)
    log(
        "GPU HDC build done: "
        f"gpu_call={build_stats['gpu_call_wall_sec_including_transfers']:.3f}s, "
        f"dense_to_sparse={build_stats['dense_to_sparse_sec']:.3f}s"
    )

    rows = []
    for label_col in EMPO_COLUMNS:
        log(f"Running {label_col}")
        labels = []
        valid_idx = []
        for i, value in enumerate(metadata[label_col].tolist()):
            label = clean_label(value)
            if label:
                valid_idx.append(i)
                labels.append(label)
        counts = Counter(labels)
        if len(counts) < 2:
            rows.append({"empo_level": label_col, "status": "skipped_one_class", "samples": len(labels), "classes": len(counts)})
            continue

        valid_idx_np = np.asarray(valid_idx, dtype=np.int32)
        train_rel, test_rel = split_for_label(labels, args.test_size, args.random_state)
        y_true, pred, train_sec, predict_sec = train_eval_hdc(
            x_hdc[valid_idx_np],
            labels,
            train_rel,
            test_rel,
            args.random_state,
        )
        acc = accuracy_score(y_true, pred)
        level_dir = args.outdir / label_col
        level_dir.mkdir(exist_ok=True)
        (level_dir / "gpu_hdc_classification_report.txt").write_text(
            classification_report(y_true, pred, zero_division=0)
        )
        rows.append(
            {
                "empo_level": label_col,
                "status": "ok",
                "samples": len(labels),
                "classes": len(counts),
                "accuracy": acc,
                "train_sec": train_sec,
                "predict_sec": predict_sec,
                "label_counts": json.dumps(dict(counts.most_common()), sort_keys=True),
            }
        )
        log(f"{label_col} GPU-HDC done: acc={acc:.4f}, train={train_sec:.2f}s, predict={predict_sec:.3f}s")

    stats.update(build_stats)
    stats.update(
        {
            "hdc_dim": args.hdc_dim,
            "active_dims_per_sequence": args.active_dims_per_sequence,
            "hdc_matrix_shape": list(x_hdc.shape),
            "hdc_matrix_nnz": int(x_hdc.nnz),
            "total_script_wall_sec": time.perf_counter() - total_started,
            "device": args.device,
        }
    )
    (args.outdir / "gpu_hdc_dataset_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))

    fieldnames = sorted({k for row in rows for k in row})
    preferred = ["empo_level", "status", "samples", "classes", "accuracy", "train_sec", "predict_sec", "label_counts"]
    fieldnames = [f for f in preferred if f in fieldnames] + [f for f in fieldnames if f not in preferred]
    with (args.outdir / "gpu_hdc_empo_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log(f"Wrote {args.outdir / 'gpu_hdc_empo_summary.csv'}")
    log(f"Wrote {args.outdir / 'gpu_hdc_dataset_stats.json'}")


if __name__ == "__main__":
    main()
