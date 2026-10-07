#!/usr/bin/env python3
"""Shared utilities for the final RF, Explicit-Vocab, and HDC benchmarks."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC


METHODS = ("Random Forest", "Explicit-Vocab (SVM)", "HDC-Linear_opt")


def require_file(path: Path, description: str) -> None:
    """Raise a short, actionable error when an input file is missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not found: {path}. Check the path and try again."
        )


def require_columns(columns, required: set[str], description: str) -> None:
    """Validate exact, case-sensitive column names before model work starts."""
    available = set(columns)
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            f"{description} is missing required column(s): {', '.join(missing)}. "
            f"Column names are case-sensitive."
        )


def require_unique(values, description: str) -> None:
    """Reject duplicate IDs where one row must represent one entity."""
    series = np.asarray(values, dtype=object)
    duplicates = [value for value, count in Counter(series.tolist()).items() if count > 1]
    if duplicates:
        preview = ", ".join(map(str, duplicates[:5]))
        suffix = " ..." if len(duplicates) > 5 else ""
        raise ValueError(
            f"{description} contains duplicate ID(s): {preview}{suffix}. "
            "Remove duplicates or aggregate them before running."
        )


def validate_labels(labels: np.ndarray, description: str) -> None:
    """Check that a stratified train/test split is possible."""
    counts = Counter(np.asarray(labels, dtype=str).tolist())
    if len(counts) < 2:
        raise ValueError(
            f"{description} needs at least 2 non-empty classes; found {len(counts)}."
        )
    small = [f"{label!r} ({count})" for label, count in counts.items() if count < 2]
    if small:
        raise ValueError(
            f"{description} needs at least 2 samples per class for the default "
            f"stratified split; too-small class(es): {', '.join(small[:5])}."
        )


def validate_split_size(
    labels: np.ndarray, test_size: float, description: str
) -> None:
    """Check that both split partitions can contain every class."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("--test-size must be greater than 0 and less than 1.")
    class_count = len(set(np.asarray(labels, dtype=str).tolist()))
    test_samples = math.ceil(len(labels) * test_size)
    train_samples = len(labels) - test_samples
    if test_samples < class_count or train_samples < class_count:
        raise ValueError(
            f"{description} has {len(labels)} samples across {class_count} classes, "
            f"but --test-size {test_size} gives {train_samples} train and "
            f"{test_samples} test samples. Both partitions need at least "
            f"{class_count} samples to contain every class. Add samples or "
            f"increase --test-size (for example, try 0.5 for a tiny demo)."
        )


def validate_positive_options(repeats: int, n_estimators: int) -> None:
    if repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if n_estimators < 1:
        raise ValueError("--n-estimators must be at least 1.")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _import_cupy():
    """Import CuPy with the CUDA runtime visible to the active environment."""
    cuda_root = Path(sys.executable).parent.parent
    os.environ.setdefault("CUDA_PATH", str(cuda_root))
    nvrtc_library = cuda_root / "lib" / "libnvrtc.so.12"
    if nvrtc_library.exists():
        import ctypes

        ctypes.CDLL(str(nvrtc_library), mode=ctypes.RTLD_GLOBAL)
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy is installed, but no CUDA device is available.")
    except ImportError as error:
        raise RuntimeError(
            "The GPU cached-input path requires CuPy. Install the optional "
            "GPU dependencies from scripts/requirements-gpu.txt."
        ) from error
    except RuntimeError as error:
        raise RuntimeError(
            "The GPU backend needs CuPy and an available CUDA device. "
            "Use --backend cpu on a CPU-only computer or install the optional "
            "GPU dependencies from scripts/requirements-gpu.txt."
        ) from error
    return cp, gpu_csr_matrix


def gpu_cached_sparse_prediction(
    cache_path: Path,
    classifier,
    classes: np.ndarray,
    repeats: int,
    device: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Predict from a cached sparse matrix using a GPU-resident linear readout.

    The cache open, host-to-device transfer, sparse matrix construction, GPU
    matrix-vector multiply, and device-to-host prediction copy are timed. The
    fitted LinearSVC coefficients are uploaded once before the timed runs,
    matching the EMP cached-input benchmark convention.
    """
    cp, gpu_csr_matrix = _import_cupy()
    cp.cuda.Device(device).use()
    coefficients = cp.asarray(
        classifier.coef_.astype(np.float32, copy=False).T, dtype=cp.float32
    )
    intercept = cp.asarray(
        classifier.intercept_.astype(np.float32, copy=False), dtype=cp.float32
    )
    classes = np.asarray(classes)

    def run_once() -> tuple[np.ndarray, dict[str, float]]:
        started = time.perf_counter()
        host_matrix = sparse.load_npz(cache_path).tocsr()
        data = cp.asarray(host_matrix.data, dtype=cp.float32)
        indices = cp.asarray(host_matrix.indices, dtype=cp.int32)
        indptr = cp.asarray(host_matrix.indptr, dtype=cp.int32)
        matrix = gpu_csr_matrix(
            (data, indices, indptr), shape=host_matrix.shape, dtype=cp.float32
        )
        scores = matrix @ coefficients
        scores += intercept
        if classifier.coef_.shape[0] == 1:
            prediction_indices = (scores[:, 0] > 0).astype(cp.int32)
        else:
            prediction_indices = cp.argmax(scores, axis=1).astype(cp.int32)
        cp.cuda.Stream.null.synchronize()
        prediction = classes[cp.asnumpy(prediction_indices)]
        return prediction, {
            "total_sec": time.perf_counter() - started,
            "cache_bytes": int(cache_path.stat().st_size),
            "nnz": int(host_matrix.nnz),
        }

    run_once()
    times = []
    prediction = None
    metadata = {}
    for _ in range(repeats):
        prediction, run_metadata = run_once()
        times.append(run_metadata["total_sec"])
        metadata = run_metadata
    cp.get_default_memory_pool().free_all_blocks()
    return np.asarray(prediction), np.asarray(times, dtype=np.float64), metadata


def timed_runs(runner, repeats: int):
    prediction = runner()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        prediction = runner()
        times.append(time.perf_counter() - started)
    return np.asarray(prediction), np.asarray(times, dtype=np.float64)


def sequence_projection(
    sequences: list[str] | np.ndarray,
    dimension: int,
    active_dims: int,
    seed: int,
) -> sparse.csr_matrix:
    """Map each ASV sequence to a deterministic sparse bipolar hypervector."""
    rows = np.repeat(np.arange(len(sequences), dtype=np.int32), active_dims)
    columns = np.empty(len(rows), dtype=np.int32)
    values = np.empty(len(rows), dtype=np.float32)
    signs = np.asarray([-1.0, 1.0], dtype=np.float32)
    offset = 0
    for sequence in sequences:
        digest = hashlib.blake2b(
            f"{seed}|asv_sequence:{sequence}".encode(), digest_size=16
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        stop = offset + active_dims
        columns[offset:stop] = rng.choice(
            dimension, size=active_dims, replace=False
        )
        values[offset:stop] = rng.choice(signs, size=active_dims)
        offset = stop
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(sequences), dimension),
        dtype=np.float32,
    )


def extract_reference_sequences(
    qza_path: Path,
    observation_ids: list[str],
    cache_path: Path | None = None,
) -> np.ndarray:
    """Extract Greengenes2 sequences corresponding to BIOM feature IDs."""
    require_file(qza_path, "Reference sequence archive")
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if len(cached) == len(observation_ids):
            return cached

    wanted = set(observation_ids)
    found: dict[str, list[str]] = {}

    def target_id(header: str) -> str | None:
        if header in wanted:
            return header
        match = re.fullmatch(r"(G\d+)_\d+", header)
        if match and match.group(1) in wanted:
            return match.group(1)
        return None

    def store(target: str | None, chunks: list[str]) -> None:
        if target is not None and chunks:
            found.setdefault(target, []).append("".join(chunks))

    with zipfile.ZipFile(qza_path) as archive:
        member = next(
            name for name in archive.namelist()
            if name.endswith("/data/dna-sequences.fasta")
        )
        with archive.open(member) as handle:
            target = None
            chunks: list[str] = []
            for raw_line in handle:
                line = raw_line.decode("ascii").strip()
                if line.startswith(">"):
                    store(target, chunks)
                    target = target_id(line[1:].split()[0])
                    chunks = []
                elif target is not None:
                    chunks.append(line)
            store(target, chunks)

    missing = [feature for feature in observation_ids if feature not in found]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} reference sequences; first IDs: {missing[:5]}"
        )
    sequences = np.asarray(
        ["N".join(sorted(found[feature])) for feature in observation_ids], dtype=str
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, sequences, allow_pickle=False)
    return sequences


def evaluate_three_methods(
    counts: sparse.csr_matrix,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    projection: sparse.csr_matrix,
    hdc_c: float,
    repeats: int,
    random_state: int,
    svc_class_weight=None,
    n_estimators: int = 300,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Train and time the final three models from the same in-memory counts."""
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    train_counts = counts[train_idx]
    test_counts = counts[test_idx]

    train_started = time.perf_counter()
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    ).fit(train_counts, y_train)
    rf_train_sec = time.perf_counter() - train_started

    explicit_tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
    train_started = time.perf_counter()
    explicit_train = explicit_tfidf.fit_transform(train_counts)
    explicit = LinearSVC(
        C=1.0,
        class_weight=svc_class_weight,
        dual="auto",
        max_iter=12000,
        random_state=random_state,
    ).fit(explicit_train, y_train)
    explicit_train_sec = time.perf_counter() - train_started

    hdc_tfidf = TfidfTransformer(norm=None, sublinear_tf=True)
    train_started = time.perf_counter()
    weighted_train = hdc_tfidf.fit_transform(train_counts).tocsr().astype(np.float32)
    hdc_train = (weighted_train @ projection).tocsr()
    normalize(hdc_train, norm="l2", copy=False)
    hdc = LinearSVC(
        C=hdc_c,
        class_weight=svc_class_weight,
        dual="auto",
        max_iter=12000,
        random_state=random_state,
    ).fit(hdc_train, y_train)
    hdc_train_sec = time.perf_counter() - train_started

    def run_rf():
        return forest.predict(test_counts)

    def run_explicit():
        return explicit.predict(explicit_tfidf.transform(test_counts))

    def run_hdc():
        weighted = hdc_tfidf.transform(test_counts).tocsr().astype(np.float32)
        encoded = (weighted @ projection).tocsr()
        normalize(encoded, norm="l2", copy=False)
        return hdc.predict(encoded)

    specifications = [
        ("Random Forest", run_rf, rf_train_sec),
        ("Explicit-Vocab (SVM)", run_explicit, explicit_train_sec),
        ("HDC-Linear_opt", run_hdc, hdc_train_sec),
    ]
    rows = []
    predictions = {}
    for method, runner, train_sec in specifications:
        prediction, times = timed_runs(runner, repeats)
        predictions[method] = prediction
        rows.append({
            "method": method,
            "accuracy": float(accuracy_score(y_test, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "train_sec": train_sec,
            "warmup_runs": 1,
            "timed_runs": repeats,
            "prediction_pipeline_mean_sec": float(times.mean()),
            "prediction_pipeline_std_sec": float(times.std()),
            "execution": "CPU in-memory prediction",
            "timing_scope": "in-memory feature transform + CPU prediction",
        })

    rf_time = rows[0]["prediction_pipeline_mean_sec"]
    for row in rows:
        row["speedup_vs_random_forest"] = (
            rf_time / row["prediction_pipeline_mean_sec"]
        )
    return rows, predictions
