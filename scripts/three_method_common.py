#!/usr/bin/env python3
"""Shared utilities for the final RF, Explicit-Vocab, and HDC benchmarks."""

from __future__ import annotations

import csv
import hashlib
import re
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC


METHODS = ("Random Forest", "Explicit-Vocab (SVM)", "HDC-Linear_opt")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
        })

    rf_time = rows[0]["prediction_pipeline_mean_sec"]
    for row in rows:
        row["speedup_vs_random_forest"] = (
            rf_time / row["prediction_pipeline_mean_sec"]
        )
    return rows, predictions
