#!/usr/bin/env python3
"""Run the final three classifiers on Marine eDNA geo_loc_name."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from scipy import sparse

from three_method_common import (
    gpu_cached_sparse_prediction,
    require_file,
    timed_runs,
    validate_labels,
    validate_positive_options,
    write_rows,
)


TAXONOMY_FIELDS = [
    "domain", "phylum", "class", "order", "family", "genus", "species"
]
MISSING = {"", "NA", "N/A", "nan", "None", "null"}
DNA_RE = re.compile(r"[^ACGTN]")


def clean(value: str | None) -> str:
    text = "" if value is None else value.strip()
    return "" if text in MISSING else text


def row_features(row: dict[str, str], kmer_size: int, max_weight: int):
    features = Counter()
    lineage = []
    for field in TAXONOMY_FIELDS:
        value = clean(row.get(field))
        if value:
            token = re.sub(r"\s+", "_", value.lower())
            features[f"{field}={token}"] += 1
            lineage.append(token)
    if lineage:
        features["lineage=" + "|".join(lineage)] += 1
    sequence = DNA_RE.sub("", clean(row.get("ASV_sequence")).upper())
    for index in range(max(0, len(sequence) - kmer_size + 1)):
        features[f"k{kmer_size}_{sequence[index:index + kmer_size]}"] += 1
    try:
        count = float(clean(row.get("count")) or 1.0)
    except ValueError:
        count = 1.0
    weight = max(1, min(max_weight, int(math.log10(max(count, 1.0))) + 1))
    return {key: value * weight for key, value in features.items()}


def load_samples(path: Path, kmer_size: int, max_weight: int):
    require_file(path, "Marine CSV")
    features = defaultdict(Counter)
    labels = defaultdict(Counter)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample", "geo_loc_name", "ASV_sequence", *TAXONOMY_FIELDS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        for row in reader:
            sample = clean(row.get("sample"))
            label = clean(row.get("geo_loc_name"))
            if not sample or not label:
                continue
            features[sample].update(row_features(row, kmer_size, max_weight))
            labels[sample][label] += 1
    if not features:
        raise ValueError(
            "Marine CSV contains no usable samples. Check sample and "
            "geo_loc_name values."
        )
    sample_ids = list(features)
    conflicting = {
        sample: sorted(values) for sample, values in labels.items() if len(values) > 1
    }
    if conflicting:
        sample, values = next(iter(conflicting.items()))
        raise ValueError(
            f"Sample {sample!r} has multiple geo_loc_name labels: {values}. "
            "Each sample must have one target label."
        )
    x = [dict(features[sample]) for sample in sample_ids]
    y = np.asarray([labels[sample].most_common(1)[0][0] for sample in sample_ids])
    return np.asarray(sample_ids), x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--kmer-size", type=int, default=6)
    parser.add_argument("--max-count-weight", type=int, default=5)
    parser.add_argument("--hdc-dim", type=int, default=32768)
    parser.add_argument("--hdc-c", type=float, default=16.0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--backend", choices=("gpu", "cpu"), default="gpu",
        help="Prediction backend for HDC-Linear_opt (default: gpu).",
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    validate_positive_options(args.repeats, args.n_estimators)

    sample_ids, features, labels = load_samples(
        args.csv, args.kmer_size, args.max_count_weight
    )
    validate_labels(labels, "Marine geo_loc_name")
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.random_state,
        stratify=labels,
    )
    x_train = [features[index] for index in train_idx]
    x_test = [features[index] for index in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    models = [
        ("Random Forest", Pipeline([
            ("vectorizer", DictVectorizer(sparse=True)),
            ("classifier", RandomForestClassifier(
                n_estimators=args.n_estimators, max_features="sqrt",
                class_weight="balanced_subsample", random_state=args.random_state,
                n_jobs=-1,
            )),
        ])),
        ("Explicit-Vocab (SVM)", Pipeline([
            ("vectorizer", DictVectorizer(sparse=True)),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
            ("classifier", LinearSVC(
                class_weight="balanced", random_state=args.random_state,
            )),
        ])),
        ("HDC-Linear_opt", Pipeline([
            ("hasher", FeatureHasher(
                n_features=args.hdc_dim, input_type="dict", alternate_sign=False,
            )),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
            ("classifier", LinearSVC(
                C=args.hdc_c, class_weight="balanced", dual="auto",
                max_iter=12000, random_state=args.random_state,
            )),
        ])),
    ]

    rows = []
    predictions = {}
    for method, model in models:
        started = time.perf_counter()
        model.fit(x_train, y_train)
        train_sec = time.perf_counter() - started
        cache_path = ""
        execution = "CPU in-memory prediction"
        cache_bytes = 0
        cache_nnz = 0
        if method == "HDC-Linear_opt":
            if args.backend == "gpu":
                # Build the exact CPU-trained HDC representation once, then use it
                # as the cached input for the GPU prediction benchmark.
                hashed_test = model.named_steps["hasher"].transform(x_test)
                hdc_test = model.named_steps["tfidf"].transform(hashed_test).tocsr()
                cache_file = args.outdir / "hdc_gpu_cached_input.npz"
                sparse.save_npz(cache_file, hdc_test, compressed=True)
                prediction, times, gpu_metadata = gpu_cached_sparse_prediction(
                    cache_file,
                    model.named_steps["classifier"],
                    model.classes_,
                    args.repeats,
                    args.device,
                )
                cache_path = str(cache_file)
                cache_bytes = gpu_metadata["cache_bytes"]
                cache_nnz = gpu_metadata["nnz"]
                execution = "GPU cached-input pipeline"
            else:
                prediction, times = timed_runs(lambda: model.predict(x_test), args.repeats)
                execution = "CPU in-memory prediction"
        else:
            prediction, times = timed_runs(lambda: model.predict(x_test), args.repeats)
        predictions[method] = prediction
        rows.append({
            "method": method,
            "accuracy": float(accuracy_score(y_test, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
            "samples": len(labels),
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "train_sec": train_sec,
            "prediction_pipeline_mean_sec": float(times.mean()),
            "prediction_pipeline_std_sec": float(times.std()),
            "speedup_vs_random_forest": 0.0,
            "execution": execution,
            "timing_scope": (
                "cache load + H2D + GPU sparse LinearSVC readout"
                if method == "HDC-Linear_opt" and args.backend == "gpu"
                else "in-memory feature transform + CPU prediction"
            ),
            "cache_path": cache_path,
            "cache_bytes": cache_bytes,
            "cache_nnz": cache_nnz,
            "warmup_runs": 1,
            "timed_runs": args.repeats,
            "gpu_device": (
                args.device
                if method == "HDC-Linear_opt" and args.backend == "gpu"
                else ""
            ),
        })
    rf_time = rows[0]["prediction_pipeline_mean_sec"]
    for row in rows:
        row["speedup_vs_random_forest"] = rf_time / row["prediction_pipeline_mean_sec"]

    write_rows(args.outdir / "results.csv", rows)
    with (args.outdir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "truth", *predictions])
        for offset, index in enumerate(test_idx):
            writer.writerow([
                sample_ids[index], y_test[offset],
                *(predictions[method][offset] for method in predictions),
            ])
    settings = {
        **vars(args),
        "samples": len(labels),
        "classes": len(np.unique(labels)),
        "evaluation": "stratified random 80/20 split",
        "count_behavior": "optional; missing or invalid count defaults to weight 1",
    }
    (args.outdir / "settings.json").write_text(
        json.dumps(settings, default=str, indent=2)
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error
