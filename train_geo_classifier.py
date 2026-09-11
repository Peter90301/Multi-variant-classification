#!/usr/bin/env python3
"""Train a sample-grouped geo_loc_name classifier from all_voyages_NEW.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DEFAULT_CSV = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/Multi_variant_classification/all_voyages_NEW.csv"
)
DEFAULT_OUTDIR = Path("/home/tsl012/Multi_variant classification/output")

TAXONOMY_FIELDS = [
    "domain",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]

MISSING_VALUES = {"", "NA", "N/A", "nan", "None", "null"}
DNA_RE = re.compile(r"[^ACGTN]")


def clean_value(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip()
    if value in MISSING_VALUES:
        return ""
    return value


def normalize_token(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip().lower())


def sequence_kmers(sequence: str, k: int) -> list[str]:
    sequence = DNA_RE.sub("", sequence.upper())
    if len(sequence) < k:
        return []
    return [f"k{k}_{sequence[i:i + k]}" for i in range(len(sequence) - k + 1)]


def row_weight(row: dict[str, str], max_weight: int) -> int:
    try:
        count = float(clean_value(row.get("count")) or 1.0)
    except ValueError:
        count = 1.0
    if count <= 0:
        return 1
    return max(1, min(max_weight, int(math.log10(count)) + 1))


def row_features(row: dict[str, str], kmer_size: int, max_weight: int) -> Counter[str]:
    features: Counter[str] = Counter()
    lineage: list[str] = []

    for field in TAXONOMY_FIELDS:
        value = clean_value(row.get(field))
        if not value:
            continue
        token_value = normalize_token(value)
        features[f"{field}={token_value}"] += 1
        lineage.append(token_value)

    if lineage:
        features["lineage=" + "|".join(lineage)] += 1

    sequence = clean_value(row.get("ASV_sequence"))
    if sequence:
        features.update(sequence_kmers(sequence, kmer_size))

    weight = row_weight(row, max_weight)
    for key in list(features):
        features[key] *= weight
    return features


def build_sample_dataset(
    csv_path: Path,
    kmer_size: int,
    max_weight: int,
    min_rows_per_sample: int,
) -> tuple[list[str], list[dict[str, int]], list[str], dict[str, object]]:
    sample_features: dict[str, Counter[str]] = defaultdict(Counter)
    sample_labels: dict[str, Counter[str]] = defaultdict(Counter)
    sample_rows: Counter[str] = Counter()
    total_rows = 0
    skipped_no_sample = 0
    skipped_no_label = 0

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample", "geo_loc_name", "ASV_sequence", *TAXONOMY_FIELDS}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {missing}")

        for row in reader:
            total_rows += 1
            sample = clean_value(row.get("sample"))
            if not sample:
                skipped_no_sample += 1
                continue

            label = clean_value(row.get("geo_loc_name"))
            if not label:
                skipped_no_label += 1
                continue

            features = row_features(row, kmer_size=kmer_size, max_weight=max_weight)
            if not features:
                continue

            sample_rows[sample] += 1
            sample_labels[sample][label] += 1
            sample_features[sample].update(features)

    sample_ids: list[str] = []
    feature_dicts: list[dict[str, int]] = []
    labels: list[str] = []

    for sample, features in sample_features.items():
        if sample_rows[sample] < min_rows_per_sample:
            continue
        if not sample_labels[sample]:
            continue
        sample_ids.append(sample)
        feature_dicts.append(dict(features))
        labels.append(sample_labels[sample].most_common(1)[0][0])

    stats = {
        "source_csv": str(csv_path),
        "total_rows": total_rows,
        "skipped_no_sample": skipped_no_sample,
        "skipped_no_label": skipped_no_label,
        "samples_before_filter": len(sample_features),
        "samples_after_filter": len(sample_ids),
        "label_counts": dict(Counter(labels).most_common()),
        "taxonomy_fields": TAXONOMY_FIELDS,
        "sequence_feature": "ASV_sequence",
        "group_column": "sample",
        "label_column": "geo_loc_name",
        "kmer_size": kmer_size,
        "max_count_weight": max_weight,
        "min_rows_per_sample": min_rows_per_sample,
    }
    return sample_ids, feature_dicts, labels, stats


def make_split(
    sample_ids: list[str],
    feature_dicts: list[dict[str, int]],
    labels: list[str],
    test_size: float,
    random_state: int,
):
    label_counts = Counter(labels)
    stratify = labels if min(label_counts.values()) >= 2 else None
    return train_test_split(
        sample_ids,
        feature_dicts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def save_predictions(path: Path, sample_ids: list[str], y_true: list[str], y_pred: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "true_geo_loc_name", "pred_geo_loc_name", "correct"])
        for sample, true, pred in zip(sample_ids, y_true, y_pred):
            writer.writerow([sample, true, pred, true == pred])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a classifier that predicts geo_loc_name from sample-grouped taxonomy and ASV_sequence."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to all_voyages_NEW.csv.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Directory for model and reports.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of samples used for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--kmer-size", type=int, default=6, help="DNA k-mer size for ASV_sequence.")
    parser.add_argument("--max-count-weight", type=int, default=5, help="Maximum token repetition from log10(count).")
    parser.add_argument("--min-rows-per-sample", type=int, default=1, help="Drop samples with fewer rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    sample_ids, feature_dicts, labels, stats = build_sample_dataset(
        args.csv,
        kmer_size=args.kmer_size,
        max_weight=args.max_count_weight,
        min_rows_per_sample=args.min_rows_per_sample,
    )
    if len(set(labels)) < 2:
        raise ValueError("Need at least two geo_loc_name classes after filtering.")

    train_ids, test_ids, x_train, x_test, y_train, y_test = make_split(
        sample_ids, feature_dicts, labels, args.test_size, args.random_state
    )

    model = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
            ("classifier", LinearSVC(class_weight="balanced", random_state=args.random_state)),
        ]
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    labels_sorted = sorted(set(labels))
    stats.update(
        {
            "train_samples": len(train_ids),
            "test_samples": len(test_ids),
            "accuracy": accuracy_score(y_test, y_pred),
            "model": "DictVectorizer + TfidfTransformer + LinearSVC(class_weight='balanced')",
        }
    )

    joblib.dump(
        {
            "model": model,
            "taxonomy_fields": TAXONOMY_FIELDS,
            "kmer_size": args.kmer_size,
            "max_count_weight": args.max_count_weight,
            "stats": stats,
        },
        args.outdir / "geo_loc_name_model.joblib",
    )

    (args.outdir / "metrics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    (args.outdir / "classification_report.txt").write_text(
        classification_report(y_test, y_pred, labels=labels_sorted, zero_division=0)
    )
    save_predictions(args.outdir / "test_predictions.csv", test_ids, y_test, y_pred)

    matrix = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    with (args.outdir / "confusion_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels_sorted])
        for label, row in zip(labels_sorted, matrix):
            writer.writerow([label, *row])

    print(f"Samples used: {len(sample_ids)}")
    print(f"Classes: {len(labels_sorted)}")
    print(f"Accuracy: {stats['accuracy']:.4f}")
    print(f"Saved model and reports to: {args.outdir}")


if __name__ == "__main__":
    main()
