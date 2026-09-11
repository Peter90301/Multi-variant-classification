#!/usr/bin/env python3
"""Predict geo_loc_name for each sample in a CSV using a trained model."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import joblib

from train_geo_classifier import clean_value, row_features


DEFAULT_MODEL = Path("/home/tsl012/Multi_variant classification/output/geo_loc_name_model.joblib")
DEFAULT_CSV = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/Multi_variant_classification/all_voyages_NEW.csv"
)


def load_sample_features(csv_path: Path, kmer_size: int, max_weight: int) -> tuple[list[str], list[dict[str, int]]]:
    sample_features: dict[str, Counter[str]] = defaultdict(Counter)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample = clean_value(row.get("sample"))
            if not sample:
                continue
            features = row_features(row, kmer_size=kmer_size, max_weight=max_weight)
            if features:
                sample_features[sample].update(features)

    sample_ids = sorted(sample_features)
    feature_dicts = [dict(sample_features[sample]) for sample in sample_ids]
    return sample_ids, feature_dicts


def top_scores(model, feature_dicts: list[dict[str, int]], top_k: int) -> list[list[tuple[str, float]]]:
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    scores = model.decision_function(feature_dicts)
    if len(classes) == 2:
        scores = [[-score, score] for score in scores]

    ranked: list[list[tuple[str, float]]] = []
    for row in scores:
        order = sorted(range(len(classes)), key=lambda i: row[i], reverse=True)[:top_k]
        ranked.append([(classes[i], float(row[i])) for i in order])
    return ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict geo_loc_name from sample-grouped taxonomy and ASV_sequence.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to geo_loc_name_model.joblib.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV with sample/taxonomy/ASV_sequence columns.")
    parser.add_argument("--output", type=Path, default=Path("geo_predictions.csv"), help="Output prediction CSV.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of ranked labels to write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    model = bundle["model"]
    kmer_size = int(bundle["kmer_size"])
    max_weight = int(bundle["max_count_weight"])

    sample_ids, feature_dicts = load_sample_features(args.csv, kmer_size=kmer_size, max_weight=max_weight)
    predictions = model.predict(feature_dicts)
    ranked = top_scores(model, feature_dicts, args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["sample", "pred_geo_loc_name"]
        for i in range(1, args.top_k + 1):
            header.extend([f"top{i}_geo_loc_name", f"top{i}_score"])
        writer.writerow(header)

        for sample, pred, top in zip(sample_ids, predictions, ranked):
            row = [sample, pred]
            for label, score in top:
                row.extend([label, score])
            writer.writerow(row)

    print(f"Wrote {len(sample_ids)} sample predictions to: {args.output}")
    print("Predicted class counts:")
    for label, count in Counter(predictions).most_common():
        print(f"{count}\t{label}")


if __name__ == "__main__":
    main()
