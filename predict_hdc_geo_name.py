#!/usr/bin/env python3
"""Predict geo_loc_name with a trained HDC prototype model."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from train_hdc_geo_classifier import DEFAULT_CSV, clean_value, row_tokens, row_weight, token_hv
from predict_geo_name import load_sample_features


DEFAULT_MODEL = Path("/home/tsl012/Multi_variant classification/hdc_output/hdc_geo_loc_name_model.joblib")


def encode_samples(
    csv_path: Path,
    dim: int,
    active_dims: int,
    kmer_size: int,
    kmer_stride: int,
    max_sequence_kmers: int,
    max_weight: int,
    seed: int,
    binarize: bool,
) -> tuple[list[str], np.ndarray]:
    token_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sample_vectors: dict[str, np.ndarray] = {}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample = clean_value(row.get("sample"))
            if not sample:
                continue
            vector = sample_vectors.setdefault(sample, np.zeros(dim, dtype=np.int32))
            weight = row_weight(row, max_weight)
            for token in row_tokens(row, kmer_size, kmer_stride, max_sequence_kmers):
                hv = token_cache.get(token)
                if hv is None:
                    hv = token_hv(token, dim=dim, active_dims=active_dims, seed=seed)
                    token_cache[token] = hv
                positions, signs = hv
                np.add.at(vector, positions, signs.astype(np.int32) * weight)

    sample_ids = sorted(sample_vectors)
    vectors = np.vstack(
        [
            np.where(sample_vectors[sample] >= 0, 1, -1).astype(np.float32)
            if binarize
            else sample_vectors[sample].astype(np.float32)
            for sample in sample_ids
        ]
    )
    return sample_ids, vectors


def predict(vectors: np.ndarray, classes: list[str], prototypes: np.ndarray) -> tuple[list[str], np.ndarray]:
    x = vectors.astype(np.float32)
    p = prototypes.astype(np.float32)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    p_norm = np.linalg.norm(p, axis=1, keepdims=True).T
    scores = (x @ p.T) / np.maximum(x_norm * p_norm, 1e-12)
    pred_indexes = np.argmax(scores, axis=1)
    return [classes[i] for i in pred_indexes], scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict geo_loc_name with HDC prototypes.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to HDC model joblib.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV.")
    parser.add_argument("--output", type=Path, default=Path("hdc_geo_predictions.csv"), help="Output prediction CSV.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of ranked labels to write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    if bundle.get("mode") == "linear-svm":
        model = bundle["model"]
        kmer_size = int(bundle["kmer_size"])
        max_weight = int(bundle["max_count_weight"])
        sample_ids, feature_dicts = load_sample_features(
            args.csv,
            kmer_size=kmer_size,
            max_weight=max_weight,
        )
        predictions = model.predict(feature_dicts)
        scores = model.decision_function(feature_dicts)
        classes = list(model.named_steps["classifier"].classes_)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            header = ["sample", "pred_geo_loc_name"]
            for i in range(1, args.top_k + 1):
                header.extend([f"top{i}_geo_loc_name", f"top{i}_score"])
            writer.writerow(header)
            for sample, pred, row_scores in zip(sample_ids, predictions, scores):
                order = np.argsort(row_scores)[::-1][:args.top_k]
                row = [sample, pred]
                for index in order:
                    row.extend([classes[index], float(row_scores[index])])
                writer.writerow(row)

        print(f"Wrote {len(sample_ids)} tuned HDC sample predictions to: {args.output}")
        print("Predicted class counts:")
        for label, count in Counter(predictions).most_common():
            print(f"{count}\t{label}")
        return

    classes = list(bundle["classes"])
    prototypes = bundle["prototypes"]
    dim = int(bundle["dimension"])
    active_dims = int(bundle["active_dims"])
    kmer_size = int(bundle["kmer_size"])
    kmer_stride = int(bundle["kmer_stride"])
    max_sequence_kmers = int(bundle["max_sequence_kmers"])
    max_weight = int(bundle["max_count_weight"])
    seed = int(bundle["seed"])
    binarize = bool(bundle.get("binarize", False))

    sample_ids, vectors = encode_samples(
        args.csv,
        dim,
        active_dims,
        kmer_size,
        kmer_stride,
        max_sequence_kmers,
        max_weight,
        seed,
        binarize,
    )
    predictions, scores = predict(vectors, classes, prototypes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["sample", "pred_geo_loc_name"]
        for i in range(1, args.top_k + 1):
            header.extend([f"top{i}_geo_loc_name", f"top{i}_similarity"])
        writer.writerow(header)
        for sample, pred, row_scores in zip(sample_ids, predictions, scores):
            order = np.argsort(row_scores)[::-1][:args.top_k]
            row = [sample, pred]
            for index in order:
                row.extend([classes[index], float(row_scores[index])])
            writer.writerow(row)

    print(f"Wrote {len(sample_ids)} HDC sample predictions to: {args.output}")
    print("Predicted class counts:")
    for label, count in Counter(predictions).most_common():
        print(f"{count}\t{label}")


if __name__ == "__main__":
    main()
