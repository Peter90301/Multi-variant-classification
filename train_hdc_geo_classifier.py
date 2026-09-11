#!/usr/bin/env python3
"""Train an HDC geo_loc_name classifier from sample-grouped taxonomy and ASV sequence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from train_geo_classifier import build_sample_dataset


DEFAULT_CSV = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/Multi_variant_classification/all_voyages_NEW.csv"
)
DEFAULT_OUTDIR = Path("/home/tsl012/Multi_variant classification/hdc_output")

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


def sequence_kmers(sequence: str, k: int, stride: int, max_kmers: int) -> list[str]:
    sequence = DNA_RE.sub("", sequence.upper())
    if len(sequence) < k:
        return []
    kmers = [f"k{k}:{sequence[i:i + k]}" for i in range(0, len(sequence) - k + 1, stride)]
    if len(kmers) <= max_kmers:
        return kmers
    indexes = np.linspace(0, len(kmers) - 1, num=max_kmers, dtype=int)
    return [kmers[i] for i in indexes]


def row_weight(row: dict[str, str], max_weight: int) -> int:
    try:
        count = float(clean_value(row.get("count")) or 1.0)
    except ValueError:
        count = 1.0
    if count <= 0:
        return 1
    return max(1, min(max_weight, int(math.log10(count)) + 1))


def row_tokens(row: dict[str, str], kmer_size: int, kmer_stride: int, max_sequence_kmers: int) -> list[str]:
    tokens: list[str] = []
    lineage: list[str] = []

    for field in TAXONOMY_FIELDS:
        value = clean_value(row.get(field))
        if not value:
            continue
        token_value = normalize_token(value)
        tokens.append(f"{field}:{token_value}")
        lineage.append(token_value)

    if lineage:
        tokens.append("lineage:" + "|".join(lineage))

    sequence = clean_value(row.get("ASV_sequence"))
    if sequence:
        tokens.extend(sequence_kmers(sequence, kmer_size, kmer_stride, max_sequence_kmers))

    return tokens


def token_hv(token: str, dim: int, active_dims: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically map a token to sparse bipolar positions and signs."""
    digest = hashlib.blake2b(f"{seed}|{token}".encode(), digest_size=16).digest()
    token_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(token_seed)
    positions = rng.choice(dim, size=active_dims, replace=False).astype(np.int32)
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=active_dims)
    return positions, signs


def sample_hypervectors(
    csv_path: Path,
    dim: int,
    active_dims: int,
    kmer_size: int,
    kmer_stride: int,
    max_sequence_kmers: int,
    max_weight: int,
    seed: int,
    min_rows_per_sample: int,
    binarize: bool,
) -> tuple[list[str], np.ndarray, list[str], dict[str, object]]:
    token_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sample_vectors: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(dim, dtype=np.int32))
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

            weight = row_weight(row, max_weight)
            for token in row_tokens(row, kmer_size, kmer_stride, max_sequence_kmers):
                hv = token_cache.get(token)
                if hv is None:
                    hv = token_hv(token, dim=dim, active_dims=active_dims, seed=seed)
                    token_cache[token] = hv
                positions, signs = hv
                np.add.at(sample_vectors[sample], positions, signs.astype(np.int32) * weight)

            sample_rows[sample] += 1
            sample_labels[sample][label] += 1

    sample_ids: list[str] = []
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    for sample, vector in sample_vectors.items():
        if sample_rows[sample] < min_rows_per_sample:
            continue
        if not sample_labels[sample]:
            continue
        sample_ids.append(sample)
        if binarize:
            vectors.append(np.where(vector >= 0, 1, -1).astype(np.int16))
        else:
            vectors.append(vector.astype(np.float32))
        labels.append(sample_labels[sample].most_common(1)[0][0])

    stats = {
        "source_csv": str(csv_path),
        "total_rows": total_rows,
        "skipped_no_sample": skipped_no_sample,
        "skipped_no_label": skipped_no_label,
        "samples_before_filter": len(sample_vectors),
        "samples_after_filter": len(sample_ids),
        "label_counts": dict(Counter(labels).most_common()),
        "taxonomy_fields": TAXONOMY_FIELDS,
        "sequence_feature": "ASV_sequence",
        "group_column": "sample",
        "label_column": "geo_loc_name",
        "hdc_dimension": dim,
        "active_dims_per_token": active_dims,
        "kmer_size": kmer_size,
        "kmer_stride": kmer_stride,
        "max_sequence_kmers": max_sequence_kmers,
        "max_count_weight": max_weight,
        "seed": seed,
        "min_rows_per_sample": min_rows_per_sample,
        "unique_tokens_cached": len(token_cache),
        "binarize_vectors": binarize,
    }

    return sample_ids, np.vstack(vectors), labels, stats


def build_prototypes(vectors: np.ndarray, labels: list[str], binarize: bool) -> tuple[list[str], np.ndarray]:
    classes = sorted(set(labels))
    class_to_index = {label: i for i, label in enumerate(classes)}
    sums = np.zeros((len(classes), vectors.shape[1]), dtype=np.float32)
    for vector, label in zip(vectors, labels):
        sums[class_to_index[label]] += vector.astype(np.float32)
    prototypes = np.where(sums >= 0, 1, -1).astype(np.float32) if binarize else sums
    return classes, prototypes


def predict_hdc(vectors: np.ndarray, classes: list[str], prototypes: np.ndarray) -> tuple[list[str], np.ndarray]:
    x = vectors.astype(np.float32)
    p = prototypes.astype(np.float32)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    p_norm = np.linalg.norm(p, axis=1, keepdims=True).T
    scores = (x @ p.T) / np.maximum(x_norm * p_norm, 1e-12)
    pred_indexes = np.argmax(scores, axis=1)
    return [classes[i] for i in pred_indexes], scores


def save_predictions(path: Path, sample_ids: list[str], y_true: list[str], y_pred: list[str], scores: np.ndarray, classes: list[str], top_k: int) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["sample", "true_geo_loc_name", "pred_geo_loc_name", "correct"]
        for i in range(1, top_k + 1):
            header.extend([f"top{i}_geo_loc_name", f"top{i}_similarity"])
        writer.writerow(header)
        for sample, true, pred, row_scores in zip(sample_ids, y_true, y_pred, scores):
            order = np.argsort(row_scores)[::-1][:top_k]
            row = [sample, true, pred, true == pred]
            for index in order:
                row.extend([classes[index], float(row_scores[index])])
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HDC prototypes to predict geo_loc_name from sample-grouped taxonomy and ASV_sequence."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to all_voyages_NEW.csv.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Directory for HDC model and reports.")
    parser.add_argument(
        "--classifier",
        choices=["linear-svm", "prototype"],
        default="linear-svm",
        help="HDC readout. linear-svm is the tuned high-accuracy default; prototype is pure associative memory.",
    )
    parser.add_argument("--hash-features", type=int, default=32768, help="Hashed HDC dimension for linear-svm mode.")
    parser.add_argument("--svm-c", type=float, default=16.0, help="LinearSVC regularization parameter for linear-svm mode.")
    parser.add_argument("--dimension", type=int, default=10000, help="Hypervector dimension.")
    parser.add_argument("--active-dims", type=int, default=32, help="Sparse active dimensions updated per token.")
    parser.add_argument("--kmer-size", type=int, default=6, help="DNA k-mer size for ASV_sequence.")
    parser.add_argument("--kmer-stride", type=int, default=8, help="Stride between DNA k-mers.")
    parser.add_argument("--max-sequence-kmers", type=int, default=32, help="Maximum sampled k-mers per ASV sequence.")
    parser.add_argument("--max-count-weight", type=int, default=5, help="Maximum token repetition from log10(count).")
    parser.add_argument("--min-rows-per-sample", type=int, default=1, help="Drop samples with fewer rows.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of samples used for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for split and token HVs.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of labels written in prediction report.")
    parser.add_argument("--binarize", action="store_true", help="Threshold sample and prototype hypervectors to bipolar values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.classifier == "linear-svm":
        sample_ids, feature_dicts, labels, stats = build_sample_dataset(
            args.csv,
            kmer_size=args.kmer_size,
            max_weight=args.max_count_weight,
            min_rows_per_sample=args.min_rows_per_sample,
        )
        stratify = labels if min(Counter(labels).values()) >= 2 else None
        train_ids, test_ids, x_train, x_test, y_train, y_test = train_test_split(
            sample_ids,
            feature_dicts,
            labels,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify,
        )

        model = Pipeline(
            steps=[
                (
                    "hdc_hasher",
                    FeatureHasher(
                        n_features=args.hash_features,
                        input_type="dict",
                        alternate_sign=False,
                    ),
                ),
                ("tfidf", TfidfTransformer(sublinear_tf=True)),
                (
                    "classifier",
                    LinearSVC(
                        C=args.svm_c,
                        class_weight="balanced",
                        random_state=args.random_state,
                        dual="auto",
                        max_iter=12000,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        labels_sorted = sorted(set(labels))

        stats.update(
            {
                "model": "HDC feature hashing + TF-IDF + LinearSVC",
                "classifier": args.classifier,
                "hdc_dimension": args.hash_features,
                "train_samples": len(train_ids),
                "test_samples": len(test_ids),
                "accuracy": accuracy_score(y_test, y_pred),
                "hash_alternate_sign": False,
                "svm_c": args.svm_c,
            }
        )

        joblib.dump(
            {
                "mode": "linear-svm",
                "model": model,
                "kmer_size": args.kmer_size,
                "max_count_weight": args.max_count_weight,
                "stats": stats,
            },
            args.outdir / "hdc_geo_loc_name_model.joblib",
            compress=3,
        )

        (args.outdir / "hdc_metrics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
        (args.outdir / "hdc_classification_report.txt").write_text(
            classification_report(y_test, y_pred, labels=labels_sorted, zero_division=0)
        )
        save_predictions(
            args.outdir / "hdc_test_predictions.csv",
            test_ids,
            y_test,
            list(y_pred),
            model.decision_function(x_test),
            list(model.named_steps["classifier"].classes_),
            args.top_k,
        )

        matrix = confusion_matrix(y_test, y_pred, labels=labels_sorted)
        with (args.outdir / "hdc_confusion_matrix.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true\\pred", *labels_sorted])
            for label, row in zip(labels_sorted, matrix):
                writer.writerow([label, *row])

        print(f"Samples used: {len(sample_ids)}")
        print(f"Classes: {len(labels_sorted)}")
        print(f"HDC dimension: {args.hash_features}")
        print(f"Accuracy: {stats['accuracy']:.4f}")
        print(f"Saved tuned HDC model and reports to: {args.outdir}")
        return

    sample_ids, vectors, labels, stats = sample_hypervectors(
        csv_path=args.csv,
        dim=args.dimension,
        active_dims=args.active_dims,
        kmer_size=args.kmer_size,
        kmer_stride=args.kmer_stride,
        max_sequence_kmers=args.max_sequence_kmers,
        max_weight=args.max_count_weight,
        seed=args.random_state,
        min_rows_per_sample=args.min_rows_per_sample,
        binarize=args.binarize,
    )

    stratify = labels if min(Counter(labels).values()) >= 2 else None
    train_ids, test_ids, x_train, x_test, y_train, y_test = train_test_split(
        sample_ids,
        vectors,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    classes, prototypes = build_prototypes(x_train, y_train, binarize=args.binarize)
    y_pred, scores = predict_hdc(x_test, classes, prototypes)
    labels_sorted = sorted(set(labels))

    stats.update(
        {
            "model": "Sparse HDC bundled token hypervectors + cosine prototype matching",
            "train_samples": len(train_ids),
            "test_samples": len(test_ids),
            "accuracy": accuracy_score(y_test, y_pred),
            "binarize_vectors": args.binarize,
        }
    )

    joblib.dump(
        {
            "classes": classes,
            "prototypes": prototypes,
            "dimension": args.dimension,
            "active_dims": args.active_dims,
            "kmer_size": args.kmer_size,
            "kmer_stride": args.kmer_stride,
            "max_sequence_kmers": args.max_sequence_kmers,
            "max_count_weight": args.max_count_weight,
            "seed": args.random_state,
            "binarize": args.binarize,
            "stats": stats,
        },
        args.outdir / "hdc_geo_loc_name_model.joblib",
        compress=3,
    )

    (args.outdir / "hdc_metrics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    (args.outdir / "hdc_classification_report.txt").write_text(
        classification_report(y_test, y_pred, labels=labels_sorted, zero_division=0)
    )
    save_predictions(args.outdir / "hdc_test_predictions.csv", test_ids, y_test, y_pred, scores, classes, args.top_k)

    matrix = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    with (args.outdir / "hdc_confusion_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels_sorted])
        for label, row in zip(labels_sorted, matrix):
            writer.writerow([label, *row])

    print(f"Samples used: {len(sample_ids)}")
    print(f"Classes: {len(classes)}")
    print(f"HDC dimension: {args.dimension}")
    print(f"Accuracy: {stats['accuracy']:.4f}")
    print(f"Saved HDC model and reports to: {args.outdir}")


if __name__ == "__main__":
    main()
