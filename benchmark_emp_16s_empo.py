#!/usr/bin/env python3
"""Benchmark Explicit-Vocab vs HDC-Hash on EMP 16S EMPO labels.

Inputs:
  - EMP deblur 90bp BIOM table
  - EMP release 1 QIIME mapping metadata

The feature table is sample x ASV abundance after filtering samples with total
count >= --min-sample-sum. Explicit-Vocab uses the ASV IDs directly. HDC-Hash
encodes each 90bp ASV sequence as a sparse hypervector built from hashed k-mers,
then projects sample abundance vectors into the HDC space.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from biom import load_table
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC


DEFAULT_BIOM = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/EMP_16S/emp_deblur_90bp.release1.biom"
)
DEFAULT_METADATA = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/EMP_16S/emp_qiime_mapping_release1.tsv"
)
DEFAULT_OUTDIR = Path("/home/tsl012/Multi_variant classification/emp_16s_benchmark")
EMPO_COLUMNS = ["empo_0", "empo_1", "empo_2", "empo_3"]
MISSING_LABELS = {"", "NA", "N/A", "nan", "None", "null", "Unknown", "unknown"}
DNA_RE = re.compile(r"[^ACGTN]")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def clean_label(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in MISSING_LABELS:
        return ""
    return text


def sequence_kmers(sequence: str, k: int, stride: int, max_kmers: int) -> list[str]:
    sequence = DNA_RE.sub("", str(sequence).upper())
    if len(sequence) < k:
        return []
    kmers = [sequence[i : i + k] for i in range(0, len(sequence) - k + 1, stride)]
    if len(kmers) <= max_kmers:
        return kmers
    idx = np.linspace(0, len(kmers) - 1, num=max_kmers, dtype=np.int32)
    return [kmers[int(i)] for i in idx]


def token_hv(token: str, dim: int, active_dims: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    digest = hashlib.blake2b(f"{seed}|{token}".encode(), digest_size=16).digest()
    token_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(token_seed)
    cols = rng.choice(dim, size=active_dims, replace=False).astype(np.int32)
    vals = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=active_dims)
    return cols, vals


def build_sequence_projection(
    sequences: list[str],
    dim: int,
    sequence_encoding: str,
    kmer_size: int,
    kmer_stride: int,
    max_sequence_kmers: int,
    active_dims_per_kmer: int,
    active_dims_per_sequence: int,
    seed: int,
) -> sparse.csr_matrix:
    """Create sparse ASV-sequence-to-HDC projection matrix."""
    if sequence_encoding == "whole-sequence":
        rows = np.repeat(np.arange(len(sequences), dtype=np.int32), active_dims_per_sequence)
        cols = np.empty(len(sequences) * active_dims_per_sequence, dtype=np.int32)
        vals = np.empty(len(sequences) * active_dims_per_sequence, dtype=np.float32)
        offset = 0
        for sequence in sequences:
            hv_cols, hv_vals = token_hv(
                f"asv_sequence:{sequence}",
                dim=dim,
                active_dims=active_dims_per_sequence,
                seed=seed,
            )
            cols[offset : offset + active_dims_per_sequence] = hv_cols
            vals[offset : offset + active_dims_per_sequence] = hv_vals
            offset += active_dims_per_sequence
        return sparse.coo_matrix((vals, (rows, cols)), shape=(len(sequences), dim), dtype=np.float32).tocsr()

    token_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    row_idx: list[np.ndarray] = []
    col_idx: list[np.ndarray] = []
    data: list[np.ndarray] = []

    for row, sequence in enumerate(sequences):
        kmers = sequence_kmers(sequence, k=kmer_size, stride=kmer_stride, max_kmers=max_sequence_kmers)
        if not kmers:
            continue
        cols_for_row: list[np.ndarray] = []
        vals_for_row: list[np.ndarray] = []
        for kmer in kmers:
            hv = token_cache.get(kmer)
            if hv is None:
                hv = token_hv(f"k{kmer_size}:{kmer}", dim=dim, active_dims=active_dims_per_kmer, seed=seed)
                token_cache[kmer] = hv
            cols, vals = hv
            cols_for_row.append(cols)
            vals_for_row.append(vals)
        cols = np.concatenate(cols_for_row)
        vals = np.concatenate(vals_for_row)
        rows = np.full(cols.shape, row, dtype=np.int32)
        row_idx.append(rows)
        col_idx.append(cols)
        data.append(vals)

    if not data:
        return sparse.csr_matrix((len(sequences), dim), dtype=np.float32)

    matrix = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(row_idx), np.concatenate(col_idx))),
        shape=(len(sequences), dim),
        dtype=np.float32,
    )
    return matrix.tocsr()


def load_emp_dataset(args: argparse.Namespace):
    started = time.perf_counter()
    table = load_table(str(args.biom))
    load_biom_sec = time.perf_counter() - started

    started = time.perf_counter()
    sample_ids = np.array([str(x) for x in table.ids(axis="sample")])
    obs_ids = [str(x) for x in table.ids(axis="observation")]
    sample_sums = np.asarray(table.sum(axis="sample"), dtype=np.float64)
    keep_mask = sample_sums >= args.min_sample_sum
    kept_sample_ids = sample_ids[keep_mask]

    metadata = pd.read_csv(args.metadata, sep="\t", dtype=str, low_memory=False)
    if "#SampleID" not in metadata.columns:
        raise ValueError("Metadata is missing #SampleID.")
    metadata = metadata.set_index("#SampleID", drop=False)

    intersect_ids = [sid for sid in kept_sample_ids if sid in metadata.index]
    sample_index = {sid: i for i, sid in enumerate(sample_ids)}
    keep_indices = np.array([sample_index[sid] for sid in intersect_ids], dtype=np.int32)

    # BIOM is observation x sample; transpose to sample x ASV.
    x_counts = table.matrix_data.tocsc()[:, keep_indices].T.tocsr().astype(np.float32)
    md = metadata.loc[intersect_ids]
    prepare_sec = time.perf_counter() - started

    stats = {
        "biom_path": str(args.biom),
        "metadata_path": str(args.metadata),
        "biom_shape_observations_samples": list(table.shape),
        "biom_nnz": int(table.matrix_data.nnz),
        "samples_total_in_biom": int(len(sample_ids)),
        "samples_with_sum_ge_min": int(keep_mask.sum()),
        "samples_after_metadata_intersection": int(len(intersect_ids)),
        "min_sample_sum": args.min_sample_sum,
        "observations": int(len(obs_ids)),
        "load_biom_sec": load_biom_sec,
        "prepare_filtered_matrix_sec": prepare_sec,
    }
    return intersect_ids, obs_ids, x_counts, md, stats


def split_for_label(sample_ids: list[str], labels: list[str], test_size: float, random_state: int):
    label_counts = Counter(labels)
    stratify = labels if min(label_counts.values()) >= 2 else None
    indices = np.arange(len(sample_ids))
    return train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def train_eval_explicit(x_counts, y, train_idx, test_idx, random_state: int):
    model = Pipeline(
        [
            ("tfidf", TfidfTransformer(norm="l2", sublinear_tf=True)),
            ("svc", LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=random_state)),
        ]
    )
    started = time.perf_counter()
    model.fit(x_counts[train_idx], [y[i] for i in train_idx])
    train_sec = time.perf_counter() - started

    started = time.perf_counter()
    pred = model.predict(x_counts[test_idx])
    predict_sec = time.perf_counter() - started
    y_true = [y[i] for i in test_idx]
    return model, pred, y_true, train_sec, predict_sec


def train_eval_hdc(x_hdc, y, train_idx, test_idx, random_state: int):
    model = Pipeline(
        [
            ("normalize", Normalizer(norm="l2")),
            ("svc", LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=random_state)),
        ]
    )
    started = time.perf_counter()
    model.fit(x_hdc[train_idx], [y[i] for i in train_idx])
    train_sec = time.perf_counter() - started

    started = time.perf_counter()
    pred = model.predict(x_hdc[test_idx])
    predict_sec = time.perf_counter() - started
    y_true = [y[i] for i in test_idx]
    return model, pred, y_true, train_sec, predict_sec


def write_report(path: Path, y_true: list[str], y_pred: np.ndarray) -> None:
    path.write_text(classification_report(y_true, y_pred, zero_division=0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biom", type=Path, default=DEFAULT_BIOM)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--min-sample-sum", type=float, default=1000.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hdc-dim", type=int, default=8192)
    parser.add_argument(
        "--sequence-encoding",
        choices=["whole-sequence", "kmer"],
        default="whole-sequence",
        help="How to encode each ASV sequence before sample-level HDC bundling.",
    )
    parser.add_argument("--kmer-size", type=int, default=6)
    parser.add_argument("--kmer-stride", type=int, default=1)
    parser.add_argument("--max-sequence-kmers", type=int, default=24)
    parser.add_argument("--active-dims-per-kmer", type=int, default=4)
    parser.add_argument("--active-dims-per-sequence", type=int, default=64)
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()

    log("Loading and filtering EMP BIOM + metadata")
    sample_ids, obs_ids, x_counts, metadata, stats = load_emp_dataset(args)
    log(
        "Loaded filtered matrix "
        f"{x_counts.shape[0]} samples x {x_counts.shape[1]} ASVs with {x_counts.nnz} nonzeros"
    )

    log(
        "Building HDC sequence projection "
        f"({args.sequence_encoding}, dim={args.hdc_dim})"
    )
    started = time.perf_counter()
    projection = build_sequence_projection(
        obs_ids,
        dim=args.hdc_dim,
        sequence_encoding=args.sequence_encoding,
        kmer_size=args.kmer_size,
        kmer_stride=args.kmer_stride,
        max_sequence_kmers=args.max_sequence_kmers,
        active_dims_per_kmer=args.active_dims_per_kmer,
        active_dims_per_sequence=args.active_dims_per_sequence,
        seed=args.random_state,
    )
    projection_build_sec = time.perf_counter() - started
    log(f"Built HDC projection in {projection_build_sec:.2f}s with {projection.nnz} nonzeros")

    log("Projecting sample abundance matrix into HDC space")
    started = time.perf_counter()
    x_hdc = (x_counts @ projection).tocsr()
    hdc_feature_build_sec = time.perf_counter() - started
    log(f"Built sample HDC matrix in {hdc_feature_build_sec:.2f}s with {x_hdc.nnz} nonzeros")

    stats.update(
        {
            "x_counts_shape": list(x_counts.shape),
            "x_counts_nnz": int(x_counts.nnz),
            "hdc_projection_shape": list(projection.shape),
            "hdc_projection_nnz": int(projection.nnz),
            "hdc_matrix_shape": list(x_hdc.shape),
            "hdc_matrix_nnz": int(x_hdc.nnz),
            "hdc_projection_build_sec": projection_build_sec,
            "hdc_sample_feature_build_sec": hdc_feature_build_sec,
            "hdc_dim": args.hdc_dim,
            "sequence_encoding": args.sequence_encoding,
            "kmer_size": args.kmer_size,
            "kmer_stride": args.kmer_stride,
            "max_sequence_kmers": args.max_sequence_kmers,
            "active_dims_per_kmer": args.active_dims_per_kmer,
            "active_dims_per_sequence": args.active_dims_per_sequence,
        }
    )

    rows: list[dict[str, object]] = []
    for label_col in EMPO_COLUMNS:
        log(f"Running {label_col}")
        if label_col not in metadata.columns:
            rows.append({"empo_level": label_col, "status": "missing_metadata_column"})
            continue

        valid_local_idx: list[int] = []
        labels: list[str] = []
        for i, value in enumerate(metadata[label_col].tolist()):
            label = clean_label(value)
            if label:
                valid_local_idx.append(i)
                labels.append(label)

        label_counts = Counter(labels)
        if len(label_counts) < 2:
            log(f"Skipping {label_col}: only {len(label_counts)} class")
            rows.append(
                {
                    "empo_level": label_col,
                    "status": "skipped_one_class",
                    "samples": len(labels),
                    "classes": len(label_counts),
                    "label_counts": json.dumps(dict(label_counts.most_common()), sort_keys=True),
                }
            )
            continue

        valid_local_idx_np = np.array(valid_local_idx, dtype=np.int32)
        train_rel, test_rel = split_for_label(
            [sample_ids[i] for i in valid_local_idx],
            labels,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        train_idx = valid_local_idx_np[train_rel]
        test_idx = valid_local_idx_np[test_rel]

        x_counts_level = x_counts[valid_local_idx_np]
        x_hdc_level = x_hdc[valid_local_idx_np]

        explicit_model, explicit_pred, y_true, explicit_train_sec, explicit_predict_sec = train_eval_explicit(
            x_counts_level, labels, train_rel, test_rel, args.random_state
        )
        log(f"{label_col} Explicit-Vocab done: train={explicit_train_sec:.2f}s predict={explicit_predict_sec:.2f}s")
        hdc_model, hdc_pred, _, hdc_train_sec, hdc_predict_sec = train_eval_hdc(
            x_hdc_level, labels, train_rel, test_rel, args.random_state
        )
        log(f"{label_col} HDC-Hash done: train={hdc_train_sec:.2f}s predict={hdc_predict_sec:.2f}s")

        explicit_acc = accuracy_score(y_true, explicit_pred)
        hdc_acc = accuracy_score(y_true, hdc_pred)
        level_dir = args.outdir / label_col
        level_dir.mkdir(exist_ok=True)
        write_report(level_dir / "explicit_vocab_classification_report.txt", y_true, explicit_pred)
        write_report(level_dir / "hdc_hash_classification_report.txt", y_true, hdc_pred)

        pred_path = level_dir / "test_predictions.csv"
        with pred_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample", "true_label", "explicit_vocab_pred", "hdc_hash_pred"])
            for idx, true, p1, p2 in zip(test_idx, y_true, explicit_pred, hdc_pred):
                writer.writerow([sample_ids[int(idx)], true, p1, p2])

        if args.save_models:
            joblib.dump(explicit_model, level_dir / "explicit_vocab_model.joblib")
            joblib.dump(hdc_model, level_dir / "hdc_hash_model.joblib")

        rows.append(
            {
                "empo_level": label_col,
                "status": "ok",
                "samples": len(labels),
                "classes": len(label_counts),
                "train_samples": len(train_rel),
                "test_samples": len(test_rel),
                "explicit_vocab_accuracy": explicit_acc,
                "hdc_hash_accuracy": hdc_acc,
                "explicit_vocab_train_sec": explicit_train_sec,
                "hdc_hash_train_sec": hdc_train_sec,
                "explicit_vocab_predict_sec": explicit_predict_sec,
                "hdc_hash_predict_sec": hdc_predict_sec,
                "hdc_train_speedup_vs_explicit": explicit_train_sec / hdc_train_sec if hdc_train_sec else None,
                "hdc_predict_speedup_vs_explicit": explicit_predict_sec / hdc_predict_sec if hdc_predict_sec else None,
                "label_counts": json.dumps(dict(label_counts.most_common()), sort_keys=True),
            }
        )

    stats["total_script_wall_sec"] = time.perf_counter() - total_started
    (args.outdir / "dataset_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))

    summary_path = args.outdir / "empo_benchmark_summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "empo_level",
        "status",
        "samples",
        "classes",
        "train_samples",
        "test_samples",
        "explicit_vocab_accuracy",
        "hdc_hash_accuracy",
        "explicit_vocab_train_sec",
        "hdc_hash_train_sec",
        "explicit_vocab_predict_sec",
        "hdc_hash_predict_sec",
        "hdc_train_speedup_vs_explicit",
        "hdc_predict_speedup_vs_explicit",
        "label_counts",
    ]
    fieldnames = [f for f in preferred if f in fieldnames] + [f for f in fieldnames if f not in preferred]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {summary_path}")
    print(f"Wrote {args.outdir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
