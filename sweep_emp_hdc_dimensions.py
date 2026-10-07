#!/usr/bin/env python3
"""Measure EMP EMPO1-3 accuracy across HDC hypervector dimensions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from biom import load_table
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

script_dir = Path(__file__).resolve().parent / "scripts"
if not script_dir.exists():
    script_dir = Path(__file__).resolve().parent / "Final_version" / "scripts"
sys.path.insert(0, str(script_dir))
from three_method_common import sequence_projection


LEVELS = ("empo_1", "empo_2", "empo_3")
DIMENSIONS = (1024, 2048, 4096, 8192, 16384, 32768)

# Keep the selected sparsity and readout regularization fixed while dimension
# changes, so the sweep isolates hypervector capacity.
LEVEL_SETTINGS = {
    "empo_1": {"active_dims": 32, "c_value": 2.0},
    "empo_2": {"active_dims": 32, "c_value": 1.0},
    "empo_3": {"active_dims": 16, "c_value": 20.0},
}
DEFAULT_BIOM = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/EMP_16S/emp_deblur_90bp.release1.biom"
)
DEFAULT_METADATA = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/EMP_16S/emp_qiime_mapping_release1.tsv"
)
MISSING_LABELS = {"", "NA", "N/A", "nan", "None", "null", "Unknown", "unknown"}


def clean_label(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text in MISSING_LABELS else text


def load_emp_dataset(args: argparse.Namespace):
    table = load_table(str(args.biom))
    sample_ids = np.asarray(list(map(str, table.ids(axis="sample"))))
    observation_ids = list(map(str, table.ids(axis="observation")))
    sample_sums = np.asarray(table.sum(axis="sample"), dtype=np.float64)
    kept = sample_ids[sample_sums >= args.min_sample_sum]
    metadata = pd.read_csv(args.metadata, sep="\t", dtype=str, low_memory=False)
    if "#SampleID" not in metadata.columns:
        raise ValueError("Metadata is missing #SampleID")
    metadata = metadata.set_index("#SampleID", drop=False)
    sample_ids = [sample for sample in kept if sample in metadata.index]
    positions = {sample: index for index, sample in enumerate(kept)}
    selected = np.asarray([positions[sample] for sample in sample_ids], dtype=np.int32)
    kept_positions = np.flatnonzero(sample_sums >= args.min_sample_sum)
    selected = kept_positions[selected]
    counts = table.matrix_data.tocsc()[:, selected].T.tocsr().astype(np.float32)
    counts.sort_indices()
    return np.asarray(sample_ids), observation_ids, counts, metadata.loc[sample_ids], {
        "samples_after_filter": len(sample_ids),
        "observations": len(observation_ids),
        "counts_nnz": int(counts.nnz),
    }


def split_for_label(sample_ids, labels, test_size: float, random_state: int):
    counts = pd.Series(labels).value_counts()
    stratify = labels if counts.min() >= 2 else None
    return train_test_split(
        np.arange(len(sample_ids)),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biom", type=Path, default=DEFAULT_BIOM)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("/home/tsl012/Multi_variant classification/emp_16s_dimension_sweep"),
    )
    parser.add_argument("--min-sample-sum", type=float, default=1000.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_dimension_accuracy(frame: pd.DataFrame, output: Path) -> None:
    plt.rcParams["svg.fonttype"] = "none"
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = {"EMPO1": "#2563eb", "EMPO2": "#f97316", "EMPO3": "#16a34a"}
    for level, display_level in zip(LEVELS, ("EMPO1", "EMPO2", "EMPO3")):
        subset = frame[frame["level"] == level.upper()].sort_values("dimension")
        line, = axis.plot(
            subset["dimension"].to_numpy(), subset["accuracy"].to_numpy(),
            marker="o", linewidth=2.5, markersize=7,
            color=colors[display_level], label=display_level, clip_on=False,
        )
        line.set_clip_path(None)
    axis.set_title("EMP 16S Accuracy vs HDC Hypervector Dimension", fontsize=20)
    axis.set_xlabel("HDC hypervector dimension", fontsize=14)
    axis.set_ylabel("Accuracy", fontsize=14)
    axis.set_xscale("log", base=2)
    axis.set_xticks(DIMENSIONS)
    axis.set_xticklabels([str(value) for value in DIMENSIONS])
    axis.set_ylim(max(0.0, frame["accuracy"].min() - 0.03), 1.0)
    axis.grid(axis="y", color="#dbe3ef", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, title=None)
    fig.savefig(output, format="svg")
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)


def prepare_level(counts, metadata: pd.DataFrame, sample_ids, level: str, args):
    valid_idx = np.asarray([
        index for index, value in enumerate(metadata[level].tolist())
        if clean_label(value)
    ], dtype=np.int32)
    labels = np.asarray([
        clean_label(metadata[level].iloc[index]) for index in valid_idx
    ])
    train_idx, test_idx = split_for_label(
        np.asarray(sample_ids)[valid_idx].tolist(), labels.tolist(),
        args.test_size, args.random_state,
    )
    level_counts = counts[valid_idx]
    transformer = TfidfTransformer(norm=None, sublinear_tf=True)
    weighted = transformer.fit_transform(level_counts[train_idx])
    weighted_all = transformer.transform(level_counts).tocsr().astype(np.float32)
    return labels, train_idx, test_idx, weighted_all


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    projection_dir = args.outdir / "projection_cache"
    projection_dir.mkdir(exist_ok=True)

    log("Loading EMP BIOM and metadata")
    sample_ids, observation_ids, counts, metadata, dataset_stats = load_emp_dataset(args)
    log(
        f"Loaded {len(sample_ids)} samples x {len(observation_ids)} ASVs; "
        f"{counts.nnz} nonzero counts"
    )

    prepared = {
        level: prepare_level(counts, metadata, sample_ids, level, args)
        for level in LEVELS
    }
    rows: list[dict] = []

    for dimension in DIMENSIONS:
        log(f"Dimension {dimension}")
        for level in LEVELS:
            labels, train_idx, test_idx, weighted = prepared[level]
            settings = LEVEL_SETTINGS[level]
            projection_path = projection_dir / (
                f"whole_sequence_d{dimension}_as{settings['active_dims']}.npz"
            )
            if projection_path.exists():
                projection = sparse.load_npz(projection_path).tocsr()
            else:
                projection = sequence_projection(
                    observation_ids,
                    dimension,
                    settings["active_dims"],
                    seed=args.random_state,
                )
                sparse.save_npz(projection_path, projection, compressed=True)

            started = time.perf_counter()
            matrix = (weighted @ projection).tocsr()
            normalize(matrix, norm="l2", copy=False)
            model = LinearSVC(
                C=settings["c_value"],
                dual="auto",
                max_iter=12000,
                random_state=args.random_state,
            )
            model.fit(matrix[train_idx], labels[train_idx])
            prediction = model.predict(matrix[test_idx])
            elapsed = time.perf_counter() - started
            rows.append({
                "level": level.upper(),
                "dimension": dimension,
                "active_dims": settings["active_dims"],
                "c_value": settings["c_value"],
                "samples": len(labels),
                "train_samples": len(train_idx),
                "test_samples": len(test_idx),
                "accuracy": float(accuracy_score(labels[test_idx], prediction)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels[test_idx], prediction)
                ),
                "build_and_train_sec": elapsed,
            })
            log(
                f"{level.upper()} d={dimension}: "
                f"accuracy={rows[-1]['accuracy']:.4f}"
            )
            del matrix, projection

        write_rows(args.outdir / "dimension_accuracy.csv", rows)

    frame = pd.DataFrame(rows)
    plot_dimension_accuracy(
        frame, args.outdir / "emp_16s_dimension_accuracy.svg"
    )

    (args.outdir / "settings.json").write_text(json.dumps({
        **vars(args),
        "dimensions": DIMENSIONS,
        "levels": LEVELS,
        "level_settings": LEVEL_SETTINGS,
        "sequence_encoding": "whole-sequence",
        "weighting": "sublinear TF-IDF fit on each level's training subset",
        "evaluation": "stratified random 80/20 split, random_state=42",
        **dataset_stats,
    }, default=str, indent=2, sort_keys=True))
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
