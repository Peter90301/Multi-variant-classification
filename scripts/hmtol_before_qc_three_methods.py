#!/usr/bin/env python3
"""Run the final three methods on pre-QC HMTOL Country prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from biom import load_table
from sklearn.model_selection import train_test_split

from three_method_common import (
    evaluate_three_methods,
    extract_reference_sequences,
    require_columns,
    require_file,
    require_unique,
    sequence_projection,
    validate_labels,
    validate_positive_options,
    write_rows,
)


def load_dataset(data_dir: Path, min_sample_sum: float):
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"HMTOL data directory was not found: {data_dir}. "
            "Provide a directory containing feature-table.biom and metadata.tsv."
        )
    biom_path = data_dir / "feature-table.biom"
    metadata_path = data_dir / "metadata.tsv"
    require_file(biom_path, "HMTOL feature table")
    require_file(metadata_path, "HMTOL metadata")
    table = load_table(str(biom_path))
    all_samples = np.asarray(list(map(str, table.ids(axis="sample"))))
    observation_ids = list(map(str, table.ids(axis="observation")))
    sums = np.asarray(table.sum(axis="sample"), dtype=np.float64)
    kept = all_samples[sums >= min_sample_sum]
    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str, low_memory=False)
    require_columns(metadata.columns, {"SampleID", "Country"}, "HMTOL metadata")
    require_unique(metadata["SampleID"].tolist(), "HMTOL metadata SampleID")
    metadata = metadata.set_index("SampleID")
    sample_ids = [
        sample for sample in kept
        if sample in metadata.index
        and pd.notna(metadata.at[sample, "Country"])
        and str(metadata.at[sample, "Country"]).strip()
        not in {"", "NA", "N/A", "nan", "None", "null"}
    ]
    if not sample_ids:
        raise ValueError(
            "No HMTOL samples have both a matching metadata SampleID and a "
            "non-empty Country label after filtering."
        )
    positions = {sample: index for index, sample in enumerate(all_samples)}
    selected = np.asarray([positions[sample] for sample in sample_ids], dtype=np.int32)
    counts = table.matrix_data.tocsc()[:, selected].T.tocsr().astype(np.float32)
    counts.sort_indices()
    labels = metadata.loc[sample_ids, "Country"].astype(str).to_numpy()
    return np.asarray(sample_ids), observation_ids, counts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sequence-qza", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--min-sample-sum", type=float, default=1000.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hdc-dim", type=int, default=32768)
    parser.add_argument("--active-dims", type=int, default=32)
    parser.add_argument("--hdc-c", type=float, default=10.0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    validate_positive_options(args.repeats, args.n_estimators)
    sequence_qza = args.sequence_qza or args.data_dir / "2024.09.seqs.fna.qza"

    sample_ids, observation_ids, counts, labels = load_dataset(
        args.data_dir, args.min_sample_sum
    )
    validate_labels(labels, "HMTOL Country")
    indices = np.arange(len(labels), dtype=np.int32)
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.random_state,
        stratify=labels,
    )
    sequences = extract_reference_sequences(
        sequence_qza, observation_ids,
        args.outdir / "selected_reference_sequences.npy",
    )
    projection = sequence_projection(
        sequences, args.hdc_dim, args.active_dims, args.random_state
    )
    rows, predictions = evaluate_three_methods(
        counts, labels, train_idx, test_idx, projection,
        hdc_c=args.hdc_c, repeats=args.repeats,
        random_state=args.random_state, n_estimators=args.n_estimators,
    )
    for row in rows:
        row.update({
            "task": "HMTOL before QC",
            "target": "Country",
            "samples": len(labels),
            "classes": len(np.unique(labels)),
            "hdc_dimension": args.hdc_dim,
            "hdc_active_dims": args.active_dims,
            "hdc_c": args.hdc_c,
        })
    write_rows(args.outdir / "results.csv", rows)
    prediction_rows = []
    for offset, index in enumerate(test_idx):
        prediction_rows.append({
            "sample": sample_ids[index],
            "truth": labels[index],
            **{method: values[offset] for method, values in predictions.items()},
        })
    write_rows(args.outdir / "predictions.csv", prediction_rows)
    settings = {
        **vars(args),
        "sequence_qza": sequence_qza,
        "evaluation": "sample-level stratified random 80/20 split",
        "warning": "Country is confounded with Study ID",
        "timing_scope": "in-memory test counts through prediction; training excluded",
        "low_abundance_behavior": f"samples with total count < {args.min_sample_sum} are removed",
        "sample_id_matching": "BIOM sample IDs are matched exactly to metadata SampleID",
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
