#!/usr/bin/env python3
"""Run RF, Explicit-Vocab, and tuned HDC-Linear on EMP EMPO1-3."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from biom import load_table
from sklearn.model_selection import train_test_split

from three_method_common import (
    evaluate_three_methods,
    sequence_projection,
    write_rows,
)


CONFIGS = {
    "empo_1": {"dimension": 32768, "active_dims": 32, "hdc_c": 2.0},
    "empo_2": {"dimension": 16384, "active_dims": 32, "hdc_c": 1.0},
    "empo_3": {"dimension": 32768, "active_dims": 16, "hdc_c": 20.0},
}
MISSING = {"", "NA", "N/A", "nan", "None", "null", "Unknown", "unknown"}


def clean_label(value) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text in MISSING else text


def load_emp(biom_path: Path, metadata_path: Path, min_sample_sum: float):
    table = load_table(str(biom_path))
    all_samples = np.asarray(list(map(str, table.ids(axis="sample"))))
    observation_ids = list(map(str, table.ids(axis="observation")))
    sample_sums = np.asarray(table.sum(axis="sample"), dtype=np.float64)
    kept = all_samples[sample_sums >= min_sample_sum]
    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str, low_memory=False)
    if "#SampleID" not in metadata.columns:
        raise ValueError("Metadata is missing #SampleID")
    metadata = metadata.set_index("#SampleID", drop=False)
    sample_ids = [sample for sample in kept if sample in metadata.index]
    positions = {sample: index for index, sample in enumerate(all_samples)}
    selected = np.asarray([positions[sample] for sample in sample_ids], dtype=np.int32)
    counts = table.matrix_data.tocsc()[:, selected].T.tocsr().astype(np.float32)
    counts.sort_indices()
    return np.asarray(sample_ids), observation_ids, counts, metadata.loc[sample_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biom", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--min-sample-sum", type=float, default=1000.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    sample_ids, observation_ids, counts, metadata = load_emp(
        args.biom, args.metadata, args.min_sample_sum
    )
    all_rows = []
    for level, config in CONFIGS.items():
        valid = np.asarray([
            index for index, value in enumerate(metadata[level].tolist())
            if clean_label(value)
        ], dtype=np.int32)
        labels = np.asarray([
            clean_label(metadata[level].iloc[index]) for index in valid
        ])
        relative = np.arange(len(valid), dtype=np.int32)
        stratify = labels if min(Counter(labels).values()) >= 2 else None
        train_idx, test_idx = train_test_split(
            relative, test_size=args.test_size, random_state=args.random_state,
            stratify=stratify,
        )
        projection = sequence_projection(
            observation_ids,
            config["dimension"],
            config["active_dims"],
            args.random_state,
        )
        rows, _ = evaluate_three_methods(
            counts[valid], labels, train_idx, test_idx, projection,
            hdc_c=config["hdc_c"], repeats=args.repeats,
            random_state=args.random_state, n_estimators=args.n_estimators,
        )
        for row in rows:
            all_rows.append({
                "task": level.upper().replace("_", ""),
                "samples": len(labels),
                "classes": len(np.unique(labels)),
                "hdc_dimension": config["dimension"],
                "hdc_active_dims": config["active_dims"],
                "hdc_c": config["hdc_c"],
                **row,
            })
        print(f"Completed {level}", flush=True)

    write_rows(args.outdir / "results.csv", all_rows)
    settings = {
        **vars(args),
        "samples_after_filter": len(sample_ids),
        "features": len(observation_ids),
        "configs": CONFIGS,
        "timing_scope": "in-memory test counts through prediction; training excluded",
    }
    (args.outdir / "settings.json").write_text(
        json.dumps(settings, default=str, indent=2)
    )
    print(json.dumps(all_rows, indent=2))


if __name__ == "__main__":
    main()
