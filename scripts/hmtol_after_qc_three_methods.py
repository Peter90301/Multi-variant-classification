#!/usr/bin/env python3
"""Run the final three methods on study-held-out HMTOL QC targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from biom import load_table
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from three_method_common import (
    METHODS,
    evaluate_three_methods,
    extract_reference_sequences,
    require_columns,
    require_file,
    require_unique,
    sequence_projection,
    validate_positive_options,
    write_rows,
)


def load_dataset(data_dir: Path):
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"HMTOL QC data directory was not found: {data_dir}. "
            "Provide feature-table.qc.min3.biom and metadata.qc.min3.tsv."
        )
    biom_path = data_dir / "feature-table.qc.min3.biom"
    metadata_path = data_dir / "metadata.qc.min3.tsv"
    require_file(biom_path, "HMTOL QC feature table")
    require_file(metadata_path, "HMTOL QC metadata")
    table = load_table(str(biom_path))
    sample_ids = np.asarray(list(map(str, table.ids(axis="sample"))))
    observation_ids = list(map(str, table.ids(axis="observation")))
    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str, low_memory=False)
    require_columns(
        metadata.columns,
        {"SampleID", "study", "Continent", "region"},
        "HMTOL QC metadata",
    )
    require_unique(metadata["SampleID"].tolist(), "HMTOL QC metadata SampleID")
    if len(sample_ids) != len(metadata):
        raise ValueError(
            "QC BIOM and metadata have different numbers of samples. "
            "They must contain the same samples in the same order."
        )
    if sample_ids.tolist() != metadata["SampleID"].tolist():
        raise ValueError(
            "QC BIOM sample order does not match metadata SampleID order. "
            "Reorder the metadata rows to exactly match the BIOM table."
        )
    counts = table.matrix_data.T.tocsr().astype(np.float32)
    counts.sort_indices()
    return sample_ids, observation_ids, counts, metadata


def balanced_study_folds(
    metadata: pd.DataFrame,
    target: str,
    study_column: str,
    n_splits: int,
):
    """Balance target counts across folds while keeping studies intact."""
    if study_column not in metadata.columns:
        raise ValueError(
            f"HMTOL QC metadata is missing study column {study_column!r}."
        )
    if target not in metadata.columns:
        raise ValueError(
            f"HMTOL QC metadata is missing target column {target!r}. "
            "Use exact case-sensitive names such as Continent or region."
        )
    for column in (study_column, target):
        values = metadata[column].astype(str).str.strip()
        if values.isin({"", "nan", "NA", "N/A", "None", "null"}).any():
            raise ValueError(
                f"HMTOL QC column {column!r} contains missing values. "
                "Fill them before running study-held-out evaluation."
            )
    studies = np.asarray(sorted(metadata[study_column].unique()))
    classes = np.asarray(sorted(metadata[target].unique()))
    if len(studies) < n_splits:
        raise ValueError(
            f"Study-held-out evaluation with {n_splits} folds needs at least "
            f"{n_splits} studies; found {len(studies)}."
        )
    study_pos = {value: index for index, value in enumerate(studies)}
    class_pos = {value: index for index, value in enumerate(classes)}
    weights = np.zeros((len(studies), len(classes)), dtype=np.float64)
    for (study, label), count in metadata.groupby([study_column, target]).size().items():
        weights[study_pos[study], class_pos[label]] = count

    support = (weights > 0).sum(axis=0)
    if np.any(support < n_splits):
        too_few = [
            f"{classes[index]!r} ({count} studies)"
            for index, count in enumerate(support) if count < n_splits
        ]
        raise ValueError(
            f"Each target class must occur in at least {n_splits} different "
            f"studies; insufficient class support: {', '.join(too_few)}."
        )

    groups = len(studies)
    class_count = len(classes)
    assignment_vars = groups * n_splits
    deviation_vars = n_splits * class_count * 2
    variable_count = assignment_vars + deviation_vars
    objective = np.zeros(variable_count)
    totals = weights.sum(axis=0)
    for fold in range(n_splits):
        for class_index in range(class_count):
            offset = fold * class_count + class_index
            penalty = 1.0 / totals[class_index]
            objective[assignment_vars + offset] = penalty
            objective[assignment_vars + n_splits * class_count + offset] = penalty

    rows = []
    lower = []
    upper = []
    for group in range(groups):
        row = sparse.lil_matrix((1, variable_count))
        for fold in range(n_splits):
            row[0, group * n_splits + fold] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)

    for fold in range(n_splits):
        for class_index in range(class_count):
            offset = fold * class_count + class_index
            row = sparse.lil_matrix((1, variable_count))
            for group in range(groups):
                row[0, group * n_splits + fold] = weights[group, class_index]
            row[0, assignment_vars + offset] = -1.0
            row[0, assignment_vars + n_splits * class_count + offset] = 1.0
            rows.append(row)
            target_count = totals[class_index] / n_splits
            lower.append(target_count)
            upper.append(target_count)

            presence = sparse.lil_matrix((1, variable_count))
            for group in np.flatnonzero(weights[:, class_index] > 0):
                presence[0, group * n_splits + fold] = 1.0
            rows.append(presence)
            lower.append(1.0)
            upper.append(np.inf)

    result = milp(
        objective,
        integrality=np.r_[np.ones(assignment_vars), np.zeros(deviation_vars)],
        bounds=Bounds(
            np.zeros(variable_count),
            np.r_[np.ones(assignment_vars), np.full(deviation_vars, np.inf)],
        ),
        constraints=LinearConstraint(
            sparse.vstack(rows, format="csc"), np.asarray(lower), np.asarray(upper)
        ),
        options={"time_limit": 60},
    )
    if result.x is None:
        raise RuntimeError(f"Unable to construct study folds: {result.message}")
    assignment = result.x[:assignment_vars].reshape(groups, n_splits).argmax(axis=1)
    fold_by_study = dict(zip(studies, assignment, strict=True))
    sample_folds = metadata[study_column].map(fold_by_study).to_numpy(np.int32)
    return classes, fold_by_study, sample_folds


def run_target(args, target, sample_ids, counts, metadata, projection):
    classes, fold_by_study, sample_folds = balanced_study_folds(
        metadata, target, args.study_column, args.folds
    )
    class_to_int = {label: index for index, label in enumerate(classes)}
    labels = metadata[target].map(class_to_int).to_numpy(np.int32)
    predictions = {
        method: np.full(len(labels), -1, dtype=np.int32) for method in METHODS
    }
    fold_rows = []
    for fold in range(args.folds):
        test_idx = np.flatnonzero(sample_folds == fold).astype(np.int32)
        train_idx = np.flatnonzero(sample_folds != fold).astype(np.int32)
        train_studies = set(metadata.iloc[train_idx][args.study_column])
        test_studies = set(metadata.iloc[test_idx][args.study_column])
        if train_studies & test_studies:
            raise AssertionError("Study leakage detected")
        rows, fold_predictions = evaluate_three_methods(
            counts, labels, train_idx, test_idx, projection,
            hdc_c=args.hdc_c, repeats=args.repeats,
            random_state=args.random_state, n_estimators=args.n_estimators,
        )
        for row in rows:
            fold_rows.append({"target": target, "fold": fold + 1, **row})
        for method, values in fold_predictions.items():
            predictions[method][test_idx] = values
        print(f"Completed {target} fold {fold + 1}", flush=True)

    summary = []
    for method in METHODS:
        method_rows = [row for row in fold_rows if row["method"] == method]
        total_time = sum(row["prediction_pipeline_mean_sec"] for row in method_rows)
        summary.append({
            "target": target,
            "method": method,
            "accuracy": float(accuracy_score(labels, predictions[method])),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels, predictions[method])
            ),
            "samples": len(labels),
            "studies": metadata[args.study_column].nunique(),
            "classes": len(classes),
            "cv_folds": args.folds,
            "prediction_pipeline_mean_total_sec": total_time,
            "speedup_vs_random_forest": 0.0,
        })
    rf_time = summary[0]["prediction_pipeline_mean_total_sec"]
    for row in summary:
        row["speedup_vs_random_forest"] = (
            rf_time / row["prediction_pipeline_mean_total_sec"]
        )

    target_dir = args.outdir / target.lower()
    write_rows(target_dir / "fold_results.csv", fold_rows)
    write_rows(target_dir / "summary.csv", summary)
    write_rows(target_dir / "study_fold_assignments.csv", [
        {"study": study, "fold": int(fold) + 1}
        for study, fold in sorted(fold_by_study.items())
    ])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sequence-qza", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", default=["Continent", "region"])
    parser.add_argument("--study-column", default="study")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hdc-dim", type=int, default=32768)
    parser.add_argument("--active-dims", type=int, default=32)
    parser.add_argument("--hdc-c", type=float, default=10.0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.folds < 2:
        raise ValueError("--folds must be at least 2 for study-held-out evaluation.")
    validate_positive_options(args.repeats, args.n_estimators)

    sample_ids, observation_ids, counts, metadata = load_dataset(args.data_dir)
    sequences = extract_reference_sequences(
        args.sequence_qza, observation_ids,
        args.outdir / "selected_reference_sequences.npy",
    )
    projection = sequence_projection(
        sequences, args.hdc_dim, args.active_dims, args.random_state
    )
    all_rows = []
    for target in args.targets:
        all_rows.extend(
            run_target(args, target, sample_ids, counts, metadata, projection)
        )
    write_rows(args.outdir / "results.csv", all_rows)
    settings = {
        **vars(args),
        "samples": len(sample_ids),
        "features": len(observation_ids),
        "evaluation": "balanced study-held-out cross-validation",
        "study_id_used_as_feature": False,
        "timing_scope": "in-memory test counts through prediction; training excluded",
        "sample_order_requirement": "BIOM sample IDs must exactly match metadata SampleID order",
        "study_requirement": "each target class must occur in at least --folds studies",
    }
    (args.outdir / "settings.json").write_text(
        json.dumps(settings, default=str, indent=2)
    )
    print(json.dumps(all_rows, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error
