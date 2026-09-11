#!/usr/bin/env python3
"""Tune EMP HDC sequence encodings on validation data, then test once."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from benchmark_emp_16s_empo import (
    DEFAULT_BIOM,
    DEFAULT_METADATA,
    build_sequence_projection,
    clean_label,
    load_emp_dataset,
    split_for_label,
)


ROOT = Path("/home/tsl012/Multi_variant classification")
OUTDIR = ROOT / "emp_16s_hdc_encoding_tuning"
LEVELS = ["empo_1", "empo_2", "empo_3"]

if not hasattr(np, "Inf"):
    np.Inf = np.inf


@dataclass(frozen=True)
class Config:
    name: str
    encoding: str
    dimension: int
    weighting: str
    active_sequence: int = 4
    kmer_size: int = 6
    kmer_stride: int = 1
    max_kmers: int = 24
    active_kmer: int = 4
    c_value: float = 1.0


CONFIGS = [
    Config("current_whole_raw_d4096_a4", "whole-sequence", 4096, "raw"),
    Config("whole_log_d4096_a4", "whole-sequence", 4096, "log1p"),
    Config("whole_tfidf_d4096_a4", "whole-sequence", 4096, "tfidf"),
    Config(
        "whole_tfidf_d16384_a16", "whole-sequence", 16384, "tfidf",
        active_sequence=16,
    ),
    Config(
        "whole_tfidf_d32768_a32", "whole-sequence", 32768, "tfidf",
        active_sequence=32,
    ),
    Config(
        "kmer5_tfidf_d8192", "kmer", 8192, "tfidf", kmer_size=5,
        kmer_stride=2, max_kmers=16, active_kmer=2,
    ),
    Config(
        "kmer6_tfidf_d8192", "kmer", 8192, "tfidf", kmer_size=6,
        kmer_stride=2, max_kmers=16, active_kmer=2,
    ),
    Config(
        "kmer8_tfidf_d8192", "kmer", 8192, "tfidf", kmer_size=8,
        kmer_stride=2, max_kmers=16, active_kmer=2,
    ),
    Config(
        "kmer6_tfidf_d16384", "kmer", 16384, "tfidf", kmer_size=6,
        kmer_stride=2, max_kmers=24, active_kmer=4,
    ),
]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biom", type=Path, default=DEFAULT_BIOM)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--min-sample-sum", type=float, default=1000.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--selection-metric", choices=["accuracy", "balanced_accuracy"],
                        default="accuracy")
    return parser.parse_args()


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def projection_path(cache_dir, config):
    fields = [config.encoding, f"d{config.dimension}"]
    if config.encoding == "whole-sequence":
        fields.append(f"as{config.active_sequence}")
    else:
        fields.extend([
            f"k{config.kmer_size}", f"s{config.kmer_stride}",
            f"m{config.max_kmers}", f"ak{config.active_kmer}",
        ])
    return cache_dir / ("_".join(fields).replace("-", "_") + ".npz")


def get_projection(cache_dir, observation_ids, config, seed):
    path = projection_path(cache_dir, config)
    if path.exists():
        return sparse.load_npz(path).tocsr()
    log(f"Building projection {path.stem}")
    projection = build_sequence_projection(
        observation_ids,
        dim=config.dimension,
        sequence_encoding=config.encoding,
        kmer_size=config.kmer_size,
        kmer_stride=config.kmer_stride,
        max_sequence_kmers=config.max_kmers,
        active_dims_per_kmer=config.active_kmer,
        active_dims_per_sequence=config.active_sequence,
        seed=seed,
    )
    sparse.save_npz(path, projection, compressed=True)
    return projection


def apply_weighting(counts, fit_indices, mode):
    if mode == "raw":
        return counts, None
    if mode == "log1p":
        weighted = counts.copy()
        weighted.data = np.log1p(weighted.data)
        return weighted, None
    transformer = TfidfTransformer(norm=None, sublinear_tf=True)
    transformer.fit(counts[fit_indices])
    return transformer.transform(counts).tocsr().astype(np.float32), transformer


def build_hdc(weighted, projection):
    matrix = (weighted @ projection).tocsr()
    normalize(matrix, norm="l2", copy=False)
    return matrix


def metrics(truth, prediction):
    return {
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "macro_f1": f1_score(truth, prediction, average="macro", zero_division=0),
    }


def prepare_levels(args, sample_ids, metadata):
    levels = {}
    sample_ids = np.asarray(sample_ids)
    for level in LEVELS:
        valid_idx = []
        labels = []
        for index, value in enumerate(metadata[level].tolist()):
            label = clean_label(value)
            if label:
                valid_idx.append(index)
                labels.append(label)
        valid_idx = np.asarray(valid_idx, dtype=np.int32)
        labels = np.asarray(labels)
        outer_train, outer_test = split_for_label(
            sample_ids[valid_idx].tolist(), labels.tolist(),
            args.test_size, args.random_state,
        )
        fit_rel, validation_rel = train_test_split(
            np.arange(len(outer_train)), test_size=args.validation_size,
            random_state=args.random_state + 1, stratify=labels[outer_train],
        )
        levels[level] = {
            "valid_idx": valid_idx,
            "labels": labels,
            "outer_train": np.asarray(outer_train),
            "outer_test": np.asarray(outer_test),
            "fit": np.asarray(outer_train)[fit_rel],
            "validation": np.asarray(outer_train)[validation_rel],
        }
    return levels


def evaluate_config(args, counts, projection, config, level, split):
    level_counts = counts[split["valid_idx"]]
    started = time.perf_counter()
    weighted, _ = apply_weighting(level_counts, split["fit"], config.weighting)
    matrix = build_hdc(weighted, projection)
    build_sec = time.perf_counter() - started
    model = LinearSVC(
        C=config.c_value, dual="auto", max_iter=12000,
        random_state=args.random_state, class_weight=None,
    )
    started = time.perf_counter()
    model.fit(matrix[split["fit"]], split["labels"][split["fit"]])
    train_sec = time.perf_counter() - started
    prediction = model.predict(matrix[split["validation"]])
    result = metrics(split["labels"][split["validation"]], prediction)
    del matrix, weighted
    return {
        "level": level.upper(),
        "stage": "encoding",
        **asdict(config),
        **{f"validation_{key}": value for key, value in result.items()},
        "feature_build_sec": build_sec,
        "train_sec": train_sec,
    }


def tune_c(args, counts, projection, config, level, split):
    level_counts = counts[split["valid_idx"]]
    weighted, _ = apply_weighting(level_counts, split["fit"], config.weighting)
    matrix = build_hdc(weighted, projection)
    rows = []
    for c_value in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        candidate = replace(config, name=f"{config.name}_c{c_value:g}", c_value=c_value)
        model = LinearSVC(
            C=c_value, dual="auto", max_iter=12000,
            random_state=args.random_state,
        )
        started = time.perf_counter()
        model.fit(matrix[split["fit"]], split["labels"][split["fit"]])
        train_sec = time.perf_counter() - started
        prediction = model.predict(matrix[split["validation"]])
        result = metrics(split["labels"][split["validation"]], prediction)
        rows.append({
            "level": level.upper(), "stage": "regularization",
            **asdict(candidate),
            **{f"validation_{key}": value for key, value in result.items()},
            "feature_build_sec": 0.0, "train_sec": train_sec,
        })
        log(f"{level.upper()} {candidate.name}: val_accuracy={result['accuracy']:.4f}")
    return rows


def timed_predict(model, matrix, repeats):
    model.predict(matrix)
    times = []
    prediction = None
    for _ in range(repeats):
        started = time.perf_counter()
        prediction = model.predict(matrix)
        times.append(time.perf_counter() - started)
    return prediction, np.asarray(times)


def final_test(args, counts, projection, config, level, split, output_dir):
    level_counts = counts[split["valid_idx"]]
    started = time.perf_counter()
    weighted, transformer = apply_weighting(
        level_counts, split["outer_train"], config.weighting
    )
    matrix = build_hdc(weighted, projection)
    build_sec = time.perf_counter() - started
    model = LinearSVC(
        C=config.c_value, dual="auto", max_iter=12000,
        random_state=args.random_state,
    )
    started = time.perf_counter()
    model.fit(matrix[split["outer_train"]], split["labels"][split["outer_train"]])
    train_sec = time.perf_counter() - started
    test_matrix = matrix[split["outer_test"]]
    prediction, times = timed_predict(model, test_matrix, args.repeats)
    truth = split["labels"][split["outer_test"]]
    result = {
        "level": level.upper(),
        **asdict(config),
        **{f"test_{key}": value for key, value in metrics(truth, prediction).items()},
        "train_samples": len(split["outer_train"]),
        "test_samples": len(split["outer_test"]),
        "full_feature_build_sec": build_sec,
        "train_sec": train_sec,
        "warmup_runs": 1,
        "timed_runs": args.repeats,
        "predict_mean_sec": times.mean(),
        "predict_std_sec": times.std(),
    }
    labels = sorted(set(split["labels"]))
    pd.DataFrame(classification_report(
        truth, prediction, labels=labels, output_dict=True, zero_division=0
    )).T.to_csv(output_dir / f"{level}_best_per_class_metrics.csv")
    raw = confusion_matrix(truth, prediction, labels=labels)
    normalized = confusion_matrix(truth, prediction, labels=labels, normalize="true")
    pd.DataFrame(raw, index=labels, columns=labels).to_csv(
        output_dir / f"{level}_best_confusion_raw.csv"
    )
    pd.DataFrame(normalized, index=labels, columns=labels).to_csv(
        output_dir / f"{level}_best_confusion_normalized.csv"
    )
    pd.DataFrame({
        "true_label": truth,
        "predicted_label": prediction,
    }).to_csv(output_dir / f"{level}_best_test_predictions.csv", index=False)
    return result


def plot_results(results, output):
    frame = pd.DataFrame(results)
    levels = [level.upper() for level in LEVELS]
    values = frame.set_index("level").loc[levels, "test_accuracy"]
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = axis.bar(levels, values, color=["#2563eb", "#f97316", "#16a34a"])
    axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=5, fontsize=13)
    axis.set_title("EMP 16S Tuned HDC Encoding Accuracy", fontsize=21)
    axis.set_xlabel("Classification level", fontsize=15)
    axis.set_ylabel("Untouched test accuracy", fontsize=15)
    axis.set_ylim(max(0, values.min() - 0.08), 1.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#dbe3ef", linewidth=0.8)
    axis.set_axisbelow(True)
    fig.savefig(output)
    plt.close(fig)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.outdir / "projection_cache"
    cache_dir.mkdir(exist_ok=True)
    log("Loading EMP BIOM and metadata")
    sample_ids, observation_ids, counts, metadata, dataset_stats = load_emp_dataset(args)
    splits = prepare_levels(args, sample_ids, metadata)
    trial_rows = []

    for config in CONFIGS:
        projection = get_projection(
            cache_dir, observation_ids, config, args.random_state
        )
        for level in LEVELS:
            row = evaluate_config(args, counts, projection, config, level, splits[level])
            trial_rows.append(row)
            write_rows(args.outdir / "validation_trials.csv", trial_rows)
            log(f"{level.upper()} {config.name}: "
                f"val_accuracy={row['validation_accuracy']:.4f}")
        del projection

    selected = {}
    for level in LEVELS:
        candidates = [row for row in trial_rows if row["level"] == level.upper()]
        winner = max(candidates, key=lambda row: (
            row[f"validation_{args.selection_metric}"], row["validation_macro_f1"]
        ))
        base = next(config for config in CONFIGS if config.name == winner["name"])
        projection = get_projection(cache_dir, observation_ids, base, args.random_state)
        c_rows = tune_c(args, counts, projection, base, level, splits[level])
        trial_rows.extend(c_rows)
        write_rows(args.outdir / "validation_trials.csv", trial_rows)
        best_c_row = max(c_rows, key=lambda row: (
            row[f"validation_{args.selection_metric}"], row["validation_macro_f1"]
        ))
        selected[level] = replace(base, c_value=float(best_c_row["c_value"]))
        del projection

    final_rows = []
    for level in LEVELS:
        config = selected[level]
        projection = get_projection(cache_dir, observation_ids, config, args.random_state)
        log(f"Final untouched test: {level.upper()} using {config.name}, C={config.c_value:g}")
        final_rows.append(final_test(
            args, counts, projection, config, level, splits[level], args.outdir
        ))
        del projection
    write_rows(args.outdir / "untouched_test_results.csv", final_rows)
    plot_results(final_rows, args.outdir / "tuned_hdc_test_accuracy.svg")
    settings = {
        **vars(args), **dataset_stats,
        "candidate_configs": [asdict(config) for config in CONFIGS],
        "selected_configs": {level: asdict(config) for level, config in selected.items()},
        "test_access_policy": "outer test used only after encoding and C selection",
    }
    (args.outdir / "settings.json").write_text(
        json.dumps(settings, default=str, indent=2, sort_keys=True)
    )
    print(pd.DataFrame(final_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
