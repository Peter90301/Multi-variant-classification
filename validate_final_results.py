#!/usr/bin/env python3
"""Validate the consolidated three-method result table."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "final_results.csv"
EXPECTED_METHODS = {
    "Random Forest",
    "Explicit-Vocab (SVM)",
    "HDC-Linear_opt",
}
EXPECTED_TASKS = {
    "Marine location",
    "EMPO1",
    "EMPO2",
    "EMPO3",
    "HMTOL before QC",
    "HMTOL QC Continent",
    "HMTOL QC Region",
}
EXPECTED_HDC_DIMENSIONS = {
    "Marine location": 32768,
    "EMPO1": 32768,
    "EMPO2": 32768,
    "EMPO3": 16384,
    "HMTOL before QC": 32768,
    "HMTOL QC Continent": 32768,
    "HMTOL QC Region": 32768,
}


def main() -> None:
    with RESULTS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 21, f"expected 21 rows, found {len(rows)}"
    assert {row["task"] for row in rows} == EXPECTED_TASKS
    assert {row["method"] for row in rows} == EXPECTED_METHODS

    task_methods: dict[str, set[str]] = defaultdict(set)
    pairs = Counter()
    for row in rows:
        task = row["task"]
        method = row["method"]
        task_methods[task].add(method)
        pairs[(task, method)] += 1

        accuracy = float(row["accuracy"])
        speedup = float(row["speedup_vs_rf"])
        assert 0.0 <= accuracy <= 1.0, (task, method, accuracy)
        assert speedup > 0.0, (task, method, speedup)

        if method == "Random Forest":
            assert speedup == 1.0, (task, speedup)
        elif method == "HDC-Linear_opt":
            dimension = int(row["hdc_dimension"])
            assert dimension == EXPECTED_HDC_DIMENSIONS[task], (
                task,
                dimension,
            )

    for task, methods in task_methods.items():
        assert methods == EXPECTED_METHODS, (task, methods)
    assert all(count == 1 for count in pairs.values()), "duplicate task/method row"

    print(
        f"Validated {len(rows)} rows: {len(EXPECTED_TASKS)} tasks x "
        f"{len(EXPECTED_METHODS)} methods."
    )


if __name__ == "__main__":
    main()
