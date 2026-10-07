"""End-to-end tests using only generated fictional data."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from synthetic_data import write_emp, write_hmtol_before, write_hmtol_qc, write_marine


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXPECTED = {"Random Forest", "Explicit-Vocab (SVM)", "HDC-Linear_opt"}


def run(script: str, *args: object) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        check=True, capture_output=True, text=True,
    )


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def assert_methods(test: unittest.TestCase, result_rows: list[dict[str, str]]) -> None:
    test.assertEqual({row["method"] for row in result_rows}, EXPECTED)
    for row in result_rows:
        test.assertGreaterEqual(float(row["accuracy"]), 0.0)
        test.assertLessEqual(float(row["accuracy"]), 1.0)
        test.assertGreater(float(row["speedup_vs_random_forest"]), 0.0)


class FinalScriptTests(unittest.TestCase):
    def test_marine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_marine(root / "marine.csv")
            output = root / "output"
            run("marine_edna_three_methods.py", "--csv", root / "marine.csv", "--outdir", output,
                "--hdc-dim", 256, "--n-estimators", 10, "--repeats", 1)
            result = rows(output / "results.csv")
            self.assertEqual(len(result), 3)
            assert_methods(self, result)

    def test_empo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            biom, metadata = write_emp(root)
            output = root / "output"
            run("empo_three_methods.py", "--biom", biom, "--metadata", metadata,
                "--outdir", output, "--min-sample-sum", 0, "--n-estimators", 10, "--repeats", 1)
            result = rows(output / "results.csv")
            self.assertEqual(len(result), 9)
            for task in {row["task"] for row in result}:
                assert_methods(self, [row for row in result if row["task"] == task])

    def test_hmtol_before_qc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qza = write_hmtol_before(root)
            output = root / "output"
            run("hmtol_before_qc_three_methods.py", "--data-dir", root, "--sequence-qza", qza,
                "--outdir", output, "--min-sample-sum", 0, "--hdc-dim", 256,
                "--active-dims", 4, "--n-estimators", 10, "--repeats", 1)
            result = rows(output / "results.csv")
            self.assertEqual(len(result), 3)
            assert_methods(self, result)

    def test_hmtol_after_qc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qza = write_hmtol_qc(root)
            output = root / "output"
            run("hmtol_after_qc_three_methods.py", "--data-dir", root, "--sequence-qza", qza,
                "--outdir", output, "--folds", 3, "--hdc-dim", 256,
                "--active-dims", 4, "--n-estimators", 10, "--repeats", 1)
            result = rows(output / "results.csv")
            self.assertEqual(len(result), 6)
            for target in {row["target"] for row in result}:
                assert_methods(self, [row for row in result if row["target"] == target])


if __name__ == "__main__":
    unittest.main()
