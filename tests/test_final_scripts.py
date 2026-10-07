from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "examples" / "generate_synthetic_data.py"


class PublicSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = Path(tempfile.mkdtemp(prefix="multi_variant_smoke_"))
        cls.data = cls.tempdir / "data"
        cls.outputs = cls.tempdir / "outputs"
        cls.run_command([sys.executable, str(GENERATOR), "--outdir", str(cls.data)])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir, ignore_errors=True)

    @staticmethod
    def run_command(command):
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise AssertionError(
                f"Command failed ({result.returncode}): {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def test_all_entrypoints_and_output_shapes(self):
        commands = [
            (
                ROOT / "scripts" / "marine_edna_three_methods.py",
                ["--csv", self.data / "marine_edna.csv", "--backend", "cpu"],
                3,
                "results.csv",
            ),
            (
                ROOT / "scripts" / "empo_three_methods.py",
                [
                    "--biom", self.data / "emp_feature_table.biom",
                    "--metadata", self.data / "emp_metadata.tsv",
                ],
                9,
                "results.csv",
            ),
            (
                ROOT / "scripts" / "hmtol_before_qc_three_methods.py",
                [
                    "--data-dir", self.data / "hmtol_before_qc",
                    "--sequence-qza", self.data / "synthetic-sequences.qza",
                    "--hdc-dim", "64", "--active-dims", "4",
                ],
                3,
                "results.csv",
            ),
            (
                ROOT / "scripts" / "hmtol_after_qc_three_methods.py",
                [
                    "--data-dir", self.data / "hmtol_qc",
                    "--sequence-qza", self.data / "synthetic-sequences.qza",
                    "--targets", "Continent", "region",
                    "--folds", "3", "--hdc-dim", "64", "--active-dims", "4",
                ],
                6,
                "results.csv",
            ),
        ]
        for index, (script, options, expected_rows, filename) in enumerate(commands):
            outdir = self.outputs / f"case_{index}"
            self.run_command([
                sys.executable, str(script), *map(str, options),
                "--outdir", str(outdir), "--n-estimators", "5", "--repeats", "1",
            ])
            output = outdir / filename
            self.assertTrue(output.is_file(), output)
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), expected_rows)
            self.assertTrue({"method", "accuracy"}.issubset(rows[0]))
            self.assertTrue(all(0.0 <= float(row["accuracy"]) <= 1.0 for row in rows))

    def test_help_for_all_entrypoints(self):
        for script in sorted((ROOT / "scripts").glob("*_three_methods.py")):
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout.lower())

    def test_missing_input_has_actionable_error(self):
        output = self.outputs / "missing"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "marine_edna_three_methods.py"),
                "--csv", str(self.data / "does-not-exist.csv"),
                "--outdir", str(output),
                "--backend", "cpu",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not found", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
