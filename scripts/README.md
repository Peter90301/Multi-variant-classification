# Dataset entrypoints

These four scripts run the same three classifier families on different input
formats. They train and evaluate from labelled data; they do not load a
pretrained model or classify unlabelled samples.

| Dataset/task | Script | Outputs |
|---|---|---|
| Marine eDNA, `geo_loc_name` | `marine_edna_three_methods.py` | 3 method rows plus sample predictions |
| EMP 16S, `empo_1`/`empo_2`/`empo_3` | `empo_three_methods.py` | 9 rows: 3 methods x 3 targets |
| HMTOL before QC, `Country` | `hmtol_before_qc_three_methods.py` | 3 method rows plus sample predictions |
| HMTOL after QC, `Continent`/`region` | `hmtol_after_qc_three_methods.py` | 6 summary rows plus per-target/per-fold files |

Run every command from the repository root after installing
`scripts/requirements.txt`.

The commands below use Bash syntax for Linux/macOS/WSL. In Windows
PowerShell, use `py` instead of `python` and paste each command on one line;
PowerShell does not use Bash's trailing `\` continuation.

## Marine eDNA

```bash
python scripts/marine_edna_three_methods.py \
  --csv /path/to/marine_table.csv \
  --outdir output/marine \
  --backend cpu
```

PowerShell:

```powershell
py scripts/marine_edna_three_methods.py --csv C:\path\to\marine_table.csv --outdir output\marine --backend cpu
```

Required, case-sensitive CSV columns:

```text
sample,geo_loc_name,ASV_sequence,domain,phylum,class,order,family,genus,species
```

`count` is optional. If it is absent or invalid, the script uses count `1` for
that row. Repeated rows for one sample are expected and their taxonomy and
sequence-token features are aggregated into one explicit sample feature
matrix. The script extracts 6-mers from `ASV_sequence`, normalizes taxonomy
tokens to lowercase, and removes non-`ACGTN` sequence characters. Random
Forest and Explicit-Vocab use this Marine feature matrix; HDC hashes the same
feature dictionaries before TF-IDF.
Empty values in taxonomy or sequence cells are ignored, but the column itself
must exist. A sample with multiple `geo_loc_name` values is rejected because a
sample needs one target label. IDs and column names are case-sensitive.

The default HDC dimension is `32768`, HDC `C` is `16`, the default split is a
stratified random 80/20 split, and prediction timing uses one warm-up plus 20
runs. The default HDC backend is GPU cached-input; use `--backend cpu` on a
CPU-only computer. `results.csv` records the selected backend.

## EMP 16S

```bash
python scripts/empo_three_methods.py \
  --biom /path/to/emp_table.biom \
  --metadata /path/to/emp_mapping.tsv \
  --outdir output/empo
```

The BIOM sample IDs must match `#SampleID` in the tab-separated metadata,
exactly and case-sensitively. Required metadata columns are:

```text
#SampleID,empo_1,empo_2,empo_3
```

Samples with total BIOM abundance below `--min-sample-sum` (default `1000`)
are filtered. Missing-like labels (`NA`, `N/A`, `Unknown`, and empty values)
are excluded from that target. Duplicate metadata `#SampleID` values are
rejected. The output has three rows for each EMPO target, plus `settings.json`.

The final HDC settings in this script are EMPO1 `32768/32/C=2`, EMPO2
`16384/32/C=1`, and EMPO3 `32768/16/C=20` (dimension/active dimensions/SVM C).
Random Forest and Explicit-Vocab use the explicit BIOM sample-by-feature
abundance matrix. HDC applies sublinear TF-IDF to that matrix before its
deterministic sparse bipolar projection.
This entrypoint's prediction timing is CPU in-memory timing; it is not the
separately recorded GPU cached-input result in the canonical table.

## HMTOL before QC

```bash
python scripts/hmtol_before_qc_three_methods.py \
  --data-dir /path/to/hmtol_before_qc \
  --sequence-qza /path/to/reference-sequences.qza \
  --outdir output/hmtol_before_qc
```

The data directory must contain:

```text
feature-table.biom
metadata.tsv
```

The metadata must contain exact, case-sensitive columns `SampleID` and
`Country`. BIOM sample IDs are matched exactly to `SampleID`; missing or
low-abundance samples are excluded. The default threshold is `1000`. The QZA
must contain `data/dna-sequences.fasta` entries matching the BIOM feature IDs.
The output contains `results.csv`, `predictions.csv`, and `settings.json`.

This uses a stratified sample-level random 80/20 split. Country is confounded
with Study ID in the original data, so do not interpret this as study-held-out
generalization.

## HMTOL after QC

```bash
python scripts/hmtol_after_qc_three_methods.py \
  --data-dir /path/to/hmtol_qc \
  --sequence-qza /path/to/reference-sequences.qza \
  --outdir output/hmtol_qc \
  --targets Continent region \
  --folds 3
```

The data directory must contain:

```text
feature-table.qc.min3.biom
metadata.qc.min3.tsv
```

The metadata must contain `SampleID`, `study`, `Continent`, and `region`. The
BIOM IDs and metadata `SampleID` values must have identical length and order.
No abundance filtering is performed by this entrypoint. Each class must occur
in at least `--folds` different studies; otherwise a study-held-out fold is
not valid and the script stops with an actionable error. `study` controls the
split and is not a model feature.

The root `results.csv` has one summary row for each target and method. Each
target directory contains `fold_results.csv`, `summary.csv`, and
`study_fold_assignments.csv`. There are more than three output rows because
this task has two targets and multiple held-out folds.

## Timing fields

All scripts use one unrecorded warm-up and then `--repeats` timed prediction
runs (default 20). `train_sec` is measured separately. Marine GPU timing
includes cache load, host-to-device transfer, GPU sparse LinearSVC readout,
and device-to-host copy. EMP and HMTOL scripts currently time their in-memory
CPU prediction path. Marine, EMP, and HMTOL before-QC `results.csv` rows
contain `prediction_pipeline_mean_sec`, `execution`, and `timing_scope`. HMTOL
QC summary rows use `prediction_pipeline_mean_total_sec`; its fold rows use
`prediction_pipeline_mean_sec`. Always read the timing field together with
`execution` and `timing_scope` before comparing speedup values.

## Common options

```bash
python scripts/<entrypoint>.py --help
```

Useful options include `--random-state 42`, `--repeats 1` for a quick run,
`--n-estimators 20` for a smaller RF demonstration, `--test-size 0.2`,
`--min-sample-sum 1000`, and the HDC dimension/active-dimension options exposed
by the selected script. Sample-level splits require both train and test to have
at least one sample for every class, so very small examples may need a larger
`--test-size` or more samples.
