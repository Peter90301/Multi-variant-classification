# Multi-Variant Classification

## Overview

This repository compares three supervised classifiers for microbial and
environmental sequencing data: Random Forest, Explicit-Vocabulary SVM, and a
tuned linear HDC representation with a LinearSVC readout. The inputs are a
labelled feature table plus sample metadata. The programs split the labelled
samples into training and test data, fit all three methods, and write accuracy
and prediction-timing results.

This is an analysis and benchmarking repository, not a prediction service. The
current scripts do not save/load a trained model and do not predict labels for
unlabelled samples. Every run below trains and evaluates using labels supplied
in the input data.

## Requirements

- Python 3.13 was verified on 64-bit Linux with the pinned dependencies in
  `scripts/requirements.txt`.
- The CPU workflow needs only Python, NumPy, pandas, SciPy, scikit-learn, and
  BIOM-Format. It does not need an NVIDIA GPU.
- The optional Marine GPU backend needs an NVIDIA GPU, a working CUDA driver,
  and the CuPy package in `scripts/requirements-gpu.txt`.
- Linux is the only operating system validated by the release smoke tests.
  CPU execution may work on other systems when compatible wheels are available;
  Windows and macOS are not claimed as tested platforms here.

All commands below are run from the repository root. Paste them into a
Terminal, PowerShell, or WSL shell, depending on your operating system.

## Installation

### CPU installation

Linux/macOS/WSL:

```bash
git clone https://github.com/Peter90301/Multi-variant-classification.git
cd Multi-variant-classification
python3 -m venv .venv
```

On Linux, macOS, or WSL:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
git clone https://github.com/Peter90301/Multi-variant-classification.git
cd Multi-variant-classification
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

If `py` is not installed, replace every `py` or `py -3` command below with
`python`.

Linux/macOS/WSL install command:

```bash
python -m pip install --upgrade pip
python -m pip install -r scripts/requirements.txt
```

Windows PowerShell install command:

```powershell
py -m pip install --upgrade pip
py -m pip install -r scripts/requirements.txt
```

### Optional GPU installation

Install the CPU dependencies first, then install CuPy for CUDA 12:

```bash
python -m pip install -r scripts/requirements-gpu.txt
```

The GPU option is explicit. On a computer without CuPy or a usable CUDA
device, use Marine with `--backend cpu`. The EMP and HMTOL entrypoints in this
repository currently use the shared CPU in-memory prediction implementation;
their research speedup tables refer to separately recorded cached GPU
benchmarks and are documented in [docs/RESULTS.md](docs/RESULTS.md).

### Docker (CPU)

The repository includes a CPU-only Docker image. From the repository root:

```bash
docker build -t multi-variant-classification .
docker run --rm multi-variant-classification
```

The default command prints the Marine entrypoint help. To run the release-safe
synthetic example, create the fictional data on the host and mount the
repository into the container:

```bash
python3 examples/generate_synthetic_data.py --outdir examples/data
docker run --rm -v "$PWD:/app" multi-variant-classification \
  python scripts/marine_edna_three_methods.py \
  --csv examples/data/marine_edna.csv \
  --outdir output/docker_quickstart \
  --backend cpu --hdc-dim 256 --n-estimators 20 --repeats 1
```

PowerShell:

```powershell
py examples/generate_synthetic_data.py --outdir examples/data
docker run --rm -v "${PWD}:/app" multi-variant-classification python scripts/marine_edna_three_methods.py --csv examples/data/marine_edna.csv --outdir output/docker_quickstart --backend cpu --hdc-dim 256 --n-estimators 20 --repeats 1
```

The Dockerfile does not install CuPy or provide GPU access. GPU execution
requires a compatible NVIDIA host, NVIDIA Container Toolkit, and a separate
CUDA-compatible image/dependency setup.

### Docker (GPU)

`Dockerfile.gpu` is a separate NVIDIA CUDA 12 image for the Marine GPU
backend. The host must have a compatible NVIDIA driver and NVIDIA Container
Toolkit. Build and check GPU visibility from the repository root:

```bash
docker build -f Dockerfile.gpu -t multi-variant-classification:gpu .
docker run --rm --gpus all multi-variant-classification:gpu
```

Run Marine with a host data directory mounted into the container:

```bash
docker run --rm --gpus all \
  -v "$PWD:/app" \
  -v /path/to/data:/data:ro \
  multi-variant-classification:gpu \
  python scripts/marine_edna_three_methods.py \
  --csv /data/marine_table.csv \
  --outdir output/marine_gpu \
  --backend gpu
```

The GPU image installs CuPy for CUDA 12. It does not make the EMP or HMTOL
public entrypoints GPU implementations; those entrypoints currently use the
shared CPU in-memory path described in [docs/RESULTS.md](docs/RESULTS.md).

### VS Code Dev Container

The repository includes `.devcontainer/devcontainer.json`, which points to
the CPU `Dockerfile`. Install the **Dev Containers** extension in VS Code,
open the repository folder, and select **Reopen in Container** when prompted.
VS Code will build the environment from the repository Dockerfile. This uses
the CPU image; GPU access still requires the separate `Dockerfile.gpu` and a
host configured with NVIDIA Container Toolkit.

## Quick Start

The following creates a small, fictional, release-safe dataset. It does not
download research data and does not require a GPU.

Linux/macOS/WSL (Bash):

```bash
python examples/generate_synthetic_data.py --outdir examples/data
python scripts/marine_edna_three_methods.py \
  --csv examples/data/marine_edna.csv \
  --outdir output/quickstart \
  --backend cpu \
  --hdc-dim 256 \
  --n-estimators 20 \
  --repeats 1
```

Windows PowerShell (paste each command as one line; PowerShell does not use
Bash's trailing `\` continuation):

```powershell
py examples/generate_synthetic_data.py --outdir examples/data
py scripts/marine_edna_three_methods.py --csv examples/data/marine_edna.csv --outdir output/quickstart --backend cpu --hdc-dim 256 --n-estimators 20 --repeats 1
```

Successful output includes:

```text
output/quickstart/results.csv
output/quickstart/predictions.csv
output/quickstart/settings.json
```

`results.csv` has one row for each of `Random Forest`, `Explicit-Vocab (SVM)`,
and `HDC-Linear_opt`, with `accuracy`, `balanced_accuracy`, training time,
prediction time, backend, and RF-relative speedup. The fictional result is an
operation check only; its accuracy and speed are not research results.

Run all release-safe end-to-end checks with:

```bash
python -m unittest discover -s tests -v
```

PowerShell version:

```powershell
py -m unittest discover -s tests -v
```

## Using Your Own Data

The four public entrypoints are:

| Analysis | Script | Target |
|---|---|---|
| Marine eDNA | `scripts/marine_edna_three_methods.py` | `geo_loc_name` |
| EMP 16S | `scripts/empo_three_methods.py` | `empo_1`, `empo_2`, `empo_3` |
| HMTOL before QC | `scripts/hmtol_before_qc_three_methods.py` | `Country` |
| HMTOL after QC | `scripts/hmtol_after_qc_three_methods.py` | `Continent`, `region` |

Input paths are command-line options. You do not need to edit Python source
code. Full field definitions, missing-value behavior, ID matching, and output
examples are in [scripts/README.md](scripts/README.md).

### Marine eDNA CSV

Required, case-sensitive columns are `sample`, `geo_loc_name`,
`ASV_sequence`, `domain`, `phylum`, `class`, `order`, `family`, `genus`, and
`species`. The optional `count` column is used when present; missing or invalid
counts behave as count `1`, which is the actual default in the script. Multiple
rows for one sample are aggregated as a long ASV table. A sample must have one
`geo_loc_name` label.

```bash
python scripts/marine_edna_three_methods.py \
  --csv /path/to/marine_table.csv \
  --outdir output/marine \
  --backend cpu
```

Use `--backend gpu` (the default) only after the optional GPU installation and
CUDA check. The CPU and GPU outputs are labelled separately in `results.csv`.

### EMP 16S BIOM plus mapping file

The BIOM sample IDs must match the metadata `#SampleID` values exactly,
including case. The metadata must contain the exact columns `#SampleID`,
`empo_1`, `empo_2`, and `empo_3`. Samples with total BIOM abundance below
`--min-sample-sum` (default `1000`) are removed. Empty, `NA`, `N/A`, `Unknown`,
and similar labels are skipped for the corresponding EMPO task.

```bash
python scripts/empo_three_methods.py \
  --biom /path/to/emp_deblur_90bp.release1.biom \
  --metadata /path/to/emp_qiime_mapping_release1.tsv \
  --outdir output/empo
```

The output `results.csv` contains nine rows: three methods for each of the
three EMPO targets. `settings.json` records the abundance threshold and HDC
configuration.

### HMTOL before QC

The data directory must contain `feature-table.biom` and `metadata.tsv`. The
metadata must contain exact, case-sensitive columns `SampleID` and `Country`.
BIOM sample IDs are matched exactly to `SampleID`; rows without a matching
sample or non-empty Country are excluded. The default low-abundance threshold
is `1000`. Reference sequences are read from the required QZA archive and must
contain the BIOM feature IDs (or the compatible `G123_1` header form).

```bash
python scripts/hmtol_before_qc_three_methods.py \
  --data-dir /path/to/hmtol_before_qc \
  --sequence-qza /path/to/reference-sequences.qza \
  --outdir output/hmtol_before_qc
```

This is a stratified sample-level 80/20 split. Country is confounded with
study in the original analysis, so this is not a study-held-out estimate.

### HMTOL after QC

The data directory must contain `feature-table.qc.min3.biom` and
`metadata.qc.min3.tsv`. The metadata must contain exact columns `SampleID`,
`study`, `Continent`, and `region`. BIOM sample IDs and metadata `SampleID`
must have the same length and exactly the same order. No additional abundance
filter is applied by this entrypoint because the input is already the QC table.

Study IDs are split as groups and are never model features. With the default
three folds, every target class must occur in at least three different studies.

```bash
python scripts/hmtol_after_qc_three_methods.py \
  --data-dir /path/to/hmtol_qc \
  --sequence-qza /path/to/reference-sequences.qza \
  --outdir output/hmtol_qc \
  --targets Continent region \
  --folds 3
```

The root `results.csv` has six rows for the two targets and three methods.
Each target directory also contains fold-level results, a summary, and the
study-to-fold assignments.

CSV files can be opened in Excel with **File > Open**. If the columns appear in
one field, import the file as comma-delimited (or tab-delimited for metadata).

## Usage and Options

All scripts support `--help`. Common options are:

| Option | Default | Meaning |
|---|---:|---|
| `--outdir` | required | Output directory; it is created if needed. |
| `--random-state` | `42` where available | Reproducible split and projection seed. |
| `--repeats` | `20` | Timed prediction repetitions after one warm-up. Use `1` for a quick smoke test. |
| `--n-estimators` | `300` | Random Forest trees. Lower it for a quick demonstration. |
| `--test-size` | `0.2` | Fraction held out for sample-level tests. Both partitions must be large enough to contain every class. |
| `--hdc-dim` | `32768` where exposed | HDC hypervector dimension. Lower dimensions are useful for a smoke test, not for canonical results. |
| `--active-dims` | `32` where exposed | Active dimensions per sparse sequence projection. |
| `--hdc-c` | dataset default | LinearSVC regularization for HDC. |
| `--min-sample-sum` | `1000` where exposed | Removes low-abundance BIOM samples. |

Marine additionally supports `--backend gpu|cpu` and `--device` (default
`0`). HMTOL QC additionally supports `--targets`, `--study-column` (default
`study`), and `--folds` (default `3`).

## Output Files

- `results.csv`: method-level metrics. `accuracy` is ordinary held-out
  accuracy; `balanced_accuracy` weights classes equally; `train_sec` excludes
  prediction timing; `prediction_pipeline_mean_sec` is the mean of the timed
  runs for Marine, EMP, and HMTOL before QC; `speedup_vs_random_forest` is RF
  time divided by that method's time. These rows also include `execution` and
  `timing_scope`.
- `predictions.csv`: sample-level truth and predictions for Marine and HMTOL
  before QC. The HMTOL QC entrypoint does not write a sample-level prediction
  file; it writes fold-level metrics and study-fold assignments instead.
- `settings.json`: input paths, thresholds, split design, model settings, and
  timing scope.
- HMTOL QC `*/fold_results.csv`: one row per target, fold, and method, including
  per-fold `prediction_pipeline_mean_sec`, `execution`, and `timing_scope`;
  `*/summary.csv`: pooled target-level accuracy and timing in
  `prediction_pipeline_mean_total_sec`, with the same execution/scope labels;
  `*/study_fold_assignments.csv`: the study grouping used for the blind test.

GPU timing includes the scope stated in the row, such as cache load, host to
device transfer, GPU readout, and device to host prediction copy. It is not
automatically raw-BIOM-to-prediction timing.

## Methods and Results

- **Random Forest** uses the explicit feature matrix produced by each input
  loader. For Marine eDNA this is the aggregated taxonomy-token and sequence
  6-mer feature matrix; for EMP and HMTOL it is the BIOM ASV/feature abundance
  matrix.
- **Explicit-Vocab (SVM)** uses an explicit sparse vocabulary built from the
  same dataset-specific input representation, TF-IDF, and `LinearSVC`.
- **HDC-Linear_opt** uses the dataset-specific tuned HDC encoding and a
  normalized linear HDC representation with `LinearSVC`.

The public scripts are the reproducible code interface. The canonical research
tables, source manifest, split caveats, timing scopes, and historical benchmark
corrections are documented separately in [docs/RESULTS.md](docs/RESULTS.md) and
[docs/METHODS.md](docs/METHODS.md). They are intentionally not regenerated by
the synthetic Quick Start.

## Limitations

- These are labelled train/test or study-held-out experiments. The repository
  does not provide a saved-model or unlabelled deployment workflow.
- Sample-random splits can be optimistic when study, country, or location is
  confounded. HMTOL QC uses study-held-out folds and is the more conservative
  cross-study evaluation.
- The canonical speedups come from recorded benchmark artifacts with different
  cache formats and backends. They should be compared within a task and with
  their timing scope, not treated as raw BIOM end-to-end speedups.
- HDC settings are dataset-specific. Changing dimension, active dimensions,
  TF-IDF settings, or SVM `C` can change both accuracy and runtime.
- The release smoke tests do not require or verify a GPU. GPU execution must be
  validated on the user's CUDA installation.

## Citation

This repository does not contain a confirmed project-paper citation or a
complete data-publication citation record. Before publishing, cite the original
data release for every dataset used and add the project citation once it has
been confirmed by the authors. `results/source_manifest.csv` identifies the
source artifact family for each canonical result without inventing
bibliographic details.

## Common Issues

- **File not found:** verify the path after `--csv`, `--biom`, `--metadata`,
  `--data-dir`, or `--sequence-qza`; all paths are resolved from your current
  shell location.
- **Missing column:** use the exact case-sensitive names listed above.
- **BIOM/metadata IDs do not match:** inspect IDs for whitespace, prefixes, and
  case differences. HMTOL QC also requires the same sample order.
- **Too few samples or classes:** the default stratified split needs at least
  two samples in each class, and both train and test partitions must be large
  enough to contain every class. For a tiny three-class demo with six total
  samples, the default `--test-size 0.2` cannot work; add samples or try
  `--test-size 0.5`.
- **Too few studies for QC:** with `--folds 3`, every class must occur in at
  least three studies. Reduce folds only when that is scientifically justified.
- **GPU unavailable:** install `scripts/requirements-gpu.txt` and verify an
  NVIDIA/CUDA device, or rerun Marine with `--backend cpu`. The program reports
  the backend instead of silently relabelling CPU timing as GPU timing.

## Acknowledgement

This work was supported in part by the Minderoo Foundation (eDNAID:
Environmental DNA and AI as Tools for Ocean Aid) under award number CLB-3502.

## License

This project is released under the BSD 3-Clause License. See
[LICENSE](LICENSE) for the full license text.
