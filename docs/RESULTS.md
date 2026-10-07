# Canonical research results

The tables in this document summarize the checked values in
`results/final_results.csv`. They are kept separate from the fictional Quick
Start.
Run `python validate_final_results.py` from the repository root to check the
table structure, method coverage, accuracy ranges, RF reference rows, and HDC
dimensions.

## Accuracy

| Dataset/task | Target | RF | Explicit-Vocab | HDC-Linear_opt | HDC dimension |
|---|---|---:|---:|---:|---:|
| Marine eDNA | `geo_loc_name` | 0.8869 | 0.9550 | **0.9640** | 32,768 |
| EMP 16S EMPO1 | `empo_1` | 0.9411 | **0.9654** | 0.9630 | 32,768 |
| EMP 16S EMPO2 | `empo_2` | 0.9350 | **0.9612** | 0.9596 | 16,384 |
| EMP 16S EMPO3 | `empo_3` | 0.9157 | **0.9523** | 0.9467 | 32,768 |
| HMTOL before QC | `Country` | 0.8559 | 0.9787 | **0.9790** | 32,768 |
| HMTOL QC | `Continent` | 0.3783 | 0.4279 | **0.4318** | 32,768 |
| HMTOL QC | `region` | 0.3160 | **0.4373** | 0.4295 | 32,768 |

The HMTOL before-QC Country result uses a sample-level random split and Country
is confounded with Study ID. HMTOL QC uses three-fold study-held-out evaluation,
so its lower accuracy is a different and more conservative generalization
question.

## RF-relative prediction speedup

The values below are the canonical recorded benchmark values. RF is `1.00x`
within each task.

| Dataset/task | RF | Explicit-Vocab | HDC-Linear_opt | HDC execution |
|---|---:|---:|---:|---|
| Marine eDNA | 1.00x | 0.88x | 9.88x | GPU cached-input pipeline |
| EMP 16S EMPO1 | 1.00x | 1.26x | 19.76x | GPU cached-input pipeline |
| EMP 16S EMPO2 | 1.00x | 1.25x | 19.93x | GPU cached-input pipeline |
| EMP 16S EMPO3 | 1.00x | 1.23x | 35.97x | GPU cached-input pipeline |
| HMTOL before QC | 1.00x | 5.33x | 15.96x | GPU cached-input pipeline |
| HMTOL QC Continent | 1.00x | 4.04x | 7.45x | GPU optimized precache pipeline |
| HMTOL QC Region | 1.00x | 4.08x | 7.78x | GPU optimized precache pipeline |

These speedups are prediction-oriented and are not automatically raw-BIOM to
prediction end-to-end measurements. Cache format, preprocessing scope, and
backend differ between the recorded benchmark families. Do not compare the
absolute speedup values across tasks without reading `timing_scope` and
`speedup_comparability` in `results/final_results.csv`.

## Reproduction status

The raw Marine, EMP, and HMTOL research inputs are not distributed in this
repository. `results/source_manifest.csv` records the source artifact family used for
the canonical values. The four public scripts are suitable for rerunning the
analysis when the corresponding input files are available, but their current
EMP/HMTOL timing path is CPU in-memory and therefore does not recreate the
recorded GPU speedups. The release-safe tests verify code behavior without
using private or restricted research inputs.

## Historical corrections

The draft table previously mixed CPU and GPU paths, cache formats, and timing
scopes. The canonical values above are preserved as data in
`results/final_results.csv`; the old audit decisions remain in
`results/legacy_speedup_audit.csv`. The EMP dimension sweep data are in
`results/emp_16s_dimension_sweep/`. Neither the Quick Start nor the smoke
tests modify these research result files.
