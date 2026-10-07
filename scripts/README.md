# Final Three-Method Scripts

This directory intentionally contains only the final three classifier families:

1. Random Forest
2. Explicit-Vocab (SVM)
3. HDC-Linear_opt


## Files

| Dataset/task | Script |
|---|---|
| Marine eDNA | `marine_edna_three_methods.py` |
| EMP EMPO1-3 | `empo_three_methods.py` |
| HMTOL before QC: Country | `hmtol_before_qc_three_methods.py` |
| HMTOL after QC: Continent and Region | `hmtol_after_qc_three_methods.py` |
| Shared HDC/model utilities | `three_method_common.py` |

All scripts output exactly three method rows to `results.csv`.

## Timing definition

The clean scripts train each model once and then perform one warm-up plus 20
timed prediction runs by default. EMP and HMTOL use the cached-input GPU
benchmark convention for their GPU HDC paths. Marine HDC-Linear_opt now uses
the same convention:

- Random Forest: test abundance matrix to prediction.
- Explicit-Vocab: TF-IDF transform plus LinearSVC prediction.
- HDC-Linear_opt on Marine: cached sparse HDC input, host-to-device transfer,
  GPU sparse linear readout, and device-to-host prediction copy.

Random Forest and Explicit-Vocab remain CPU in-memory prediction baselines in
the Marine script. Accuracy and model hyperparameters are unchanged, but
speedup values should always be reported with their timing scope.

## Environment

```bash
cd "/home/tsl012/Multi_variant classification/Final_version"
../.venv/bin/python scripts/marine_edna_three_methods.py --help
```

Install CPU dependencies with:

```bash
python -m pip install -r scripts/requirements.txt
```

## Changing the dataset paths

The dataset locations are supplied when launching each script. They are not
stored in `three_method_common.py`, so do not edit the shared module when
switching datasets.

### Marine eDNA

Change the value after `--csv`:

```bash
../.venv/bin/python scripts/marine_edna_three_methods.py \
  --csv /path/to/another_marine_table.csv \
  --outdir output/marine_edna
```

The CSV must contain `sample`, `geo_loc_name`, `ASV_sequence`, and the seven
taxonomy columns: `domain`, `phylum`, `class`, `order`, `family`, `genus`, and
`species`.

### EMP 16S

Change the values after `--biom` and `--metadata`:

```bash
../.venv/bin/python scripts/empo_three_methods.py \
  --biom /path/to/another_feature_table.biom \
  --metadata /path/to/another_mapping.tsv \
  --outdir output/empo
```

The metadata file must contain `#SampleID`, `empo_1`, `empo_2`, and `empo_3`.
The BIOM table and metadata IDs must overlap.

### HMTOL before QC

Change `--data-dir` to the directory containing `feature-table.biom` and
`metadata.tsv`. Change `--sequence-qza` to the matching reference sequence
archive:

```bash
../.venv/bin/python scripts/hmtol_before_qc_three_methods.py \
  --data-dir /path/to/another_hmtol_dataset \
  --sequence-qza /path/to/another_reference_sequences.qza \
  --outdir output/hmtol_before_qc
```

The metadata must contain `SampleID` and `Country`.

### HMTOL after QC

Change `--data-dir` to the directory containing
`feature-table.qc.min3.biom` and `metadata.qc.min3.tsv`. Always provide the
matching `--sequence-qza` file:

```bash
../.venv/bin/python scripts/hmtol_after_qc_three_methods.py \
  --data-dir /path/to/another_hmtol_qc_dataset \
  --sequence-qza /path/to/another_reference_sequences.qza \
  --outdir output/hmtol_after_qc \
  --targets Continent region
```

The QC metadata must contain `SampleID`, `study`, `Continent`, and `region`.
Study IDs are used only to create held-out folds and are not model features.

## Marine eDNA

```bash
../.venv/bin/python scripts/marine_edna_three_methods.py \
  --csv /path/to/all_voyages_NEW.csv \
  --outdir output/marine_edna \
  --device 0
```

The input features are taxonomy ranks plus 6-mer sequence tokens grouped by
sample. HDC uses a 32,768-dimensional non-negative FeatureHasher and
`LinearSVC(C=16)`. Its test representation is saved as
`hdc_gpu_cached_input.npz`; the timed GPU path loads this cache, transfers it
to the selected CUDA device, and performs the sparse linear readout there.

Install the optional CUDA dependency for the Marine GPU path with:

```bash
python -m pip install -r scripts/requirements-gpu.txt
```

## EMP EMPO1-3

```bash
../.venv/bin/python scripts/empo_three_methods.py \
  --biom /path/to/emp_deblur_90bp.release1.biom \
  --metadata /path/to/emp_qiime_mapping_release1.tsv \
  --outdir output/empo
```

The script applies the final per-level HDC settings:

| Level | Dimension | Active dimensions | LinearSVC C |
|---|---:|---:|---:|
| EMPO1 | 32,768 | 32 | 2 |
| EMPO2 | 32,768 | 32 | 1 |
| EMPO3 | 16,384 | 16 | 20 |

## HMTOL before QC

```bash
../.venv/bin/python scripts/hmtol_before_qc_three_methods.py \
  --data-dir /path/to/hmtol_before_qc \
  --sequence-qza /path/to/2024.09.seqs.fna.qza \
  --outdir output/hmtol_before_qc
```

This reproduces the sample-level stratified Country task. Country is confounded
with Study ID, so this result is not a cross-study generalization estimate.

## HMTOL after QC

```bash
../.venv/bin/python scripts/hmtol_after_qc_three_methods.py \
  --data-dir /path/to/hmtol_qc \
  --sequence-qza /path/to/2024.09.seqs.fna.qza \
  --outdir output/hmtol_after_qc \
  --targets Continent region --folds 3
```

This uses balanced study-held-out folds. Study ID controls the split and is
not included as an input feature.

