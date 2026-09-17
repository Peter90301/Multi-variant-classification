# Multi-variant Classification with HDC

This repository contains experimental fixed-dimensional classification
pipelines for taxonomic and amplicon sequence variant (ASV) features. The Geo
and EMP experiments use different encodings and should not be treated as one
identical HDC algorithm.

## Method Definitions

### Geo default: nonnegative feature hashing

The default Geo classifier is:

```text
sample-level taxonomy and sequence tokens
-> FeatureHasher(alternate_sign=False)
-> sublinear TF-IDF
-> LinearSVC
```

This is a fixed-dimensional **nonnegative feature-hashing** pipeline. It is not
the sparse bipolar random-indexing representation used by the EMP experiments.

### EMP tuned: sparse bipolar random indexing

The tuned EMP classifier is:

```text
sample x ASV abundance
-> training-fitted abundance weighting
-> sparse bipolar whole-ASV or k-mer projection
-> sample-wise L2 normalization
-> LinearSVC
```

The selected EMP configurations use sublinear TF-IDF weighting and sparse
bipolar whole-ASV projections. This representation is referred to as
**HDC-Hash** in the EMP results. GPU experiments use CuPy CUDA kernels and,
where indicated, cuML.

Neither primary pipeline implements random Fourier features, an RBF feature
map, or a kernel bandwidth parameter. Separate nonlinear experiments, when
reported, use an exact RBF-SVM readout and are not random Fourier features.

## Scripts

### `train_hdc_geo_classifier.py`

Groups observations by sample and trains a `geo_loc_name` classifier from:

- taxonomic hierarchy: domain, phylum, class, order, family, genus, species
- `ASV_sequence`

The default high-accuracy mode uses nonnegative feature hashing, TF-IDF
weighting, and LinearSVC. The script also includes a separate prototype-based
HDC readout.

### `predict_hdc_geo_name.py`

Loads a model produced by `train_hdc_geo_classifier.py`, constructs the same
sample-level features, and writes predicted `geo_loc_name` labels.

### `tune_emp_16s_hdc_encodings.py`

Tunes HDC sequence encodings on the Earth Microbiome Project (EMP) 16S data for
EMPO1, EMPO2, and EMPO3 classification. It compares raw, log-transformed, and
sublinear TF-IDF abundance weighting; multiple HDC dimensions and densities;
whole-ASV and k-mer representations; and LinearSVC regularization values. Model
selection uses an inner validation split before one final untouched-test
evaluation.

### `benchmark_emp_tuned_hdc_gpu.py`

Benchmarks the currently selected whole-ASV, TF-IDF-weighted EMP HDC
representations with a GPU-resident readout. It compares the tuned GPU HDC
pipeline with Random Forest while recording accuracy, stage-level runtime, and
prediction speedup. See the compatibility and timing limitations below.

### `benchmark_emp_16s_optimized_precache.py`

Benchmarks the optimized deployment-oriented EMP prediction path. It uses an
uncompressed memory-mapped cache, compact array types, CSR-aware CUDA HDC
accumulation, GPU normalization, and a pretrained cuML LinearSVC. Training is
performed once and excluded from prediction timing.

## Dependencies

The CPU scripts use Python, NumPy, SciPy, pandas, joblib, matplotlib,
biom-format, and scikit-learn. GPU scripts additionally require an NVIDIA GPU,
a CUDA 12-compatible driver, CuPy, and RAPIDS cuML.

The repository includes the local helper modules imported by the five main
scripts. The default paths in the scripts refer to the original workstation,
so the commands below explicitly pass portable dataset, cache, and output
paths.

## Installation

Clone the repository and create the tested Conda environment:

```bash
git clone https://github.com/Peter90301/Multi-variant-classification.git
cd Multi-variant-classification
conda env create -f environment.yml
conda activate multi-variant-hdc
```

Alternatively, install into an existing Python 3.10 CUDA 12 environment:

```bash
python -m pip install -r requirements.txt
```

The pinned package versions reproduce the software environment used for the
reported GPU experiments. The NVIDIA driver and CUDA-compatible hardware are
not installed by these files.

## EMP Raw Data

Download the published EMP release 1 deblur 90 bp BIOM feature table and QIIME
mapping metadata:

```bash
mkdir -p data/emp cache results

curl -L --fail --retry 3 \
  -o data/emp/emp_deblur_90bp.release1.biom \
  https://ftp.microbio.me/emp/release1/otu_tables/deblur/emp_deblur_90bp.release1.biom

curl -L --fail --retry 3 \
  -o data/emp/emp_qiime_mapping_release1.tsv \
  https://ftp.microbio.me/emp/release1/mapping_files/emp_qiime_mapping_release1.tsv
```

Samples with a total count below 1,000 are removed by the commands below.

## Build EMP Precache From Raw Data

Build the standard 4,096-dimensional HDC precache directly from the downloaded
BIOM table and metadata. This performs BIOM/metadata parsing, sample filtering,
ASV hypervector generation, label encoding, and train/test split once:

```bash
python benchmark_emp_16s_full_gpu_cuml.py \
  --biom data/emp/emp_deblur_90bp.release1.biom \
  --metadata data/emp/emp_qiime_mapping_release1.tsv \
  --precache-dir cache/emp_16s_gpu_precache_4096 \
  --build-precache-only \
  --outdir results/emp_precache_build \
  --min-sample-sum 1000 \
  --test-size 0.2 \
  --random-state 42 \
  --hdc-dim 4096 \
  --active-dims-per-sequence 4 \
  --device 0
```

Convert that cache to the compact prediction-only memory-mapped layout and run
one warm-up plus 20 timed cached-input prediction-pipeline repetitions:

```bash
python benchmark_emp_16s_optimized_precache.py \
  --source-cache cache/emp_16s_gpu_precache_4096 \
  --optimized-cache cache/emp_16s_gpu_precache_4096_optimized \
  --outdir results/emp_optimized_precache \
  --repeats 20 \
  --hdc-dim 4096 \
  --active-dims-per-sequence 4 \
  --device 0
```

The second command trains deployment models once before timing. Its measured
**cached-input prediction pipeline** includes memory-map opening,
host-to-device transfer, HDC accumulation, L2 normalization, GPU prediction,
and returning labels to CPU. Raw BIOM parsing, metadata alignment, filtering,
splitting, and model training are excluded because they were completed in the
one-time preparation stage. The standard 4,096-dimensional cache contains raw
counts; the separate tuned GPU benchmark caches the already TF-IDF-weighted
test input, so its TF-IDF transformation is also outside the timed region. This
timing must not be reported as raw-BIOM end-to-end runtime.

## Tune And Benchmark EMP HDC

Tune TF-IDF weighting, HDC dimensions/density, sequence encoding, and LinearSVC
regularization using an inner validation split:

```bash
python tune_emp_16s_hdc_encodings.py \
  --biom data/emp/emp_deblur_90bp.release1.biom \
  --metadata data/emp/emp_qiime_mapping_release1.tsv \
  --outdir results/emp_hdc_tuning \
  --min-sample-sum 1000 \
  --test-size 0.2 \
  --validation-size 0.2 \
  --random-state 42 \
  --repeats 20
```

Benchmark the validation-selected whole-ASV TF-IDF encodings with the
GPU-resident linear readout:

```bash
python benchmark_emp_tuned_hdc_gpu.py \
  --biom data/emp/emp_deblur_90bp.release1.biom \
  --metadata data/emp/emp_qiime_mapping_release1.tsv \
  --tuning-dir results/emp_hdc_tuning \
  --outdir results/emp_tuned_gpu \
  --min-sample-sum 1000 \
  --test-size 0.2 \
  --random-state 42 \
  --repeats 20 \
  --device 0
```

The current GPU benchmark reconstructs a `whole_sequence_...npz` projection and
always applies sublinear TF-IDF. It therefore supports the configurations that
won the reported EMP tuning runs:

| Target | Encoding | Weighting |
|---|---|---|
| EMPO1 | whole ASV sequence | sublinear TF-IDF |
| EMPO2 | whole ASV sequence | sublinear TF-IDF |
| EMPO3 | whole ASV sequence | sublinear TF-IDF |

It does not yet dispatch generically on every candidate emitted by the tuning
script. If a future run selects raw counts, log1p weighting, or a k-mer
projection, `benchmark_emp_tuned_hdc_gpu.py` must be extended before that
selection can be benchmarked faithfully.

## Geo-location Training And Prediction

The input CSV must include a sample identifier, `geo_loc_name`, taxonomy
columns, and `ASV_sequence`. Train the default nonnegative feature-hashing,
TF-IDF, and LinearSVC model:

```bash
python train_hdc_geo_classifier.py \
  --csv data/all_voyages_NEW.csv \
  --outdir results/geo_hdc \
  --classifier linear-svm \
  --hash-features 32768 \
  --svm-c 16 \
  --test-size 0.2 \
  --random-state 42
```

Predict labels for samples represented in another compatible CSV:

```bash
python predict_hdc_geo_name.py \
  --model results/geo_hdc/hdc_geo_loc_name_model.joblib \
  --csv data/new_samples.csv \
  --output results/geo_hdc/new_sample_predictions.csv \
  --top-k 3
```

## Evaluation Scope

The Geo and EMP scripts in this repository use sample-level random or
stratified random splits. Rows are aggregated by sample before Geo splitting,
but samples are not grouped by Voyage. EMP samples are not grouped by Study.
Consequently, their reported test accuracies estimate performance on held-out
samples drawn from the same collection of voyages or studies; they do not
establish generalization to unseen voyages or studies.

Cross-voyage or cross-study claims require group-aware outer splits such as
`GroupKFold` or `StratifiedGroupKFold`, with all preprocessing and
hyperparameter selection repeated inside each outer training partition. The
EMP tuning script does use an inner validation split and fits TF-IDF only on
the applicable training subset, but its outer split is still sample-level.

## Benchmark Scope And Known Limitations

- The tuned GPU HDC timing begins from an already transformed cache. It covers
  cache opening, transfer, HDC construction, normalization, and prediction,
  not preprocessing from raw BIOM and metadata.
- The Random Forest cached-input path reads a compressed SciPy `.npz`, whereas
  the optimized GPU path reads uncompressed memory-mapped `.npy` arrays. The
  reported total therefore includes both compute and different cache-format
  I/O costs. Classification-only timing and per-stage timing should be reported
  separately from this cached-input total.
- `gpu_linear_predict()` currently assumes a multiclass LinearSVC readout and
  applies `argmax` across class-score columns. A binary LinearSVC has only one
  score column, so the current helper would always return class index 0 for a
  binary task. Binary support must threshold the single score at zero before
  this helper is used for two-class experiments. The reported EMP tasks have
  3, 6, and 19 classes, so this limitation does not affect those results.

## Supporting Modules

- `train_geo_classifier.py`: CSV cleaning and sample-level feature aggregation.
- `predict_geo_name.py`: prediction-time explicit feature loading utilities.
- `benchmark_emp_16s_empo.py`: EMP BIOM/metadata loading, filtering, splitting,
  and sequence projection.
- `benchmark_emp_16s_full_gpu_cuml.py`: raw-data GPU pipeline and standard
  precache creation/loading.
- `benchmark_emp_16s_gpu_hdc.py`: shared CSR and CUDA HDC build utilities.
- `benchmark_human_gut_168k_five_methods.py`: shared optimized GPU matrix and
  linear-readout utilities.
- `cuda_emp_hdc_build.cu`: native CUDA implementation used by the standalone
  GPU HDC feature-build benchmark.

## Data

Large datasets, trained models, and generated benchmark caches are not included
in this repository.
