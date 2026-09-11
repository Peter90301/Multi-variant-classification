# Multi-variant Classification with HDC

This repository contains the main scripts used to predict sample labels from
taxonomic and amplicon sequence variant (ASV) features with hyperdimensional
computing (HDC).

The HDC method used here is **TF-IDF-weighted sparse bipolar random indexing
with a LinearSVC readout**, referred to as **HDC-Hash**. GPU experiments use
CuPy CUDA kernels and, where indicated, cuML.

## Scripts

### `train_hdc_geo_classifier.py`

Groups observations by sample and trains a `geo_loc_name` classifier from:

- taxonomic hierarchy: domain, phylum, class, order, family, genus, species
- `ASV_sequence`

The high-accuracy mode uses feature hashing, TF-IDF weighting, and LinearSVC.
The script also includes a prototype-based HDC readout.

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

Benchmarks the validation-selected EMP HDC representations with a GPU-resident
readout. It compares the tuned GPU HDC pipeline with Random Forest while
recording accuracy, stage-level runtime, and prediction speedup.

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
one warm-up plus 20 timed full prediction-pipeline repetitions:

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
prediction pipeline includes memory-map opening, host-to-device transfer, HDC
accumulation, L2 normalization, GPU prediction, and returning labels to CPU.
Raw BIOM parsing and model training are intentionally excluded from deployment
prediction timing because they were completed in the one-time preparation
stage.

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

Benchmark the validation-selected encodings with the GPU-resident linear
readout:

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

## Geo-location Training And Prediction

The input CSV must include a sample identifier, `geo_loc_name`, taxonomy
columns, and `ASV_sequence`. Train the default TF-IDF HDC-Hash plus LinearSVC
model:

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
