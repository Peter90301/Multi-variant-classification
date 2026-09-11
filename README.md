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

The CPU scripts use Python, NumPy, SciPy, pandas, joblib, matplotlib, and
scikit-learn. GPU scripts additionally require a CUDA-compatible CuPy build,
RAPIDS cuML, and an NVIDIA GPU.

These files were extracted from a larger experimental workspace. Some scripts
import local helper modules such as `train_geo_classifier.py`,
`predict_geo_name.py`, `benchmark_emp_16s_empo.py`,
`benchmark_emp_16s_full_gpu_cuml.py`, and
`benchmark_human_gut_168k_five_methods.py`. Dataset and output defaults also
refer to the original workstation paths, so those arguments or constants must
be updated when running in another environment.

## Data

Large datasets, trained models, and generated benchmark caches are not included
in this repository.
