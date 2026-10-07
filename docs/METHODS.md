# Methods and evaluation design

## Shared classifier comparison

Each labelled dataset is split or grouped before fitting. The three reported
methods are:

1. **Random Forest**: a 300-tree `RandomForestClassifier` with square-root
   feature selection and balanced subsampling, fitted to the explicit sparse
   sample-by-feature abundance representation.
2. **Explicit-Vocab (SVM)**: an explicit sparse feature vocabulary, TF-IDF,
   and `LinearSVC`.
3. **HDC-Linear_opt**: a fixed-dimensional sparse HDC representation,
   L2-normalized before a tuned `LinearSVC` readout.

The HDC encoding is dataset-specific. That distinction matters:

- **Marine eDNA** builds taxonomy tokens and sequence 6-mer tokens from the
  long CSV table, hashes them with `FeatureHasher(alternate_sign=False)`,
  applies TF-IDF, and fits `LinearSVC(C=16)`.
- **EMP 16S** starts from BIOM abundance features and deterministically maps
  each BIOM observation ID to sparse bipolar active dimensions. It applies
  sublinear TF-IDF to abundance before sparse projection and uses the
  dataset-specific dimension, active-dimension, and SVM `C` settings in
  `scripts/empo_three_methods.py`. The public EMP entrypoint does not parse a
  sequence archive.
- **HMTOL** extracts reference sequences from the supplied QZA archive and
  maps each sequence deterministically to sparse bipolar active dimensions.
  Abundance is weighted with sublinear TF-IDF before projection and the result
  is L2-normalized.

The projection is deterministic for a fixed `--random-state`; it is not a
learned phylogenetic tree or a pretrained embedding.

## Evaluation units

- Marine eDNA, EMP, and HMTOL before QC use a stratified sample-level random
  80/20 split. These measurements can be optimistic when study and geography
  are confounded.
- HMTOL QC keeps studies intact and assigns them to balanced held-out folds.
  `study` is used only for grouping. It is not a feature. A target class must
  occur in at least as many studies as the requested number of folds.

## Timing scope

The public scripts fit each model once, warm up prediction once, and time the
prediction pipeline repeatedly. Training is reported separately. Marine GPU
timing includes loading the cached sparse test representation, host-to-device
transfer, GPU sparse linear readout, synchronization, and copying predictions
back to the host.

The EMP and HMTOL public scripts currently use the shared CPU in-memory
prediction implementation. The GPU speedups in `final_results.csv` came from
separate benchmark artifacts listed in `source_manifest.csv`; they should not
be described as timings produced by those four scripts.

## What is not implemented

There is no model serialization, model loading, unlabelled inference command,
RBF random Fourier feature path, or phylogenetic-tree encoder in the public
entrypoints. Such methods should not be inferred from the name HDC alone.
