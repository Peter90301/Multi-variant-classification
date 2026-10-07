# Reproducing research analyses

The commands below are templates. Replace every `/path/to/...` value with a
local copy of the corresponding research input. The repository does not ship
those datasets.

## Marine eDNA

```bash
python scripts/marine_edna_three_methods.py \
  --csv /path/to/marine_table.csv \
  --outdir output/marine_research \
  --backend gpu \
  --hdc-dim 32768 \
  --repeats 20
```

The canonical table records a GPU cached-input Marine HDC timing. A CPU run is
available with `--backend cpu`, but its timing must not be labelled as GPU.

## EMP 16S

```bash
python scripts/empo_three_methods.py \
  --biom /path/to/emp_deblur_90bp.release1.biom \
  --metadata /path/to/emp_qiime_mapping_release1.tsv \
  --outdir output/emp_research \
  --min-sample-sum 1000 \
  --repeats 20
```

The script uses the per-target HDC settings in `scripts/empo_three_methods.py`
and currently times CPU in-memory prediction. It does not recreate the
canonical cached GPU speedup numbers.

## HMTOL before QC

```bash
python scripts/hmtol_before_qc_three_methods.py \
  --data-dir /path/to/hmtol_before_qc \
  --sequence-qza /path/to/reference-sequences.qza \
  --outdir output/hmtol_before_qc_research \
  --hdc-dim 32768 \
  --active-dims 32 \
  --hdc-c 10 \
  --repeats 20
```

This is the sample-random Country task and is confounded by Study ID.

## HMTOL after QC

```bash
python scripts/hmtol_after_qc_three_methods.py \
  --data-dir /path/to/hmtol_qc \
  --sequence-qza /path/to/reference-sequences.qza \
  --outdir output/hmtol_qc_research \
  --targets Continent region \
  --folds 3 \
  --hdc-dim 32768 \
  --active-dims 32 \
  --hdc-c 10 \
  --repeats 20
```

This is the study-held-out evaluation. The QC BIOM and metadata must be in
exactly the same sample order, and each target class must be represented in at
least three studies.
