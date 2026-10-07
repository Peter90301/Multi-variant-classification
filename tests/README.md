# Release-safe test data

The tests generate all inputs at runtime. They do not read, copy, sample, or
transform Marine eDNA, EMP, HMTOL, or any other research dataset.

The fixtures contain only fictional `PUBLIC_*` sample IDs, `Synthetic *` labels,
and short hand-authored DNA strings. Temporary CSV, BIOM, TSV, and QZA files
are deleted after each test.
