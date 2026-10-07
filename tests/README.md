# Public smoke tests

These tests generate fictional data at runtime. They do not download or use
the Marine eDNA, EMP, or HMTOL research datasets. The tests run all four public
entrypoints with CPU prediction, validate the output row counts and fields,
check `--help`, and verify one actionable missing-file error.

Run from the repository root:

```bash
python -m unittest discover -s tests -v
```

The Marine GPU backend is intentionally not required for the public tests.
