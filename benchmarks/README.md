# Benchmark data

This directory contains the five tabular datasets used by the repository's
regression and smoke-validation scripts. They are development/benchmark assets,
not runtime dependencies of the installable `fixait` package.

Run the quick single-model check with:

```bash
python scripts/validate_datasets.py
```

Run the multi-model matrix with:

```bash
python scripts/benchmark_matrix.py --max-rows 800 --output benchmark.json
```

Global and local faithfulness optimization can be enabled independently:

```bash
python scripts/benchmark_matrix.py --optimize-faithfulness --optimize-local-faithfulness
```
