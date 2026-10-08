# Release validation — 2026-10-08

## Installation and end-to-end smoke test

A new Python venv was created with `include-system-site-packages = false`. Dependencies were
installed into that environment, a wheel was built, and that wheel was installed normally
(**not editable**). The test switched to a new temporary directory outside the repository.
`PYTHONPATH` was removed; no research CSV, imaging input, server mount or patient identifier was used.

Environment: Python 3.9.23, PyTorch 2.5.1+cpu, NumPy 1.26.4, pandas 2.2.3, SciPy 1.13.1.
The declared runtime environment is in `pyproject.toml`; this is separate from the historical GPU environment.

| Check | Result |
|---|---|
| Wheel build / normal installation | Passed |
| Import and `--help` for all 24 CLI commands | Passed |
| Generated synthetic CSV | 8 subjects × 7 TEs × 2 metrics × 2 ROIs = 224 rows |
| Input matrix | (8, 7, 4), all finite |
| Duplicate-measurement input | Correctly rejected |
| DLinear training/prediction | All 8 LOSO folds completed, one epoch per fold, CPU |
| iTransformer training/prediction | All 8 LOSO folds completed, one epoch per fold, CPU |
| TimesNet training/prediction | All 8 LOSO folds completed, one epoch per fold, CPU |
| Saved predictions for each model | (8, 2, 4), all finite, 8 unique held-out subjects |
| Model/target summaries and consistency checks | Passed |
| ROI and synthetic-group summaries / paired FDR | Passed |
| 49-person split invariants | Every fold 38/10/1, disjoint, all subjects held out once |
| 42-person split invariants | Every fold 33/8/1, disjoint, all subjects held out once |
| Training-only scaling | Altering validation/test data did not change training mean/SD |

Run the checks with `python tests/smoke.py` after installing the package. The test creates and
cleans a private temporary directory and can optionally save JSON with `--report PATH`.
The synthetic one-epoch errors are not estimates of study performance and must not be compared
with the paper's metrics. The complete check executes 24 training folds, not 49 real-subject folds.

## Refactor and publication checks

- Eight central function bodies were compared as ASTs to the original research code and were
  unchanged: training-only standardization, regression metrics, neural training/aggregation,
  MoLE training, fold generation, canonical train-fold dispatch and cohort selection.
- All nine retained vendor Python files were verified byte-for-byte against the original snapshots.
- Own source filenames contain no numeric experiment prefixes; dynamic imports and source-hash
  paths refer to the new names. The mapping is recorded in `RENAMING.csv`.
- Public source was scanned for original server paths, recognizable patient identifiers and
  credential patterns; none were found. Data/output directories are ignored by default.
- Two integration defects found by testing were fixed: optional MoLE dependency checking when
  another model is selected, and legacy import-path collisions between project and vendor models.

## Limits

MoLE's optional upstream source was not bundled or exercised by this public smoke test.
The real 49-person study and every full extension experiment were not retrained. CPU/current
library execution does not establish bitwise agreement with the historical CUDA/PyTorch setup.
Atlas registration was **not rerun**: the author confirmed all 49 formal participants used FA,
and requested no server check. No research result or original manuscript was modified.
