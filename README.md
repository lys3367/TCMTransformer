# TCMTransformer: ROI-level multi-TE prediction

Code accompanying **Predicting Diffusion MRI Microstructural Metrics at Target Echo Times with a Variable-Centric Transformer**, submitted to *Biomedical Signal Processing and Control* (BSPC).

TCMTransformer is the study framework built on iTransformer. The code identifier `itransformer` corresponds to the paper label **TCMTransformer**; the other three formal models are TimesNet, MoLE and DLinear. This release starts from **already fitted, spatially processed and aggregated ROI-level multi-TE metric tables**. It implements prediction and numerical extension analyses. Raw MRI preprocessing, registration, fitting, patient data and publication-layout rendering are outside its scope.

## Installation and synthetic quick start

Python 3.9–3.12 is supported by the declared dependencies. The release test uses a fresh Python 3.9.23 environment with PyTorch 2.5.1 on Windows CPU. The historical experiments used Python 3.10.20 and PyTorch 1.11.0+cu115; the public environment is an execution baseline, not a guarantee of identical GPU floating-point results.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install .
python -m teprediction synthetic --output example
python -m teprediction prepare --config example/config.json
python -m teprediction train --model dlinear --config example/config.json --manifest example/cohort.csv --expected-subjects 8 --output-dir outputs/example --device cpu --max-epochs 1 --patience 1
python -m teprediction evaluate --input-dir outputs/example --output-dir outputs/example_summary --models dlinear --groups example/groups.csv
```

The example generator creates eight fictitious subjects, seven TEs and four variables. It includes a portable config and cohort/group tables. The command trains eight LOSO folds and writes predictions. The same command accepts `itransformer` and `timesnet`; MoLE needs the separate dependency below. A one-epoch synthetic test verifies execution, **not reproduction of paper accuracy**.

## Input contract and formal experiment

The CSV requires exactly these measurement fields (additional columns are allowed):

| Column | Meaning |
|---|---|
| `Subject` | De-identified string, consistent across all files |
| `TE` | Integer milliseconds, normally 75,85,95,105,115,125,135 |
| `Model`, `Metric` | Fitting family and parameter name |
| `ROI_ID`, `ROI_Name` | Consistent ROI identity and label |
| `Mean`, `Std` | Already computed ROI statistics; Std may be empty |

Duplicate subject/TE/model/metric/ROI keys are rejected. Preparation retains complete-TE
subjects and excludes a metric family/parameter across all ROIs if its mean is missing anywhere
in that complete cohort, matching the original audit rule. Supply the same input scope and
cohort/feature decisions for exact study comparison. `data/cohort.csv` must contain a unique
`subject` column with the 49 approved pseudonyms. **No original patient list is distributed.**

Edit `config.json` to locate your CSV, matrices and outputs. All relative paths are relative
to the working directory, including paths inside a config; run consistently from your project
root. The configuration is portable and contains no original server paths.

```bash
python -m teprediction prepare --config config.json
python -m teprediction train --model itransformer --config config.json --manifest data/cohort.csv
# Repeat train for timesnet, mole and dlinear after supplying the MoLE dependency.
python -m teprediction summarize --input-dir outputs/table1_clean49
python -m teprediction evaluate --groups data/groups.csv
```

Canonical settings: 49 subjects; 38 metrics × 296 BN/JHU ROIs = 11,248 variables; five input
TEs (75–115 ms) predicting 125/135 ms; subject-level LOSO with 38/10/1 train/validation/test;
seed 20260623, batch 1, AdamW lr 7e-4, weight decay 1e-4, up to 100 epochs, patience 12.
Standardization uses the current training subjects across their seven TEs, not validation/test
subjects. The same trained predictions supply overall and target-specific summaries.
Cohort inclusion must follow the approved study manifest; completeness alone does not establish eligibility.

## Model sources and MoLE setup

Nine unchanged Python source files provide the runtime import dependencies of iTransformer (MIT), TimesNet (MIT) and DLinear (Apache-2.0); their original licenses are bundled. See [third-party notices](THIRD_PARTY_NOTICES.md). Complete third-party repositories are unnecessary.

MoLE is the official **MoLE_DLinear** implementation from [RogerNi/MoLE](https://github.com/RogerNi/MoLE), *Mixture-of-Linear-Experts for Long-term Time Series Forecasting* (AISTATS 2024). Its two runtime files are **not redistributed** here: no explicit upstream license was located. Obtain/use them under applicable permission; public availability is not a redistribution license.

For a separately obtained local dependency, the following command downloads only those two files, verifies normalized SHA256 values, and refuses to overwrite different files:

```bash
python -m teprediction install-mole --destination model_sources/MoLE
```

Set `external_repos_dir` to `model_sources` in your chosen config. The downloader pins commit `a4f9fa8a6640598e953464f39f021a641cd9146c`. Its model file matches the research copy after CRLF/LF normalization (SHA256 `fe5dad01208b0f775923cde1beb5edd1fd80032fd6fb06a0b316300fb6f93c90`). This is a verified reproducible source pin, **not a recovered historical checkout commit**. MoLE uses two candidate predictions (`t_dim=2` for the primary task); unused legacy config attributes do not define its architecture.

Total parameter counts at 11,248 variables are 2,386 (iTransformer), 742,746 (TimesNet), 506,205,040 (MoLE), and 24 (DLinear). TimesNet has 741,530 trainable parameters plus 1,216 fixed embedding parameters; the submitted manuscript labels the total as trainable. The other three totals are also their trainable counts. The full MoLE model is large; the synthetic smoke uses four variables and is not a full-dimension training benchmark.

## Paper extension analyses

| Analysis | Commands / required input |
|---|---|
| Lesion clean-42 | Use a separate config pointing to a **precomputed** lesion ROI table with the same schema, one `L_1` ROI and the canonical 38 metrics. Supply its 42-person manifest; `prepare`, then `train --expected-subjects 42 --output-dir outputs/lesion`; the canonical split is 33/8/1. Use `summarize --input-dir outputs/lesion --expected-subjects 42 --expected-variables 38`. No lesion resampling is performed here. |
| ROI and disease groups | `evaluate`, optionally `--groups data/groups.csv` (`subject,group`); emits held-out ROI means, group statistics and paired Wilcoxon/BH-FDR across all supplied group/comparator pairs, with 10,000 participant bootstrap resamples. Templates/atlas brain rendering are not bundled. |
| Population-template controls | `population-template --config config.json --table1-dir outputs/table1_clean49`; uses saved fold scaling and predictions, compares train38/dev48 templates with paired tests and bootstrap intervals |
| Linear extrapolation reference | `linear-reference` using canonical fold manifest and the prepared matrix |
| Leave-one-TE-out | `leave-one-te --model itransformer --mode loso` (also TimesNet/DLinear); `leave-one-te-mole --model mole --mode loso` for MoLE. Input must already reflect the approved complete cohort. |
| Input-TE counts / pairs | `input-te-count`, `te-pairs`; each requires canonical cohort selection and the prepared matrix |
| External ScienceDB | `interpolate` for the external ROI table's TE grid if needed; `external --model ...` uses internal canonical cohort and a separate external table. No image fitting or external labels are downloaded. |
| Incomplete-TE pretraining | `prepare-incomplete`, then `pretrain-itransformer --mode loso`, `pretrain-baselines --mode loso` or `pretrain-mole --mode loso`; `matched-pretraining --max-folds 0` implements the full matched control. Supply the authorized incomplete/complete cohort and use the CLI's explicit exclusion option if required. |
| Native-unit / metric summaries | `native-units`, `metric-errors`, using saved predictions |
| WMTI-GM exclusion | `wmti-sensitivity`, `summarize-wmti`; strict original 49-person/11,248-variable checks retained |
| Moving-average kernel sensitivity | `kernel-sensitivity`, `summarize-kernel`; reuses canonical k=25 and tests k=3/5 |

For extension-level paired comparisons, `paired-statistics --input measurements.csv` accepts `subject,model,MAE` (lowercase model identifiers). With `--loto`, also provide `target_te`: exactly seven canonical targets are required and averaged within participant before Wilcoxon/BH-FDR across the three reference comparisons. Historical tables must be converted explicitly to this schema; there is no silent discovery of older results.

All commands expose `--help`. Extension training has its own options; use the manuscript's authorized cohort and the documented experiment settings. The primary CLI's default batch size is 1. The historical base-model LOTO run used `--batch-size 1` for iTransformer and `--batch-size 8` for TimesNet/DLinear, with training seed `20260623 + 100*fold + target_te`. Set these explicitly; the paper's primary-task batch/seed description must not be generalized to LOTO. The exact historical MoLE LOTO batch override remains unverified (its CLI default is 16). LOTO table SD describes 343 fold-target measurements; paired tests average the seven targets first, yielding 49 independent participant pairs. These are distinct aggregation levels.

## Reproduction checks

After installation and separate MoLE setup:

```bash
python tests/smoke.py --mole-dir model_sources/MoLE --report outputs/smoke_report.json
```

The test runs from an unrelated temporary directory, checks imports and command help, validates 49/42-person split sizes and train-only scaling, rejects duplicate measurements, and runs all four models on eight synthetic LOSO folds each. It checks finite predictions and ROI/group summaries. Real-data results require authorized data and matched ordering, cohort, preprocessing and historical settings; they are not established by this test. Full extension retraining is not part of the minimal smoke test.

## Privacy and scope

Use de-identified IDs throughout. Do not commit patient tables, groups, images, checkpoints, logs or predictions. Generated paths and local model sources are ignored; ignore rules do not anonymize content. Load NPZ artifacts only from trusted sources (the historical matrix schema includes object arrays).

The author confirmed that all 49 formal participants used the FWDTI FA atlas-registration path. No new image registration or independent per-subject imaging audit is claimed here. The generators of the upstream QC reports remain unconfirmed and are outside this package's input boundary. No MRI data or internal research documents are included.

## License and citation

Original project code is under the [MIT License](LICENSE). Third-party files retain their own licenses; the project MIT license does not cover externally fetched MoLE.

Until a published DOI is available, cite the submitted manuscript without implying acceptance:

> Li, Y., Zhang, C., Cao, S., Li, J., Lei, Z., Liu, X., and Wu, Y. Predicting Diffusion MRI Microstructural Metrics at Target Echo Times with a Variable-Centric Transformer. Manuscript submitted to Biomedical Signal Processing and Control, 2026.

Also cite the upstream model papers when using their implementations. Repository: https://github.com/lys3367/TCMTransformer.
