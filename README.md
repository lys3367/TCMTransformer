# ROI-level multi-TE prediction

This repository starts from **already fitted and aggregated ROI-level multi-TE metric tables**.
It reproduces TE prediction and the study's numerical extension analyses; it is not a raw-dMRI
preprocessing, fitting or atlas-registration pipeline. The repository name TCMTransformer does
not imply a new model architecture. The four study models are iTransformer, TimesNet, MoLE and DLinear.

## Install and run an entirely synthetic example

Use Python 3.9–3.12 in a new environment (tested versions/results in `docs/VALIDATION.md`).
Dependencies are declared once, in `pyproject.toml`. CPU is sufficient for this example.

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

This generates 8 fictitious subjects × 7 TEs × 2 metrics × 2 ROIs, trains all eight LOSO
folds, and saves predictions and ROI/group summaries. One epoch verifies execution, not accuracy.
DLinear, iTransformer and TimesNet run without fetching complete external repositories.
`python -m teprediction --help` lists every command; append `--help` to a command for arguments.
To repeat the release checks: `python tests/smoke.py` after installation.

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
The initial historical audit contained 51 complete-TE subjects before the clean-49 selection.
Do not replace the cohort selection with the assumption that every complete subject is eligible.

## Third-party dependencies and MoLE

Only the transitive model-source dependencies (9 Python files) of the three licensed snapshots
are bundled; see `THIRD_PARTY_NOTICES.md`. Their source contents are unchanged. MoLE's training
adapter is included, but its upstream source has no located explicit license and is not redistributed.
Obtain the appropriate official [RogerNi/MoLE](https://github.com/RogerNi/MoLE) snapshot under its
applicable terms in `model_sources/MoLE`, and set `external_repos_dir` in your config to
`model_sources`. Bundled models remain available as fallback. The adapter loads exactly
`models/MoLE_DLinear.py`, not a heuristically chosen alternative. Its verified study SHA256 is:

```text
7179a807348870723434723cac2292e4401f96354190c46c6b24ae5933df1f74
```

The exact historical MoLE upstream commit remains unknown. Validate local dependencies before
reproducing results; a current upstream checkout need not match. The public tested environment
uses PyTorch 2.5.1, whereas the recorded original GPU training used PyTorch 1.11.0+cu115.
A smoke pass does not establish bitwise reproduction of the historical GPU results.

## Paper extension analyses

| Analysis | Commands / required input |
|---|---|
| Lesion clean-42 | Use a separate config pointing to a **precomputed** lesion ROI table with the same schema, one `L_1` ROI and the canonical 38 metrics. Supply its 42-person manifest; `prepare`, then `train --expected-subjects 42 --output-dir outputs/lesion`; the canonical split is 33/8/1. Use `summarize --input-dir outputs/lesion --expected-subjects 42 --expected-variables 38`. No lesion resampling is performed here. |
| ROI and disease groups | `evaluate`, optionally `--groups data/groups.csv` (`subject,group`); emits held-out ROI means, group statistics and paired Wilcoxon/BH-FDR. Templates/atlas brain rendering are not bundled. |
| Linear extrapolation reference | `linear-reference` using canonical fold manifest and the prepared matrix |
| Leave-one-TE-out | `leave-one-te --model itransformer --mode loso` (also TimesNet/DLinear); `leave-one-te-mole --model mole --mode loso` for MoLE. Input must already reflect the approved complete cohort. |
| Input-TE counts / pairs | `input-te-count`, `te-pairs`; each requires canonical cohort selection and the prepared matrix |
| External ScienceDB | `interpolate` for the external ROI table's TE grid if needed; `external --model ...` uses internal canonical cohort and a separate external table. No image fitting or external labels are downloaded. |
| Incomplete-TE pretraining | `prepare-incomplete`, then `pretrain-itransformer --mode loso`, `pretrain-baselines --mode loso` or `pretrain-mole --mode loso`; `matched-pretraining --max-folds 0` implements the full matched control. Supply the authorized incomplete/complete cohort and use the CLI's explicit exclusion option if required. |
| Native-unit / metric summaries | `native-units`, `metric-errors`, using saved predictions |
| WMTI-GM exclusion | `wmti-sensitivity`, `summarize-wmti`; strict original 49-person/11,248-variable checks retained |
| Moving-average kernel sensitivity | `kernel-sensitivity`, `summarize-kernel`; reuses canonical k=25 and tests k=3/5 |

Numerical analysis is retained; old publication-layout renderers, manuscript directories and
historical models are removed. See [release audit](docs/RELEASE_AUDIT.md) and
[rename mapping](docs/RENAMING.csv). Module helpers are intentionally kept separate where
merging would obscure distinct training/validation procedures.

## Provenance and privacy

On 2026-10-08 the author confirmed that all 49 formal participants' BN/JHU atlases were generated
using FWDTI FA registration. **This is author confirmation, not a new registration run or independent
per-subject imaging audit.** No original results were overwritten. Earlier evidence supports
resample-then-mean lesion extraction. The upstream raw-dMRI execution record and the generators of
`QC_Flagged_Report.csv` / `QC_Final_Report.csv` remain unresolved and outside this package's input boundary.
Earlier wording describing FSL thresholding as replacement by numerical bounds was incorrect:
`-thr/-uthr` sets out-of-range voxels to zero. This release begins after that stage.

Do not commit patient data, cohort/group tables, predictions or credentials. Generated output is
ignored by default, but Git ignore rules do not anonymize data. Example generation uses no patient
records. Original project code is MIT-licensed; third-party exceptions are listed in `THIRD_PARTY_NOTICES.md`.

## 中文说明

公开入口已经收敛为“拟合、空间处理和ROI统计已完成的多TE指标CSV”。源代码按实际功能命名，
数字编号、服务器路径、历史模型和整仓第三方副本已清理；示例完全生成。49人FA路径由作者于
2026-10-08确认，本轮未重跑影像配准。论文原始数值复现仍需获授权的数据、相同队列/指标规则
及对应依赖版本，不能用小型CPU smoke test替代正式实验复现。
