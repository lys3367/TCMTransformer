# TCMTransformer

Research code for target-TE prediction of diffusion MRI microstructural metrics.
The canonical experiment compares **iTransformer, TimesNet, MoLE and DLinear** on
49 participants, seven echo times and 296 BN/JHU ROIs. The repository name does not
identify a newly implemented Transformer architecture: these are the four existing
baseline models used in the study.

## Scope and reproducibility

- Main task: TE 75/85/95/105/115 ms → TE 125/135 ms; 38 metrics × 296 ROIs = 11,248 variables.
- Canonical validation: clean-49, 49-fold subject-level LOSO, 38/10/1 train/validation/test.
- Lesion extension: clean-42, one lesion ROI, 38 metrics, 42-fold LOSO (33/8/1).
- Original research Python and shell filenames and contents are preserved. Some scripts
  retain original server paths; inspect/configure those paths before use. The direct
  Python commands below avoid the hard-coded `cd` in historical shell wrappers.
- Patient images, metric tables, participant manifests, predictions and training logs
  are **not distributed**. This is a code release, not a complete executable data bundle.
- Read [processing provenance and unresolved items](docs/PROCESSING_PROVENANCE.md).
  Full raw-dMRI preprocessing execution and the generators of the two internal QC reports
  have not been recovered. Do not describe this release as a complete reconstruction of every historical step.

## Layout

| Path | Purpose |
|---|---|
| `code/clean_code/01_train_table1_clean49.py` | Canonical four-model training entry point |
| `code/03_train_neural.py`, `code/73_train_extra_models.py` | Model/training wrappers |
| `code/01_audit_prepare.py` | Input audit and matrix construction |
| `code/07_prepare_lesion_csv.py` | Aggregate existing per-subject resampled lesion means |
| `code/clean_code/06_prepare_lesion_clean42.py` | Prepare complete clean-42 lesion cohort |
| `code/clean_code/` | Main results, figures and sensitivity/extension analyses |
| `preprocessing/BN_JHU/` | FA-based atlas registration, label splitting and extraction |
| `preprocessing/dipy/` | Metric fitting, QC cleaning and candidate lesion extraction scripts |
| `scripts/` | Original orchestration and extension scripts |
| `external_repos/` | Three third-party source snapshots with their original licenses |
| `docs/run_settings/` | Public settings with participant identities/signatures removed |
| `MAIN_EXPERIMENT_HASH_CHECK.csv` | Historical core model/wrapper hash verification |

## Environments

Training uses `mte`; MRI fitting uses `dipy`. Current server exports are supplied as
`environment_mte_server.yml` and `environment_dipy_server.yml` with machine-specific
conda `prefix` fields removed. They are environment records, not cross-platform lockfiles.
The older `environment.yml`, `mte_environment.yml` and `mte_requirements.txt` are retained
as historical records; old exports can include machine-specific package locations.

Current recorded versions: mte Python 3.10.20 / PyTorch 1.11.0+cu115; dipy Python 3.9.25 /
DIPY 1.10.0 / nibabel 5.3.3 / AMICO 2.1.1. See `mri_software_versions.txt` for FLIRT,
ANTs and MRtrix3. Current exports do not establish every historical fitting version.
Linux, a compatible CUDA installation and separately installed MRI tools are expected.

```bash
conda env create -f environment_mte_server.yml
conda activate mte
```

The original MoLE source is not redistributed because no explicit license was found
in the provided snapshot or upstream root. Obtain it from [RogerNi/MoLE](https://github.com/RogerNi/MoLE)
under applicable terms and place it at `external_repos/MoLE`. Its exact historical upstream
commit is unknown. Compare `models/MoLE_DLinear.py` against the recorded SHA256 in
`MAIN_EXPERIMENT_HASH_CHECK.csv`; a current checkout is not guaranteed to match.
Your project MoLE wrapper is included. See [third-party notices](THIRD_PARTY_NOTICES.md).
`scripts/04_setup_extra_repos.sh` is a historical optional-model helper, not the installation
entry point for these four canonical models.

## Prepare data and run the main experiment

Supply an authorized, de-identified ROI table at `data/raw/BN_JHU_metric_results_raw.csv`.
The audited table requires `Subject`, `TE`, `Model`, `Metric`, `ROI_ID`, `ROI_Name`,
`Mean` and `Std`. Refer to the reader in `code/01_audit_prepare.py`
and the provided ROI mapping; TE values must correspond to the seven configured TEs.
Use consistent pseudonyms across input tables and cohort manifests.

```bash
python code/01_audit_prepare.py --config config.json
```

Supply `data/processed/clean49_subjects.csv` with a `subject` column containing exactly
49 authorized de-identified participants selected according to the study cohort rules.
The audit can initially contain 51 complete-TE participants: the canonical trainer applies
the clean-49 manifest. `00_build_clean49_manifest.py` needs a historical result table that
is not public, so it is not a from-scratch cohort discovery tool.

From the repository root, after supplying data, the manifest and all four model dependencies:

```bash
for model in itransformer timesnet mole dlinear; do
  python code/clean_code/01_train_table1_clean49.py \
    --model "$model" --config config.json \
    --manifest data/processed/clean49_subjects.csv \
    --device cuda --batch-size 1 --max-epochs 100 --patience 12
done
python code/clean_code/02_summarize_table1_s1.py
```

For lesions, `07_prepare_lesion_csv.py` expects existing
`<subject>/rs/lesion_metrics_mean.csv` files; it does not resample MRI itself. See the
processing document before running `scripts/08_prepare_lesion_csv.sh` and the clean-42
preparation/training scripts. Atlas files and lesion masks must be obtained separately.

## Verification and licensing

`RELEASE_FILE_MANIFEST.csv` records shipped file hashes and whether each copy is byte-identical
to the local release source. `docs/RELEASE_VALIDATION.md` states the checks performed.
No training or MRI preprocessing was rerun for this release.

Third-party code retains its own licenses. A license for the original project code has
not yet been selected by the maintainer; public availability alone does not grant an
additional reuse license. See `THIRD_PARTY_NOTICES.md`.

## 中文说明

本次公开的是正式四模型及相关处理/分析代码，原脚本名称与内容未改。
BN/JHU正式方案有文档和代码支持为FWDTI FA配准，但所有最终图谱的逐例来源仍无法确认；
clean-42病灶分析支持重采样后统计。上游预处理完整执行记录、两份内部QC报告的生成程序仍缺失。
不包含患者影像、患者姓名/编号、训练名单或内部日志。MoLE第三方源码需从官方来源另行取得；
其训练封装和已核验哈希保留。详细证据边界见[流程说明](docs/PROCESSING_PROVENANCE.md)。
