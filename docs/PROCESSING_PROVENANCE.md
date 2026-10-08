# Processing provenance and evidence limits / 实际流程及证据边界

Release audit: 2026-10-08. This public summary distinguishes implementation, recorded
formal workflow, and missing historical execution evidence. Original participant-level
records and internal drafting documents are not distributed.

## Raw dMRI preprocessing / 原始预处理

Internal Methods records describe conversion and orientation, per-TE denoising/Gibbs
correction, joint multi-TE eddy/motion correction, anatomical distortion correction,
gradient reorientation, bias correction and splitting back into individual TEs.
The available `preprocessing/dipy/fit_*.py` modules operate on prepared diffusion inputs;
they do not establish that every upstream step ran successfully. **Execution cannot be confirmed.**
The released fitting implementations cover DTI, RESTORE, free-water DTI, DKI, WMTI,
MSDKI, GQI, FORECAST and NODDI. IVIM was extracted initially but excluded from the final
38-metric matrix by completeness filtering (`code/01_audit_prepare.py`).

## BN/JHU: formal FA scheme / 正式FA方案

The internal BN_JHU project record explicitly states a 2026-07-01 change from mean-b0
to FWDTI FA. The matching released implementation is:

- `preprocessing/BN_JHU/scripts/reg_bn_jhu.sh`, lines 27–48: T1 input and FA/mask choice,
  preferring TE135, falling back through 125/115/105/95/85/75.
- Lines 83–107: T1-to-MNI nonlinear registration, FA-to-T1 affine registration,
  inverse transforms of standard atlas labels onto the FA reference using nearest-neighbor interpolation.
- Lines 110–113: brain masking and FA geometry.
- `run_bn_jhu.sh` invokes registration and `split_bn_jhu.sh` to form 50 JHU WM and
  246 BN GM ROI masks. `extract_bn_jhu_metrics.sh`, lines 49–74, reuses masks across TE
  and obtains nonzero masked means/SD with `fslstats -M -S`.

**B: FA is the documented formal choice.** However, lines 61–64 skip existing final atlas
files unless forced. Historical mean-b0 scripts use the same final naming. Without the
per-subject transform/log provenance, **all final atlases having been regenerated via FA
cannot be confirmed**. Cross-TE geometry and visual registration QC are not proven by
script existence. `register_all_subjects_atlas_to_dwi_parallel.sh` is a historical alternative.
Original wrappers use server paths; remap them before running in a new installation.

## Lesions: resample then mean / 重采样后均值

`code/07_prepare_lesion_csv.py`, lines 50–54, reads
`<subject>/rs/lesion_metrics_mean.csv`; the required columns are
`subject,TE,model,metric,mean`. It aggregates existing means and does not read/resample NIfTI.
`scripts/08_prepare_lesion_csv.sh` copies the aggregate to project `data/raw`.
`code/clean_code/06_prepare_lesion_clean42.py`, lines 29–103, applies cohort/metric rules,
requires complete seven-TE coverage and verifies 42×7×38 entries.

Internal experiment record section 32.6 (line 2880) documents repairing one corrupt
resampled DTI FA image, recomputing its masked mean, and regenerating the final clean-42
input. The subsequent audit and 42-fold logs support this downstream chain. **B: the final
clean-42 analysis used resampled means.** Patient identifiers and logs are not published.

Two released implementations can generate the required per-subject CSV:

- `preprocessing/dipy/mte3_clean_rs_then_mean.sh`, lines 177–233: `mrgrid regrid -template`
  followed by `mrstats -mask -ignorezero`. This call has no explicit interpolation flag.
- `preprocessing/dipy/mte_lesion_metrics_rs_mean.sh`, lines 162–170: explicitly uses
  `-interp linear`, then masked nonzero means.

The precise original writer/version for every subject cannot be confirmed. Regridding
to the lesion template is not estimation of a new anatomical registration transform;
prior mask/image alignment and its QC remain unverified. Direct-mask extraction in
`scripts/05_add_lesion_masks.sh` and `scripts/06_extract_lesion_metrics.sh` is an alternative
implementation, not evidence overriding the actual `rs` input of the formal clean-42 task.

## QC / 质控

`preprocessing/dipy/QC_MTE3_clean.py`, line 12, **reads** `QC_Flagged_Report.csv`.
Lines 70–77 threshold rows whose status contains `Physics Violation` or `Stat Warning`;
other existing inputs are copied. It neither generates that report nor writes
`QC_Final_Report.csv`. **Generators of both internal reports cannot be confirmed.**
External-cohort QC must not be substituted for internal report provenance.

`fslmaths -thr/-uthr` sets out-of-range voxels to zero, not to the numerical bounds.
This corrects an error in earlier internal Methods notes. Nonzero ROI means also exclude
legitimate zero values. See the [official FSL interface](https://pages.fmrib.ox.ac.uk/fsl/fslpy/_modules/fsl/wrappers/fslmaths.html)
and [FSL practical](https://fsl.fmrib.ox.ac.uk/fslcourse/graduate/lectures/practicals/intro3/).
The cleaning script suppresses subprocess output and does not check the return code;
its `FIXED` counter alone does not establish successful processing.

## Formal analysis / 正式分析

Initial completeness auditing produced a 51×7×11248 matrix; the canonical manifest selects
49 participants for main LOSO. The lesion extension uses a distinct 42×7×38 matrix.
Formal core model/wrapper hashes were verified against original run settings before
publication. Public settings omit identities and subject-dependent signatures and must
not be substituted into an old output directory to resume training. No upstream commit
is inferred from missing `.git` history, including MoLE.

结论：正式FA方案明确，但逐例最终atlas来源未全部证实；正式clean-42支持重采样分支；
原始预处理逐项执行、两份QC报告生成程序、全部病灶CSV的精确生成版本及空间QC仍无法确认。
