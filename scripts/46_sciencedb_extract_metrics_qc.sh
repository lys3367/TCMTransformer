#!/usr/bin/env bash
set -euo pipefail

cd /media/UG1/lys/dipy/BN_JHU296
mkdir -p logs data/raw

ROOT="${ROOT:-/media/UG1/lys/dipy/data/ScienceDB_MTE}"
MASK_ROOT="${MASK_ROOT:-/media/UG1/lys/dipy/data/ScienceDB_MTE_clean}"
METRIC_ROOT="${METRIC_ROOT:-/media/UG1/lys/dipy/data/ScienceDB_MTE_clean}"
CLEAN_ROOT="${CLEAN_ROOT:-/media/UG1/lys/dipy/data/ScienceDB_MTE_clean_qc}"
EXTRACT_JOBS="${EXTRACT_JOBS:-12}"
LOG="logs/46_sciencedb_extract_metrics_qc_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "Log file: $LOG"
echo "[46.1] Extract ScienceDB BN/JHU ROI metrics after physical-bound QC"
python code/46_extract_sciencedb_bn_jhu_metrics_qc.py \
  --root "$ROOT" \
  --mask-root "$MASK_ROOT" \
  --metric-root "$METRIC_ROOT" \
  --clean-root "$CLEAN_ROOT" \
  --roi-mapping data/raw/BN_JHU_roi_mapping.csv \
  --output data/raw/ScienceDB_BN_JHU_metric_results_native_qc.csv \
  --jobs "$EXTRACT_JOBS"

echo "[46.2] Quick completeness check"
python - <<'PY'
import pandas as pd
from pathlib import Path

path = Path("data/raw/ScienceDB_BN_JHU_metric_results_native_qc.csv")
df = pd.read_csv(path)
print(f"CSV: {path.resolve()}")
print(f"Rows: {len(df)}")
print(f"Subjects: {df['Subject'].nunique()}")
print(f"TEs: {sorted(df['TE'].unique().tolist())}")
print(f"Model+Metric: {df[['Model', 'Metric']].drop_duplicates().shape[0]}")
print(f"ROIs: {df['ROI_ID'].nunique()}")
print("Rows per subject-TE min/max:")
counts = df.groupby(["Subject", "TE"]).size()
print(counts.min(), counts.max())
print("Extreme absolute means after QC:")
tmp = df.assign(abs_mean=df["Mean"].abs()).sort_values("abs_mean", ascending=False)
print(tmp[["Subject", "TE", "Model", "Metric", "ROI_ID", "Mean"]].head(20).to_string(index=False))
PY

echo "[46] Done."
echo "Output: /media/UG1/lys/dipy/BN_JHU296/data/raw/ScienceDB_BN_JHU_metric_results_native_qc.csv"
