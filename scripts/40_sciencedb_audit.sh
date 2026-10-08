#!/usr/bin/env bash
set -euo pipefail

cd /media/UG1/lys/dipy/BN_JHU296
mkdir -p logs outputs_sciencedb

ROOT="${ROOT:-/media/UG1/lys/dipy/data/ScienceDB_MTE}"
LOG="logs/40_sciencedb_audit_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "Log file: $LOG"
echo "[40] Audit ScienceDB DWI files"
python code/40_audit_sciencedb_dataset.py \
  --root "$ROOT" \
  --out-dir outputs_sciencedb

echo "[40] Done."
