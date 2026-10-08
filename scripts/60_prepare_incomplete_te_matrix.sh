#!/usr/bin/env bash
set -euo pipefail

cd /media/UG1/lys/dipy/BN_JHU296
mkdir -p logs data/processed outputs

MIN_TE="${MIN_TE:-3}"
PYTHON_BIN="${PYTHON_BIN:-/opt/fsl/bin/python}"
LOG="logs/60_prepare_incomplete_te_matrix_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "Log file: $LOG"
echo "[60] Build incomplete-TE matrix"
"$PYTHON_BIN" code/60_prepare_incomplete_te_matrix.py --min-te "$MIN_TE"
echo "[60] Done."
