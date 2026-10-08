#!/usr/bin/env bash
set -euo pipefail

cd /media/UG1/lys/dipy/BN_JHU296
mkdir -p logs outputs

GPU_ID="${GPU_ID:-0}"
BS="${BS:-1}"
MAX_FOLDS="${MAX_FOLDS:-1}"
PYTHON_TORCH="${PYTHON_TORCH:-/opt/fsl/bin/python}"
LOG="logs/91_pretraining_matched_control_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1
echo "Log file: $LOG"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_TORCH" code/91_pretraining_matched_control.py \
  --device cuda \
  --batch-size "$BS" \
  --max-folds "$MAX_FOLDS"

echo "[91] Done."
