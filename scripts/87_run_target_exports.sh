#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
GPU_ID="${GPU_ID:-0}"
PYTHON="${PYTHON:-/opt/fsl/bin/python}"
LOG="logs/87_target_exports_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
for model in itransformer timesnet mole dlinear; do
  echo "[87] Export $model target-wise LOSO predictions"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -u code/87_export_loso_target_predictions.py --model "$model" --device cuda --batch-size "${BS:-1}" --max-epochs "${EPOCHS:-100}" --patience "${PATIENCE:-12}"
done
echo "[87] Done. Download outputs/target_predictions/*.npz for local plotting."
