#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/89_minimum_te_ablation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
PYTHON="${PYTHON:-/opt/fsl/bin/python}"
CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" "$PYTHON" -u code/89_minimum_te_ablation.py --device cuda --batch-size "${BS:-1}" --max-epochs "${EPOCHS:-100}" --patience "${PATIENCE:-12}"
