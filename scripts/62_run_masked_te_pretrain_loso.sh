#!/usr/bin/env bash
set -euo pipefail

cd /media/UG1/lys/dipy/BN_JHU296
mkdir -p logs outputs

GPU_ID="${GPU_ID:-0}"
BS="${BS:-1}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-80}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-100}"
PATIENCE="${PATIENCE:-12}"
EXCLUDE_PREFIX="${EXCLUDE_PREFIX:-}"
LOSO_MAX_FOLDS="${LOSO_MAX_FOLDS:-0}"
PYTHON_TORCH="${PYTHON_TORCH:-/opt/fsl/bin/python}"
LOG="logs/62_masked_te_pretrain_loso_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "Log file: $LOG"
echo "GPU_ID=$GPU_ID"
echo "BS=$BS"
echo "PRETRAIN_EPOCHS=$PRETRAIN_EPOCHS"
echo "FINETUNE_EPOCHS=$FINETUNE_EPOCHS"
echo "EXCLUDE_PREFIX=$EXCLUDE_PREFIX"
echo "LOSO_MAX_FOLDS=$LOSO_MAX_FOLDS"
echo "PYTHON_TORCH=$PYTHON_TORCH"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_TORCH" code/61_masked_te_pretrain_itransformer.py \
  --mode loso \
  --device cuda \
  --batch-size "$BS" \
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --patience "$PATIENCE" \
  --loso-max-folds "$LOSO_MAX_FOLDS" \
  ${EXCLUDE_PREFIX:+--exclude-subject-prefix $EXCLUDE_PREFIX}

echo "[62 LOSO] Done."
