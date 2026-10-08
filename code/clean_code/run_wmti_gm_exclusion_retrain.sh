#!/usr/bin/env bash
set -euo pipefail

cd /media/UG1/lys/dipy/BN_JHU296
mkdir -p logs outputs/wmti_gm_exclusion_retrain

if [[ "${CONDA_DEFAULT_ENV:-}" != "mte" ]]; then
  echo "ERROR: activate the mte environment before running this script." >&2
  echo "Run: conda activate mte" >&2
  exit 2
fi

GPU_ID="${GPU_ID:-0}"
MODELS="${MODELS:-itransformer timesnet mole dlinear}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"
SUMMARIZE="${SUMMARIZE:-1}"
MODEL_TAG="${MODELS// /_}"
LOG="logs/wmti_gm_exclusion_retrain_gpu${GPU_ID}_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1
echo "Log file: $LOG"
echo "Python: $(command -v python)"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-none}"
echo "GPU_ID=$GPU_ID MODELS=$MODELS AUDIT_ONLY=$AUDIT_ONLY SUMMARIZE=$SUMMARIZE"

python -c "import sys, numpy, pandas, scipy, torch; print(sys.executable); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('scipy', scipy.__version__); print('torch', torch.__version__); print('CUDA', torch.cuda.is_available())"

python -u code/clean_code/14_train_wmti_gm_exclusion_retrain.py --audit-only

if [[ "$AUDIT_ONLY" == "1" ]]; then
  echo "Preflight audit complete. No model was trained."
  exit 0
fi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable in the active mte environment'"

for model in $MODELS; do
  echo "[WMTI-GM exclusion retraining] clean-49 LOSO: $model"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python -u \
    code/clean_code/14_train_wmti_gm_exclusion_retrain.py \
    --model "$model" \
    --device cuda \
    --batch-size 1 \
    --max-epochs 100 \
    --patience 12
done

if [[ "$SUMMARIZE" == "1" ]]; then
  python -u code/clean_code/15_summarize_wmti_gm_exclusion_retrain.py
fi

echo "Done: outputs/wmti_gm_exclusion_retrain"
