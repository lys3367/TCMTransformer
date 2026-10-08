#!/usr/bin/env bash
set -euo pipefail

cd /media/UG1/lys/dipy/BN_JHU296
mkdir -p logs outputs/table1_clean49

GPU_ID="${GPU_ID:-0}"
BS="${BS:-1}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-12}"
MODELS="${MODELS:-itransformer timesnet mole dlinear}"
BUILD_MANIFEST="${BUILD_MANIFEST:-1}"
SUMMARIZE="${SUMMARIZE:-1}"
MODEL_TAG="${MODELS// /_}"
LOG="logs/table1_clean49_gpu${GPU_ID}_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1
echo "Log file: $LOG"
echo "Python: $(command -v python)"
echo "GPU_ID=$GPU_ID BS=$BS EPOCHS=$EPOCHS PATIENCE=$PATIENCE"
echo "MODELS=$MODELS BUILD_MANIFEST=$BUILD_MANIFEST SUMMARIZE=$SUMMARIZE"

python -c "import sys, numpy, pandas, torch; print(sys.executable); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('torch', torch.__version__); print('CUDA', torch.cuda.is_available()); assert torch.cuda.is_available()"

if [[ "$BUILD_MANIFEST" == "1" ]]; then
  python -u code/clean_code/00_build_clean49_manifest.py
fi

for model in $MODELS; do
  echo "[Table 1] clean-49 LOSO: $model"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python -u \
    code/clean_code/01_train_table1_clean49.py \
    --model "$model" \
    --device cuda \
    --batch-size "$BS" \
    --max-epochs "$EPOCHS" \
    --patience "$PATIENCE"
done

if [[ "$SUMMARIZE" == "1" ]]; then
  python -u code/clean_code/02_summarize_table1_s1.py
fi
echo "Done: outputs/table1_clean49"
