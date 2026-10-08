#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
GPU_ID="${GPU_ID:-0}"
MODELS="${MODELS:-itransformer mole dlinear timesnet}"
BS="${BS:-1}"
mkdir -p logs outputs_lesion_clean42
LOG_FILE="logs/s3_lesion_clean42_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Python: $(command -v python)"
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available())"
if [[ "${PREPARE:-1}" == "1" ]]; then
  python code/clean_code/06_prepare_lesion_clean42.py
  python code/01_audit_prepare.py --config config_lesion_clean42.json
fi
for model in ${MODELS}; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python code/clean_code/01_train_table1_clean49.py \
    --model "${model}" \
    --config config_lesion_clean42.json \
    --manifest data/processed_lesion/06_clean42_subjects.csv \
    --expected-subjects 42 \
    --output-dir outputs_lesion_clean42 \
    --device cuda --batch-size "${BS}"
done
if [[ "${SUMMARIZE:-1}" == "1" ]]; then
  python code/clean_code/07_summarize_s3_models.py \
    --input-dir outputs_lesion_clean42 --expected-subjects 42
fi
