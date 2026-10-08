#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
GPU_ID="${GPU_ID:-0}"
MODELS="${MODELS:-itransformer mole dlinear timesnet}"
BS="${BS:-1}"
mkdir -p logs outputs_sciencedb_clean49
LOG_FILE="logs/s3_sciencedb_clean49_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Python: $(command -v python)"
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available())"
if [[ "${PREPARE:-1}" == "1" && ! -f data/raw/ScienceDB_BN_JHU_metric_results_interpolated_qc.csv ]]; then
  python code/41_interpolate_sciencedb_metrics.py \
    --input data/raw/ScienceDB_BN_JHU_metric_results_native_qc.csv \
    --output data/raw/ScienceDB_BN_JHU_metric_results_interpolated_qc.csv \
    --drop-repeat
fi
for model in ${MODELS}; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python code/clean_code/08_sciencedb_external_clean49.py \
    --model "${model}" --device cuda --batch-size "${BS}"
done
if [[ "${SUMMARIZE:-1}" == "1" ]]; then
  python code/clean_code/09_summarize_sciencedb_clean49.py
fi
