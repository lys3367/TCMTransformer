#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python}"
ROOT="${ROOT:-/media/UG1/lys/dipy/data/MTE3_clean}"
RUN_LIST="${RUN_LIST:-${ROOT}/run_list.csv}"
OUTPUT="${OUTPUT:-${ROOT}/lesion_metric_results_raw.csv}"

"${PYTHON}" code/07_prepare_lesion_csv.py \
  --root "${ROOT}" \
  --run-list "${RUN_LIST}" \
  --output "${OUTPUT}"

mkdir -p data/raw
cp "${OUTPUT}" data/raw/lesion_metric_results_raw.csv
echo "Project copy: data/raw/lesion_metric_results_raw.csv"
