#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs outputs/linear_extrapolation_clean49
LOG_FILE="logs/linear_extrapolation_clean49_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log file: ${PROJECT_ROOT}/${LOG_FILE}"
echo "Python: $(command -v python)"
python -c "import sys; print('Executable:', sys.executable)"
python code/clean_code/05_linear_extrapolation_clean49.py
