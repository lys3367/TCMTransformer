#!/usr/bin/env bash
set -euo pipefail

cd /media/UG1/lys/dipy/BN_JHU296
mkdir -p logs outputs/kernel_sensitivity_clean49

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
CHECKPOINT_DTYPE="${CHECKPOINT_DTYPE:-fp16}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="logs/kernel_sensitivity_clean49_2gpu_${STAMP}.log"

exec > >(tee -a "$MASTER_LOG") 2>&1
echo "Master log: $MASTER_LOG"
echo "Python: $(command -v python)"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-<none>}"
echo "GPU0=$GPU0 GPU1=$GPU1 CHECKPOINT_DTYPE=$CHECKPOINT_DTYPE"

if [[ "${CONDA_DEFAULT_ENV:-}" != "mte" ]]; then
  echo "ERROR: activate the mte environment before launching this script." >&2
  exit 2
fi

python - <<'PY'
import sys
import numpy
import pandas
import scipy
import torch
print(sys.executable)
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("CUDA", torch.cuda.is_available())
assert torch.cuda.is_available()
PY

echo "[Preflight] nvidia-smi"
nvidia-smi
echo "[Preflight] host memory"
free -h
echo "[Preflight] output filesystem"
df -h outputs/kernel_sensitivity_clean49

AVAILABLE_BYTES="$(df -PB1 outputs/kernel_sensitivity_clean49 | awk 'NR==2 {print $4}')"
if [[ "$CHECKPOINT_DTYPE" == "fp16" ]]; then
  REQUIRED_BYTES=120000000000
else
  REQUIRED_BYTES=220000000000
fi
if (( AVAILABLE_BYTES < REQUIRED_BYTES )); then
  echo "ERROR: insufficient free disk for $CHECKPOINT_DTYPE checkpoints." >&2
  echo "Available bytes: $AVAILABLE_BYTES; required minimum: $REQUIRED_BYTES" >&2
  exit 4
fi

ACTIVE_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
if [[ -n "$ACTIVE_PIDS" && "$ALLOW_BUSY_GPU" != "1" ]]; then
  echo "ERROR: active GPU compute processes were found: $ACTIVE_PIDS" >&2
  echo "Set ALLOW_BUSY_GPU=1 only after confirming they will not be disturbed." >&2
  exit 3
fi

echo "[Sanity] full 11,248-variable CPU forward for DLinear/MoLE k=3/5"
python -u code/clean_code/16_train_kernel_sensitivity_clean49.py \
  --sanity-only \
  --sanity-variables 11248

run_lane() {
  local gpu="$1"
  local kernel="$2"
  local lane_log="logs/kernel_sensitivity_gpu${gpu}_k${kernel}_${STAMP}.log"
  {
    echo "Lane log: $lane_log"
    echo "GPU=$gpu kernel=$kernel"
    CUDA_VISIBLE_DEVICES="$gpu" python -u \
      code/clean_code/16_train_kernel_sensitivity_clean49.py \
      --model mole \
      --kernel "$kernel" \
      --device cuda \
      --batch-size 1 \
      --max-epochs 100 \
      --patience 12 \
      --checkpoint-dtype "$CHECKPOINT_DTYPE"

    CUDA_VISIBLE_DEVICES="$gpu" python -u \
      code/clean_code/16_train_kernel_sensitivity_clean49.py \
      --model dlinear \
      --kernel "$kernel" \
      --device cuda \
      --batch-size 1 \
      --max-epochs 100 \
      --patience 12 \
      --checkpoint-dtype "$CHECKPOINT_DTYPE"
  } 2>&1 | tee -a "$lane_log"
}

echo "[Training] GPU $GPU0: MoLE k=3 then DLinear k=3"
run_lane "$GPU0" 3 &
PID0=$!
echo "[Training] GPU $GPU1: MoLE k=5 then DLinear k=5"
run_lane "$GPU1" 5 &
PID1=$!

STATUS=0
wait "$PID0" || STATUS=$?
wait "$PID1" || STATUS=$?
if [[ "$STATUS" != "0" ]]; then
  echo "ERROR: at least one GPU lane failed; summary was not run." >&2
  exit "$STATUS"
fi

echo "[Summary] validate 196 new folds, reuse 98 canonical k=25 folds, compute statistics"
python -u code/clean_code/17_summarize_kernel_sensitivity_clean49.py

echo "Done: outputs/kernel_sensitivity_clean49"
