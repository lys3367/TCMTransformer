#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

LOG="logs/90_two_te_pair_ablation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

PYTHON="${PYTHON:-/opt/fsl/bin/python}"
TORCH_WHEEL="${TORCH_WHEEL:-/media/UG1/lys/Single_Shell_FWDTI/torch/torch-1.11.0+cu115-cp310-cp310-linux_x86_64.whl}"
TORCH_RUNTIME="${TORCH_RUNTIME:-$PWD/.runtime/torch111_cu115_py310}"

# /opt/fsl/bin/python otherwise sees ~/.local packages. That currently mixes
# NumPy 2.x and CUDA 13 PyTorch with FSL's NumPy 1.x binary extensions.
if [ ! -f "$TORCH_RUNTIME/torch/__init__.py" ]; then
  if [ ! -f "$TORCH_WHEEL" ]; then
    echo "ERROR: compatible PyTorch wheel not found: $TORCH_WHEEL"
    exit 2
  fi
  echo "Preparing isolated PyTorch runtime (one-time setup): $TORCH_RUNTIME"
  mkdir -p "$TORCH_RUNTIME"
  PYTHONNOUSERSITE=1 "$PYTHON" -m pip install \
    --no-deps --target "$TORCH_RUNTIME" "$TORCH_WHEEL"
fi

run_python() {
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$TORCH_RUNTIME" \
  "$PYTHON" "$@"
}

echo "Log file: $LOG"
echo "Python: $PYTHON"
echo "GPU: ${GPU_ID:-0}"
echo "Isolated torch runtime: $TORCH_RUNTIME"
echo "Experiment: all 10 two-TE combinations, 49-fold LOSO, target TE125/TE135"
run_python -c 'import sys, numpy, pandas, torch; print("Python:", sys.executable); print("NumPy:", numpy.__version__); print("pandas:", pandas.__version__); print("PyTorch:", torch.__version__); print("Torch CUDA:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available()); assert torch.cuda.is_available(), "Compatible CUDA runtime is unavailable"'

CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" run_python \
  -u code/90_two_te_pair_ablation.py \
  --device cuda \
  --batch-size "${BS:-1}" \
  --max-epochs "${EPOCHS:-100}" \
  --patience "${PATIENCE:-12}"
