#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/media/UG1/lys/dipy/data/MTE3}"
RUN_LIST="${RUN_LIST:-${DATA_ROOT}/run_list.csv}"
CLEAN_ROOT="${CLEAN_ROOT:-/media/UG1/lys/dipy/data/MTE3_clean}"
MNI="${MNI:-/media/UG1/lys/dipy/data/MNI}"
MNI_TEMPLATE="${MNI_TEMPLATE:-/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz}"
JOBS="${JOBS:-2}"
THREADS="${THREADS:-4}"
FORCE="${FORCE:-0}"

REGISTER_SCRIPT="/media/UG1/lys/dipy/reg_bn_jhu.sh"
SPLIT_SCRIPT="/media/UG1/lys/dipy/split_bn_jhu.sh"

echo "============================================================"
echo "[1/2] Register BN/JHU atlases to DWI space"
echo "DATA_ROOT=${DATA_ROOT}"
echo "RUN_LIST=${RUN_LIST}"
echo "MNI=${MNI}"
echo "JOBS=${JOBS}, THREADS=${THREADS}, FORCE=${FORCE}"
echo "============================================================"

DATA_ROOT="${DATA_ROOT}" \
RUN_LIST="${RUN_LIST}" \
MNI="${MNI}" \
MNI_TEMPLATE="${MNI_TEMPLATE}" \
JOBS="${JOBS}" \
THREADS="${THREADS}" \
FORCE="${FORCE}" \
bash "${REGISTER_SCRIPT}"

echo "============================================================"
echo "[2/2] Split BN/JHU atlases into single ROI masks"
echo "CLEAN_ROOT=${CLEAN_ROOT}"
echo "============================================================"

DATA_ROOT="${DATA_ROOT}" \
RUN_LIST="${RUN_LIST}" \
MNI="${MNI}" \
CLEAN_ROOT="${CLEAN_ROOT}" \
bash "${SPLIT_SCRIPT}"

echo "Done: ${CLEAN_ROOT}/<subject>/BN_JHU/"
