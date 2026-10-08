#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/media/UG1/lys/dipy/data/MTE3}"
RUN_LIST="${RUN_LIST:-${DATA_ROOT}/run_list.csv}"
MNI="${MNI:-/media/UG1/lys/dipy/data/MNI}"
MNI_TEMPLATE="${MNI_TEMPLATE:-/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz}"
JOBS="${JOBS:-2}"
THREADS="${THREADS:-4}"
FORCE="${FORCE:-0}"
TE_LIST="${TE_LIST:-135 125 115 105 95 85 75}"

JHU="${MNI}/JHU-ICBM-labels-1mm.nii.gz"
BN="${MNI}/BN_Atlas_246_1mm.nii.gz"

for f in "${MNI_TEMPLATE}" "${JHU}" "${BN}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f}"; exit 1; }
done

process_subject() {
  local SUB="$1"
  local IDX="$2"
  local TOTAL="$3"
  local SID
  SID="$(basename "${SUB}")"

  local T1="${SUB}/PA_TE135/t1w_std_align_center_unring_unbiased_brain_in_dwi.nii.gz"
  local MRI="${SUB}/PA_TE135/freesurfer/mri"
  local JHU_OUT="${MRI}/JHU_in_DWI_masked.nii.gz"
  local BN_OUT="${MRI}/BN_in_DWI_masked.nii.gz"
  local FA=""
  local MASK=""
  local FA_TE=""

  if [[ ! -f "${T1}" ]]; then
    echo "[${IDX}/${TOTAL}] [SKIP] ${SID}: missing T1"
    return 0
  fi

  for te in ${TE_LIST}; do
    local fa_try="${SUB}/PA_TE${te}/FWDTI/FA.nii.gz"
    local mask_try="${SUB}/PA_TE${te}/mask.nii.gz"
    if [[ -f "${fa_try}" && -f "${mask_try}" ]]; then
      FA="${fa_try}"
      MASK="${mask_try}"
      FA_TE="${te}"
      break
    fi
  done

  if [[ -z "${FA}" || -z "${MASK}" ]]; then
    echo "[${IDX}/${TOTAL}] [SKIP] ${SID}: missing FA/mask in TE list: ${TE_LIST}"
    return 0
  fi

  local WORK="${MRI}/registration_work_fa_TE${FA_TE}"
  local FA_WORK="${WORK}/fa_to_t1"
  local T1_WORK="${MRI}/registration_work_fa/t1_to_mni"
  local LOG="${WORK}/ants.log"

  if [[ "${FORCE}" != "1" && -f "${JHU_OUT}" && -f "${BN_OUT}" ]]; then
    echo "[${IDX}/${TOTAL}] [SKIP] ${SID}: final outputs exist"
    return 0
  fi

  mkdir -p "${MRI}" "${FA_WORK}" "${T1_WORK}"
  : > "${LOG}"
  export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${THREADS}"

  echo "------------------------------------------------------------"
  echo "[${IDX}/${TOTAL}] [START] ${SID}"
  echo "FA reference: ${FA}"
  echo "FA TE: ${FA_TE}"
  echo "Mask: ${MASK}"
  echo "ANTs log: ${LOG}"
  echo "------------------------------------------------------------"

  if [[ "${FORCE}" == "1" ]]; then
    rm -f "${FA_WORK}"/fa2t1_* "${WORK}/JHU_in_DWI.nii.gz" "${WORK}/BN_in_DWI.nii.gz"
    rm -f "${JHU_OUT}" "${BN_OUT}"
  fi

  if [[ ! -f "${T1_WORK}/t12mni_1InverseWarp.nii.gz" ]]; then
    echo "[${IDX}/${TOTAL}] ${SID} | 1/4 T1 -> MNI nonlinear registration"
    antsRegistrationSyN.sh -d 3 -f "${MNI_TEMPLATE}" -m "${T1}" -t s -o "${T1_WORK}/t12mni_" >> "${LOG}" 2>&1
  else
    echo "[${IDX}/${TOTAL}] ${SID} | 1/4 T1 -> MNI exists"
  fi

  if [[ ! -f "${FA_WORK}/fa2t1_0GenericAffine.mat" ]]; then
    echo "[${IDX}/${TOTAL}] ${SID} | 2/4 FA -> T1 affine registration"
    antsRegistrationSyNQuick.sh -d 3 -f "${T1}" -m "${FA}" -t a -o "${FA_WORK}/fa2t1_" >> "${LOG}" 2>&1
  else
    echo "[${IDX}/${TOTAL}] ${SID} | 2/4 FA -> T1 exists"
  fi

  echo "[${IDX}/${TOTAL}] ${SID} | 3/4 apply transforms: JHU -> FA space"
  antsApplyTransforms -d 3 -i "${JHU}" -r "${FA}" -o "${WORK}/JHU_in_DWI.nii.gz" -n NearestNeighbor \
    -t "[${FA_WORK}/fa2t1_0GenericAffine.mat,1]" \
    -t "[${T1_WORK}/t12mni_0GenericAffine.mat,1]" \
    -t "${T1_WORK}/t12mni_1InverseWarp.nii.gz" >> "${LOG}" 2>&1

  echo "[${IDX}/${TOTAL}] ${SID} | 3/4 apply transforms: BN -> FA space"
  antsApplyTransforms -d 3 -i "${BN}" -r "${FA}" -o "${WORK}/BN_in_DWI.nii.gz" -n NearestNeighbor \
    -t "[${FA_WORK}/fa2t1_0GenericAffine.mat,1]" \
    -t "[${T1_WORK}/t12mni_0GenericAffine.mat,1]" \
    -t "${T1_WORK}/t12mni_1InverseWarp.nii.gz" >> "${LOG}" 2>&1

  echo "[${IDX}/${TOTAL}] ${SID} | 4/4 mask + copy FA geometry"
  fslmaths "${WORK}/JHU_in_DWI.nii.gz" -mas "${MASK}" "${JHU_OUT}"
  fslmaths "${WORK}/BN_in_DWI.nii.gz" -mas "${MASK}" "${BN_OUT}"
  fslcpgeom "${FA}" "${JHU_OUT}"
  fslcpgeom "${FA}" "${BN_OUT}"

  echo "[${IDX}/${TOTAL}] [DONE] ${SID}"
}

[[ -f "${RUN_LIST}" ]] || { echo "ERROR: missing run list: ${RUN_LIST}"; exit 1; }
mapfile -t SUBJECTS < <(awk -F, -v root="${DATA_ROOT}" '
  NF {
    gsub(/\r/, "", $1)
    gsub(/^[ \t]+|[ \t]+$/, "", $1)
    if ($1 == "" || $1 ~ /^#/) next
    if (tolower($1) ~ /^(subject|sub|sid|id)$/) next
    print root "/" $1
  }
' "${RUN_LIST}")
TOTAL="${#SUBJECTS[@]}"
IDX=0

for SUB in "${SUBJECTS[@]}"; do
  [[ -d "${SUB}" ]] || continue
  IDX=$((IDX + 1))
  SID="$(basename "${SUB}")"
  (process_subject "${SUB}" "${IDX}" "${TOTAL}" || echo "[${IDX}/${TOTAL}] [FAIL] ${SID}") &
  while [[ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]]; do
    sleep 10
  done
done

wait
echo "Registration done."
