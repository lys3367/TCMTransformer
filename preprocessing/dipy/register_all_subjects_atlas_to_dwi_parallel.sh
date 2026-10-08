#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/media/UG1/lys/dipy/data/MTE3}"
MNI="${MNI:-/media/UG1/lys/dipy/data/MNI}"
MNI_TEMPLATE="${MNI_TEMPLATE:-/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz}"
JOBS="${JOBS:-2}"
THREADS="${THREADS:-4}"
FORCE="${FORCE:-0}"

JHU="${MNI}/JHU-ICBM-labels-1mm.nii.gz"
BN="${MNI}/BN_Atlas_246_1mm.nii.gz"

for f in "${MNI_TEMPLATE}" "${JHU}" "${BN}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f}"; exit 1; }
done

process_subject() {
  local SUB="$1"
  local SID
  SID="$(basename "${SUB}")"

  local T1="${SUB}/PA_TE135/t1w_std_align_center_unring_unbiased_brain_in_dwi.nii.gz"
  local DWI="${SUB}/PA_TE135/dwi_std_align_center_denoise_unring_preproc_preproc.nii.gz"
  local BVAL="${SUB}/PA_TE135/dwi_std_align_center_denoise_unring_preproc_preproc.bval"
  local MASK="${SUB}/PA_TE135/mask.nii.gz"
  local MRI="${SUB}/PA_TE135/freesurfer/mri"
  local WORK="${MRI}/registration_work_shared_dwi"
  local DWI_WORK="${WORK}/dwi_to_t1"
  local T1_WORK="${WORK}/t1_to_mni"
  local JHU_OUT="${MRI}/JHU_in_DWI_masked.nii.gz"
  local BN_OUT="${MRI}/BN_in_DWI_masked.nii.gz"

  if [[ ! -f "${T1}" || ! -f "${DWI}" || ! -f "${BVAL}" || ! -f "${MASK}" ]]; then
    echo "[SKIP] ${SID}: missing T1/DWI/bval/mask"
    return 0
  fi

  if [[ "${FORCE}" != "1" && -f "${JHU_OUT}" && -f "${BN_OUT}" ]]; then
    echo "[SKIP] ${SID}: final outputs exist"
    return 0
  fi

  mkdir -p "${MRI}" "${DWI_WORK}" "${T1_WORK}"
  export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${THREADS}"

  echo "===================================================================="
  echo "[START] ${SID}"
  echo "===================================================================="

  if [[ "${FORCE}" == "1" ]]; then
    rm -f "${DWI_WORK}/dwi_ref.nii.gz" "${DWI_WORK}/b0_all.nii.gz" "${DWI_WORK}"/b0_*.nii.gz
    rm -f "${DWI_WORK}"/dwi2t1_*
    rm -f "${WORK}/JHU_in_DWI.nii.gz" "${WORK}/BN_in_DWI.nii.gz"
    rm -f "${JHU_OUT}" "${BN_OUT}"
  fi

  echo "[${SID}] 1/5 mean b0"
  if [[ ! -f "${DWI_WORK}/dwi_ref.nii.gz" ]]; then
    python - "${BVAL}" > "${DWI_WORK}/b0_indices.txt" <<'PY'
import sys
vals = [float(x) for x in open(sys.argv[1]).read().split()]
idx = [str(i) for i, v in enumerate(vals) if v < 50]
if not idx:
    raise SystemExit("No b0 volume found")
print(" ".join(idx))
PY
    read -ra B0_INDICES < "${DWI_WORK}/b0_indices.txt"
    for idx in "${B0_INDICES[@]}"; do
      fslroi "${DWI}" "${DWI_WORK}/b0_${idx}.nii.gz" "${idx}" 1
    done
    fslmerge -t "${DWI_WORK}/b0_all.nii.gz" "${DWI_WORK}"/b0_*.nii.gz
    fslmaths "${DWI_WORK}/b0_all.nii.gz" -Tmean "${DWI_WORK}/dwi_ref.nii.gz"
  fi

  echo "[${SID}] 2/5 T1 -> MNI"
  if [[ ! -f "${T1_WORK}/t12mni_1InverseWarp.nii.gz" ]]; then
    antsRegistrationSyN.sh \
      -d 3 \
      -f "${MNI_TEMPLATE}" \
      -m "${T1}" \
      -t s \
      -o "${T1_WORK}/t12mni_"
  fi

  echo "[${SID}] 3/5 DWI -> T1"
  if [[ ! -f "${DWI_WORK}/dwi2t1_0GenericAffine.mat" ]]; then
    antsRegistrationSyNQuick.sh \
      -d 3 \
      -f "${T1}" \
      -m "${DWI_WORK}/dwi_ref.nii.gz" \
      -t a \
      -o "${DWI_WORK}/dwi2t1_"
  fi

  echo "[${SID}] 4/5 atlas -> DWI"
  antsApplyTransforms \
    -d 3 \
    -i "${JHU}" \
    -r "${DWI_WORK}/dwi_ref.nii.gz" \
    -o "${WORK}/JHU_in_DWI.nii.gz" \
    -n NearestNeighbor \
    -t "[${DWI_WORK}/dwi2t1_0GenericAffine.mat,1]" \
    -t "[${T1_WORK}/t12mni_0GenericAffine.mat,1]" \
    -t "${T1_WORK}/t12mni_1InverseWarp.nii.gz"

  antsApplyTransforms \
    -d 3 \
    -i "${BN}" \
    -r "${DWI_WORK}/dwi_ref.nii.gz" \
    -o "${WORK}/BN_in_DWI.nii.gz" \
    -n NearestNeighbor \
    -t "[${DWI_WORK}/dwi2t1_0GenericAffine.mat,1]" \
    -t "[${T1_WORK}/t12mni_0GenericAffine.mat,1]" \
    -t "${T1_WORK}/t12mni_1InverseWarp.nii.gz"

  echo "[${SID}] 5/5 mask + geometry"
  fslmaths "${WORK}/JHU_in_DWI.nii.gz" -mas "${MASK}" "${JHU_OUT}"
  fslmaths "${WORK}/BN_in_DWI.nii.gz" -mas "${MASK}" "${BN_OUT}"
  fslcpgeom "${DWI_WORK}/dwi_ref.nii.gz" "${JHU_OUT}"
  fslcpgeom "${DWI_WORK}/dwi_ref.nii.gz" "${BN_OUT}"

  echo "[DONE] ${SID}"
}

for SUB in "${DATA_ROOT}"/*; do
  [[ -d "${SUB}" ]] || continue
  process_subject "${SUB}" &
  while [[ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]]; do
    sleep 10
  done
done

wait
echo "All subjects finished."
