#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/media/UG1/lys/dipy/data/ScienceDB_MTE}"
CLEAN_ROOT="${CLEAN_ROOT:-/media/UG1/lys/dipy/data/ScienceDB_MTE_clean}"
MNI="${MNI:-/media/UG1/lys/dipy/data/MNI}"
MNI_TEMPLATE="${MNI_TEMPLATE:-/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz}"
REF_TE="${REF_TE:-62}"
JOBS="${JOBS:-1}"
THREADS="${THREADS:-4}"
FORCE="${FORCE:-0}"

JHU_ATLAS="${MNI}/JHU-ICBM-labels-1mm.nii.gz"
BN_ATLAS="${MNI}/BN_Atlas_246_1mm.nii.gz"
JHU_ID_LIST="${MNI}/JHU_id_list.txt"
BN_ID_LIST="${MNI}/BN_id_list.txt"

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$THREADS"

run_subject() {
  local sub="$1"
  local subject
  subject="$(basename "$sub")"
  local dwi_dir="${sub}/dwi"
  local out_root="${CLEAN_ROOT}/${subject}"
  local out_mask="${out_root}/BN_JHU"
  local work="${out_root}/registration_work_b0"

  if [ "$FORCE" = "0" ] && [ -d "$out_mask" ]; then
    local nw ng
    nw=$(find "$out_mask" -maxdepth 1 -name 'W_*.nii.gz' 2>/dev/null | wc -l)
    ng=$(find "$out_mask" -maxdepth 1 -name 'G_*.nii.gz' 2>/dev/null | wc -l)
    if [ "$nw" -ge 50 ] && [ "$ng" -ge 246 ]; then
      echo "[SKIP] ${subject}: masks complete W=${nw} G=${ng}"
      return 0
    fi
  fi

  local ref="${dwi_dir}/${subject}_acq-TE${REF_TE}_B0ave.nii.gz"
  local brain_mask="${dwi_dir}/${subject}_acq-TE${REF_TE}_mask.nii.gz"
  if [ ! -f "$ref" ]; then
    ref="$(find "$dwi_dir" -maxdepth 1 -name "${subject}_acq-TE*_B0ave.nii.gz" ! -name '*R2*' | sort | head -n 1)"
  fi
  if [ -z "$ref" ] || [ ! -f "$ref" ]; then
    echo "[SKIP] ${subject}: missing B0ave reference"
    return 0
  fi
  if [ ! -f "$brain_mask" ]; then
    brain_mask="$(find "$dwi_dir" -maxdepth 1 -name "${subject}_acq-TE*_mask.nii.gz" ! -name '*R2*' | sort | head -n 1)"
  fi
  if [ -z "$brain_mask" ] || [ ! -f "$brain_mask" ]; then
    echo "[SKIP] ${subject}: missing DWI mask"
    return 0
  fi

  mkdir -p "$work" "$out_mask"
  echo "[START] ${subject}"
  echo "  reference: $ref"

  # Register this subject's B0 reference to MNI. Then apply inverse transforms
  # to bring MNI-space atlases back into this subject's DWI/B0 space.
  antsRegistrationSyNQuick.sh \
    -d 3 \
    -f "$MNI_TEMPLATE" \
    -m "$ref" \
    -o "${work}/b0_to_mni_" \
    -t s \
    > "${work}/ants.log" 2>&1

  antsApplyTransforms \
    -d 3 \
    -i "$JHU_ATLAS" \
    -r "$ref" \
    -o "${work}/JHU_in_DWI.nii.gz" \
    -n NearestNeighbor \
    -t ["${work}/b0_to_mni_0GenericAffine.mat",1] \
    -t "${work}/b0_to_mni_1InverseWarp.nii.gz" \
    >> "${work}/ants.log" 2>&1

  antsApplyTransforms \
    -d 3 \
    -i "$BN_ATLAS" \
    -r "$ref" \
    -o "${work}/BN_in_DWI.nii.gz" \
    -n NearestNeighbor \
    -t ["${work}/b0_to_mni_0GenericAffine.mat",1] \
    -t "${work}/b0_to_mni_1InverseWarp.nii.gz" \
    >> "${work}/ants.log" 2>&1

  # Mask and copy geometry so viewers and fslstats use the same DWI header.
  fslmaths "${work}/JHU_in_DWI.nii.gz" -mas "$brain_mask" "${work}/JHU_in_DWI_masked.nii.gz"
  fslmaths "${work}/BN_in_DWI.nii.gz" -mas "$brain_mask" "${work}/BN_in_DWI_masked.nii.gz"
  fslcpgeom "$ref" "${work}/JHU_in_DWI_masked.nii.gz"
  fslcpgeom "$ref" "${work}/BN_in_DWI_masked.nii.gz"

  cp "${work}/JHU_in_DWI_masked.nii.gz" "${out_root}/JHU_in_DWI_masked.nii.gz"
  cp "${work}/BN_in_DWI_masked.nii.gz" "${out_root}/BN_in_DWI_masked.nii.gz"

  while read -r id; do
    [ -z "$id" ] && continue
    fslmaths "${work}/JHU_in_DWI_masked.nii.gz" -thr "$id" -uthr "$id" -bin "${out_mask}/W_${id}.nii.gz"
  done < "$JHU_ID_LIST"

  while read -r id; do
    [ -z "$id" ] && continue
    fslmaths "${work}/BN_in_DWI_masked.nii.gz" -thr "$id" -uthr "$id" -bin "${out_mask}/G_${id}.nii.gz"
  done < "$BN_ID_LIST"

  local final_w final_g
  final_w=$(find "$out_mask" -maxdepth 1 -name 'W_*.nii.gz' | wc -l)
  final_g=$(find "$out_mask" -maxdepth 1 -name 'G_*.nii.gz' | wc -l)
  echo "[DONE] ${subject}: W=${final_w} G=${final_g} -> ${out_mask}"
}

export -f run_subject
export ROOT CLEAN_ROOT MNI MNI_TEMPLATE REF_TE FORCE JHU_ATLAS BN_ATLAS JHU_ID_LIST BN_ID_LIST

mapfile -t SUBJECTS < <(find "$ROOT" -maxdepth 1 -type d -name 'sub-*' | sort)
total="${#SUBJECTS[@]}"
echo "ScienceDB BN/JHU mask registration"
echo "Subjects: $total"
echo "ROOT=$ROOT"
echo "CLEAN_ROOT=$CLEAN_ROOT"
echo "REF_TE=$REF_TE"
echo "JOBS=$JOBS THREADS=$THREADS FORCE=$FORCE"

printf '%s\n' "${SUBJECTS[@]}" | xargs -I{} -P "$JOBS" bash -c 'run_subject "$@"' _ {}

echo "All done."
