#!/usr/bin/env bash
set -euo pipefail

LABEL_ROOT="${LABEL_ROOT:-/media/UG1/lys/dipy/data/MTE_label}"
CLEAN_ROOT="${CLEAN_ROOT:-/media/UG1/lys/dipy/data/MTE3_clean}"
RUN_LIST="${RUN_LIST:-${CLEAN_ROOT}/run_list.csv}"
OUT_CSV="${OUT_CSV:-${CLEAN_ROOT}/lesion_mask_summary.csv}"

echo "subject,source_mask,output_mask,status" > "${OUT_CSV}"

mapfile -t SUBJECTS < <(awk -F, '
  NF {
    gsub(/\r/, "", $1)
    gsub(/^[ \t]+|[ \t]+$/, "", $1)
    if ($1 == "" || $1 ~ /^#/) next
    if (tolower($1) ~ /^(subject|sub|sid|id)$/) next
    print $1
  }
' "${RUN_LIST}")

for sub in "${SUBJECTS[@]}"; do
  src=""
  for candidate in \
    "${LABEL_ROOT}/${sub}/Segmentation.nii.gz" \
    "${LABEL_ROOT}/${sub}/Segmentation.nii"; do
    if [[ -f "${candidate}" ]]; then
      src="${candidate}"
      break
    fi
  done

  out_dir="${CLEAN_ROOT}/${sub}/Lesion"
  out="${out_dir}/L_1.nii.gz"

  if [[ -z "${src}" ]]; then
    echo "${sub},,${out},missing_source" >> "${OUT_CSV}"
    echo "[SKIP] ${sub}: missing lesion mask"
    continue
  fi

  mkdir -p "${out_dir}"

  fslmaths "${src}" -bin "${out}"
  echo "${sub},${src},${out},complete" >> "${OUT_CSV}"
  echo "[DONE] ${sub}: ${out}"
done

echo "All done: ${OUT_CSV}"
