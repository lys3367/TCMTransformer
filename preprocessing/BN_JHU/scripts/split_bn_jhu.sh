#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/media/UG1/lys/dipy/data/MTE3}"
CLEAN_ROOT="${CLEAN_ROOT:-/media/UG1/lys/dipy/data/MTE3_clean}"
RUN_LIST="${RUN_LIST:-${DATA_ROOT}/run_list.csv}"
MNI="${MNI:-/media/UG1/lys/dipy/data/MNI}"

JHU_ID_LIST="${JHU_ID_LIST:-${MNI}/JHU_id_list.txt}"
BN_ID_LIST="${BN_ID_LIST:-${MNI}/BN_id_list.txt}"

split_atlas() {
  local atlas="$1"
  local id_list="$2"
  local out_dir="$3"
  local prefix="$4"
  local sid="$5"

  [[ -f "${atlas}" ]] || { echo "ERROR: missing atlas: ${atlas}"; return 1; }
  [[ -f "${id_list}" ]] || { echo "ERROR: missing id list: ${id_list}"; return 1; }

  local total
  total="$(grep -Ev '^[[:space:]]*$|^[[:space:]]*#' "${id_list}" | wc -l)"
  local n=0

  while read -r id; do
    [[ -z "${id}" || "${id}" =~ ^# ]] && continue
    n=$((n + 1))
    if [[ "${n}" == "1" || "${n}" == "${total}" || $((n % 50)) == 0 ]]; then
      echo "[${sid}] ${prefix}: ${n}/${total}"
    fi
    fslmaths "${atlas}" -thr "${id}" -uthr "${id}" -bin "${out_dir}/${prefix}_${id}.nii.gz"
  done < "${id_list}"
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

for sub_dir in "${SUBJECTS[@]}"; do
  [[ -d "${sub_dir}" ]] || continue
  IDX=$((IDX + 1))

  sid="$(basename "${sub_dir}")"
  mri="${sub_dir}/PA_TE135/freesurfer/mri"
  jhu_atlas="${mri}/JHU_in_DWI_masked.nii.gz"
  bn_atlas="${mri}/BN_in_DWI_masked.nii.gz"
  out_dir="${CLEAN_ROOT}/${sid}/BN_JHU"

  if [[ ! -f "${jhu_atlas}" || ! -f "${bn_atlas}" ]]; then
    echo "[${IDX}/${TOTAL}] [SKIP] ${sid}: missing registered JHU/BN atlas"
    continue
  fi

  mkdir -p "${out_dir}"
  w_count="$(find "${out_dir}" -maxdepth 1 -name 'W_*.nii.gz' | wc -l)"
  g_count="$(find "${out_dir}" -maxdepth 1 -name 'G_*.nii.gz' | wc -l)"

  if [[ "${w_count}" -eq 50 && "${g_count}" -eq 246 ]]; then
    echo "[${IDX}/${TOTAL}] [SKIP] ${sid}: masks complete W=50 G=246"
    continue
  fi

  echo "------------------------------------------------------------"
  echo "[${IDX}/${TOTAL}] [RUN] ${sid} -> ${out_dir}"
  echo "Current masks: W=${w_count}/50, G=${g_count}/246"
  echo "------------------------------------------------------------"

  split_atlas "${jhu_atlas}" "${JHU_ID_LIST}" "${out_dir}" "W" "${sid}" || { echo "[${IDX}/${TOTAL}] [FAIL] ${sid}: split W"; continue; }
  split_atlas "${bn_atlas}" "${BN_ID_LIST}" "${out_dir}" "G" "${sid}" || { echo "[${IDX}/${TOTAL}] [FAIL] ${sid}: split G"; continue; }

  echo "[${IDX}/${TOTAL}] [DONE] ${sid}"
done

echo "Split done."
