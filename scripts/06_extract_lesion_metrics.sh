#!/usr/bin/env bash
set -euo pipefail

export LC_NUMERIC=C

ROOT="${ROOT:-/media/UG1/lys/dipy/data/MTE3_clean}"
LABEL_ROOT="${LABEL_ROOT:-/media/UG1/lys/dipy/data/MTE_label}"
RUN_LIST="${RUN_LIST:-${ROOT}/run_list.csv}"
OUT_CSV="${OUT_CSV:-${ROOT}/lesion_metric_results_raw.csv}"
TMP_DIR="${TMP_DIR:-${ROOT}/tmp_lesion_metrics_$(date +%s)}"
MAX_JOBS="${MAX_JOBS:-25}"

mkdir -p "${TMP_DIR}"
echo "Subject,TE,Model,Metric,ROI_ID,ROI_Name,Mean,Std" > "${OUT_CSV}"
echo "ROOT=${ROOT}"
echo "LABEL_ROOT=${LABEL_ROOT}"
echo "RUN_LIST=${RUN_LIST}"
echo "OUT_CSV=${OUT_CSV}"
echo "TMP_DIR=${TMP_DIR}"

TEs=("75" "85" "95" "105" "115" "125" "135")

declare -A METRICS=(
  [DTI]="FA MD RD AD"
  [DTI_RESTORE]="FA MD RD AD"
  [FWDTI]="AD FA FW MD RD"
  [DKI]="DKI_AD DKI_AK DKI_FA DKI_MD DKI_MK DKI_RD DKI_RK"
  [WMTI]="AWF Axonal Hindered_AD Hindered_RD"
  [MSDKI]="DI F MSD MSK uFA"
  [IVIM]="IVIM_D IVIM_D_star IVIM_f IVIM_S0"
  [GQI]="GFA QA"
  ["AMICO/NODDI"]="fit_FWF fit_NDI fit_ODI"
  [FORECAST]="d_par d_perp fa md"
)

[[ -f "${RUN_LIST}" ]] || { echo "ERROR: missing run list: ${RUN_LIST}"; exit 1; }

mapfile -t SUBJECTS < <(awk -F, '
  NF {
    gsub(/\r/, "", $1)
    gsub(/^[ \t]+|[ \t]+$/, "", $1)
    if ($1 == "" || $1 ~ /^#/) next
    if (tolower($1) ~ /^(subject|sub|sid|id)$/) next
    print $1
  }
' "${RUN_LIST}")

echo "Subjects in run list: ${#SUBJECTS[@]}"

for sub in "${SUBJECTS[@]}"; do
(
  echo "[RUN] ${sub}"

  sub_csv="${TMP_DIR}/${sub}.csv"
  : > "${sub_csv}"

  mask=""
  for candidate in \
    "${ROOT}/${sub}/Lesion/L_1.nii.gz" \
    "${LABEL_ROOT}/${sub}/Segmentation.nii.gz" \
    "${LABEL_ROOT}/${sub}/Segmentation.nii"; do
    if [[ -f "${candidate}" ]]; then
      mask="${candidate}"
      break
    fi
  done

  if [[ -z "${mask}" ]]; then
    echo "[SKIP] ${sub}: missing lesion mask in ${ROOT}/${sub}/Lesion or ${LABEL_ROOT}/${sub}"
    exit 0
  fi

  echo "[MASK] ${sub}: ${mask}"
  row_count=0
  missing_img_count=0
  invalid_stats_count=0

  for te in "${TEs[@]}"; do
    for mod in "${!METRICS[@]}"; do
      for met in ${METRICS[$mod]}; do
        img="${ROOT}/${sub}/PA_TE${te}/${mod}/${met}.nii.gz"
        if [[ ! -f "${img}" ]]; then
          missing_img_count=$((missing_img_count + 1))
          continue
        fi

        stats="$(fslstats "${img}" -k "${mask}" -M -S 2>/dev/null || true)"
        read -r mean std <<< "${stats}"

        if [[ -z "${mean:-}" || "${mean}" == "NaN" || -z "${std:-}" || "${std}" == "NaN" ]]; then
          invalid_stats_count=$((invalid_stats_count + 1))
          continue
        fi

        mean_fmt="$(awk -v x="${mean}" 'BEGIN{a=(x<0?-x:x); if(a>0 && a<1e-5) printf "%.10e",x; else printf "%.10f",x}')"
        std_fmt="$(awk -v x="${std}" 'BEGIN{a=(x<0?-x:x); if(a>0 && a<1e-5) printf "%.10e",x; else printf "%.10f",x}')"

        echo "${sub},${te},${mod},${met},L_1,Lesion_mask,${mean_fmt},${std_fmt}" >> "${sub_csv}"
        row_count=$((row_count + 1))
      done
    done
  done

  if [[ "${row_count}" -eq 0 ]]; then
    rm -f "${sub_csv}"
    echo "[WARN] ${sub}: lesion mask found but no valid metric rows. missing_images=${missing_img_count}, invalid_stats=${invalid_stats_count}"
  else
    echo "[DONE] ${sub}: rows=${row_count}, missing_images=${missing_img_count}, invalid_stats=${invalid_stats_count}"
  fi
) &

  while (( $(jobs -r -p | wc -l) >= MAX_JOBS )); do
    sleep 1
  done
done

wait

valid_files=()
shopt -s nullglob
for file in "${TMP_DIR}"/*.csv; do
  if [[ -s "${file}" ]]; then
    valid_files+=("${file}")
  fi
done
shopt -u nullglob

if [[ "${#valid_files[@]}" -gt 0 ]]; then
  cat "${valid_files[@]}" >> "${OUT_CSV}"
  total_rows=$(( $(wc -l < "${OUT_CSV}") - 1 ))
  echo "All done: ${OUT_CSV}"
  echo "Valid subject CSV files: ${#valid_files[@]}"
  echo "Total metric rows: ${total_rows}"
else
  echo "No valid lesion results generated."
  echo "Please check:"
  echo "  1) lesion masks exist under ${LABEL_ROOT}/<subject>/Segmentation.nii(.gz)"
  echo "  2) subject names in ${RUN_LIST} match folders under ${LABEL_ROOT}"
  echo "  3) metric images exist under ${ROOT}/<subject>/PA_TE*/"
fi
