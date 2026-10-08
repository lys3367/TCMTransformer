#!/usr/bin/env bash
set -euo pipefail

export LC_NUMERIC=C

ROOT="${ROOT:-/media/UG1/lys/dipy/data/MTE3_clean}"
RUN_LIST="${RUN_LIST:-${ROOT}/run_list.csv}"
OUT_CSV="${OUT_CSV:-${ROOT}/BN_JHU_metric_results_raw.csv}"
TMP_DIR="${TMP_DIR:-${ROOT}/tmp_bn_jhu_metrics_$(date +%s)}"
MAX_JOBS="${MAX_JOBS:-25}"

mkdir -p "${TMP_DIR}"
echo "Subject,TE,Model,Metric,ROI_ID,ROI_Name,Mean,Std" > "${OUT_CSV}"

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

for sub in "${SUBJECTS[@]}"; do
(
  echo "[RUN] ${sub}"

  sub_csv="${TMP_DIR}/${sub}.csv"
  : > "${sub_csv}"

  mask_dir="${ROOT}/${sub}/BN_JHU"
  if [[ ! -d "${mask_dir}" ]]; then
    echo "[SKIP] ${sub}: missing mask dir ${mask_dir}"
    exit 0
  fi

  shopt -s nullglob
  masks=("${mask_dir}"/W_*.nii.gz "${mask_dir}"/G_*.nii.gz)
  shopt -u nullglob

  if [[ "${#masks[@]}" -eq 0 ]]; then
    echo "[SKIP] ${sub}: no BN/JHU masks"
    exit 0
  fi

  for te in "${TEs[@]}"; do
    for mod in "${!METRICS[@]}"; do
      for met in ${METRICS[$mod]}; do
        img="${ROOT}/${sub}/PA_TE${te}/${mod}/${met}.nii.gz"
        [[ -f "${img}" ]] || continue

        for mask in "${masks[@]}"; do
          roi_id="$(basename "${mask}" .nii.gz)"
          roi_name="${roi_id}"

          stats="$(fslstats "${img}" -k "${mask}" -M -S 2>/dev/null || true)"
          read -r mean std <<< "${stats}"

          [[ -n "${mean:-}" && "${mean}" != "NaN" ]] || continue
          [[ -n "${std:-}" && "${std}" != "NaN" ]] || continue

          mean_fmt="$(awk -v x="${mean}" 'BEGIN{a=(x<0?-x:x); if(a>0 && a<1e-5) printf "%.10e",x; else printf "%.10f",x}')"
          std_fmt="$(awk -v x="${std}" 'BEGIN{a=(x<0?-x:x); if(a>0 && a<1e-5) printf "%.10e",x; else printf "%.10f",x}')"

          echo "${sub},${te},${mod},${met},${roi_id},${roi_name},${mean_fmt},${std_fmt}" >> "${sub_csv}"
        done
      done
    done
  done

  echo "[DONE] ${sub}"
) &

  while (( $(jobs -r -p | wc -l) >= MAX_JOBS )); do
    sleep 1
  done
done

wait

if ls "${TMP_DIR}"/*.csv >/dev/null 2>&1; then
  cat "${TMP_DIR}"/*.csv >> "${OUT_CSV}"
  echo "All done: ${OUT_CSV}"
else
  echo "No valid results generated."
fi
