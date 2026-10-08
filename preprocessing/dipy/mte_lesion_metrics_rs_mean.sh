#!/usr/bin/env bash
set -uo pipefail

# ================================================================
# 功能：
#   1) 按 run_list.csv 第一列读取被试 ID
#   2) 只处理存在病灶 mask 的被试：/media/UG1/lys/dipy/data/MTE_label/<sub>/Segmentation.nii
#   3) 以病灶 mask 的尺寸/空间信息为模板，对每个 TE 的 10 个模型、42 个指标重采样
#   4) 用病灶 mask 计算每个重采样指标在病灶区域内的平均值
#   5) 每个被试输出到：/media/UG1/lys/dipy/data/MTE3_clean/<sub>/rs/
#
# 依赖：MRtrix3: mrgrid, mrstats
# ================================================================

ROOT_DIR="/media/UG1/lys/dipy/data/MTE3_clean"
LABEL_DIR="/media/UG1/lys/dipy/data/MTE_label"
SUBJECT_LIST_CSV="${ROOT_DIR}/run_list.csv"

# 如果 run_list.csv 不在 ROOT_DIR 下，可运行时传入：
#   bash mte_lesion_metrics_rs_mean.sh /path/to/run_list.csv
if [[ $# -ge 1 ]]; then
    SUBJECT_LIST_CSV="$1"
fi

MAX_JOBS=20
NTHREADS_PER_JOB=2
TE_LIST=("PA_TE75" "PA_TE85" "PA_TE95" "PA_TE105" "PA_TE115" "PA_TE125" "PA_TE135")

# 每一行格式：model_dir|csv_model_name|csv_metric_name|input_basename_without_nii_gz
# 说明：
#   - model_dir 是指标原始文件所在的模型目录，可包含 AMICO/NODDI 这种多级目录
#   - csv_model_name/csv_metric_name 是统计表里显示的名称
#   - input_basename 是真实 nii.gz 文件名去掉 .nii.gz 后的部分
METRIC_ITEMS=(
    "DTI|DTI|FA|FA"
    "DTI|DTI|MD|MD"
    "DTI|DTI|RD|RD"
    "DTI|DTI|AD|AD"

    "DTI_RESTORE|DTI_RESTORE|FA|FA"
    "DTI_RESTORE|DTI_RESTORE|MD|MD"
    "DTI_RESTORE|DTI_RESTORE|RD|RD"
    "DTI_RESTORE|DTI_RESTORE|AD|AD"

    "FWDTI|FWDTI|AD|AD"
    "FWDTI|FWDTI|FA|FA"
    "FWDTI|FWDTI|FW|FW"
    "FWDTI|FWDTI|MD|MD"
    "FWDTI|FWDTI|RD|RD"

    "DKI|DKI|AD|DKI_AD"
    "DKI|DKI|AK|DKI_AK"
    "DKI|DKI|FA|DKI_FA"
    "DKI|DKI|MD|DKI_MD"
    "DKI|DKI|MK|DKI_MK"
    "DKI|DKI|RD|DKI_RD"
    "DKI|DKI|RK|DKI_RK"

    "WMTI|WMTI|AWF|AWF"
    "WMTI|WMTI|Axonal|Axonal"
    "WMTI|WMTI|AD|Hindered_AD"
    "WMTI|WMTI|RD|Hindered_RD"

    "MSDKI|MSDKI|DI|DI"
    "MSDKI|MSDKI|F|F"
    "MSDKI|MSDKI|MSD|MSD"
    "MSDKI|MSDKI|MSK|MSK"
    "MSDKI|MSDKI|uFA|uFA"

    "IVIM|IVIM|D|IVIM_D"
    "IVIM|IVIM|D_star|IVIM_D_star"
    "IVIM|IVIM|f|IVIM_f"
    "IVIM|IVIM|S0|IVIM_S0"

    "GQI|GQI|GFA|GFA"
    "GQI|GQI|QA|QA"

    "AMICO/NODDI|NODDI|FWF|fit_FWF"
    "AMICO/NODDI|NODDI|NDI|fit_NDI"
    "AMICO/NODDI|NODDI|ODI|fit_ODI"

    "FORECAST|FORECAST|d_par|d_par"
    "FORECAST|FORECAST|d_perp|d_perp"
    "FORECAST|FORECAST|FA|fa"
    "FORECAST|FORECAST|MD|md"
)

log() { echo "[$(date '+%F %T')] $*"; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || { echo "❌ 找不到命令：$1，请先加载/安装 MRtrix3" >&2; exit 1; }
}

csv_escape() {
    local s="$1"
    s="${s//\"/\"\"}"
    printf '"%s"' "$s"
}

# 读取 run_list.csv 的第一列，自动跳过表头 Subject/Subject_id/ID 和空行。
read_subjects() {
    local csv="$1"
    awk -F',' '
        NR==1 {
            gsub(/^\xef\xbb\xbf/, "", $1)
            header=tolower($1)
            if (header ~ /^(subject|subject_id|id|sub|sub_id)$/) next
        }
        {
            gsub(/^\xef\xbb\xbf/, "", $1)
            gsub(/\r/, "", $1)
            gsub(/^[ \t]+|[ \t]+$/, "", $1)
            if ($1 != "") print $1
        }
    ' "$csv"
}

process_subject() {
    local sub="$1"
    local sub_dir="${ROOT_DIR}/${sub}"
    local mask="${LABEL_DIR}/${sub}/Segmentation.nii"
    local rs_root="${sub_dir}/rs"
    local csv_out="${rs_root}/lesion_metrics_mean.csv"
    local missing_log="${rs_root}/missing_metrics.log"

    if [[ ! -d "$sub_dir" ]]; then
        log "⚠️ 跳过 ${sub}: 找不到被试目录 ${sub_dir}"
        return 0
    fi

    if [[ ! -f "$mask" ]]; then
        log "⚠️ 跳过 ${sub}: 找不到病灶 mask ${mask}"
        return 0
    fi

    mkdir -p "$rs_root"
    : > "$missing_log"
    echo "subject,TE,model,metric,mean,input_file,resampled_file,mask_file" > "$csv_out"

    log "⏳ 开始处理 ${sub}"

    local te item model_dir model_name metric_name input_base in_file out_dir out_file mean_value

    for te in "${TE_LIST[@]}"; do
        for item in "${METRIC_ITEMS[@]}"; do
            IFS='|' read -r model_dir model_name metric_name input_base <<< "$item"

            in_file="${sub_dir}/${te}/${model_dir}/${input_base}.nii.gz"
            out_dir="${rs_root}/${te}/${model_dir}"
            out_file="${out_dir}/${input_base}_rs.nii.gz"

            if [[ ! -f "$in_file" ]]; then
                echo "MISS_INPUT,${te},${model_name},${metric_name},${in_file}" >> "$missing_log"
                continue
            fi

            mkdir -p "$out_dir"

            if [[ ! -f "$out_file" ]]; then
                # 以病灶 mask 的尺寸/空间信息作为模板重采样。
                # 指标图用 linear 插值；如果后续重采样 mask，mask 才应使用 nearest。
                if ! mrgrid "$in_file" regrid "$out_file" -template "$mask" -interp linear -force -quiet -nthreads "$NTHREADS_PER_JOB"; then
                    echo "FAIL_REGRID,${te},${model_name},${metric_name},${in_file}" >> "$missing_log"
                    continue
                fi
            fi

            # 计算病灶区域平均值。-ignorezero 可避免指标图中 0 背景进入统计；
            # 如你需要把病灶内真实 0 值也计入均值，请删除 -ignorezero。
            mean_value=$(mrstats "$out_file" -mask "$mask" -ignorezero -output mean 2>/dev/null | awk 'NF{print $1; exit}')
            if [[ -z "$mean_value" ]]; then
                echo "FAIL_STATS,${te},${model_name},${metric_name},${out_file}" >> "$missing_log"
                continue
            fi

            {
                csv_escape "$sub"; printf ','
                csv_escape "$te"; printf ','
                csv_escape "$model_name"; printf ','
                csv_escape "$metric_name"; printf ','
                printf '%s,' "$mean_value"
                csv_escape "$in_file"; printf ','
                csv_escape "$out_file"; printf ','
                csv_escape "$mask"; printf '\n'
            } >> "$csv_out"
        done
    done

    log "✅ 完成 ${sub}: ${csv_out}"
}

main() {
    need_cmd mrgrid
    need_cmd mrstats

    if [[ ! -f "$SUBJECT_LIST_CSV" ]]; then
        echo "❌ 找不到 run_list 文件：$SUBJECT_LIST_CSV" >&2
        echo "用法：bash $0 /path/to/run_list.csv" >&2
        exit 1
    fi

    mapfile -t SUBJECTS < <(read_subjects "$SUBJECT_LIST_CSV")
    if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
        echo "❌ run_list 中没有读取到被试 ID：$SUBJECT_LIST_CSV" >&2
        exit 1
    fi

    log "读取到 ${#SUBJECTS[@]} 个被试；只会处理存在病灶 mask 的被试。并发=${MAX_JOBS}, 单任务线程=${NTHREADS_PER_JOB}"

    local sub
    for sub in "${SUBJECTS[@]}"; do
        process_subject "$sub" &
        while (( $(jobs -r -p | wc -l) >= MAX_JOBS )); do
            sleep 1
        done
    done

    wait
    log "🎉 全部任务结束"
}

main "$@"
