#!/usr/bin/env bash
set -u

# ============================================================
# 功能：
# 1) 从 run_list.csv 读取被试
# 2) 只处理存在病灶 mask 的被试：
#    /media/UG1/lys/dipy/data/MTE_label/<sub>/Segmentation.nii 或 .nii.gz
# 3) 第一阶段：以病灶 mask 为 template，并行重采样所有 TE/模型/指标
#    核心命令：
#    mrgrid "$in_file" regrid "$out_file" -template "$TEMPLATE" -force -quiet -nthreads 2
# 4) 第二阶段：重采样全部完成后，再按固定顺序计算病灶均值并输出 CSV
# 5) 每个被试输出：
#    /media/UG1/lys/dipy/data/MTE3_clean/<sub>/rs/lesion_metrics_mean.csv
#
# 依赖：MRtrix3: mrgrid, mrstats
# ============================================================

ROOT_DIR="/media/UG1/lys/dipy/data/MTE3_clean"
LABEL_DIR="/media/UG1/lys/dipy/data/MTE_label"
RUN_LIST="${1:-${ROOT_DIR}/run_list.csv}"

# ============================================================
# 并发设置建议：
# 80 核 + 503G 内存：推荐 MAX_JOBS=24 或 32，MRGRID_THREADS=2
# 如果磁盘 IO 很快且服务器无人使用，可尝试 MAX_JOBS=32
# ============================================================
MAX_JOBS=24
MRGRID_THREADS=2

TE_LIST=("PA_TE75" "PA_TE85" "PA_TE95" "PA_TE105" "PA_TE115" "PA_TE125" "PA_TE135")

# 是否覆盖已有重采样文件：
# 0 = 已存在则跳过
# 1 = 强制重新生成
OVERWRITE=0

command -v mrgrid >/dev/null 2>&1 || {
    echo "❌ 找不到 mrgrid，请先加载 MRtrix3 环境"
    exit 1
}

command -v mrstats >/dev/null 2>&1 || {
    echo "❌ 找不到 mrstats，请先加载 MRtrix3 环境"
    exit 1
}

if [ ! -f "$RUN_LIST" ]; then
    echo "❌ 找不到 run_list 文件：$RUN_LIST"
    exit 1
fi

# ============================================================
# 10 个模型，42 个指标
#
# 格式：
# model_key|model_dir|metric_label|candidate_file_stems
#
# candidate_file_stems 支持多个候选文件名，用逗号分隔。
# 例如 DKI 的 AD 可能叫 AD.nii.gz，也可能叫 DKI_AD.nii.gz。
# 脚本会自动寻找第一个存在的 nii.gz 或 nii 文件。
# ============================================================

METRIC_ITEMS=(
    # DTI, 4
    "DTI|DTI|FA|FA"
    "DTI|DTI|MD|MD"
    "DTI|DTI|RD|RD"
    "DTI|DTI|AD|AD"

    # DTI_RESTORE, 4
    "DTI_RESTORE|DTI_RESTORE|FA|FA"
    "DTI_RESTORE|DTI_RESTORE|MD|MD"
    "DTI_RESTORE|DTI_RESTORE|RD|RD"
    "DTI_RESTORE|DTI_RESTORE|AD|AD"

    # FWDTI, 5
    "FWDTI|FWDTI|AD|AD"
    "FWDTI|FWDTI|FA|FA"
    "FWDTI|FWDTI|FW|FW"
    "FWDTI|FWDTI|MD|MD"
    "FWDTI|FWDTI|RD|RD"

    # DKI, 7
    "DKI|DKI|AD|AD,DKI_AD"
    "DKI|DKI|AK|AK,DKI_AK"
    "DKI|DKI|FA|FA,DKI_FA"
    "DKI|DKI|MD|MD,DKI_MD"
    "DKI|DKI|MK|MK,DKI_MK"
    "DKI|DKI|RD|RD,DKI_RD"
    "DKI|DKI|RK|RK,DKI_RK"

    # WMTI, 4
    "WMTI|WMTI|AWF|AWF"
    "WMTI|WMTI|Axonal|Axonal"
    "WMTI|WMTI|AD|AD,Hindered_AD"
    "WMTI|WMTI|RD|RD,Hindered_RD"

    # MSDKI, 5
    "MSDKI|MSDKI|DI|DI"
    "MSDKI|MSDKI|F|F"
    "MSDKI|MSDKI|MSD|MSD"
    "MSDKI|MSDKI|MSK|MSK"
    "MSDKI|MSDKI|uFA|uFA"

    # IVIM, 4
    "IVIM|IVIM|D|D,IVIM_D"
    "IVIM|IVIM|D_star|D_star,IVIM_D_star"
    "IVIM|IVIM|f|f,IVIM_f"
    "IVIM|IVIM|S0|S0,IVIM_S0"

    # GQI, 2
    "GQI|GQI|GFA|GFA"
    "GQI|GQI|QA|QA"

    # NODDI, 3
    # 实际路径示例：AMICO/NODDI/fit_FWF.nii.gz
    "NODDI|AMICO/NODDI|FWF|fit_FWF,FWF"
    "NODDI|AMICO/NODDI|NDI|fit_NDI,NDI"
    "NODDI|AMICO/NODDI|ODI|fit_ODI,ODI,odi"

    # FORECAST, 4
    "FORECAST|FORECAST|d_par|d_par"
    "FORECAST|FORECAST|d_perp|d_perp"
    "FORECAST|FORECAST|FA|FA,fa"
    "FORECAST|FORECAST|MD|MD,md"
)

wait_for_slot() {
    while [ "$(jobs -r -p | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 0.5
    done
}

find_metric_file() {
    local base_dir="$1"
    local stems_csv="$2"
    local stem candidate

    IFS=',' read -ra stems <<< "$stems_csv"

    for stem in "${stems[@]}"; do
        for candidate in \
            "${base_dir}/${stem}.nii.gz" \
            "${base_dir}/${stem}.nii"; do
            if [ -f "$candidate" ]; then
                echo "$candidate"
                return 0
            fi
        done
    done

    echo ""
    return 1
}

resample_one_metric() {
    local sub="$1"
    local te="$2"
    local model_key="$3"
    local model_dir="$4"
    local metric_label="$5"
    local stems_csv="$6"
    local TEMPLATE="$7"
    local miss_log="$8"

    local in_base in_file out_dir out_file status

    in_base="${ROOT_DIR}/${sub}/${te}/${model_dir}"
    in_file=$(find_metric_file "$in_base" "$stems_csv")

    if [ -z "$in_file" ]; then
        echo "[缺失原始指标] ${sub},${te},${model_key},${metric_label},${in_base},候选:${stems_csv}" >> "$miss_log"
        return 0
    fi

    # 输出到每个被试新建 rs 文件夹下；model_dir 可为 AMICO/NODDI 多级目录
    out_dir="${ROOT_DIR}/${sub}/rs/${te}/${model_dir}"
    mkdir -p "$out_dir"

    out_file="${out_dir}/${metric_label}_rs.nii.gz"

    if [ "$OVERWRITE" -eq 0 ] && [ -f "$out_file" ]; then
        return 0
    fi

    # ========================================================
    # 你指定的核心重采样命令
    # TEMPLATE 是每个被试的病灶 mask
    # ========================================================
    mrgrid "$in_file" regrid "$out_file" -template "$TEMPLATE" -force -quiet -nthreads "$MRGRID_THREADS"

    status=$?
    if [ "$status" -ne 0 ] || [ ! -f "$out_file" ]; then
        echo "[重采样失败] ${sub},${te},${model_key},${metric_label},${in_file}" >> "$miss_log"
        return 0
    fi
}

write_mean_csv_after_resampling() {
    local sub="$1"
    local mask="$2"
    local csv_file="$3"
    local miss_log="$4"

    local te item model_key model_dir metric_label stems_csv
    local in_base in_file out_file mean_val

    echo "subject,TE,model,metric,mean,input_file,resampled_file,mask_file" > "$csv_file"

    # 这里不并发写 CSV，而是严格按 TE_LIST 和 METRIC_ITEMS 顺序写入，保证排列整齐
    for te in "${TE_LIST[@]}"; do
        for item in "${METRIC_ITEMS[@]}"; do
            IFS='|' read -r model_key model_dir metric_label stems_csv <<< "$item"

            in_base="${ROOT_DIR}/${sub}/${te}/${model_dir}"
            in_file=$(find_metric_file "$in_base" "$stems_csv")
            out_file="${ROOT_DIR}/${sub}/rs/${te}/${model_dir}/${metric_label}_rs.nii.gz"

            if [ -z "$in_file" ]; then
                echo "${sub},${te},${model_key},${metric_label},NA,NA,NA,${mask}" >> "$csv_file"
                continue
            fi

            if [ ! -f "$out_file" ]; then
                echo "[缺失重采样结果] ${sub},${te},${model_key},${metric_label},${out_file}" >> "$miss_log"
                echo "${sub},${te},${model_key},${metric_label},NA,${in_file},${out_file},${mask}" >> "$csv_file"
                continue
            fi

            # -ignorezero 避免指标图背景 0 影响均值。
            # 如果病灶内真实 0 值需要参与均值，请删除 -ignorezero。
            mean_val=$(mrstats "$out_file" -mask "$mask" -output mean -ignorezero 2>/dev/null | awk 'NR==1{print $1}')

            if [ -z "$mean_val" ]; then
                mean_val="NA"
                echo "[均值失败] ${sub},${te},${model_key},${metric_label},${out_file}" >> "$miss_log"
            fi

            echo "${sub},${te},${model_key},${metric_label},${mean_val},${in_file},${out_file},${mask}" >> "$csv_file"
        done
    done
}

# ============================================================
# 读取 run_list.csv
# ============================================================
SUBJECTS=()

while IFS=',' read -r sub_id _rest || [ -n "$sub_id" ]; do
    sub_clean=$(echo "$sub_id" | tr -d '\r' | xargs)

    if [ -n "$sub_clean" ] \
        && [ "$sub_clean" != "Subject" ] \
        && [ "$sub_clean" != "ID" ] \
        && [ "$sub_clean" != "subject" ] \
        && [ "$sub_clean" != "id" ]; then
        SUBJECTS+=("$sub_clean")
    fi
done < "$RUN_LIST"

if [ "${#SUBJECTS[@]}" -eq 0 ]; then
    echo "❌ run_list 中没有读取到被试 ID：$RUN_LIST"
    exit 1
fi

echo "============================================================"
echo "ROOT_DIR        : $ROOT_DIR"
echo "LABEL_DIR       : $LABEL_DIR"
echo "RUN_LIST        : $RUN_LIST"
echo "被试数量        : ${#SUBJECTS[@]}"
echo "TE 数量         : ${#TE_LIST[@]}"
echo "指标数量        : ${#METRIC_ITEMS[@]}"
echo "MAX_JOBS        : $MAX_JOBS"
echo "MRGRID_THREADS  : $MRGRID_THREADS"
echo "总线程上限约    : $((MAX_JOBS * MRGRID_THREADS))"
echo "OVERWRITE       : $OVERWRITE"
echo "============================================================"

processed_subjects=0
skipped_subjects=0

for sub in "${SUBJECTS[@]}"; do
    mask="${LABEL_DIR}/${sub}/Segmentation.nii"

    if [ ! -f "$mask" ] && [ -f "${LABEL_DIR}/${sub}/Segmentation.nii.gz" ]; then
        mask="${LABEL_DIR}/${sub}/Segmentation.nii.gz"
    fi

    if [ ! -f "$mask" ]; then
        echo "⚠️ [跳过] ${sub}: 找不到病灶 mask：${LABEL_DIR}/${sub}/Segmentation.nii 或 .nii.gz"
        skipped_subjects=$((skipped_subjects + 1))
        continue
    fi

    rs_root="${ROOT_DIR}/${sub}/rs"
    mkdir -p "$rs_root"

    csv_file="${rs_root}/lesion_metrics_mean.csv"
    miss_log="${rs_root}/missing_metrics.log"
    : > "$miss_log"

    echo "▶ 被试 ${sub}: 第一阶段开始重采样"

    # ========================================================
    # 第一阶段：只重采样，不写 CSV，不计算均值
    # ========================================================
    for te in "${TE_LIST[@]}"; do
        for item in "${METRIC_ITEMS[@]}"; do
            IFS='|' read -r model_key model_dir metric_label stems_csv <<< "$item"

            wait_for_slot

            resample_one_metric \
                "$sub" \
                "$te" \
                "$model_key" \
                "$model_dir" \
                "$metric_label" \
                "$stems_csv" \
                "$mask" \
                "$miss_log" &
        done
    done

    # 当前被试全部重采样完成，再开始统计均值
    wait
    echo "✅ 被试 ${sub}: 重采样完成"

    # ========================================================
    # 第二阶段：按固定顺序计算均值并写 CSV，保证排列整齐
    # ========================================================
    echo "▶ 被试 ${sub}: 第二阶段开始计算病灶均值并写 CSV"
    write_mean_csv_after_resampling "$sub" "$mask" "$csv_file" "$miss_log"

    n_rows=$(($(wc -l < "$csv_file") - 1))
    n_missing=$(wc -l < "$miss_log")

    echo "✅ 完成被试：${sub} | CSV 行数: ${n_rows} | 缺失/失败日志: ${n_missing} 条 | 输出: ${csv_file}"

    processed_subjects=$((processed_subjects + 1))
done

echo "============================================================"
echo "🎉 全部完成"
echo "完成被试数：$processed_subjects"
echo "跳过被试数：$skipped_subjects"
echo "============================================================"
