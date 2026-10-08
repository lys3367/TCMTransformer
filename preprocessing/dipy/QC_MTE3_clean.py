import os
import shutil
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 路径与核心配置
# ==========================================
INPUT_DIR = r'/media/UG1/lys/dipy/data/MTE3'
OUTPUT_DIR = r'/media/UG1/lys/dipy/data/MTE3_clean'
QC_CSV = r'/media/UG1/lys/dipy/data/MTE3/QC_Flagged_Report.csv'

# 🚀 核心加速开关：设置同时运行的线程数
# 建议设置为你服务器 CPU 核心数的一半到两倍（比如 16, 24 或 32）
MAX_WORKERS = 34

PHYSICAL_BOUNDS = {
    "FRACTION": {
        "metrics": {"FA", "DKI_FA", "GFA", "uFA", "fa", "fit_FWF", "FW", "AWF", "F", "IVIM_f", "fit_NDI", "fit_ODI"},
        "min": -0.1, "max": 1.5
    },
    "DIFFUSIVITY": {
        "metrics": {"MD", "RD", "AD", "md", "DKI_MD", "DKI_AD", "DKI_RD", "Hindered_AD", "Hindered_RD", "Axonal", "DI", "IVIM_D", "IVIM_D_star", "d_par", "d_perp"},
        "min": -0.01, "max": 0.05
    },
    "KURTOSIS": {
        "metrics": {"DKI_MK", "DKI_AK", "DKI_RK", "MSK"},
        "min": -2.0, "max": 5.0
    },
    "UNBOUNDED": {
        "metrics": {"QA", "IVIM_S0", "MSD"},
        "min": -100.0, "max": float('inf')
    }
}

def get_bounds(metric_name):
    for category, config in PHYSICAL_BOUNDS.items():
        if metric_name in config["metrics"]:
            return config["min"], config["max"]
    return -1000.0, 100000.0

# ==========================================
# 2. 单个文件的处理逻辑 (被多线程调用的工作函数)
# ==========================================
def process_single_file(row):
    sub, te, model, metric, status = row['Subject'], row['TE'], row['Model'], row['Metric'], row['QC_Status']

    in_file = os.path.join(INPUT_DIR, sub, te, model, f"{metric}.nii.gz")
    out_folder = os.path.join(OUTPUT_DIR, sub, te, model)
    out_file = os.path.join(out_folder, f"{metric}.nii.gz")

    if not os.path.exists(in_file):
        return "SKIPPED"

    os.makedirs(out_folder, exist_ok=True)

    # 顺手携带 mask
    in_mask = os.path.join(INPUT_DIR, sub, te, "mask.nii.gz")
    out_mask_folder = os.path.join(OUTPUT_DIR, sub, te)
    out_mask = os.path.join(out_mask_folder, "mask.nii.gz")
    if os.path.exists(in_mask) and not os.path.exists(out_mask):
        os.makedirs(out_mask_folder, exist_ok=True)
        try:
            shutil.copy2(in_mask, out_mask)
        except:
            pass # 多线程下可能发生冲突，忽略即可

    # 判定与执行
    if "Physics Violation" in status or "Stat Warning" in status:
        min_val, max_val = get_bounds(metric)
        if max_val == float('inf'):
            cmd = f"fslmaths {in_file} -thr {min_val} {out_file}"
        else:
            cmd = f"fslmaths {in_file} -thr {min_val} -uthr {max_val} {out_file}"
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "FIXED"
    else:
        shutil.copy2(in_file, out_file)
        if "Fatal" in status or "Warning" in status:
            return "WARNING_COPIED"
        return "COPIED"

# ==========================================
# 3. 多线程主程序
# ==========================================
if __name__ == '__main__':
    print(f"📖 正在读取质控报告: {QC_CSV}")
    df = pd.read_csv(QC_CSV)

    tasks = [row for _, row in df.iterrows()]
    total_tasks = len(tasks)

    print(f"🚀 开始构建 MTE3_clean 镜像数据舱 (并发线程数: {MAX_WORKERS})...")

    count_copied = 0
    count_fixed = 0
    count_skipped = 0
    count_warnings = 0

    # 启用线程池
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 将所有任务提交给线程池
        futures = {executor.submit(process_single_file, task): task for task in tasks}

        # 实时收集结果并显示进度
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res == "FIXED": count_fixed += 1
            elif res == "COPIED": count_copied += 1
            elif res == "SKIPPED": count_skipped += 1
            elif res == "WARNING_COPIED": count_warnings += 1

            # 每处理 500 个文件打印一次进度，防止刷屏
            if i % 500 == 0 or i == total_tasks:
                print(f"   ⏳ 进度: {i}/{total_tasks} ({(i/total_tasks)*100:.1f}%) | 修复: {count_fixed} | 拷贝: {count_copied + count_warnings}")

    print("\n==============================================")
    print("🎉 数据清洗与镜像构建全部极速完成！")
    print(f"📁 干净的数据已存放于: {OUTPUT_DIR}")
    print(f"   - 完美复制: {count_copied} 个")
    print(f"   - 带伤保留(无法修复): {count_warnings} 个")
    print(f"   - 成功置零修复: {count_fixed} 个")
    print(f"   - 文件丢失跳过: {count_skipped} 个")
    print("==============================================")