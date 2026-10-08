from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Subject", "TE", "Model", "Metric", "ROI_ID", "ROI_Name", "Mean", "Std"]
TARGET_TE = np.asarray([75, 85, 95, 105, 115, 125, 135], dtype=float)


def interp_extrapolate(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x = x[order].astype(float)
    y = y[order].astype(float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) < 2:
        return np.full(len(target), np.nan, dtype=float)

    result = np.interp(target, x, y)
    left = target < x[0]
    right = target > x[-1]
    if left.any():
        slope = (y[1] - y[0]) / (x[1] - x[0])
        result[left] = y[0] + slope * (target[left] - x[0])
    if right.any():
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        result[right] = y[-1] + slope * (target[right] - x[-1])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpolate ScienceDB native-TE ROI metrics to the internal 7TE grid."
    )
    parser.add_argument(
        "--input",
        default="data/raw/ScienceDB_BN_JHU_metric_results_native.csv",
        help="Native ScienceDB ROI metric CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/ScienceDB_BN_JHU_metric_results_interpolated.csv",
        help="Interpolated ScienceDB ROI metric CSV.",
    )
    parser.add_argument(
        "--drop-repeat",
        action="store_true",
        default=True,
        help="Ignore repeated TE labels such as 62R2/132R2 if present.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = root / input_path
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["TE_raw"] = df["TE"].astype(str)
    df["is_repeat"] = df["TE_raw"].str.contains("R2", case=False, regex=False)
    if args.drop_repeat:
        df = df[~df["is_repeat"]].copy()
    df["TE_num"] = df["TE_raw"].str.replace("R2", "", regex=False).astype(float)
    df["ROI_ID"] = df["ROI_ID"].astype(str)

    rows = []
    keys = ["Subject", "Model", "Metric", "ROI_ID", "ROI_Name"]
    for key, group in df.groupby(keys, sort=True, dropna=False):
        mean_values = interp_extrapolate(
            group["TE_num"].to_numpy(dtype=float),
            group["Mean"].to_numpy(dtype=float),
            TARGET_TE,
        )
        std_values = interp_extrapolate(
            group["TE_num"].to_numpy(dtype=float),
            group["Std"].to_numpy(dtype=float),
            TARGET_TE,
        )
        for te, mean, std in zip(TARGET_TE.astype(int), mean_values, std_values):
            rows.append(
                {
                    "Subject": key[0],
                    "TE": int(te),
                    "Model": key[1],
                    "Metric": key[2],
                    "ROI_ID": key[3],
                    "ROI_Name": key[4],
                    "Mean": mean,
                    "Std": std,
                }
            )

    out = pd.DataFrame(rows)
    if out["Mean"].isna().any():
        n_missing = int(out["Mean"].isna().sum())
        raise ValueError(f"Interpolated table contains {n_missing} missing Mean values.")
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    report = [
        "# 41 ScienceDB TE 插值",
        "",
        f"- 输入：`{input_path}`",
        f"- 输出：`{output_path}`",
        f"- 原始受试者数：{df['Subject'].nunique()}",
        f"- 输出记录数：{len(out):,}",
        f"- 目标 TE：{TARGET_TE.astype(int).tolist()}",
        "",
        "该步骤只改变 TE 采样网格，不改变 ROI、模型和指标定义。TE135 是从 ScienceDB TE132 轻微外推得到，论文中需要明确说明。",
    ]
    text = "\n".join(report) + "\n"
    (output_path.parent / "41_sciencedb_interpolation_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
