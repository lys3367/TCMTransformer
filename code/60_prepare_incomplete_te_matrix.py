from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd


common = importlib.import_module("00_common")


REQUIRED_COLUMNS = ["Subject", "TE", "Model", "Metric", "ROI_ID", "ROI_Name", "Mean", "Std"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a variable-availability TE matrix for complete and incomplete BN/JHU subjects."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--min-te", type=int, default=3)
    parser.add_argument(
        "--output",
        default="data/processed/60_model_matrix_incomplete_te.npz",
    )
    args = parser.parse_args()

    config = common.load_config(args.config)
    raw_csv = common.resolve_path(config, "raw_csv")
    metadata_path = common.resolve_path(config, "metadata_file")
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = common.ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_csv)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    metadata = pd.read_csv(metadata_path)
    metadata = metadata.sort_values("variable_index").reset_index(drop=True)
    expected_te = np.asarray(config["expected_te"], dtype=np.int16)
    variables = metadata["variable"].astype(str).to_numpy(dtype=object)

    df = df.copy()
    df["ROI_ID"] = df["ROI_ID"].astype(str)
    df["feature"] = df["Model"].astype(str) + "__" + df["Metric"].astype(str)
    df["variable"] = df["feature"] + "__" + df["ROI_ID"].astype(str)
    df = df[df["variable"].isin(set(variables))].copy()
    df = df[df["TE"].astype(int).isin(expected_te.astype(int))].copy()

    te_count = df.groupby("Subject")["TE"].nunique()
    subjects = np.asarray(
        sorted(te_count[te_count >= args.min_te].index.astype(str)),
        dtype=object,
    )
    full_index = pd.MultiIndex.from_product(
        [subjects, expected_te.astype(int), variables],
        names=["Subject", "TE", "variable"],
    )
    series = df.set_index(["Subject", "TE", "variable"])["Mean"].reindex(full_index)
    x_raw = series.to_numpy(dtype=np.float32).reshape(
        len(subjects), len(expected_te), len(variables)
    )
    observed_mask = np.isfinite(x_raw).all(axis=2)
    te_available_count = observed_mask.sum(axis=1).astype(np.int16)
    complete7_mask = te_available_count == len(expected_te)

    np.savez_compressed(
        output_path,
        x_raw=x_raw,
        observed_mask=observed_mask,
        te_available_count=te_available_count,
        complete7_mask=complete7_mask,
        subjects=subjects,
        te_values=expected_te,
        variables=variables,
        models=metadata["model"].to_numpy(dtype=object),
        metrics=metadata["metric"].to_numpy(dtype=object),
        features=metadata["feature"].to_numpy(dtype=object),
        roi_ids=metadata["roi_id"].to_numpy(dtype=object),
        roi_labels=metadata["roi_label"].to_numpy(dtype=object),
    )

    coverage = pd.Series(te_available_count).value_counts().sort_index()
    report = [
        "# 60 不完整 TE 矩阵准备",
        "",
        f"- 输入 CSV：`{raw_csv}`",
        f"- 输出 NPZ：`{output_path}`",
        f"- 最少 TE 要求：{args.min_te}",
        f"- 纳入受试者数：{len(subjects)}",
        f"- 完整 7TE 受试者数：{int(complete7_mask.sum())}",
        f"- 不完整 TE 受试者数：{int((~complete7_mask).sum())}",
        f"- TE：{expected_te.astype(int).tolist()}",
        f"- 变量数：{len(variables)}",
        f"- 矩阵形状：{tuple(x_raw.shape)}，subject × TE × variable",
        "",
        "## TE 覆盖",
        "",
    ]
    for te_n, count in coverage.items():
        report.append(f"- {int(te_n)} 个 TE：{int(count)} 名")
    report.append("")
    report.append("说明：缺失 TE 在矩阵中保留为 NaN；后续 masked-TE 训练会用 observed_mask 控制可用 TE。")

    report_path = common.resolve_path(config, "output_dir") / "60_incomplete_te_matrix_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
