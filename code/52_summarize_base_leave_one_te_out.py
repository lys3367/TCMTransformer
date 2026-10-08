from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def format_value(mean: float, sd: float) -> str:
    if pd.isna(sd):
        return f"{mean:.6f}"
    return f"{mean:.6f} ± {sd:.6f}"


def main() -> None:
    files = sorted(OUTPUT_DIR.glob("51_*_leave_one_te_out_*_full*.csv"))
    files = [
        path
        for path in files
        if not path.name.endswith("_by_roi.csv")
        and not path.name.endswith("_by_te_summary.csv")
    ]
    if not files:
        print("No 51-series base leave-one-TE-out result files found.")
        return

    rows = []
    by_te_tables = []
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        rows.append(df)
        by_te_path = path.with_name(path.stem + "_by_te_summary.csv")
        if by_te_path.exists():
            by_te = pd.read_csv(by_te_path)
            by_te.insert(0, "source_file", path.name)
            by_te_tables.append(by_te)

    if not rows:
        print("No valid rows found.")
        return

    table = pd.concat(rows, ignore_index=True)
    summary = (
        table.groupby(["task", "model", "model_name", "grouping"], as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            MAE_sd=("MAE", "std"),
            RMSE=("RMSE", "mean"),
            RMSE_sd=("RMSE", "std"),
            R2=("R2", "mean"),
            R2_sd=("R2", "std"),
            n=("MAE", "size"),
        )
        .sort_values(["task", "MAE"])
    )
    summary.to_csv(
        OUTPUT_DIR / "52_base_leave_one_te_out_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if by_te_tables:
        pd.concat(by_te_tables, ignore_index=True).to_csv(
            OUTPUT_DIR / "52_base_leave_one_te_out_by_te_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    report = ["# 52 基础模型 Leave-one-TE-out 汇总", ""]
    for task, group in summary.groupby("task", sort=True):
        report.extend([f"## {task}", ""])
        report.append("| 排名 | 模型 | MAE | RMSE | R2 | n |")
        report.append("|---:|---|---:|---:|---:|---:|")
        for rank, row in enumerate(group.sort_values("MAE").itertuples(index=False), start=1):
            report.append(
                f"| {rank} | {row.model_name} / {row.grouping} | "
                f"{format_value(row.MAE, row.MAE_sd)} | "
                f"{format_value(row.RMSE, row.RMSE_sd)} | "
                f"{format_value(row.R2, row.R2_sd)} | {int(row.n)} |"
            )
        report.append("")

    text = "\n".join(report)
    (OUTPUT_DIR / "52_base_leave_one_te_out_summary.md").write_text(
        text + "\n", encoding="utf-8"
    )
    print(text)


if __name__ == "__main__":
    main()
