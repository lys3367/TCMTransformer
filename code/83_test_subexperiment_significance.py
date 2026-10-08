from __future__ import annotations

import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "stat_tests"
MODELS = ["TimesNet", "MoLE", "DLinear"]


def bh_fdr(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def compare_task(task: str, table: pd.DataFrame, complete_family: bool = True) -> list[dict]:
    rows = []
    for model in MODELS:
        if model not in table.columns:
            continue
        pair = table[["iTransformer", model]].dropna()
        if pair.empty:
            continue
        difference = pair[model] - pair["iTransformer"]
        statistic, p_value = wilcoxon(
            pair["iTransformer"], pair[model], alternative="two-sided", zero_method="wilcox"
        )
        rows.append(
            {
                "task": task,
                "comparator": model,
                "n_pairs": len(pair),
                "mean_mae_itransformer": pair["iTransformer"].mean(),
                "mean_mae_comparator": pair[model].mean(),
                "mean_difference_comparator_minus_itransformer": difference.mean(),
                "median_difference_comparator_minus_itransformer": difference.median(),
                "itransformer_win_rate": (pair["iTransformer"] < pair[model]).mean(),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "fdr_family_complete": complete_family,
            }
        )
    if rows:
        q_values = bh_fdr([row["p_value"] for row in rows])
        for row, q_value in zip(rows, q_values):
            row["q_fdr"] = q_value
            row["itransformer_significantly_better"] = bool(
                q_value < 0.05
                and row["mean_difference_comparator_minus_itransformer"] > 0
            )
    return rows


def main_loso() -> pd.DataFrame:
    fold_map = pd.read_csv(ROOT / "outputs" / "18_loso_fold_subject_map.csv")
    tables = []
    for source, label in [
        ("04_loso_itransformer_full.csv", "iTransformer"),
        ("04_loso_timesnet_full.csv", "TimesNet"),
        ("04_loso_dlinear_full.csv", "DLinear"),
    ]:
        data = pd.read_csv(ROOT / "outputs" / source).merge(
            fold_map[["fold", "subject"]], on="fold", validate="one_to_one"
        )
        data = data[~data["subject"].str.startswith(("43_", "48_"))]
        tables.append(data[["subject", "MAE"]].rename(columns={"MAE": label}).set_index("subject"))
    for source, label in [("80_extra_loso_mole_full.csv", "MoLE")]:
        data = pd.read_csv(ROOT / "outputs" / source)
        tables.append(
            data[["test_subject", "MAE"]]
            .rename(columns={"test_subject": "subject", "MAE": label})
            .set_index("subject")
        )
    return pd.concat(tables, axis=1)


def lesion_loso() -> pd.DataFrame:
    fold_map = pd.read_csv(ROOT / "outputs_lesion" / "18_loso_fold_subject_map.csv")
    tables = []
    for source, label in [
        ("04_loso_itransformer_full.csv", "iTransformer"),
        ("04_loso_timesnet_full.csv", "TimesNet"),
        ("04_loso_dlinear_full.csv", "DLinear"),
    ]:
        data = pd.read_csv(ROOT / "outputs_lesion" / source).merge(
            fold_map[["fold", "subject"]], on="fold", validate="one_to_one"
        )
        data = data[~data["subject"].str.startswith(("43_", "48_"))]
        tables.append(data[["subject", "MAE"]].rename(columns={"MAE": label}).set_index("subject"))
    for source, label in [("73_extra_loso_mole_full.csv", "MoLE")]:
        path = ROOT / "outputs_lesion" / source
        if path.exists():
            data = pd.read_csv(path)
            key = "test_subject" if "test_subject" in data.columns else "fold"
            if key == "fold":
                config = json.loads(
                    (ROOT / "config_lesion.json").read_text(encoding="utf-8-sig")
                )
                matrix = np.load(
                    ROOT / config["matrix_file"], allow_pickle=True
                )
                subjects = matrix["subjects"].astype(str)
                keep = ~np.char.startswith(subjects, "43_") & ~np.char.startswith(
                    subjects, "48_"
                )
                subjects = subjects[keep]
                order = np.random.default_rng(int(config["random_seed"])).permutation(
                    len(subjects)
                )
                extra_fold_map = pd.DataFrame(
                    {"fold": np.arange(1, len(subjects) + 1), "subject": subjects[order]}
                )
                data = data.merge(extra_fold_map, on="fold", validate="one_to_one")
            else:
                data = data.rename(columns={key: "subject"})
            tables.append(data[["subject", "MAE"]].rename(columns={"MAE": label}).set_index("subject"))
    return pd.concat(tables, axis=1)


def leave_one_te_loso() -> pd.DataFrame:
    base = pd.read_csv(ROOT / "outputs" / "52_base_leave_one_te_out_from_log_parsed.csv")
    base = base[base["mode"].eq("loso") & base["model"].isin(["itransformer", "timesnet", "dlinear"])]
    base["model"] = base["model"].map(
        {"itransformer": "iTransformer", "timesnet": "TimesNet", "dlinear": "DLinear"}
    )
    base = base.groupby(["fold", "model"], as_index=False)["MAE"].mean()
    table = base.pivot(index="fold", columns="model", values="MAE")
    for source, label in [("74_extra_loso_leave_one_te_out_mole_full_exclude_43_48.csv", "MoLE")]:
        data = pd.read_csv(ROOT / "outputs" / source)
        table[label] = data.groupby("fold")["MAE"].mean()
    return table


def parse_itransformer_pretraining_log() -> pd.DataFrame:
    text = (ROOT / "logs" / "62_masked_te_pretrain_loso_20260711_013347.log").read_text(
        encoding="utf-8"
    )
    records = re.findall(r"loso fold=(\d+) MAE=([0-9.]+)", text)
    return pd.DataFrame(records, columns=["fold", "iTransformer"]).astype(
        {"fold": int, "iTransformer": float}
    ).set_index("fold")


def masked_pretraining_loso() -> pd.DataFrame:
    table = parse_itransformer_pretraining_log()
    for source, label in [
        ("81_masked_te_pretrain_base_timesnet_full_loso_exclude_43_48.csv", "TimesNet"),
        ("76_masked_te_pretrain_extra_mole_full_loso_exclude_43_48.csv", "MoLE"),
        ("81_masked_te_pretrain_base_dlinear_full_loso_exclude_43_48.csv", "DLinear"),
    ]:
        data = pd.read_csv(ROOT / "outputs" / source).set_index("fold")
        table[label] = data["MAE"]
    return table


def format_p(value: float) -> str:
    return f"{value:.3e}" if value < 0.001 else f"{value:.4f}"


def write_report(results: pd.DataFrame, missing_lesion: list[str]) -> None:
    lines = [
        "# 83 iTransformer full 跨子实验显著性检验",
        "",
        "主要统计单位为受试者。采用双侧配对 Wilcoxon 符号秩检验，并在每个子实验内对 iTransformer 与 TimesNet、MoLE 和 DLinear 的三项检验进行 Benjamini-Hochberg FDR 校正。leave-one-TE-out 先对每名受试者的 7 个目标 TE 求平均，再进行受试者级配对检验。",
        "",
    ]
    for task, group in results.groupby("task", sort=False):
        lines.extend(
            [
                f"## {task}",
                "",
                "| 比较模型 | 配对数 | iTransformer MAE | 比较模型 MAE | MAE 差值 | iTransformer 胜率 | p | FDR q | 显著更优 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in group.itertuples(index=False):
            conclusion = "是" if row.itransformer_significantly_better else "否"
            if not row.fdr_family_complete:
                conclusion += "（暂定）"
            lines.append(
                f"| {row.comparator} | {row.n_pairs} | {row.mean_mae_itransformer:.6f} | "
                f"{row.mean_mae_comparator:.6f} | {row.mean_difference_comparator_minus_itransformer:.6f} | "
                f"{row.itransformer_win_rate:.1%} | {format_p(row.p_value)} | {format_p(row.q_fdr)} | {conclusion} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 当前不能作正式显著性检验的结果",
            "",
            "- 固定划分只有一次测试划分，不以单个汇总 MAE 进行显著性检验。",
            "- ScienceDB 仅有 3 名外部受试者，统计功效不足；当前只作描述性外部可行性比较。",
        ]
    )
    if missing_lesion:
        lines.append(
            "- 病灶 LOSO 尚缺逐折文件："
            + "、".join(f"`{name}`" for name in missing_lesion)
            + "。因此病灶任务对这些模型的 FDR 结论需要补齐文件后更新。"
        )
    (ROOT / "outputs" / "83_subexperiment_significance_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    missing_lesion = [
        name
        for name in ["73_extra_loso_mole_full.csv"]
        if not (ROOT / "outputs_lesion" / name).exists()
    ]
    rows = []
    rows += compare_task("主预测 LOSO", main_loso())
    rows += compare_task("病灶 mask LOSO", lesion_loso(), complete_family=not missing_lesion)
    rows += compare_task("Leave-one-TE-out LOSO", leave_one_te_loso())
    rows += compare_task("不完整 TE masked-TE 预训练 LOSO", masked_pretraining_loso())
    results = pd.DataFrame(rows)
    output = OUTPUT_DIR / "83_itransformer_subexperiment_pairwise_mae.csv"
    results.to_csv(output, index=False, encoding="utf-8-sig")
    write_report(results, missing_lesion)
    print(results.to_string(index=False))
    print(f"\nSaved: {output}")
    print(f"Saved: {ROOT / 'outputs' / '83_subexperiment_significance_summary.md'}")


if __name__ == "__main__":
    main()
