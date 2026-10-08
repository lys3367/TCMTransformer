from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from importlib import import_module


common = import_module("00_common")


REQUIRED_COLUMNS = [
    "Subject",
    "TE",
    "Model",
    "Metric",
    "ROI_ID",
    "ROI_Name",
    "Mean",
    "Std",
]


def make_subject_split(
    subjects: list[str], seed: int, train_ratio: float, val_ratio: float
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(subjects, dtype=object)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    labels = ["train"] * n_train + ["val"] * n_val
    labels += ["test"] * (n_total - len(labels))
    return pd.DataFrame({"subject_id": shuffled, "split": labels})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the BN/JHU ROI table and build a complete-7TE NPZ matrix."
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = common.load_config(args.config)
    raw_path = common.resolve_path(config, "raw_csv")
    matrix_path = common.resolve_path(config, "matrix_file")
    split_path = common.resolve_path(config, "split_file")
    metadata_path = common.resolve_path(config, "metadata_file")
    output_dir = common.resolve_path(config, "output_dir")
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["ROI_ID"] = df["ROI_ID"].astype(str)
    df["ROI_Name"] = df["ROI_Name"].astype(str)
    if "roi_mapping_csv" in config:
        roi_map_path = common.resolve_path(config, "roi_mapping_csv")
        roi_map = pd.read_csv(roi_map_path, dtype=str)
        required_map_columns = {"ROI_ID", "roi_name"}
        missing_map_columns = sorted(required_map_columns - set(roi_map.columns))
        if missing_map_columns:
            raise ValueError(f"ROI mapping missing columns: {missing_map_columns}")
        roi_map = roi_map[["ROI_ID", "roi_name"]].rename(
            columns={"roi_name": "ROI_Name"}
        )
    else:
        roi_map = df[["ROI_ID", "ROI_Name"]].drop_duplicates()
    roi_map = roi_map.sort_values("ROI_ID").reset_index(drop=True)
    if roi_map["ROI_ID"].duplicated().any():
        duplicated = roi_map.loc[roi_map["ROI_ID"].duplicated(), "ROI_ID"].tolist()
        raise ValueError(f"ROI_ID maps to multiple names: {duplicated[:10]}")
    roi_lookup = dict(zip(roi_map["ROI_ID"], roi_map["ROI_Name"]))

    csv_roi_ids = set(df["ROI_ID"].dropna().astype(str))
    map_roi_ids = set(roi_lookup)
    if csv_roi_ids != map_roi_ids:
        raise ValueError(
            "CSV and ROI mapping do not contain the same ROI IDs. "
            f"Only in CSV={sorted(csv_roi_ids-map_roi_ids)[:20]}, "
            f"only in map={sorted(map_roi_ids-csv_roi_ids)[:20]}"
        )

    expected_te = sorted(int(x) for x in config["expected_te"])
    observed_te = sorted(df["TE"].dropna().astype(int).unique().tolist())
    if observed_te != expected_te:
        raise ValueError(f"Unexpected TE values: observed={observed_te}, expected={expected_te}")

    duplicate_count = int(
        df.duplicated(["Subject", "TE", "Model", "Metric", "ROI_ID"]).sum()
    )
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate measurement keys.")

    te_count = df.groupby("Subject")["TE"].nunique()
    complete_subjects = sorted(te_count[te_count == len(expected_te)].index.astype(str))
    complete = df[df["Subject"].astype(str).isin(complete_subjects)].copy()
    complete["feature"] = (
        complete["Model"].astype(str) + "__" + complete["Metric"].astype(str)
    )

    expected_rows_per_feature = len(complete_subjects) * len(expected_te) * len(roi_lookup)
    feature_audit = (
        complete.groupby(["Model", "Metric", "feature"], sort=True)["Mean"]
        .agg(n_rows="size", n_missing=lambda values: int(values.isna().sum()))
        .reset_index()
    )
    feature_audit["n_expected"] = expected_rows_per_feature
    feature_audit["n_absent"] = feature_audit["n_expected"] - feature_audit["n_rows"]
    feature_audit["missing_rate"] = (
        (feature_audit["n_missing"] + feature_audit["n_absent"].clip(lower=0))
        / feature_audit["n_expected"]
    )
    excluded = feature_audit[
        (feature_audit["n_missing"] > 0) | (feature_audit["n_rows"] != feature_audit["n_expected"])
    ].copy()
    included_features = sorted(
        feature_audit.loc[
            (feature_audit["n_missing"] == 0)
            & (feature_audit["n_rows"] == feature_audit["n_expected"]),
            "feature",
        ].tolist()
    )

    model_data = complete[complete["feature"].isin(included_features)].copy()
    model_data["roi_id"] = model_data["ROI_ID"].astype(str)
    model_data["roi_label"] = model_data["roi_id"].map(roi_lookup)
    model_data["variable"] = (
        model_data["feature"] + "__" + model_data["roi_id"].astype(str)
    )

    metadata = (
        model_data[
            ["Model", "Metric", "feature", "roi_id", "roi_label", "variable"]
        ]
        .drop_duplicates()
        .sort_values(["feature", "roi_id"])
        .reset_index(drop=True)
        .rename(columns={"Model": "model", "Metric": "metric"})
    )
    metadata.insert(0, "variable_index", np.arange(len(metadata), dtype=int))

    subjects = np.asarray(complete_subjects, dtype=object)
    variables = metadata["variable"].to_numpy(dtype=object)
    full_index = pd.MultiIndex.from_product(
        [subjects, expected_te, variables], names=["Subject", "TE", "variable"]
    )
    series = (
        model_data.set_index(["Subject", "TE", "variable"])["Mean"]
        .reindex(full_index)
    )
    x_raw = series.to_numpy(dtype=np.float32).reshape(
        len(subjects), len(expected_te), len(variables)
    )
    if np.isnan(x_raw).any():
        missing = int(np.isnan(x_raw).sum())
        raise ValueError(f"Prepared matrix still contains {missing} NaN values.")

    split = make_subject_split(
        complete_subjects,
        int(config["random_seed"]),
        float(config["train_ratio"]),
        float(config["val_ratio"]),
    )
    split.to_csv(split_path, index=False, encoding="utf-8-sig")
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    excluded.to_csv(
        output_dir / "01_excluded_features.csv", index=False, encoding="utf-8-sig"
    )

    np.savez_compressed(
        matrix_path,
        x_raw=x_raw,
        subjects=subjects,
        te_values=np.asarray(expected_te, dtype=np.int16),
        variables=variables,
        models=metadata["model"].to_numpy(dtype=object),
        metrics=metadata["metric"].to_numpy(dtype=object),
        features=metadata["feature"].to_numpy(dtype=object),
        roi_ids=metadata["roi_id"].to_numpy(dtype=object),
        roi_labels=metadata["roi_label"].to_numpy(dtype=object),
    )

    coverage = te_count.value_counts().sort_index()
    report = [
        "# 01 数据审计与建模矩阵报告",
        "",
        "## 数据概况",
        "",
        f"- 原始记录数：{len(df):,}",
        f"- 受试者数：{df['Subject'].nunique()}",
        f"- 完整 7TE 受试者数：{len(subjects)}",
        f"- TE：{expected_te}",
        f"- 扩散模型数：{df['Model'].nunique()}",
        f"- Model+Metric 指标数：{feature_audit.shape[0]}",
        f"- BN/JHU ROI 数：{len(roi_map)}",
        f"- 原始候选变量数：{feature_audit.shape[0] * len(roi_map)}",
        "",
        "## 完整性规则",
        "",
        "第一阶段只使用具有全部 7 个 TE 的受试者。只要某个 Model+Metric 指标在任一受试者、TE 或 ROI 中出现 Mean 空值，就排除该指标在全部 BN/JHU ROI 中的变量。",
        "",
        f"- 被排除指标数：{len(excluded)}",
        f"- 最终保留指标数：{len(included_features)}",
        f"- 最终变量数：{len(variables)}",
        f"- 矩阵形状：{tuple(x_raw.shape)}，依次为 subject × TE × variable",
        "",
        "## 受试者 TE 覆盖",
        "",
    ]
    report.extend([f"- {int(te)} 个 TE：{int(count)} 名" for te, count in coverage.items()])
    report.extend(
        [
            "",
            "## 被排除指标",
            "",
        ]
    )
    if excluded.empty:
        report.append("- 无")
    else:
        for row in excluded.itertuples(index=False):
            report.append(
                f"- `{row.feature}`：缺失 {row.n_missing}/{row.n_rows} "
                f"({row.missing_rate:.2%})"
            )
    report.extend(
        [
            "",
            "## 关键输出",
            "",
            f"- `{matrix_path.relative_to(common.ROOT)}`：模型输入矩阵。",
            f"- `{split_path.relative_to(common.ROOT)}`：固定 subject-level 划分。",
            f"- `{metadata_path.relative_to(common.ROOT)}`：变量与 ROI 标签表。",
            f"- `{(output_dir / '01_excluded_features.csv').relative_to(common.ROOT)}`：少量被排除指标。",
            "",
            "标准化不写死在 NPZ 中。每次固定划分或交叉验证都只用当前训练受试者重新计算每个变量的均值和标准差，避免数据泄漏。",
        ]
    )
    report_path = output_dir / "01_data_audit_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
