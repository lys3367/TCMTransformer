from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd


common = importlib.import_module("00_common")


def load_external_matrix(csv_path: Path, metadata: pd.DataFrame, expected_te: list[int]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["ROI_ID"] = df["ROI_ID"].astype(str)
    df["feature"] = df["Model"].astype(str) + "__" + df["Metric"].astype(str)
    df["variable"] = df["feature"] + "__" + df["ROI_ID"]

    variables = metadata.sort_values("variable_index")["variable"].astype(str).to_numpy()
    subjects = np.asarray(sorted(df["Subject"].astype(str).unique()), dtype=object)
    full_index = pd.MultiIndex.from_product(
        [subjects, expected_te, variables], names=["Subject", "TE", "variable"]
    )
    series = df.set_index(["Subject", "TE", "variable"])["Mean"].reindex(full_index)
    if series.isna().any():
        missing = int(series.isna().sum())
        missing_vars = series[series.isna()].reset_index()["variable"].drop_duplicates().head(20).tolist()
        raise ValueError(
            f"External matrix contains {missing} missing values. "
            f"First missing variables: {missing_vars}"
        )
    external = series.to_numpy(dtype=np.float32).reshape(len(subjects), len(expected_te), len(variables))
    return external, subjects


def metric_row(model: str, variant: str, true: np.ndarray, pred: np.ndarray, n_train: int, n_val: int, n_test: int, details: dict) -> dict:
    result = common.regression_metrics(true, pred)
    row = {
        "model": model,
        "variant": variant,
        "task": "sciencedb_external_fixed_target",
        "n_train_internal": n_train,
        "n_val_internal": n_val,
        "n_external_test": n_test,
    }
    row.update({k: v for k, v in details.items() if isinstance(v, (int, float, str))})
    row.update(result)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on internal BN/JHU296 split and test on ScienceDB external data."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--external-csv",
        default="data/raw/ScienceDB_BN_JHU_metric_results_interpolated.csv",
    )
    parser.add_argument("--output-dir", default="outputs_sciencedb")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["itransformer_full", "modified_feature_te", "modified_roi_feature_te"],
        choices=[
            "itransformer_full",
            "patchtst_full",
            "timesnet_full",
            "dlinear_full",
            "modified_feature_te",
            "modified_roi_feature_te",
        ],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    config = common.load_config(args.config)
    matrix = common.load_matrix(config)
    internal_raw = matrix["x_raw"].astype(np.float32)
    internal_subjects = matrix["subjects"].astype(str)
    te_values = matrix["te_values"].astype(int)
    metadata = pd.read_csv(common.resolve_path(config, "metadata_file"))
    split_labels = common.load_split_labels(config, internal_subjects)
    repos_dir = common.resolve_path(config, "external_repos_dir")
    input_idx = common.te_indices(te_values, config["fixed_input_te"])
    target_idx = common.te_indices(te_values, config["fixed_target_te"])

    external_csv = Path(args.external_csv)
    if not external_csv.is_absolute():
        external_csv = common.ROOT / external_csv
    external_raw, external_subjects = load_external_matrix(
        external_csv, metadata, list(map(int, te_values))
    )

    raw = np.concatenate([internal_raw, external_raw], axis=0)
    train_idx = np.where(split_labels == "train")[0]
    val_idx = np.where(split_labels == "val")[0]
    external_idx = np.arange(len(internal_raw), len(internal_raw) + len(external_raw))

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = common.ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    base_models = {
        "itransformer_full": "itransformer",
        "patchtst_full": "patchtst",
        "timesnet_full": "timesnet",
        "dlinear_full": "dlinear",
    }
    selected_base_models = [name for name in args.models if name in base_models]
    if selected_base_models:
        neural = importlib.import_module("03_train_neural")
        for model_key in selected_base_models:
            model_name = base_models[model_key]
            settings = neural.TrainSettings(
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                seed=int(config["random_seed"]),
                device=args.device,
            )
            true, pred, details = neural.run_neural(
                model_name,
                "full",
                raw,
                metadata,
                train_idx,
                val_idx,
                external_idx,
                input_idx,
                target_idx,
                repos_dir,
                settings,
            )
            rows.append(
                metric_row(
                    model_name,
                    "full",
                    true,
                    pred,
                    len(train_idx),
                    len(val_idx),
                    len(external_idx),
                    {"val_RMSE": float(details["val_RMSE"].mean())},
                )
            )
            common.summarize_by_roi(true, pred, metadata).to_csv(
                output_dir / f"42_sciencedb_{model_name}_full_by_roi.csv",
                index=False,
                encoding="utf-8-sig",
            )

    modified_variants = [name.replace("modified_", "modified_", 1) for name in args.models if name.startswith("modified_")]
    if modified_variants:
        modified = importlib.import_module("13_train_modified_official_itransformer")
        model_file = Path(__file__).resolve().with_name("13_modified_official_iTransformer_model.py")
        modified.write_modified_source(model_file)
        for variant in modified_variants:
            settings = modified.TrainSettings(
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                seed=int(config["random_seed"]),
                device=args.device,
            )
            true, pred, details = modified.train_one(
                variant,
                raw,
                train_idx,
                val_idx,
                external_idx,
                input_idx,
                target_idx,
                te_values,
                metadata,
                model_file,
                settings,
            )
            rows.append(metric_row("modified_official_itransformer", variant, true, pred, len(train_idx), len(val_idx), len(external_idx), details))
            common.summarize_by_roi(true, pred, metadata).to_csv(
                output_dir / f"42_sciencedb_{variant}_by_roi.csv",
                index=False,
                encoding="utf-8-sig",
            )

    summary = pd.DataFrame(rows).sort_values("MAE")
    summary.to_csv(output_dir / "42_sciencedb_external_summary.csv", index=False, encoding="utf-8-sig")
    external_subject_table = pd.DataFrame({"external_subject": external_subjects})
    external_subject_table.to_csv(output_dir / "42_sciencedb_external_subjects.csv", index=False, encoding="utf-8-sig")

    lines = ["# 42 ScienceDB 外部测试结果", ""]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.model}` / `{row.variant}`：MAE={row.MAE:.6f}，RMSE={row.RMSE:.6f}，R2={row.R2:.6f}"
        )
    text = "\n".join(lines) + "\n"
    (output_dir / "42_sciencedb_external_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
