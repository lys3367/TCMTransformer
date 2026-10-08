#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native-unit summary for all 38 diffusion MRI metrics in the clean-49 LOSO results.

Project:
    <project-root>

Expected prediction files:
    outputs/table1_clean49/itransformer_predictions.npz
    outputs/table1_clean49/timesnet_predictions.npz
    outputs/table1_clean49/mole_predictions.npz
    outputs/table1_clean49/dlinear_predictions.npz

Each NPZ is expected to contain:
    subjects      : (49,)
    target_te     : (2,) -> [125, 135]
    true_native   : (49, 2, 11248)
    pred_native   : (49, 2, 11248)
    models        : (11248,)
    metrics       : (11248,)
    roi_ids       : (11248,)
    roi_labels    : (11248,)

The script groups variables by (diffusion model, metric), yielding 38 groups.
For each group it reports:
    - pooled native-unit MAE, RMSE, Bias, R2
    - observed-value mean/SD/min/max
    - TE125- and TE135-specific native-unit metrics
    - per-subject native-unit metrics
    - mean/SD/median of subject-level metrics across 49 held-out participants

No retraining is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path.cwd()
DEFAULT_INPUT_DIR = PROJECT / "outputs" / "table1_clean49"
DEFAULT_OUTPUT_DIR = PROJECT / "outputs" / "native_unit_all38"

DEFAULT_MODELS = ("itransformer",)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Native-unit pooled metrics after removing non-finite pairs."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[keep]
    y_pred = y_pred[keep]

    if len(y_true) == 0:
        return {
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Bias": np.nan,
            "R2": np.nan,
            "TrueMean": np.nan,
            "TrueSD": np.nan,
            "TrueMin": np.nan,
            "TrueMax": np.nan,
        }

    error = y_pred - y_true
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    return {
        "N": int(len(y_true)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "Bias": float(np.mean(error)),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "TrueMean": float(np.mean(y_true)),
        "TrueSD": float(np.std(y_true, ddof=1)) if len(y_true) > 1 else np.nan,
        "TrueMin": float(np.min(y_true)),
        "TrueMax": float(np.max(y_true)),
    }


def load_prediction_file(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    z = np.load(path, allow_pickle=True)
    required = [
        "subjects",
        "target_te",
        "true_native",
        "pred_native",
        "models",
        "metrics",
        "roi_ids",
        "roi_labels",
    ]
    missing = [key for key in required if key not in z.files]
    if missing:
        raise KeyError(
            f"{path} is missing required arrays: {missing}\n"
            f"Available keys: {z.files}"
        )

    data = {key: z[key] for key in z.files}

    true_native = np.asarray(data["true_native"])
    pred_native = np.asarray(data["pred_native"])

    if true_native.shape != pred_native.shape:
        raise ValueError(
            f"true_native/pred_native shape mismatch: "
            f"{true_native.shape} vs {pred_native.shape}"
        )
    if true_native.ndim != 3:
        raise ValueError(
            f"Expected native arrays shaped (subject, target_TE, variable), "
            f"got {true_native.shape}"
        )
    if true_native.shape[0] != 49:
        raise ValueError(f"Expected 49 clean participants, got {true_native.shape[0]}")
    if true_native.shape[1] != 2:
        raise ValueError(f"Expected 2 target TEs, got {true_native.shape[1]}")
    if true_native.shape[2] != 11248:
        raise ValueError(f"Expected 11248 variables, got {true_native.shape[2]}")

    target_te = np.asarray(data["target_te"]).astype(int)
    if list(target_te) != [125, 135]:
        raise ValueError(f"Expected target TEs [125, 135], got {target_te.tolist()}")

    return data


def build_metric_groups(data: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build one metadata row per variable and identify the 38 model-metric groups."""
    n_variables = data["true_native"].shape[2]

    meta = pd.DataFrame(
        {
            "variable_index": np.arange(n_variables, dtype=int),
            "diffusion_model": np.asarray(data["models"]).astype(str),
            "metric": np.asarray(data["metrics"]).astype(str),
            "roi_id": np.asarray(data["roi_ids"]).astype(str),
            "roi_label": np.asarray(data["roi_labels"]).astype(str),
        }
    )

    if len(meta) != 11248:
        raise ValueError(f"Expected 11248 metadata rows, got {len(meta)}")

    group_sizes = (
        meta.groupby(["diffusion_model", "metric"], sort=False)
        .size()
        .rename("n_variables")
        .reset_index()
    )

    if len(group_sizes) != 38:
        raise ValueError(
            f"Expected exactly 38 Model-Metric groups, got {len(group_sizes)}.\n"
            f"Groups found:\n{group_sizes.to_string(index=False)}"
        )

    # In the current 38 x 296 construction, each metric should contribute 296 ROIs.
    bad = group_sizes[group_sizes["n_variables"] != 296]
    if not bad.empty:
        raise ValueError(
            "Expected every Model-Metric group to contain 296 ROI variables, "
            f"but found:\n{bad.to_string(index=False)}"
        )

    return meta


def analyze_one_model(
    model_name: str,
    prediction_file: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_prediction_file(prediction_file)
    meta = build_metric_groups(data)

    subjects = np.asarray(data["subjects"]).astype(str)
    target_te = np.asarray(data["target_te"]).astype(int)
    true_native = np.asarray(data["true_native"], dtype=np.float64)
    pred_native = np.asarray(data["pred_native"], dtype=np.float64)

    pooled_rows = []
    target_rows = []
    subject_rows = []

    groups = meta.groupby(["diffusion_model", "metric"], sort=False)

    for (diffusion_model, metric), group in groups:
        idx = group["variable_index"].to_numpy(dtype=int)

        # ----------------------------
        # 1) Pooled summary
        # 49 subjects x 2 target TEs x 296 ROIs = 29,008 pairs / metric
        # ----------------------------
        pooled = regression_metrics(
            true_native[:, :, idx],
            pred_native[:, :, idx],
        )
        pooled_rows.append(
            {
                "forecast_model": model_name,
                "diffusion_model": diffusion_model,
                "metric": metric,
                "n_rois": len(idx),
                **pooled,
            }
        )

        # ----------------------------
        # 2) Target-TE-specific summary
        # 49 subjects x 296 ROIs = 14,504 pairs / metric / TE
        # ----------------------------
        for te_pos, te in enumerate(target_te):
            res = regression_metrics(
                true_native[:, te_pos, idx],
                pred_native[:, te_pos, idx],
            )
            target_rows.append(
                {
                    "forecast_model": model_name,
                    "diffusion_model": diffusion_model,
                    "metric": metric,
                    "target_te": int(te),
                    "n_rois": len(idx),
                    **res,
                }
            )

        # ----------------------------
        # 3) Subject-level summary
        # each subject: 2 target TEs x 296 ROIs = 592 pairs / metric
        # ----------------------------
        for subject_pos, subject in enumerate(subjects):
            res = regression_metrics(
                true_native[subject_pos, :, idx],
                pred_native[subject_pos, :, idx],
            )
            subject_rows.append(
                {
                    "forecast_model": model_name,
                    "subject": subject,
                    "diffusion_model": diffusion_model,
                    "metric": metric,
                    "n_rois": len(idx),
                    **res,
                }
            )

    pooled_df = pd.DataFrame(pooled_rows)
    target_df = pd.DataFrame(target_rows)
    subject_df = pd.DataFrame(subject_rows)

    # ----------------------------
    # 4) Summarize subject-level metrics across 49 held-out participants
    # This is useful because participants, rather than the 29,008 pooled points,
    # are the independent validation units.
    # ----------------------------
    summary_rows = []
    metric_cols = ["MAE", "RMSE", "Bias", "R2"]

    for (forecast_model, diffusion_model, metric), g in subject_df.groupby(
        ["forecast_model", "diffusion_model", "metric"], sort=False
    ):
        row = {
            "forecast_model": forecast_model,
            "diffusion_model": diffusion_model,
            "metric": metric,
            "n_subjects": int(g["subject"].nunique()),
        }
        for col in metric_cols:
            vals = g[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{col}_sd"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            )
            row[f"{col}_median"] = float(np.median(vals)) if len(vals) else np.nan
            row[f"{col}_q25"] = float(np.quantile(vals, 0.25)) if len(vals) else np.nan
            row[f"{col}_q75"] = float(np.quantile(vals, 0.75)) if len(vals) else np.nan
        summary_rows.append(row)

    subject_summary_df = pd.DataFrame(summary_rows)

    # Sort in the same order the groups occur in the model matrix.
    order = (
        meta[["diffusion_model", "metric"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(metric_order=lambda d: np.arange(len(d)))
    )

    def add_order(df: pd.DataFrame) -> pd.DataFrame:
        out = df.merge(order, on=["diffusion_model", "metric"], how="left")
        sort_cols = ["metric_order"]
        if "target_te" in out.columns:
            sort_cols.append("target_te")
        if "subject" in out.columns:
            sort_cols.append("subject")
        out = out.sort_values(sort_cols).drop(columns="metric_order").reset_index(drop=True)
        return out

    pooled_df = add_order(pooled_df)
    target_df = add_order(target_df)
    subject_df = add_order(subject_df)
    subject_summary_df = add_order(subject_summary_df)

    model_out = output_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    pooled_df.to_csv(
        model_out / "native_unit_all38_pooled.csv",
        index=False,
        encoding="utf-8-sig",
    )
    target_df.to_csv(
        model_out / "native_unit_all38_by_target_te.csv",
        index=False,
        encoding="utf-8-sig",
    )
    subject_df.to_csv(
        model_out / "native_unit_all38_by_subject.csv",
        index=False,
        encoding="utf-8-sig",
    )
    subject_summary_df.to_csv(
        model_out / "native_unit_all38_subject_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return pooled_df, target_df, subject_df, subject_summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute native-unit summaries for all 38 Model-Metric groups."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=["itransformer", "timesnet", "mole", "dlinear"],
        help="Forecasting models to summarize. Default: itransformer",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing <model>_predictions.npz",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pooled = []
    all_target = []
    all_subject = []
    all_subject_summary = []

    for model_name in args.models:
        prediction_file = input_dir / f"{model_name}_predictions.npz"
        print(f"\n===== {model_name} =====")
        print(f"Input : {prediction_file}")

        pooled_df, target_df, subject_df, subject_summary_df = analyze_one_model(
            model_name=model_name,
            prediction_file=prediction_file,
            output_dir=output_dir,
        )

        all_pooled.append(pooled_df)
        all_target.append(target_df)
        all_subject.append(subject_df)
        all_subject_summary.append(subject_summary_df)

        print(
            f"Done: {model_name} | "
            f"{len(pooled_df)} metrics | "
            f"N pooled per metric = {pooled_df['N'].unique().tolist()}"
        )

    # Combined tables when more than one forecasting model is requested.
    pd.concat(all_pooled, ignore_index=True).to_csv(
        output_dir / "native_unit_all38_pooled_ALL_MODELS.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_target, ignore_index=True).to_csv(
        output_dir / "native_unit_all38_by_target_te_ALL_MODELS.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_subject, ignore_index=True).to_csv(
        output_dir / "native_unit_all38_by_subject_ALL_MODELS.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_subject_summary, ignore_index=True).to_csv(
        output_dir / "native_unit_all38_subject_summary_ALL_MODELS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nAll outputs saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()
