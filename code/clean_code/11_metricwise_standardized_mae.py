#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_metricwise_standardized_mae.py

Purpose
-------
Compute the 38 Model-Metric groups' standardized MAE from the existing
clean-49 LOSO prediction files.

This is a pure post-hoc analysis:
- NO retraining
- Uses saved true_z / pred_z
- Groups by diffusion_model + metric
- Each metric contains 296 ROI variables
- Each subject-level metric MAE pools 2 target TEs x 296 ROIs = 592 values
- Summary is reported across the 49 held-out subjects

Expected input files
--------------------
/media/UG1/lys/dipy/BN_JHU296/outputs/table1_clean49/
    itransformer_predictions.npz
    timesnet_predictions.npz
    mole_predictions.npz
    dlinear_predictions.npz

Expected NPZ keys
-----------------
subjects, target_te, true_z, pred_z, models, metrics, roi_ids, roi_labels

Outputs
-------
outputs/metricwise_standardized_mae/
    metricwise_standardized_mae_by_subject.csv
    metricwise_standardized_mae_summary.csv
    metricwise_standardized_mae_winners.csv
    metricwise_standardized_mae_audit.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT = Path("/media/UG1/lys/dipy/BN_JHU296")
DEFAULT_INPUT_DIR = PROJECT / "outputs" / "table1_clean49"
DEFAULT_OUTPUT_DIR = PROJECT / "outputs" / "metricwise_standardized_mae"

FORECAST_MODELS = ("itransformer", "timesnet", "mole", "dlinear")

EXPECTED_N_SUBJECTS = 49
EXPECTED_N_VARIABLES = 11248
EXPECTED_N_METRICS = 38
EXPECTED_N_ROIS_PER_METRIC = 296
EXPECTED_TARGET_TE = [125, 135]


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    z = np.load(path, allow_pickle=True)
    required = [
        "subjects", "target_te", "true_z", "pred_z",
        "models", "metrics", "roi_ids", "roi_labels"
    ]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(
            f"{path} is missing keys: {missing}\nAvailable keys: {z.files}"
        )

    data = {k: z[k] for k in z.files}

    true_z = np.asarray(data["true_z"])
    pred_z = np.asarray(data["pred_z"])

    if true_z.shape != pred_z.shape:
        raise ValueError(
            f"true_z/pred_z shape mismatch: {true_z.shape} vs {pred_z.shape}"
        )

    if true_z.shape != (EXPECTED_N_SUBJECTS, 2, EXPECTED_N_VARIABLES):
        raise ValueError(
            f"Expected shape ({EXPECTED_N_SUBJECTS}, 2, {EXPECTED_N_VARIABLES}), "
            f"got {true_z.shape}"
        )

    target_te = np.asarray(data["target_te"]).astype(int).tolist()
    if target_te != EXPECTED_TARGET_TE:
        raise ValueError(
            f"Expected target TEs {EXPECTED_TARGET_TE}, got {target_te}"
        )

    return data


def build_metadata(data: dict[str, np.ndarray]) -> pd.DataFrame:
    meta = pd.DataFrame({
        "variable_index": np.arange(EXPECTED_N_VARIABLES),
        "diffusion_model": np.asarray(data["models"]).astype(str),
        "metric": np.asarray(data["metrics"]).astype(str),
        "roi_id": np.asarray(data["roi_ids"]).astype(str),
        "roi_label": np.asarray(data["roi_labels"]).astype(str),
    })

    groups = (
        meta.groupby(["diffusion_model", "metric"], sort=False)
        .size()
        .reset_index(name="n_variables")
    )

    if len(groups) != EXPECTED_N_METRICS:
        raise ValueError(
            f"Expected {EXPECTED_N_METRICS} Model-Metric groups, "
            f"found {len(groups)}.\n{groups.to_string(index=False)}"
        )

    bad = groups[groups["n_variables"] != EXPECTED_N_ROIS_PER_METRIC]
    if not bad.empty:
        raise ValueError(
            "Not every metric contains 296 ROI variables:\n"
            + bad.to_string(index=False)
        )

    return meta


def verify_same_metadata(reference: dict, other: dict, model_name: str) -> None:
    for key in ["models", "metrics", "roi_ids", "roi_labels", "target_te"]:
        a = np.asarray(reference[key]).astype(str)
        b = np.asarray(other[key]).astype(str)
        if a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(
                f"Metadata mismatch for forecasting model {model_name}: {key}"
            )


def reorder_subjects(reference_subjects: np.ndarray, data: dict) -> dict:
    ref = np.asarray(reference_subjects).astype(str)
    cur = np.asarray(data["subjects"]).astype(str)

    if set(ref) != set(cur):
        raise ValueError("Subject sets differ across forecasting models.")

    lookup = {s: i for i, s in enumerate(cur)}
    order = np.asarray([lookup[s] for s in ref], dtype=int)

    out = dict(data)
    out["subjects"] = cur[order]
    out["true_z"] = np.asarray(data["true_z"])[order]
    out["pred_z"] = np.asarray(data["pred_z"])[order]
    return out


def analyze_one_model(
    forecast_model: str,
    data: dict[str, np.ndarray],
    meta: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    subjects = np.asarray(data["subjects"]).astype(str)
    true_z = np.asarray(data["true_z"], dtype=np.float64)
    pred_z = np.asarray(data["pred_z"], dtype=np.float64)

    subject_rows = []
    summary_rows = []

    for (diffusion_model, metric), g in meta.groupby(
        ["diffusion_model", "metric"], sort=False
    ):
        idx = g["variable_index"].to_numpy(dtype=int)

        abs_err = np.abs(pred_z[:, :, idx] - true_z[:, :, idx])

        # Each subject: average over 2 target TEs x 296 ROIs
        subject_mae = abs_err.mean(axis=(1, 2))

        # Target-specific subject-level MAE
        subject_mae_125 = abs_err[:, 0, :].mean(axis=1)
        subject_mae_135 = abs_err[:, 1, :].mean(axis=1)

        for i, subject in enumerate(subjects):
            subject_rows.append({
                "forecast_model": forecast_model,
                "subject": subject,
                "diffusion_model": diffusion_model,
                "metric": metric,
                "standardized_MAE": float(subject_mae[i]),
                "standardized_MAE_TE125": float(subject_mae_125[i]),
                "standardized_MAE_TE135": float(subject_mae_135[i]),
            })

        summary_rows.append({
            "forecast_model": forecast_model,
            "diffusion_model": diffusion_model,
            "metric": metric,
            "n_subjects": len(subjects),
            "n_rois": len(idx),
            "standardized_MAE_mean": float(subject_mae.mean()),
            "standardized_MAE_sd": float(subject_mae.std(ddof=1)),
            "standardized_MAE_median": float(np.median(subject_mae)),
            "standardized_MAE_q25": float(np.quantile(subject_mae, 0.25)),
            "standardized_MAE_q75": float(np.quantile(subject_mae, 0.75)),
            "standardized_MAE_TE125_mean": float(subject_mae_125.mean()),
            "standardized_MAE_TE125_sd": float(subject_mae_125.std(ddof=1)),
            "standardized_MAE_TE135_mean": float(subject_mae_135.mean()),
            "standardized_MAE_TE135_sd": float(subject_mae_135.std(ddof=1)),
        })

    return pd.DataFrame(subject_rows), pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR)
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR)
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(FORECAST_MODELS),
        choices=list(FORECAST_MODELS)
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for model in args.models:
        path = input_dir / f"{model}_predictions.npz"
        print(f"Loading {model}: {path}", flush=True)
        loaded[model] = load_prediction(path)

    ref_model = args.models[0]
    ref = loaded[ref_model]
    ref_subjects = np.asarray(ref["subjects"]).astype(str)
    meta = build_metadata(ref)

    for model in args.models[1:]:
        verify_same_metadata(ref, loaded[model], model)
        loaded[model] = reorder_subjects(ref_subjects, loaded[model])

    subject_all = []
    summary_all = []

    for model in args.models:
        subj, summ = analyze_one_model(model, loaded[model], meta)
        subject_all.append(subj)
        summary_all.append(summ)

    subject_df = pd.concat(subject_all, ignore_index=True)
    summary_df = pd.concat(summary_all, ignore_index=True)

    subject_df.to_csv(
        output_dir / "metricwise_standardized_mae_by_subject.csv",
        index=False,
        encoding="utf-8-sig"
    )
    summary_df.to_csv(
        output_dir / "metricwise_standardized_mae_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Winner among the requested forecasting models for each of the 38 metrics.
    winner_rows = []
    for (dm, metric), g in summary_df.groupby(
        ["diffusion_model", "metric"], sort=False
    ):
        g = g.sort_values("standardized_MAE_mean").reset_index(drop=True)

        row = {
            "diffusion_model": dm,
            "metric": metric,
            "best_forecast_model": g.loc[0, "forecast_model"],
            "best_standardized_MAE_mean": float(
                g.loc[0, "standardized_MAE_mean"]
            ),
        }

        for model in args.models:
            gm = g[g["forecast_model"] == model]
            if not gm.empty:
                row[f"{model}_MAE_mean"] = float(
                    gm["standardized_MAE_mean"].iloc[0]
                )
                row[f"{model}_MAE_sd"] = float(
                    gm["standardized_MAE_sd"].iloc[0]
                )

        winner_rows.append(row)

    winner_df = pd.DataFrame(winner_rows)

    if len(winner_df) != EXPECTED_N_METRICS:
        raise ValueError(
            f"Expected {EXPECTED_N_METRICS} winner rows, got {len(winner_df)}"
        )

    winner_df.to_csv(
        output_dir / "metricwise_standardized_mae_winners.csv",
        index=False,
        encoding="utf-8-sig"
    )

    counts = winner_df["best_forecast_model"].value_counts()

    audit_lines = [
        "Metric-wise standardized MAE audit",
        "==================================",
        f"Forecasting models: {args.models}",
        f"Subjects: {EXPECTED_N_SUBJECTS}",
        f"Target TEs: {EXPECTED_TARGET_TE}",
        f"Variables: {EXPECTED_N_VARIABLES}",
        f"Model-Metric groups: {EXPECTED_N_METRICS}",
        f"ROIs per metric: {EXPECTED_N_ROIS_PER_METRIC}",
        "",
        "Winner counts:",
        counts.to_string(),
        "",
    ]
    (output_dir / "metricwise_standardized_mae_audit.txt").write_text(
        "\n".join(audit_lines),
        encoding="utf-8"
    )

    print("\n=== Winner counts across 38 metrics ===")
    print(counts.to_string())

    print("\nSaved to:")
    print(output_dir)


if __name__ == "__main__":
    main()
