#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_wmti_gm_sensitivity.py

Purpose
-------
Evaluate whether the primary clean-49 model comparison changes after
excluding WMTI-derived variables from gray-matter ROIs.

This is an evaluation-only sensitivity analysis:
- NO retraining
- Uses existing true_z / pred_z
- Excludes WMTI x Brainnetome GM variables only
- Keeps WMTI variables in the 50 JHU WM ROIs
- Recomputes subject-level standardized MAE, RMSE, and R2
- Repeats paired Wilcoxon tests vs iTransformer with BH-FDR correction

Expected ROI-ID convention
--------------------------
G_<id> : Brainnetome gray-matter ROI
W_<id> : JHU white-matter ROI

Expected counts
---------------
296 total ROIs = 246 GM + 50 WM
4 WMTI metrics
Excluded: 4 x 246 = 984 variables
Retained: 11248 - 984 = 10264 variables

Expected input files
--------------------
outputs/table1_clean49/
    itransformer_predictions.npz
    timesnet_predictions.npz
    mole_predictions.npz
    dlinear_predictions.npz

Outputs
-------
outputs/wmti_gm_sensitivity/
    wmti_gm_excluded_variables.csv
    wmti_gm_sensitivity_by_subject.csv
    wmti_gm_sensitivity_summary.csv
    wmti_gm_pairwise_vs_itransformer.csv
    wmti_gm_sensitivity_audit.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT = Path("/media/UG1/lys/dipy/BN_JHU296")
DEFAULT_INPUT_DIR = PROJECT / "outputs" / "table1_clean49"
DEFAULT_OUTPUT_DIR = PROJECT / "outputs" / "wmti_gm_sensitivity"

FORECAST_MODELS = ("itransformer", "timesnet", "mole", "dlinear")

EXPECTED_N_SUBJECTS = 49
EXPECTED_N_VARIABLES = 11248
EXPECTED_TARGET_TE = [125, 135]
EXPECTED_N_GM_ROIS = 246
EXPECTED_N_WM_ROIS = 50
EXPECTED_N_WMTI_METRICS = 4
EXPECTED_N_EXCLUDED = 984
EXPECTED_N_RETAINED = 10264


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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
            "R2": np.nan,
        }

    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    return {
        "N": int(len(y_true)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)

    finite = np.isfinite(p)
    pv = p[finite]
    if len(pv) == 0:
        return out

    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    q_sorted = ranked * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    restored = np.empty_like(q_sorted)
    restored[order] = q_sorted
    out[finite] = restored
    return out


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
    return pd.DataFrame({
        "variable_index": np.arange(EXPECTED_N_VARIABLES),
        "diffusion_model": np.asarray(data["models"]).astype(str),
        "metric": np.asarray(data["metrics"]).astype(str),
        "roi_id": np.asarray(data["roi_ids"]).astype(str),
        "roi_label": np.asarray(data["roi_labels"]).astype(str),
    })


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


def make_wmti_gm_mask(meta: pd.DataFrame):
    unique_rois = meta[["roi_id", "roi_label"]].drop_duplicates()

    gm_rois = set(
        unique_rois.loc[
            unique_rois["roi_id"].str.startswith("G_", na=False),
            "roi_id"
        ].astype(str)
    )
    wm_rois = set(
        unique_rois.loc[
            unique_rois["roi_id"].str.startswith("W_", na=False),
            "roi_id"
        ].astype(str)
    )

    if len(gm_rois) != EXPECTED_N_GM_ROIS:
        raise ValueError(
            f"Detected {len(gm_rois)} GM ROIs; expected {EXPECTED_N_GM_ROIS}"
        )
    if len(wm_rois) != EXPECTED_N_WM_ROIS:
        raise ValueError(
            f"Detected {len(wm_rois)} WM ROIs; expected {EXPECTED_N_WM_ROIS}"
        )

    is_wmti = meta["diffusion_model"].str.upper().eq("WMTI")
    is_gm = meta["roi_id"].isin(gm_rois)

    wmti_metrics = sorted(
        meta.loc[is_wmti, "metric"].drop_duplicates().astype(str).tolist()
    )
    if len(wmti_metrics) != EXPECTED_N_WMTI_METRICS:
        raise ValueError(
            f"Expected {EXPECTED_N_WMTI_METRICS} WMTI metrics, "
            f"found {len(wmti_metrics)}: {wmti_metrics}"
        )

    exclude_mask = (is_wmti & is_gm).to_numpy(dtype=bool)
    keep_mask = ~exclude_mask

    if int(exclude_mask.sum()) != EXPECTED_N_EXCLUDED:
        raise ValueError(
            f"Expected {EXPECTED_N_EXCLUDED} excluded WMTI-GM variables, "
            f"found {int(exclude_mask.sum())}"
        )

    if int(keep_mask.sum()) != EXPECTED_N_RETAINED:
        raise ValueError(
            f"Expected {EXPECTED_N_RETAINED} retained variables, "
            f"found {int(keep_mask.sum())}"
        )

    excluded = meta.loc[exclude_mask].copy()
    excluded["reason"] = "WMTI metric in Brainnetome gray-matter ROI"

    return keep_mask, exclude_mask, wmti_metrics, excluded


def analyze_subjects(
    forecast_model: str,
    data: dict[str, np.ndarray],
    keep_mask: np.ndarray,
    exclude_mask: np.ndarray
) -> pd.DataFrame:

    subjects = np.asarray(data["subjects"]).astype(str)
    true_z = np.asarray(data["true_z"], dtype=np.float64)
    pred_z = np.asarray(data["pred_z"], dtype=np.float64)

    rows = []

    for i, subject in enumerate(subjects):
        original = regression_metrics(true_z[i], pred_z[i])

        restricted = regression_metrics(
            true_z[i, :, keep_mask],
            pred_z[i, :, keep_mask]
        )

        excluded_only = regression_metrics(
            true_z[i, :, exclude_mask],
            pred_z[i, :, exclude_mask]
        )

        rows.append({
            "forecast_model": forecast_model,
            "subject": subject,

            "original_MAE": original["MAE"],
            "restricted_MAE": restricted["MAE"],
            "delta_MAE_restricted_minus_original":
                restricted["MAE"] - original["MAE"],

            "original_RMSE": original["RMSE"],
            "restricted_RMSE": restricted["RMSE"],

            "original_R2": original["R2"],
            "restricted_R2": restricted["R2"],

            "WMTI_GM_only_MAE": excluded_only["MAE"],

            "n_original_variables": EXPECTED_N_VARIABLES,
            "n_restricted_variables": EXPECTED_N_RETAINED,
            "n_excluded_WMTI_GM_variables": EXPECTED_N_EXCLUDED,
        })

    return pd.DataFrame(rows)


def summarize(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model, g in subject_df.groupby("forecast_model", sort=False):
        row = {
            "forecast_model": model,
            "n_subjects": len(g),
        }

        for col in [
            "original_MAE", "restricted_MAE",
            "original_RMSE", "restricted_RMSE",
            "original_R2", "restricted_R2",
            "WMTI_GM_only_MAE"
        ]:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_sd"] = float(g[col].std(ddof=1))

        row["MAE_change_mean"] = float(
            g["delta_MAE_restricted_minus_original"].mean()
        )
        row["MAE_change_percent"] = float(
            100.0
            * (row["restricted_MAE_mean"] - row["original_MAE_mean"])
            / row["original_MAE_mean"]
        )

        rows.append(row)

    out = pd.DataFrame(rows)
    out["restricted_MAE_rank"] = (
        out["restricted_MAE_mean"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    return out.sort_values("restricted_MAE_rank").reset_index(drop=True)


def pairwise_vs_itransformer(subject_df: pd.DataFrame) -> pd.DataFrame:
    ref = (
        subject_df[subject_df["forecast_model"] == "itransformer"]
        [["subject", "restricted_MAE"]]
        .rename(columns={"restricted_MAE": "itransformer_MAE"})
    )

    rows = []

    for comparator in ["timesnet", "mole", "dlinear"]:
        comp = (
            subject_df[subject_df["forecast_model"] == comparator]
            [["subject", "restricted_MAE"]]
            .rename(columns={"restricted_MAE": "comparator_MAE"})
        )

        merged = ref.merge(
            comp,
            on="subject",
            how="inner",
            validate="one_to_one"
        )

        if len(merged) != EXPECTED_N_SUBJECTS:
            raise ValueError(
                f"{comparator}: expected {EXPECTED_N_SUBJECTS} paired subjects, "
                f"found {len(merged)}"
            )

        diff = (
            merged["comparator_MAE"] - merged["itransformer_MAE"]
        ).to_numpy(dtype=float)

        stat, p = wilcoxon(
            diff,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto"
        )

        rows.append({
            "reference": "iTransformer",
            "comparator": comparator,
            "n_subjects": len(merged),
            "itransformer_MAE_mean":
                float(merged["itransformer_MAE"].mean()),
            "comparator_MAE_mean":
                float(merged["comparator_MAE"].mean()),
            "mean_difference_comparator_minus_itransformer":
                float(diff.mean()),
            "wilcoxon_statistic": float(stat),
            "p_value": float(p),
        })

    out = pd.DataFrame(rows)
    out["FDR_q_value"] = benjamini_hochberg(
        out["p_value"].to_numpy(dtype=float)
    )
    return out


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
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for model in FORECAST_MODELS:
        path = input_dir / f"{model}_predictions.npz"
        print(f"Loading {model}: {path}", flush=True)
        loaded[model] = load_prediction(path)

    ref = loaded["itransformer"]
    ref_subjects = np.asarray(ref["subjects"]).astype(str)
    meta = build_metadata(ref)

    for model in FORECAST_MODELS[1:]:
        verify_same_metadata(ref, loaded[model], model)
        loaded[model] = reorder_subjects(ref_subjects, loaded[model])

    keep_mask, exclude_mask, wmti_metrics, excluded = make_wmti_gm_mask(meta)

    excluded.to_csv(
        output_dir / "wmti_gm_excluded_variables.csv",
        index=False,
        encoding="utf-8-sig"
    )

    all_subject = []
    for model in FORECAST_MODELS:
        all_subject.append(
            analyze_subjects(
                model,
                loaded[model],
                keep_mask,
                exclude_mask
            )
        )

    subject_df = pd.concat(all_subject, ignore_index=True)
    summary_df = summarize(subject_df)
    pairwise_df = pairwise_vs_itransformer(subject_df)

    subject_df.to_csv(
        output_dir / "wmti_gm_sensitivity_by_subject.csv",
        index=False,
        encoding="utf-8-sig"
    )
    summary_df.to_csv(
        output_dir / "wmti_gm_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )
    pairwise_df.to_csv(
        output_dir / "wmti_gm_pairwise_vs_itransformer.csv",
        index=False,
        encoding="utf-8-sig"
    )

    audit = [
        "WMTI-GM sensitivity audit",
        "=========================",
        f"Subjects: {EXPECTED_N_SUBJECTS}",
        f"Target TEs: {EXPECTED_TARGET_TE}",
        f"Original variables: {EXPECTED_N_VARIABLES}",
        f"GM ROIs: {EXPECTED_N_GM_ROIS}",
        f"WM ROIs: {EXPECTED_N_WM_ROIS}",
        f"WMTI metrics: {wmti_metrics}",
        f"Excluded WMTI-GM variables: {int(exclude_mask.sum())}",
        f"Retained variables: {int(keep_mask.sum())}",
        "",
        "Restricted MAE summary:",
        summary_df.to_string(index=False),
        "",
        "Paired comparisons vs iTransformer:",
        pairwise_df.to_string(index=False),
    ]

    (output_dir / "wmti_gm_sensitivity_audit.txt").write_text(
        "\n".join(audit),
        encoding="utf-8"
    )

    print("\n=== WMTI-GM variable audit ===")
    print(f"Excluded: {int(exclude_mask.sum())}")
    print(f"Retained: {int(keep_mask.sum())}")

    print("\n=== Restricted MAE summary ===")
    print(
        summary_df[
            [
                "forecast_model",
                "original_MAE_mean",
                "restricted_MAE_mean",
                "MAE_change_percent",
                "restricted_MAE_rank",
            ]
        ].to_string(index=False)
    )

    print("\n=== Pairwise vs iTransformer ===")
    print(
        pairwise_df[
            ["comparator", "p_value", "FDR_q_value"]
        ].to_string(index=False)
    )

    print("\nSaved to:")
    print(output_dir)


if __name__ == "__main__":
    main()
