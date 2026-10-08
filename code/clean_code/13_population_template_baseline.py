#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_population_template_baseline.py

Purpose
-------
Test whether the strong clean-49 LOSO prediction performance could be explained
mainly by a stable population-level ROI × metric × target-TE profile.

This script DOES NOT retrain iTransformer.

For each of the 49 established iTransformer LOSO folds, it builds two completely
subject-blind population-template baselines:

1) train38 template
   For each target TE and each of the 11,248 variables, predict the held-out
   participant using the mean target value across the 38 model-fitting subjects.

2) dev48 template
   A stronger/conservative baseline using all 48 non-test participants
   (38 train + 10 validation) to form the population mean. This baseline still
   never sees the held-out test participant.

The template is constructed in native space and then transformed into the SAME
fold-specific standardized space used by the trained model, using the mean/std
saved in each iTransformer fold file.

Primary comparison
------------------
Participant-level standardized MAE:
    iTransformer vs train38 population template
    iTransformer vs dev48 population template

Secondary outputs
-----------------
- Target-specific MAE (TE125 / TE135)
- Metric-wise MAE for all 38 Model-Metric groups
- Paired Wilcoxon tests and BH-FDR
- Paired participant bootstrap CI for mean MAE differences
- ROI-wise across-subject R2 at fixed Metric × ROI × target TE
  (diagnostic against inflated pooled R2 from stable between-ROI differences)
- Raw-vs-saved native-value audit using np.allclose(rtol=1e-5, atol=1e-4)
  to tolerate harmless float32 round-off across differently scaled metrics

Expected project structure
--------------------------
/media/UG1/lys/dipy/BN_JHU296/
    config.json
    code/00_common.py
    outputs/table1_clean49/
        itransformer/
            fold_01.npz
            ...
            fold_49.npz
        itransformer_predictions.npz

Expected fold NPZ keys
----------------------
subject, train_subjects, val_subjects, target_te,
true_z, pred_z, true_native, pred_native, mean, std

Outputs
-------
outputs/population_template_baseline/
    population_template_by_subject.csv
    population_template_summary.csv
    population_template_pairwise_vs_itransformer.csv
    population_template_metricwise_by_subject.csv
    population_template_metricwise_summary.csv
    population_template_roiwise_across_subject_r2.csv
    population_template_roiwise_r2_summary.csv
    population_template_predictions.npz
    population_template_audit.txt

No GPU is required.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT = Path("/media/UG1/lys/dipy/BN_JHU296")
CODE_DIR = PROJECT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

common = importlib.import_module("00_common")

DEFAULT_TABLE1_DIR = PROJECT / "outputs" / "table1_clean49"
DEFAULT_OUTPUT_DIR = PROJECT / "outputs" / "population_template_baseline"

EXPECTED_N_SUBJECTS = 49
EXPECTED_N_TRAIN = 38
EXPECTED_N_VAL = 10
EXPECTED_N_VARIABLES = 11248
EXPECTED_N_MODEL_METRICS = 38
EXPECTED_N_ROIS = 296
EXPECTED_TARGET_TE = [125, 135]
BOOTSTRAP_SEED = 20260623
BOOTSTRAP_N = 10000


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE, RMSE, R2 after flattening finite paired observations."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[keep]
    y_pred = y_pred[keep]

    if len(y_true) == 0:
        return {"N": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    return {
        "N": int(len(y_true)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def r2_fixed_variable(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R2 across subjects for ONE fixed Metric × ROI × target TE variable.
    This removes the large between-ROI component that can inflate pooled R2.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[keep]
    y_pred = y_pred[keep]
    if len(y_true) < 3:
        return np.nan
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return np.nan
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    return float(1.0 - ss_res / ss_tot)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH-FDR correction without statsmodels."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    finite = np.isfinite(p)
    pv = p[finite]
    if len(pv) == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[finite] = restored
    return q


def paired_bootstrap_mean_diff(
    comparator: np.ndarray,
    reference: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """
    Bootstrap CI for mean(comparator - reference) across participants.
    Positive values favor iTransformer when reference=iTransformer.
    """
    comparator = np.asarray(comparator, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    diff = comparator - reference

    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = diff[idx].mean()

    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def load_project_matrix(config_path: str | None):
    config = common.load_config(config_path)
    matrix = common.load_matrix(config)

    raw = np.asarray(matrix["x_raw"], dtype=np.float32)
    subjects = np.asarray(matrix["subjects"]).astype(str)
    te_values = np.asarray(matrix["te_values"]).astype(int)

    if raw.ndim != 3:
        raise ValueError(f"x_raw must be [subject, TE, variable], got {raw.shape}")
    if raw.shape[2] != EXPECTED_N_VARIABLES:
        raise ValueError(
            f"Expected {EXPECTED_N_VARIABLES} variables, got {raw.shape[2]}"
        )

    return config, matrix, raw, subjects, te_values


def build_subject_lookup(subjects: np.ndarray) -> Dict[str, int]:
    subjects = np.asarray(subjects).astype(str)
    if len(set(subjects)) != len(subjects):
        raise ValueError("Matrix subject IDs are not unique.")
    return {subject: i for i, subject in enumerate(subjects)}


def target_indices(te_values: np.ndarray, target_te: np.ndarray) -> np.ndarray:
    out = []
    for te in np.asarray(target_te).astype(int):
        hit = np.where(te_values == te)[0]
        if len(hit) != 1:
            raise ValueError(f"Could not uniquely locate target TE={te} in {te_values}")
        out.append(int(hit[0]))
    return np.asarray(out, dtype=int)


def metadata_frame(pred: np.lib.npyio.NpzFile) -> pd.DataFrame:
    required = ["models", "metrics", "roi_ids", "roi_labels"]
    missing = [k for k in required if k not in pred.files]
    if missing:
        raise KeyError(
            f"Prediction NPZ is missing metadata keys {missing}; "
            f"available keys={pred.files}"
        )

    meta = pd.DataFrame(
        {
            "variable_index": np.arange(EXPECTED_N_VARIABLES, dtype=int),
            "diffusion_model": np.asarray(pred["models"]).astype(str),
            "metric": np.asarray(pred["metrics"]).astype(str),
            "roi_id": np.asarray(pred["roi_ids"]).astype(str),
            "roi_label": np.asarray(pred["roi_labels"]).astype(str),
        }
    )

    groups = (
        meta.groupby(["diffusion_model", "metric"], sort=False)
        .size()
        .reset_index(name="n_variables")
    )
    if len(groups) != EXPECTED_N_MODEL_METRICS:
        raise ValueError(
            f"Expected {EXPECTED_N_MODEL_METRICS} Model-Metric groups, "
            f"found {len(groups)}."
        )
    bad = groups[groups["n_variables"] != EXPECTED_N_ROIS]
    if not bad.empty:
        raise ValueError(
            "Expected 296 ROI variables for every Model-Metric group:\n"
            + bad.to_string(index=False)
        )

    return meta


def load_fold_paths(table1_dir: Path) -> List[Path]:
    model_dir = table1_dir / "itransformer"
    paths = sorted(model_dir.glob("fold_*.npz"))
    if len(paths) != EXPECTED_N_SUBJECTS:
        raise FileNotFoundError(
            f"Expected {EXPECTED_N_SUBJECTS} iTransformer fold files in {model_dir}, "
            f"found {len(paths)}."
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Subject-blind population-template baseline for the canonical clean-49 "
            "iTransformer LOSO experiment."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Project config.json; default uses the project's normal config loading.",
    )
    parser.add_argument(
        "--table1-dir",
        default=str(DEFAULT_TABLE1_DIR),
        help="Directory containing itransformer fold files and prediction NPZ.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=BOOTSTRAP_N,
        help="Paired participant bootstrap repetitions.",
    )
    args = parser.parse_args()

    table1_dir = Path(args.table1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw complete matrix and metadata.
    config, matrix, raw_all, subjects_all, te_values = load_project_matrix(args.config)
    subject_lookup = build_subject_lookup(subjects_all)

    pred_path = table1_dir / "itransformer_predictions.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}")
    pred_meta_npz = np.load(pred_path, allow_pickle=True)
    meta = metadata_frame(pred_meta_npz)

    fold_paths = load_fold_paths(table1_dir)

    # Store aligned out-of-sample arrays across 49 folds.
    all_subjects: List[str] = []
    true_z_all: List[np.ndarray] = []
    it_z_all: List[np.ndarray] = []
    template38_z_all: List[np.ndarray] = []
    template48_z_all: List[np.ndarray] = []

    subject_rows: List[dict] = []

    for fold_path in fold_paths:
        fold = np.load(fold_path, allow_pickle=True)

        required = [
            "fold",
            "subject",
            "train_subjects",
            "val_subjects",
            "target_te",
            "true_z",
            "pred_z",
            "true_native",
            "mean",
            "std",
        ]
        missing = [k for k in required if k not in fold.files]
        if missing:
            raise KeyError(
                f"{fold_path} missing keys {missing}; available={fold.files}"
            )

        fold_number = int(np.asarray(fold["fold"]).reshape(-1)[0])
        test_subject = str(np.asarray(fold["subject"]).astype(str).reshape(-1)[0])
        train_subjects = np.asarray(fold["train_subjects"]).astype(str)
        val_subjects = np.asarray(fold["val_subjects"]).astype(str)
        target_te = np.asarray(fold["target_te"]).astype(int)

        if len(train_subjects) != EXPECTED_N_TRAIN:
            raise ValueError(
                f"Fold {fold_number}: expected {EXPECTED_N_TRAIN} train subjects, "
                f"got {len(train_subjects)}"
            )
        if len(val_subjects) != EXPECTED_N_VAL:
            raise ValueError(
                f"Fold {fold_number}: expected {EXPECTED_N_VAL} val subjects, "
                f"got {len(val_subjects)}"
            )
        if target_te.tolist() != EXPECTED_TARGET_TE:
            raise ValueError(
                f"Fold {fold_number}: target TE mismatch {target_te.tolist()}"
            )

        overlap = (
            set(train_subjects) & set(val_subjects)
            or set(train_subjects) & {test_subject}
            or set(val_subjects) & {test_subject}
        )
        if overlap:
            raise ValueError(f"Fold {fold_number}: train/val/test overlap: {overlap}")

        # Exact row indices in the raw matrix.
        try:
            train_rows = np.asarray(
                [subject_lookup[s] for s in train_subjects], dtype=int
            )
            val_rows = np.asarray(
                [subject_lookup[s] for s in val_subjects], dtype=int
            )
            test_row = int(subject_lookup[test_subject])
        except KeyError as exc:
            raise KeyError(
                f"Fold {fold_number}: subject not found in x_raw matrix: {exc}"
            ) from exc

        t_idx = target_indices(te_values, target_te)

        # Subject-blind target templates in NATIVE space.
        template38_native = raw_all[train_rows][:, t_idx, :].mean(axis=0)
        dev_rows = np.concatenate([train_rows, val_rows])
        template48_native = raw_all[dev_rows][:, t_idx, :].mean(axis=0)

        # Fold-specific scaler used by the actual iTransformer training.
        mean = np.asarray(fold["mean"], dtype=np.float64).reshape(-1)
        std = np.asarray(fold["std"], dtype=np.float64).reshape(-1)
        if mean.shape[0] != EXPECTED_N_VARIABLES or std.shape[0] != EXPECTED_N_VARIABLES:
            raise ValueError(
                f"Fold {fold_number}: scaler shape mismatch "
                f"mean={mean.shape}, std={std.shape}"
            )
        if np.any(std <= 0):
            raise ValueError(f"Fold {fold_number}: non-positive std detected.")

        template38_z = (template38_native - mean[None, :]) / std[None, :]
        template48_z = (template48_native - mean[None, :]) / std[None, :]

        true_z = np.asarray(fold["true_z"], dtype=np.float64)
        pred_z = np.asarray(fold["pred_z"], dtype=np.float64)

        if true_z.shape != (1, 2, EXPECTED_N_VARIABLES):
            raise ValueError(
                f"Fold {fold_number}: true_z shape={true_z.shape}, expected "
                f"(1, 2, {EXPECTED_N_VARIABLES})"
            )
        if pred_z.shape != true_z.shape:
            raise ValueError(
                f"Fold {fold_number}: pred_z shape={pred_z.shape}, "
                f"true_z shape={true_z.shape}"
            )

        # Strong audit: saved true_native should numerically match the raw-matrix
        # test target values. Because some retained metrics have much larger native
        # scales (for example GQI QA), float32 round-off can produce absolute
        # differences larger than 1e-4 even when the two arrays are effectively
        # identical. Therefore use a combined relative + absolute tolerance rather
        # than a single absolute threshold.
        raw_test_native = raw_all[test_row, t_idx, :].astype(np.float64)
        saved_true_native = np.asarray(fold["true_native"], dtype=np.float64)[0]

        native_diff = np.abs(raw_test_native - saved_true_native)
        native_max_abs_error = float(np.max(native_diff))

        native_is_close = np.allclose(
            raw_test_native,
            saved_true_native,
            rtol=1e-5,
            atol=1e-4,
            equal_nan=True,
        )

        if not native_is_close:
            # Extra diagnostics to distinguish harmless floating-point differences
            # from a genuine subject / TE / data-version mismatch.
            denom = np.maximum(
                np.maximum(np.abs(raw_test_native), np.abs(saved_true_native)),
                1e-12,
            )
            native_rel_diff = native_diff / denom
            native_max_rel_error = float(np.nanmax(native_rel_diff))

            worst_flat = int(np.nanargmax(native_diff))
            worst_te_pos, worst_var = np.unravel_index(
                worst_flat, native_diff.shape
            )

            raise RuntimeError(
                f"Fold {fold_number}: raw-vs-saved true_native mismatch beyond "
                f"allclose tolerance (rtol=1e-5, atol=1e-4). "
                f"max_abs={native_max_abs_error:.6g}, "
                f"max_rel={native_max_rel_error:.6g}, "
                f"worst_target_TE={int(target_te[worst_te_pos])}, "
                f"worst_variable_index={int(worst_var)}"
            )

        y = true_z[0]
        p_it = pred_z[0]
        p_38 = template38_z
        p_48 = template48_z

        m_it = regression_metrics(y, p_it)
        m_38 = regression_metrics(y, p_38)
        m_48 = regression_metrics(y, p_48)

        t_it = [regression_metrics(y[i], p_it[i]) for i in range(2)]
        t_38 = [regression_metrics(y[i], p_38[i]) for i in range(2)]
        t_48 = [regression_metrics(y[i], p_48[i]) for i in range(2)]

        subject_rows.append(
            {
                "fold": fold_number,
                "test_subject": test_subject,
                "n_train_template38": len(train_rows),
                "n_dev_template48": len(dev_rows),

                "itransformer_MAE": m_it["MAE"],
                "template_train38_MAE": m_38["MAE"],
                "template_dev48_MAE": m_48["MAE"],

                "itransformer_RMSE": m_it["RMSE"],
                "template_train38_RMSE": m_38["RMSE"],
                "template_dev48_RMSE": m_48["RMSE"],

                "itransformer_R2": m_it["R2"],
                "template_train38_R2": m_38["R2"],
                "template_dev48_R2": m_48["R2"],

                "itransformer_MAE_TE125": t_it[0]["MAE"],
                "template_train38_MAE_TE125": t_38[0]["MAE"],
                "template_dev48_MAE_TE125": t_48[0]["MAE"],

                "itransformer_MAE_TE135": t_it[1]["MAE"],
                "template_train38_MAE_TE135": t_38[1]["MAE"],
                "template_dev48_MAE_TE135": t_48[1]["MAE"],

                "delta_MAE_train38_minus_itransformer": m_38["MAE"] - m_it["MAE"],
                "delta_MAE_dev48_minus_itransformer": m_48["MAE"] - m_it["MAE"],
                "native_audit_max_abs_error": native_max_abs_error,
            }
        )

        all_subjects.append(test_subject)
        true_z_all.append(y[None, ...])
        it_z_all.append(p_it[None, ...])
        template38_z_all.append(p_38[None, ...])
        template48_z_all.append(p_48[None, ...])

    subject_df = pd.DataFrame(subject_rows).sort_values("fold").reset_index(drop=True)

    true_z_all = np.concatenate(true_z_all, axis=0)
    it_z_all = np.concatenate(it_z_all, axis=0)
    template38_z_all = np.concatenate(template38_z_all, axis=0)
    template48_z_all = np.concatenate(template48_z_all, axis=0)
    all_subjects_arr = np.asarray(all_subjects).astype(str)

    if len(set(all_subjects_arr)) != EXPECTED_N_SUBJECTS:
        raise RuntimeError("The 49 fold test subjects are not unique.")

    # ------------------------------------------------------------------
    # Overall subject-level summary
    # ------------------------------------------------------------------
    summary_rows = []
    model_cols = {
        "iTransformer": "itransformer",
        "Population template (train38)": "template_train38",
        "Population template (dev48)": "template_dev48",
    }

    for display_name, prefix in model_cols.items():
        row = {"method": display_name, "n_subjects": EXPECTED_N_SUBJECTS}
        for metric in ["MAE", "RMSE", "R2", "MAE_TE125", "MAE_TE135"]:
            col = f"{prefix}_{metric}"
            values = subject_df[col].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            row[f"{metric}_q75"] = float(np.quantile(values, 0.75))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # Paired overall tests vs iTransformer
    # ------------------------------------------------------------------
    it_mae = subject_df["itransformer_MAE"].to_numpy(dtype=float)
    pair_rows = []

    comparisons = [
        ("Population template (train38)", "template_train38_MAE", 0),
        ("Population template (dev48)", "template_dev48_MAE", 1),
    ]

    for name, col, seed_offset in comparisons:
        comp = subject_df[col].to_numpy(dtype=float)
        diff = comp - it_mae  # positive => iTransformer lower error

        stat, p = wilcoxon(
            diff,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        ci_low, ci_high = paired_bootstrap_mean_diff(
            comp,
            it_mae,
            n_boot=args.bootstrap,
            seed=BOOTSTRAP_SEED + seed_offset,
        )

        comp_mean = float(np.mean(comp))
        it_mean = float(np.mean(it_mae))
        relative_reduction = (
            100.0 * (comp_mean - it_mean) / comp_mean
            if comp_mean != 0
            else np.nan
        )

        pair_rows.append(
            {
                "reference": "iTransformer",
                "comparator": name,
                "n_subjects": EXPECTED_N_SUBJECTS,
                "itransformer_MAE_mean": it_mean,
                "comparator_MAE_mean": comp_mean,
                "mean_difference_comparator_minus_itransformer": float(np.mean(diff)),
                "bootstrap95CI_low": ci_low,
                "bootstrap95CI_high": ci_high,
                "relative_MAE_reduction_percent": relative_reduction,
                "subjects_itransformer_lower_MAE": int(np.sum(it_mae < comp)),
                "subjects_equal_MAE": int(np.sum(np.isclose(it_mae, comp))),
                "subjects_template_lower_MAE": int(np.sum(comp < it_mae)),
                "wilcoxon_statistic": float(stat),
                "p_value": float(p),
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    pair_df["FDR_q_value"] = benjamini_hochberg(
        pair_df["p_value"].to_numpy(dtype=float)
    )

    # ------------------------------------------------------------------
    # Metric-wise analysis: 38 Model-Metric groups
    # ------------------------------------------------------------------
    metric_subject_rows = []
    metric_summary_rows = []

    for (diffusion_model, metric), g in meta.groupby(
        ["diffusion_model", "metric"], sort=False
    ):
        idx = g["variable_index"].to_numpy(dtype=int)
        if len(idx) != EXPECTED_N_ROIS:
            raise RuntimeError(
                f"{diffusion_model}/{metric}: expected {EXPECTED_N_ROIS} variables, "
                f"got {len(idx)}"
            )

        mae_it = np.mean(
            np.abs(it_z_all[:, :, idx] - true_z_all[:, :, idx]),
            axis=(1, 2),
        )
        mae_38 = np.mean(
            np.abs(template38_z_all[:, :, idx] - true_z_all[:, :, idx]),
            axis=(1, 2),
        )
        mae_48 = np.mean(
            np.abs(template48_z_all[:, :, idx] - true_z_all[:, :, idx]),
            axis=(1, 2),
        )

        for i, subject in enumerate(all_subjects_arr):
            metric_subject_rows.append(
                {
                    "subject": subject,
                    "diffusion_model": diffusion_model,
                    "metric": metric,
                    "itransformer_MAE": float(mae_it[i]),
                    "template_train38_MAE": float(mae_38[i]),
                    "template_dev48_MAE": float(mae_48[i]),
                    "delta_train38_minus_itransformer": float(mae_38[i] - mae_it[i]),
                    "delta_dev48_minus_itransformer": float(mae_48[i] - mae_it[i]),
                }
            )

        stat38, p38 = wilcoxon(
            mae_38 - mae_it,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        stat48, p48 = wilcoxon(
            mae_48 - mae_it,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        metric_summary_rows.append(
            {
                "diffusion_model": diffusion_model,
                "metric": metric,
                "n_subjects": EXPECTED_N_SUBJECTS,
                "n_rois": EXPECTED_N_ROIS,

                "itransformer_MAE_mean": float(np.mean(mae_it)),
                "itransformer_MAE_sd": float(np.std(mae_it, ddof=1)),

                "template_train38_MAE_mean": float(np.mean(mae_38)),
                "template_train38_MAE_sd": float(np.std(mae_38, ddof=1)),
                "train38_relative_reduction_percent": float(
                    100.0 * (np.mean(mae_38) - np.mean(mae_it)) / np.mean(mae_38)
                ),
                "train38_subject_wins_itransformer": int(np.sum(mae_it < mae_38)),
                "train38_wilcoxon_statistic": float(stat38),
                "train38_p_value": float(p38),

                "template_dev48_MAE_mean": float(np.mean(mae_48)),
                "template_dev48_MAE_sd": float(np.std(mae_48, ddof=1)),
                "dev48_relative_reduction_percent": float(
                    100.0 * (np.mean(mae_48) - np.mean(mae_it)) / np.mean(mae_48)
                ),
                "dev48_subject_wins_itransformer": int(np.sum(mae_it < mae_48)),
                "dev48_wilcoxon_statistic": float(stat48),
                "dev48_p_value": float(p48),
            }
        )

    metric_subject_df = pd.DataFrame(metric_subject_rows)
    metric_summary_df = pd.DataFrame(metric_summary_rows)

    metric_summary_df["train38_FDR_q"] = benjamini_hochberg(
        metric_summary_df["train38_p_value"].to_numpy(dtype=float)
    )
    metric_summary_df["dev48_FDR_q"] = benjamini_hochberg(
        metric_summary_df["dev48_p_value"].to_numpy(dtype=float)
    )

    # ------------------------------------------------------------------
    # ROI-wise across-subject R2 diagnostic
    #
    # Fix metric + ROI + target TE, then calculate R2 across the 49 held-out
    # participants. This removes stable between-ROI differences from the R2.
    # ------------------------------------------------------------------
    roi_r2_rows = []

    for (diffusion_model, metric), g in meta.groupby(
        ["diffusion_model", "metric"], sort=False
    ):
        for row in g.itertuples(index=False):
            j = int(row.variable_index)

            for target_pos, te in enumerate(EXPECTED_TARGET_TE):
                y = true_z_all[:, target_pos, j]
                p_it = it_z_all[:, target_pos, j]
                p_38 = template38_z_all[:, target_pos, j]
                p_48 = template48_z_all[:, target_pos, j]

                roi_r2_rows.append(
                    {
                        "diffusion_model": diffusion_model,
                        "metric": metric,
                        "variable_index": j,
                        "roi_id": row.roi_id,
                        "roi_label": row.roi_label,
                        "target_te": te,
                        "R2_itransformer": r2_fixed_variable(y, p_it),
                        "R2_template_train38": r2_fixed_variable(y, p_38),
                        "R2_template_dev48": r2_fixed_variable(y, p_48),
                    }
                )

    roi_r2_df = pd.DataFrame(roi_r2_rows)

    roi_summary_rows = []
    for (diffusion_model, metric), g in roi_r2_df.groupby(
        ["diffusion_model", "metric"], sort=False
    ):
        row = {
            "diffusion_model": diffusion_model,
            "metric": metric,
            "n_roi_target_cells": len(g),
        }
        for col in [
            "R2_itransformer",
            "R2_template_train38",
            "R2_template_dev48",
        ]:
            vals = g[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{col}_median"] = float(np.median(vals)) if len(vals) else np.nan
            row[f"{col}_q25"] = float(np.quantile(vals, 0.25)) if len(vals) else np.nan
            row[f"{col}_q75"] = float(np.quantile(vals, 0.75)) if len(vals) else np.nan
            row[f"{col}_fraction_gt0"] = (
                float(np.mean(vals > 0)) if len(vals) else np.nan
            )
        roi_summary_rows.append(row)

    roi_summary_df = pd.DataFrame(roi_summary_rows)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    subject_df.to_csv(
        output_dir / "population_template_by_subject.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        output_dir / "population_template_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pair_df.to_csv(
        output_dir / "population_template_pairwise_vs_itransformer.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_subject_df.to_csv(
        output_dir / "population_template_metricwise_by_subject.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_summary_df.to_csv(
        output_dir / "population_template_metricwise_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    roi_r2_df.to_csv(
        output_dir / "population_template_roiwise_across_subject_r2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    roi_summary_df.to_csv(
        output_dir / "population_template_roiwise_r2_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        output_dir / "population_template_predictions.npz",
        subjects=all_subjects_arr,
        target_te=np.asarray(EXPECTED_TARGET_TE, dtype=int),
        true_z=true_z_all.astype(np.float32),
        itransformer_pred_z=it_z_all.astype(np.float32),
        template_train38_pred_z=template38_z_all.astype(np.float32),
        template_dev48_pred_z=template48_z_all.astype(np.float32),
        models=np.asarray(pred_meta_npz["models"]).astype(str),
        metrics=np.asarray(pred_meta_npz["metrics"]).astype(str),
        roi_ids=np.asarray(pred_meta_npz["roi_ids"]).astype(str),
        roi_labels=np.asarray(pred_meta_npz["roi_labels"]).astype(str),
    )

    # Compact audit text.
    sig_train38 = int(
        np.sum(metric_summary_df["train38_FDR_q"].to_numpy(dtype=float) < 0.05)
    )
    sig_dev48 = int(
        np.sum(metric_summary_df["dev48_FDR_q"].to_numpy(dtype=float) < 0.05)
    )
    lower_train38 = int(
        np.sum(
            metric_summary_df["itransformer_MAE_mean"].to_numpy(dtype=float)
            < metric_summary_df["template_train38_MAE_mean"].to_numpy(dtype=float)
        )
    )
    lower_dev48 = int(
        np.sum(
            metric_summary_df["itransformer_MAE_mean"].to_numpy(dtype=float)
            < metric_summary_df["template_dev48_MAE_mean"].to_numpy(dtype=float)
        )
    )

    audit_lines = [
        "Population-template baseline audit",
        "==================================",
        f"Project: {PROJECT}",
        f"Table1 dir: {table1_dir}",
        f"Subjects/folds: {EXPECTED_N_SUBJECTS}",
        f"Variables: {EXPECTED_N_VARIABLES}",
        f"Target TEs: {EXPECTED_TARGET_TE}",
        f"Template train-only N: {EXPECTED_N_TRAIN}",
        f"Template development-pool N: {EXPECTED_N_TRAIN + EXPECTED_N_VAL}",
        f"Bootstrap repetitions: {args.bootstrap}",
        "",
        "OVERALL SUMMARY",
        summary_df.to_string(index=False),
        "",
        "PAIRED COMPARISONS VS iTransformer",
        pair_df.to_string(index=False),
        "",
        "METRIC-WISE COUNTS",
        f"iTransformer lower mean MAE than train38 template: "
        f"{lower_train38}/{EXPECTED_N_MODEL_METRICS}",
        f"iTransformer lower mean MAE than dev48 template: "
        f"{lower_dev48}/{EXPECTED_N_MODEL_METRICS}",
        f"Metric-wise FDR significant vs train38 template: "
        f"{sig_train38}/{EXPECTED_N_MODEL_METRICS}",
        f"Metric-wise FDR significant vs dev48 template: "
        f"{sig_dev48}/{EXPECTED_N_MODEL_METRICS}",
        "",
        "INTERPRETATION RULE",
        "If iTransformer has lower participant-level MAE than the subject-blind "
        "population template, especially the stronger dev48 template, then the "
        "primary prediction cannot be explained solely by a stable population "
        "ROI × metric × target-TE profile.",
        "",
        "IMPORTANT",
        "The ROI-wise across-subject R2 is a stricter diagnostic than pooled R2: "
        "each R2 is calculated across 49 participants at a fixed metric, ROI, "
        "and target TE, so between-ROI anatomical differences cannot inflate it.",
    ]

    (output_dir / "population_template_audit.txt").write_text(
        "\n".join(audit_lines) + "\n",
        encoding="utf-8",
    )

    # Console summary.
    print("\n=== Overall participant-level standardized MAE ===")
    print(
        summary_df[
            ["method", "MAE_mean", "MAE_sd", "MAE_TE125_mean", "MAE_TE135_mean"]
        ].to_string(index=False)
    )

    print("\n=== Paired comparisons vs iTransformer ===")
    print(
        pair_df[
            [
                "comparator",
                "mean_difference_comparator_minus_itransformer",
                "bootstrap95CI_low",
                "bootstrap95CI_high",
                "relative_MAE_reduction_percent",
                "subjects_itransformer_lower_MAE",
                "p_value",
                "FDR_q_value",
            ]
        ].to_string(index=False)
    )

    print("\n=== Metric-wise summary ===")
    print(
        f"iTransformer lower mean MAE than train38 template: "
        f"{lower_train38}/{EXPECTED_N_MODEL_METRICS}"
    )
    print(
        f"iTransformer lower mean MAE than dev48 template: "
        f"{lower_dev48}/{EXPECTED_N_MODEL_METRICS}"
    )
    print(
        f"FDR-significant metric-wise differences vs train38: "
        f"{sig_train38}/{EXPECTED_N_MODEL_METRICS}"
    )
    print(
        f"FDR-significant metric-wise differences vs dev48: "
        f"{sig_dev48}/{EXPECTED_N_MODEL_METRICS}"
    )

    print("\nSaved outputs to:")
    print(output_dir)


if __name__ == "__main__":
    main()
