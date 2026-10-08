from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

common = importlib.import_module("00_common")


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT / value


def subject_indices(all_subjects: np.ndarray, names: list[str]) -> np.ndarray:
    lookup = {subject: index for index, subject in enumerate(all_subjects.astype(str))}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise ValueError(f"Subjects missing from matrix: {missing}")
    return np.asarray([lookup[name] for name in names], dtype=int)


def fit_linear_extrapolation(observed: np.ndarray, observed_te: np.ndarray, target_te: np.ndarray) -> np.ndarray:
    """Fit one OLS line per variable using one subject's observed TE values."""
    centered_te = observed_te.astype(np.float64) - observed_te.mean()
    denominator = float(np.sum(centered_te**2))
    slope = np.sum(centered_te[:, None] * observed.astype(np.float64), axis=0) / denominator
    intercept = observed.mean(axis=0, dtype=np.float64) - slope * observed_te.mean()
    return (intercept[None, :] + target_te[:, None] * slope[None, :]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean-49 subject-wise linear TE extrapolation baseline.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--fold-manifest", default="outputs/table1_clean49/fold_manifest.csv")
    parser.add_argument("--table1-summary", default="outputs/table1_clean49/table1_clean49_summary.csv")
    parser.add_argument("--itransformer-folds", default="outputs/table1_clean49/itransformer_fold_metrics.csv")
    parser.add_argument("--output-dir", default="outputs/linear_extrapolation_clean49")
    args = parser.parse_args()

    config = common.load_config(args.config)
    matrix = common.load_matrix(config)
    raw = matrix["x_raw"].astype(np.float32)
    subjects = matrix["subjects"].astype(str)
    te_values = matrix["te_values"].astype(int)
    input_idx = common.te_indices(te_values, config["fixed_input_te"])
    target_idx = common.te_indices(te_values, config["fixed_target_te"])
    input_te = te_values[input_idx]
    target_te = te_values[target_idx]

    manifest = pd.read_csv(resolve(args.fold_manifest))
    if len(manifest) != 49 or manifest["test_subject"].nunique() != 49:
        raise ValueError("Fold manifest must contain 49 unique clean-49 test subjects.")

    rows: list[dict[str, float | int | str]] = []
    true_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []
    test_subjects: list[str] = []
    for row in manifest.itertuples(index=False):
        train_names = str(row.train_subjects).split(";")
        train_idx = subject_indices(subjects, train_names)
        test_idx = subject_indices(subjects, [str(row.test_subject)])
        standardized, _, _ = common.standardize_from_train(raw, train_idx)
        observed = standardized[test_idx[0], input_idx, :]
        true = standardized[test_idx[0], target_idx, :]
        prediction = fit_linear_extrapolation(observed, input_te, target_te)
        metrics = common.regression_metrics(true, prediction)
        mae_by_te = np.mean(np.abs(prediction - true), axis=1)
        identity_error = abs(metrics["MAE"] - float(mae_by_te.mean()))
        rows.append(
            {
                "model": "Simple Linear Extrapolation",
                "fold": int(row.fold),
                "test_subject": str(row.test_subject),
                "fold_seed": int(row.fold_seed),
                "n_train": int(row.n_train),
                "n_val": int(row.n_val),
                "n_test": int(row.n_test),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "MAE_TE125": float(mae_by_te[0]),
                "MAE_TE135": float(mae_by_te[1]),
                "MAE_identity_error": identity_error,
            }
        )
        true_all.append(true)
        pred_all.append(prediction)
        test_subjects.append(str(row.test_subject))

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(output_dir / "linear_fold_metrics.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output_dir / "linear_predictions.npz",
        subjects=np.asarray(test_subjects),
        target_te=target_te,
        true_z=np.stack(true_all),
        pred_z=np.stack(pred_all),
        variables=matrix["variables"],
        models=matrix["models"],
        metrics=matrix["metrics"],
        features=matrix["features"],
        roi_ids=matrix["roi_ids"],
        roi_labels=matrix["roi_labels"],
    )

    itransformer = pd.read_csv(resolve(args.itransformer_folds))[["test_subject", "MAE"]].rename(
        columns={"MAE": "MAE_iTransformer"}
    )
    paired = folds[["test_subject", "MAE"]].rename(columns={"MAE": "MAE_Linear"}).merge(
        itransformer, on="test_subject", validate="one_to_one"
    )
    paired["delta_MAE_Linear_minus_iTransformer"] = paired["MAE_Linear"] - paired["MAE_iTransformer"]
    statistic, p_value = wilcoxon(
        paired["MAE_Linear"], paired["MAE_iTransformer"], alternative="two-sided", method="auto"
    )
    comparison = pd.DataFrame(
        [
            {
                "comparison": "Simple Linear Extrapolation vs iTransformer",
                "n": len(paired),
                "mean_delta_MAE_Linear_minus_iTransformer": paired["delta_MAE_Linear_minus_iTransformer"].mean(),
                "median_delta_MAE_Linear_minus_iTransformer": paired["delta_MAE_Linear_minus_iTransformer"].median(),
                "linear_better_subjects": int((paired["delta_MAE_Linear_minus_iTransformer"] < 0).sum()),
                "itransformer_better_subjects": int((paired["delta_MAE_Linear_minus_iTransformer"] > 0).sum()),
                "wilcoxon_statistic": float(statistic),
                "wilcoxon_p": float(p_value),
            }
        ]
    )
    paired.to_csv(output_dir / "linear_vs_itransformer_paired_mae.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "linear_vs_itransformer_wilcoxon.csv", index=False, encoding="utf-8-sig")

    table1 = pd.read_csv(resolve(args.table1_summary))
    linear_summary = {
        "model": "Simple Linear Extrapolation",
        "n": len(folds),
        "MAE_mean": folds["MAE"].mean(),
        "MAE_std": folds["MAE"].std(ddof=1),
        "RMSE_mean": folds["RMSE"].mean(),
        "RMSE_std": folds["RMSE"].std(ddof=1),
        "R2_mean": folds["R2"].mean(),
        "R2_std": folds["R2"].std(ddof=1),
        "MAE_TE125_mean": folds["MAE_TE125"].mean(),
        "MAE_TE125_std": folds["MAE_TE125"].std(ddof=1),
        "MAE_TE135_mean": folds["MAE_TE135"].mean(),
        "MAE_TE135_std": folds["MAE_TE135"].std(ddof=1),
    }
    for column in linear_summary:
        if column not in table1.columns:
            table1[column] = np.nan
    supplementary = pd.concat([table1, pd.DataFrame([linear_summary])], ignore_index=True, sort=False)
    supplementary.to_csv(output_dir / "supplementary_model_summary.csv", index=False, encoding="utf-8-sig")

    result = comparison.iloc[0]
    report = f"""# Clean-49 simple linear extrapolation baseline

## Linear result

- Subjects: {len(folds)}
- MAE: {linear_summary['MAE_mean']:.6f} +/- {linear_summary['MAE_std']:.6f}
- RMSE: {linear_summary['RMSE_mean']:.6f} +/- {linear_summary['RMSE_std']:.6f}
- R2: {linear_summary['R2_mean']:.6f} +/- {linear_summary['R2_std']:.6f}
- TE125 MAE: {linear_summary['MAE_TE125_mean']:.6f} +/- {linear_summary['MAE_TE125_std']:.6f}
- TE135 MAE: {linear_summary['MAE_TE135_mean']:.6f} +/- {linear_summary['MAE_TE135_std']:.6f}
- Maximum overall-MAE identity error: {folds['MAE_identity_error'].max():.3e}

## Paired comparison with iTransformer

- Mean delta MAE (Linear - iTransformer): {result['mean_delta_MAE_Linear_minus_iTransformer']:.6f}
- Median delta MAE: {result['median_delta_MAE_Linear_minus_iTransformer']:.6f}
- Linear lower MAE: {int(result['linear_better_subjects'])}/{len(folds)} subjects
- iTransformer lower MAE: {int(result['itransformer_better_subjects'])}/{len(folds)} subjects
- Paired two-sided Wilcoxon p: {result['wilcoxon_p']:.6g}

This is a subject-wise extrapolation baseline. It fits no cross-subject prediction parameters and is reported only in the Supplementary/Appendix.
"""
    (output_dir / "linear_extrapolation_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
