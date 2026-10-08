from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT = Path(__file__).resolve().parents[2]
MODELS = ("itransformer", "timesnet", "mole", "dlinear")
DISPLAY = {
    "itransformer": "iTransformer",
    "timesnet": "TimesNet",
    "mole": "MoLE",
    "dlinear": "DLinear",
}


def markdown_table(table: pd.DataFrame) -> str:
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in table.itertuples(index=False, name=None):
        cells = [f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bh_fdr(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Table 1 and S1 from identical predictions.")
    parser.add_argument("--input-dir", default="outputs/table1_clean49")
    args = parser.parse_args()
    output_dir = Path(args.input_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT / output_dir

    tables = {}
    prediction_subjects = {}
    for model in MODELS:
        table = pd.read_csv(output_dir / f"{model}_fold_metrics.csv")
        if len(table) != 49 or table["test_subject"].nunique() != 49:
            raise ValueError(f"{model} does not contain 49 unique LOSO folds.")
        table = table.sort_values("fold").reset_index(drop=True)
        prediction = np.load(output_dir / f"{model}_predictions.npz", allow_pickle=True)
        prediction_subjects[model] = prediction["subjects"].astype(str).tolist()
        if prediction["true_z"].shape != (49, 2, 11248):
            raise ValueError(f"Unexpected prediction shape for {model}: {prediction['true_z'].shape}")
        if not np.array_equal(prediction["target_te"].astype(int), [125, 135]):
            raise ValueError(f"Unexpected target order for {model}.")
        pooled = np.mean(np.abs(prediction["pred_z"] - prediction["true_z"]), axis=(1, 2))
        if not np.allclose(pooled, table["MAE"], atol=1e-7):
            raise ValueError(f"Fold CSV and NPZ MAE differ for {model}.")
        tables[model] = table

    reference = tables[MODELS[0]][["fold", "test_subject", "fold_seed"]]
    reference_prediction_order = prediction_subjects[MODELS[0]]
    for model in MODELS[1:]:
        candidate = tables[model][["fold", "test_subject", "fold_seed"]]
        if not candidate.equals(reference):
            raise ValueError(f"Fold/test subject/seed mismatch for {model}.")
        if prediction_subjects[model] != reference_prediction_order:
            raise ValueError(f"Prediction subject order mismatch for {model}.")

    table1_rows = []
    s1_rows = []
    consistency_rows = []
    for model in MODELS:
        table = tables[model]
        row = {"model": DISPLAY[model], "n": len(table)}
        for metric in ("MAE", "RMSE", "R2"):
            row[f"{metric}_mean"] = table[metric].mean()
            row[f"{metric}_std"] = table[metric].std(ddof=1)
        table1_rows.append(row)

        s1_rows.append(
            {
                "model": DISPLAY[model],
                "n": len(table),
                "TE125_MAE_mean": table["MAE_TE125"].mean(),
                "TE125_MAE_std": table["MAE_TE125"].std(ddof=1),
                "TE135_MAE_mean": table["MAE_TE135"].mean(),
                "TE135_MAE_std": table["MAE_TE135"].std(ddof=1),
                "delta_MAE": table["MAE_TE135"].mean() - table["MAE_TE125"].mean(),
            }
        )
        implied = (table["MAE_TE125"] + table["MAE_TE135"]) / 2.0
        consistency_rows.append(
            {
                "model": DISPLAY[model],
                "max_subject_identity_error": np.max(np.abs(table["MAE"] - implied)),
                "overall_MAE_mean": table["MAE"].mean(),
                "target_average_MAE_mean": implied.mean(),
            }
        )

    table1 = pd.DataFrame(table1_rows).sort_values("MAE_mean").reset_index(drop=True)
    s1 = pd.DataFrame(s1_rows).sort_values("TE125_MAE_mean").reset_index(drop=True)
    consistency = pd.DataFrame(consistency_rows)
    if consistency["max_subject_identity_error"].max() > 1e-7:
        raise RuntimeError("Table 1/S1 MAE identity check failed.")

    pair_rows = []
    p_values = []
    for left, right in combinations(MODELS, 2):
        left_values = tables[left]["MAE"].to_numpy()
        right_values = tables[right]["MAE"].to_numpy()
        result = wilcoxon(left_values, right_values, alternative="two-sided")
        p_values.append(float(result.pvalue))
        pair_rows.append(
            {
                "model_a": DISPLAY[left],
                "model_b": DISPLAY[right],
                "mean_MAE_a": left_values.mean(),
                "mean_MAE_b": right_values.mean(),
                "mean_difference_a_minus_b": (left_values - right_values).mean(),
                "wilcoxon_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    q_values = bh_fdr(p_values)
    for row, q in zip(pair_rows, q_values):
        row["fdr_q_value"] = q
        row["significant_q_lt_0_05"] = bool(q < 0.05)

    table1.to_csv(output_dir / "table1_clean49_summary.csv", index=False, encoding="utf-8-sig")
    s1.to_csv(output_dir / "s1_target_specific_summary.csv", index=False, encoding="utf-8-sig")
    consistency.to_csv(
        output_dir / "table1_s1_consistency_check.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(pair_rows).to_csv(
        output_dir / "table1_pairwise_wilcoxon_fdr.csv", index=False, encoding="utf-8-sig"
    )

    lines = ["# Canonical clean-49 Table 1 and S1", "", "## Table 1", ""]
    lines.append(markdown_table(table1))
    lines.extend(["", "## Supplementary Table S1", ""])
    lines.append(markdown_table(s1))
    lines.extend(["", "All values above come from the same 49-fold prediction tensors.", ""])
    (output_dir / "table1_s1_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(table1.to_string(index=False))
    print("\nTarget-specific summary")
    print(s1.to_string(index=False))
    print(f"\nMaximum identity error: {consistency['max_subject_identity_error'].max():.3e}")


if __name__ == "__main__":
    main()
