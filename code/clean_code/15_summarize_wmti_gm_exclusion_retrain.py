from __future__ import annotations

import argparse
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
EXPECTED_FOLDS = 49
EXPECTED_VARIABLES = 10264
EXPECTED_TARGET_TE = [125, 135]


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def validate_model(output_dir: Path, model: str, reference_manifest: pd.DataFrame) -> pd.DataFrame:
    metrics_path = output_dir / f"{model}_fold_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing fold metrics: {metrics_path}")
    table = pd.read_csv(metrics_path).sort_values("fold").reset_index(drop=True)
    if len(table) != EXPECTED_FOLDS or table["fold"].nunique() != EXPECTED_FOLDS:
        raise ValueError(f"{model}: expected 49 unique folds, found {len(table)} rows.")
    if set(table["n_variables"].astype(int)) != {EXPECTED_VARIABLES}:
        raise ValueError(f"{model}: fold metrics do not all contain {EXPECTED_VARIABLES} variables.")
    for column in ["n_train", "n_val", "n_test"]:
        expected = {"n_train": 38, "n_val": 10, "n_test": 1}[column]
        if set(table[column].astype(int)) != {expected}:
            raise ValueError(f"{model}: invalid {column} values: {sorted(table[column].unique())}")

    if not np.array_equal(table["test_subject"].astype(str), reference_manifest["test_subject"].astype(str)):
        raise RuntimeError(f"{model}: test-subject order differs from the canonical manifest.")
    if not np.array_equal(table["fold_seed"].astype(int), reference_manifest["fold_seed"].astype(int)):
        raise RuntimeError(f"{model}: fold seeds differ from the canonical manifest.")

    model_dir = output_dir / model
    for row in reference_manifest.itertuples(index=False):
        fold_path = model_dir / f"fold_{int(row.fold):02d}.npz"
        if not fold_path.exists():
            raise FileNotFoundError(f"Missing fold file: {fold_path}")
        z = np.load(fold_path, allow_pickle=True)
        if z["true_z"].shape != (1, 2, EXPECTED_VARIABLES):
            raise ValueError(f"{fold_path}: invalid true_z shape {z['true_z'].shape}")
        if z["pred_z"].shape != z["true_z"].shape:
            raise ValueError(f"{fold_path}: pred_z/true_z shape mismatch.")
        if int(z["n_variables"][0]) != EXPECTED_VARIABLES:
            raise ValueError(f"{fold_path}: n_variables is not {EXPECTED_VARIABLES}.")
        if z["target_te"].astype(int).tolist() != EXPECTED_TARGET_TE:
            raise ValueError(f"{fold_path}: target TE mismatch.")
        if str(z["subject"][0]) != str(row.test_subject):
            raise ValueError(f"{fold_path}: test subject mismatch.")
        if int(z["fold_seed"][0]) != int(row.fold_seed):
            raise ValueError(f"{fold_path}: fold seed mismatch.")
        if ";".join(z["train_subjects"].astype(str)) != str(row.train_subjects):
            raise ValueError(f"{fold_path}: train subjects mismatch.")
        if ";".join(z["val_subjects"].astype(str)) != str(row.val_subjects):
            raise ValueError(f"{fold_path}: validation subjects mismatch.")
    return table


def rebuild_primary_summary(
    primary_dir: Path, reference_manifest: pd.DataFrame
) -> pd.DataFrame:
    """Rebuild the canonical MAE summary from its auditable fold-level files."""
    rows = []
    for model in MODELS:
        path = primary_dir / f"{model}_fold_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing canonical fold metrics: {path}")
        table = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
        if len(table) != EXPECTED_FOLDS or table["fold"].nunique() != EXPECTED_FOLDS:
            raise ValueError(f"Canonical {model}: expected 49 unique folds.")
        if not np.array_equal(
            table["test_subject"].astype(str),
            reference_manifest["test_subject"].astype(str),
        ):
            raise RuntimeError(f"Canonical {model}: test-subject order differs from manifest.")
        if not np.array_equal(
            table["fold_seed"].astype(int),
            reference_manifest["fold_seed"].astype(int),
        ):
            raise RuntimeError(f"Canonical {model}: fold seeds differ from manifest.")
        rows.append(
            {
                "model": DISPLAY[model],
                "model_key": model,
                "n": len(table),
                "MAE_mean": float(table["MAE"].mean()),
                "MAE_std": float(table["MAE"].std(ddof=1)),
            }
        )

    rebuilt = pd.DataFrame(rows)
    summary_path = primary_dir / "table1_clean49_summary.csv"
    if summary_path.exists():
        reported = pd.read_csv(summary_path)
        reported["model_key"] = reported["model"].map({v: k for k, v in DISPLAY.items()})
        if reported["model_key"].isna().any():
            raise ValueError("Could not map all models in canonical summary.")
        checked = rebuilt.merge(
            reported[["model_key", "MAE_mean", "MAE_std"]],
            on="model_key",
            suffixes=("_rebuilt", "_reported"),
            validate="one_to_one",
        )
        for metric in ("MAE_mean", "MAE_std"):
            if not np.allclose(
                checked[f"{metric}_rebuilt"],
                checked[f"{metric}_reported"],
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"Canonical {metric} differs between fold metrics and precomputed summary."
                )
    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/wmti_gm_exclusion_retrain")
    parser.add_argument("--primary-dir", default="outputs/table1_clean49")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT / output_dir
    primary_dir = Path(args.primary_dir)
    if not primary_dir.is_absolute():
        primary_dir = PROJECT / primary_dir

    reference_manifest = pd.read_csv(primary_dir / "fold_manifest.csv").sort_values("fold").reset_index(drop=True)
    reduced = {model: validate_model(output_dir, model, reference_manifest) for model in MODELS}

    summary_rows = []
    for model, table in reduced.items():
        row = {"model": DISPLAY[model], "model_key": model, "n": len(table)}
        for metric in ["MAE", "RMSE", "R2", "MAE_TE125", "MAE_TE135"]:
            row[f"{metric}_mean"] = float(table[metric].mean())
            row[f"{metric}_sd"] = float(table[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary["MAE_rank"] = summary["MAE_mean"].rank(method="min", ascending=True).astype(int)
    summary = summary.sort_values("MAE_rank").reset_index(drop=True)

    primary_summary = rebuild_primary_summary(primary_dir, reference_manifest)
    comparison = summary.merge(
        primary_summary[["model_key", "MAE_mean", "MAE_std"]].rename(
            columns={"MAE_mean": "original_MAE_mean", "MAE_std": "original_MAE_sd"}
        ),
        on="model_key",
        how="left",
        validate="one_to_one",
    )
    comparison["retrained_minus_original_MAE"] = comparison["MAE_mean"] - comparison["original_MAE_mean"]
    comparison["MAE_change_percent"] = (
        100.0 * comparison["retrained_minus_original_MAE"] / comparison["original_MAE_mean"]
    )

    reference = reduced["itransformer"][["test_subject", "MAE"]].rename(
        columns={"MAE": "itransformer_MAE"}
    )
    pair_rows = []
    for comparator in ("timesnet", "mole", "dlinear"):
        paired = reference.merge(
            reduced[comparator][["test_subject", "MAE"]].rename(columns={"MAE": "comparator_MAE"}),
            on="test_subject",
            validate="one_to_one",
        )
        difference = paired["comparator_MAE"] - paired["itransformer_MAE"]
        statistic, p_value = wilcoxon(
            paired["itransformer_MAE"],
            paired["comparator_MAE"],
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        pair_rows.append(
            {
                "reference": "iTransformer",
                "comparator": DISPLAY[comparator],
                "n": len(paired),
                "itransformer_MAE_mean": float(paired["itransformer_MAE"].mean()),
                "comparator_MAE_mean": float(paired["comparator_MAE"].mean()),
                "mean_difference_comparator_minus_itransformer": float(difference.mean()),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    pairwise = pd.DataFrame(pair_rows)
    pairwise["fdr_q_value"] = bh_fdr(pairwise["p_value"].to_numpy())
    pairwise["significant_q_lt_0_05"] = pairwise["fdr_q_value"] < 0.05

    summary.to_csv(output_dir / "wmti_gm_exclusion_retrain_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(
        output_dir / "wmti_gm_exclusion_retrain_vs_original.csv", index=False, encoding="utf-8-sig"
    )
    pairwise.to_csv(
        output_dir / "wmti_gm_exclusion_retrain_pairwise_wilcoxon_fdr.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ranking_unchanged = summary["model_key"].tolist() == ["itransformer", "mole", "dlinear", "timesnet"]
    itransformer_best = summary.iloc[0]["model_key"] == "itransformer"
    all_significant = bool(pairwise["significant_q_lt_0_05"].all())
    audit_lines = [
        "WMTI-GM exclusion retraining sensitivity analysis audit",
        "=======================================================",
        f"Validated models: {', '.join(DISPLAY[m] for m in MODELS)}",
        f"Validated folds per model: {EXPECTED_FOLDS}",
        f"Validated total fold files: {EXPECTED_FOLDS * len(MODELS)}",
        f"Variables per fold: {EXPECTED_VARIABLES}",
        "Input TEs: 75, 85, 95, 105, 115",
        "Target TEs: 125, 135",
        "Split per fold: 38 train / 10 validation / 1 test",
        "All test subjects, train/validation subjects, and fold seeds match the canonical manifest: YES",
        "All per-fold prediction tensors have shape (1, 2, 10264): YES",
        "",
        f"Ranking unchanged from primary experiment: {'YES' if ranking_unchanged else 'NO'}",
        f"iTransformer has the lowest retrained overall MAE: {'YES' if itransformer_best else 'NO'}",
        f"All three iTransformer comparisons remain significant after BH-FDR: {'YES' if all_significant else 'NO'}",
        "",
        "Retrained summary:",
        summary.to_string(index=False),
        "",
        "Change versus original 11,248-variable clean-49 experiment:",
        comparison.to_string(index=False),
        "",
        "Paired two-sided Wilcoxon tests with BH-FDR over three comparisons:",
        pairwise.to_string(index=False),
    ]
    (output_dir / "wmti_gm_exclusion_retrain_audit.txt").write_text(
        "\n".join(audit_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(audit_lines), flush=True)


if __name__ == "__main__":
    main()
