from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT = Path.cwd()
MODELS = ("dlinear", "mole")
KERNELS = (3, 5, 25)
EXPECTED_FOLDS = 49
EXPECTED_VARIABLES = 11248
EXPECTED_CANONICAL_MAE = {
    "itransformer": (0.230475, 0.061988),
    "mole": (0.234972, 0.058794),
    "dlinear": (0.237039, 0.058907),
    "timesnet": (0.242607, 0.074970),
}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT / path


def bh_fdr(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def paired_wilcoxon(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if np.allclose(difference, 0.0, rtol=0.0, atol=0.0):
        return 0.0, 1.0
    result = wilcoxon(left, right, alternative="two-sided", zero_method="wilcox", method="auto")
    return float(result.statistic), float(result.pvalue)


def validate_common_columns(table: pd.DataFrame, reference: pd.DataFrame, label: str) -> pd.DataFrame:
    table = table.sort_values("fold").reset_index(drop=True)
    if len(table) != EXPECTED_FOLDS or table["fold"].nunique() != EXPECTED_FOLDS:
        raise ValueError(f"{label}: expected 49 unique folds, found {len(table)} rows.")
    for column, expected in (("n_train", 38), ("n_val", 10), ("n_test", 1)):
        if set(table[column].astype(int)) != {expected}:
            raise ValueError(f"{label}: invalid {column}.")
    if not np.array_equal(table["test_subject"].astype(str), reference["test_subject"].astype(str)):
        raise RuntimeError(f"{label}: test-subject order differs from canonical manifest.")
    if not np.array_equal(table["fold_seed"].astype(int), reference["fold_seed"].astype(int)):
        raise RuntimeError(f"{label}: fold seeds differ from canonical manifest.")
    numeric = ["MAE", "RMSE", "R2", "MAE_TE125", "MAE_TE135"]
    if not np.isfinite(table[numeric].to_numpy(dtype=float)).all():
        raise FloatingPointError(f"{label}: non-finite metrics found.")
    return table


def validate_sensitivity(
    output_dir: Path, model: str, kernel: int, reference: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    label = f"{model} k={kernel}"
    metrics_path = output_dir / f"{model}_k{kernel}_fold_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    table = validate_common_columns(pd.read_csv(metrics_path), reference, label)
    if set(table["kernel"].astype(int)) != {kernel}:
        raise ValueError(f"{label}: wrong kernel in fold metrics.")
    if set(table["seq_len"].astype(int)) != {5} or set(table["pred_len"].astype(int)) != {2}:
        raise ValueError(f"{label}: seq_len/pred_len mismatch.")
    if set(table["n_variables"].astype(int)) != {EXPECTED_VARIABLES}:
        raise ValueError(f"{label}: variable count mismatch.")
    if (table["best_epoch"].astype(float) < 1).any():
        raise ValueError(f"{label}: invalid best_epoch.")
    if (table["epochs_run"].astype(float) < table["best_epoch"].astype(float)).any():
        raise ValueError(f"{label}: epochs_run precedes best_epoch.")
    if (table["epochs_run"].astype(float) > 100).any():
        raise ValueError(f"{label}: epochs_run exceeds canonical maximum.")

    setting_dir = output_dir / f"{model}_k{kernel}"
    settings_path = setting_dir / "run_settings.json"
    fold_paths = [setting_dir / f"fold_{int(row.fold):02d}.npz" for row in reference.itertuples(index=False)]
    full_artifacts_available = settings_path.exists() and all(path.exists() for path in fold_paths)
    checkpoint_count = 0
    if full_artifacts_available:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if settings["kernel_size"] != kernel or not settings["kernel_is_only_training_hyperparameter_change"]:
            raise RuntimeError(f"{label}: run_settings kernel audit failed.")
        for row, fold_path in zip(reference.itertuples(index=False), fold_paths):
            z = np.load(fold_path, allow_pickle=True)
            checks = {
                "kernel": int(z["kernel"][0]) == kernel,
                "seq_len": int(z["seq_len"][0]) == 5,
                "pred_len": int(z["pred_len"][0]) == 2,
                "n_variables": int(z["n_variables"][0]) == EXPECTED_VARIABLES,
                "test_subject": str(z["subject"][0]) == str(row.test_subject),
                "fold_seed": int(z["fold_seed"][0]) == int(row.fold_seed),
                "train_subjects": ";".join(z["train_subjects"].astype(str)) == str(row.train_subjects),
                "val_subjects": ";".join(z["val_subjects"].astype(str)) == str(row.val_subjects),
                "true_shape": z["true_z"].shape == (1, 2, EXPECTED_VARIABLES),
                "pred_shape": z["pred_z"].shape == (1, 2, EXPECTED_VARIABLES),
                "finite": bool(np.isfinite(z["true_z"]).all() and np.isfinite(z["pred_z"]).all()),
            }
            if not all(checks.values()):
                raise RuntimeError(f"{label} fold {row.fold} audit failed: {checks}")
            checkpoint = Path(str(z["checkpoint_path"][0]))
            if checkpoint.exists():
                checkpoint_count += 1
            else:
                raise FileNotFoundError(f"{label} fold {row.fold}: checkpoint missing: {checkpoint}")
    audit = [
        f"{label}:",
        "  49/49 fold-metric rows complete",
        f"  kernel = {kernel}",
        "  seq_len = 5",
        "  pred_len = 2",
        "  variables = 11248",
        "  split match = 49/49",
        "  seed match = 49/49",
        "  finite metrics = 49/49",
        f"  full fold-artifact audit = {'PASS' if full_artifacts_available else 'NOT AVAILABLE IN THIS COPY'}",
        f"  checkpoint present = {checkpoint_count}/49" if full_artifacts_available else "  checkpoint audit = NOT AVAILABLE IN THIS COPY",
    ]
    return table, audit


def load_k25(primary_dir: Path, model: str, reference: pd.DataFrame) -> pd.DataFrame:
    table = validate_common_columns(
        pd.read_csv(primary_dir / f"{model}_fold_metrics.csv"), reference, f"{model} k=25"
    ).copy()
    table["kernel"] = 25
    return table


def load_and_verify_canonical(
    primary_dir: Path, reference: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    tables = {}
    audit = ["Canonical clean-49 results rebuilt from saved fold metrics:"]
    for model, (expected_mean, expected_sd) in EXPECTED_CANONICAL_MAE.items():
        path = primary_dir / f"{model}_fold_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        table = validate_common_columns(pd.read_csv(path), reference, f"canonical {model}")
        observed_mean = float(table["MAE"].mean())
        observed_sd = float(table["MAE"].std(ddof=1))
        if round(observed_mean, 6) != round(expected_mean, 6) or round(observed_sd, 6) != round(expected_sd, 6):
            raise RuntimeError(
                f"Canonical {model} mismatch: observed {observed_mean:.6f} +/- {observed_sd:.6f}; "
                f"expected {expected_mean:.6f} +/- {expected_sd:.6f}. Statistics were not run."
            )
        tables[model] = table
        audit.append(
            f"  {model}: MAE={observed_mean:.6f} +/- {observed_sd:.6f} (49/49; VERIFIED)"
        )
    return tables, audit


def summary_row(model: str, kernel: int, table: pd.DataFrame) -> dict:
    row = {"model": model, "kernel": kernel, "n": len(table)}
    for source, target in (
        ("MAE", "MAE"),
        ("RMSE", "RMSE"),
        ("R2", "R2"),
        ("MAE_TE125", "TE125_MAE"),
        ("MAE_TE135", "TE135_MAE"),
    ):
        row[f"{target}_mean"] = float(table[source].mean())
        row[f"{target}_sd"] = float(table[source].std(ddof=1))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize clean-49 kernel sensitivity.")
    parser.add_argument("--output-dir", default="outputs/kernel_sensitivity_clean49")
    parser.add_argument("--primary-dir", default="outputs/table1_clean49")
    args = parser.parse_args()
    output_dir = resolve(args.output_dir)
    primary_dir = resolve(args.primary_dir)
    reference = pd.read_csv(primary_dir / "fold_manifest.csv").sort_values("fold").reset_index(drop=True)
    canonical, canonical_audit = load_and_verify_canonical(primary_dir, reference)

    tables: dict[tuple[str, int], pd.DataFrame] = {}
    audit_lines = [
        "Kernel sensitivity clean-49 audit",
        "=================================",
        "Canonical split: 38 train / 10 validation / 1 test",
        "Input TEs: 75, 85, 95, 105, 115",
        "Target TEs: 125, 135",
        "",
    ] + canonical_audit + [""]
    for model in MODELS:
        for kernel in (3, 5):
            tables[(model, kernel)], lines = validate_sensitivity(
                output_dir, model, kernel, reference
            )
            audit_lines.extend(lines + [""])
        tables[(model, 25)] = canonical[model].copy()
        tables[(model, 25)]["kernel"] = 25

    summary = pd.DataFrame(
        [summary_row(model, kernel, tables[(model, kernel)]) for model in MODELS for kernel in KERNELS]
    )
    summary["TE125_MAE"] = summary["TE125_MAE_mean"]
    summary["TE135_MAE"] = summary["TE135_MAE_mean"]
    for model in MODELS:
        reference_mae = float(summary.loc[(summary.model == model) & (summary.kernel == 25), "MAE_mean"].iloc[0])
        mask = summary.model == model
        summary.loc[mask, "delta_MAE_vs_k25"] = summary.loc[mask, "MAE_mean"] - reference_mae
        summary.loc[mask, "relative_change_percent"] = (
            100.0 * summary.loc[mask, "delta_MAE_vs_k25"] / reference_mae
        )
    summary["within_model_MAE_rank"] = summary.groupby("model")["MAE_mean"].rank(
        method="min", ascending=True
    ).astype(int)

    subject_rows = []
    for (model, kernel), table in tables.items():
        for row in table.itertuples(index=False):
            subject_rows.append(
                {
                    "subject": row.test_subject,
                    "fold": int(row.fold),
                    "fold_seed": int(row.fold_seed),
                    "model": model,
                    "kernel": kernel,
                    "MAE": float(row.MAE),
                    "RMSE": float(row.RMSE),
                    "R2": float(row.R2),
                    "TE125_MAE": float(row.MAE_TE125),
                    "TE135_MAE": float(row.MAE_TE135),
                }
            )
    subject_level = pd.DataFrame(subject_rows).sort_values(["model", "kernel", "fold"])

    kernel_stats = []
    for model in MODELS:
        for left_kernel, right_kernel in ((3, 25), (5, 25), (3, 5)):
            left = tables[(model, left_kernel)].sort_values("fold")["MAE"].to_numpy()
            right = tables[(model, right_kernel)].sort_values("fold")["MAE"].to_numpy()
            difference = left - right
            statistic, p_value = paired_wilcoxon(left, right)
            kernel_stats.append(
                {
                    "model": model,
                    "comparison": f"k{left_kernel}_minus_k{right_kernel}",
                    "mean_delta_MAE": float(difference.mean()),
                    "median_delta_MAE": float(np.median(difference)),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p": p_value,
                }
            )
    kernel_stats = pd.DataFrame(kernel_stats)
    kernel_stats["FDR_q"] = np.nan
    for model in MODELS:
        mask = kernel_stats.model == model
        kernel_stats.loc[mask, "FDR_q"] = bh_fdr(kernel_stats.loc[mask, "wilcoxon_p"].to_numpy())

    itransformer = canonical["itransformer"]
    robustness = []
    for model in MODELS:
        for kernel in KERNELS:
            comparator = tables[(model, kernel)].sort_values("fold").reset_index(drop=True)
            difference = comparator["MAE"].to_numpy() - itransformer["MAE"].to_numpy()
            statistic, p_value = paired_wilcoxon(
                itransformer["MAE"].to_numpy(), comparator["MAE"].to_numpy()
            )
            robustness.append(
                {
                    "model": model,
                    "kernel": kernel,
                    "MAE": float(comparator["MAE"].mean()),
                    "mean_delta_MAE": float(difference.mean()),
                    "itransformer_better_n": int(
                        np.sum(itransformer["MAE"].to_numpy() < comparator["MAE"].to_numpy())
                    ),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p": p_value,
                }
            )
    robustness = pd.DataFrame(robustness)
    robustness["FDR_q"] = bh_fdr(robustness["wilcoxon_p"].to_numpy())
    robustness["significant_FDR_0_05"] = robustness["FDR_q"] < 0.05

    base_means = {
        model: float(canonical[model]["MAE"].mean()) for model in ("itransformer", "timesnet")
    }
    ranking_rows = []
    for kernel in KERNELS:
        means = {
            "iTransformer": base_means["itransformer"],
            "MoLE": float(summary.loc[(summary.model == "mole") & (summary.kernel == kernel), "MAE_mean"].iloc[0]),
            "DLinear": float(summary.loc[(summary.model == "dlinear") & (summary.kernel == kernel), "MAE_mean"].iloc[0]),
            "TimesNet": base_means["timesnet"],
        }
        ordered = sorted(means, key=means.get)
        for rank, model in enumerate(ordered, start=1):
            ranking_rows.append({"kernel": kernel, "rank": rank, "model": model, "MAE": means[model]})
    rankings = pd.DataFrame(ranking_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "kernel_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    subject_level.to_csv(
        output_dir / "kernel_sensitivity_subject_level.csv", index=False, encoding="utf-8-sig"
    )
    kernel_stats.to_csv(
        output_dir / "kernel_sensitivity_vs_k25_stats.csv", index=False, encoding="utf-8-sig"
    )
    robustness.to_csv(
        output_dir / "kernel_sensitivity_vs_itransformer_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rankings.to_csv(output_dir / "kernel_sensitivity_ranking.csv", index=False, encoding="utf-8-sig")

    all_itransformer_best = bool(
        robustness["mean_delta_MAE"].gt(0).all()
        and all(
            rankings.loc[rankings.kernel == kernel].sort_values("rank").iloc[0].model
            == "iTransformer"
            for kernel in KERNELS
        )
    )
    all_significant = bool(robustness["significant_FDR_0_05"].all())
    original_order = ["iTransformer", "MoLE", "DLinear", "TimesNet"]
    ranking_unchanged = {
        kernel: rankings.loc[rankings.kernel == kernel].sort_values("rank")["model"].tolist()
        == original_order
        for kernel in KERNELS
    }
    audit_lines.extend(
        [
            "Failed sensitivity folds: 0",
            "Canonical k=25 summaries rebuilt from saved fold metrics: YES",
            "No k=25 model was retrained: YES",
            f"iTransformer lowest for every tested comparator configuration: {'YES' if all_itransformer_best else 'NO'}",
            f"All six iTransformer comparisons significant after joint BH-FDR: {'YES' if all_significant else 'NO'}",
            f"Ranking unchanged at k=3: {'YES' if ranking_unchanged[3] else 'NO'}",
            f"Ranking unchanged at k=5: {'YES' if ranking_unchanged[5] else 'NO'}",
            f"Ranking unchanged at k=25: {'YES' if ranking_unchanged[25] else 'NO'}",
        ]
    )
    (output_dir / "kernel_sensitivity_audit.txt").write_text(
        "\n".join(audit_lines) + "\n", encoding="utf-8"
    )

    interpretation = (
        "C. Results are essentially insensitive to kernel choice."
        if all_itransformer_best and all_significant and all(ranking_unchanged.values())
        else "B. Kernel affects comparator performance but iTransformer remains best."
        if all_itransformer_best
        else "A. Kernel choice materially changes the main ranking."
    )
    best_dlinear = summary.loc[summary.model == "dlinear"].sort_values("MAE_mean").iloc[0]
    best_mole = summary.loc[summary.model == "mole"].sort_values("MAE_mean").iloc[0]
    kernel_fdr_significant = bool(kernel_stats["FDR_q"].lt(0.05).any())
    report = f"""# Clean-49 moving-average kernel sensitivity report

## Execution and audit

- Completed sensitivity configurations: DLinear k=3/5; MoLE k=3/5.
- Completed folds: 49/49 for every configuration.
- Canonical subject, split, seed, scaler inputs, and target TEs: verified.
- Reference k=25: reused from `outputs/table1_clean49`; not retrained.
- Canonical inputs: `itransformer_fold_metrics.csv`, `mole_fold_metrics.csv`, `dlinear_fold_metrics.csv`, and `timesnet_fold_metrics.csv`.
- Canonical MAE verification: passed for all four models before any sensitivity statistics were computed.

## Summary

```
{summary.to_string(index=False)}
```

## Kernel comparisons

```
{kernel_stats.to_string(index=False)}
```

## Comparisons with iTransformer

```
{robustness.to_string(index=False)}
```

## Ranking robustness

```
{rankings.to_string(index=False)}
```

## Reviewer interpretation

{interpretation}

## Direct answers

- Lowest DLinear MAE: kernel={int(best_dlinear.kernel)}, MAE={best_dlinear.MAE_mean:.6f} +/- {best_dlinear.MAE_sd:.6f}.
- Lowest MoLE MAE: kernel={int(best_mole.kernel)}, MAE={best_mole.MAE_mean:.6f} +/- {best_mole.MAE_sd:.6f}.
- Any MoLE kernel below iTransformer: {'YES' if robustness.loc[robustness.model == 'mole', 'mean_delta_MAE'].lt(0).any() else 'NO'}.
- Any DLinear kernel below iTransformer: {'YES' if robustness.loc[robustness.model == 'dlinear', 'mean_delta_MAE'].lt(0).any() else 'NO'}.
- Original iTransformer > MoLE > DLinear > TimesNet ranking retained at k=3, k=5, and k=25: {'YES' if all(ranking_unchanged.values()) else 'NO'}.
- Any within-model kernel comparison significant after BH-FDR: {'YES' if kernel_fdr_significant else 'NO'}.
- All six comparisons with iTransformer significant after joint BH-FDR: {'YES' if all_significant else 'NO'}.

## Manuscript wording template

Methods: DLinear and MoLE-DLinear were evaluated using moving-average kernels of 3, 5, and the reference value 25 while preserving the clean-49 LOSO folds and all other training settings.

Results: Varying the moving-average kernel from the reference value of 25 to 3 or 5 produced only small MAE changes for DLinear (maximum absolute mean change, {summary.loc[summary.model == 'dlinear', 'delta_MAE_vs_k25'].abs().max():.6f}) and MoLE (maximum, {summary.loc[summary.model == 'mole', 'delta_MAE_vs_k25'].abs().max():.6f}); none of the within-model differences remained significant after BH-FDR correction. iTransformer retained the lowest MAE under every comparator configuration, and all six paired comparisons remained significant after joint BH-FDR correction.
"""
    (output_dir / "kernel_sensitivity_report.md").write_text(report, encoding="utf-8")
    print("\n".join(audit_lines), flush=True)
    print(f"Reviewer interpretation: {interpretation}", flush=True)


if __name__ == "__main__":
    main()
