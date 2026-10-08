from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd


common = importlib.import_module("00_common")
pretrain = importlib.import_module("61_masked_te_pretrain_itransformer")
neural = importlib.import_module("03_train_neural")


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_resamples: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, 1000):
        size = min(1000, n_resamples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start : start + size] = values[indices].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).astype(float))


def wilcoxon_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.allclose(values, 0.0):
        return np.nan
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(values, alternative="two-sided").pvalue)
    except ImportError:
        return np.nan


def fit_condition(
    condition: str,
    x: np.ndarray,
    observed_mask: np.ndarray,
    pretrain_idx: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    target_idx: np.ndarray,
    repos_dir: Path,
    device,
    args: argparse.Namespace,
    fold_seed: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    target_te_indices = list(range(x.shape[1]))
    pretrain_x, pretrain_y, _ = pretrain.make_masked_samples(
        x, observed_mask, pretrain_idx, target_te_indices, args.min_context
    )
    val_mask_x, val_mask_y, _ = pretrain.make_masked_samples(
        x, observed_mask, val_idx, target_te_indices, args.min_context
    )

    common.set_seed(fold_seed)
    pretrain_model = neural.build_model(
        "itransformer", x.shape[1], 1, x.shape[2], repos_dir
    )
    pretrain_model, pretrain_details = pretrain.train_model(
        pretrain_model,
        pretrain_x,
        pretrain_y,
        val_mask_x,
        val_mask_y,
        device,
        args.batch_size,
        args.pretrain_epochs,
        args.patience,
        lr=7e-4,
        weight_decay=1e-4,
    )

    train_x, train_y = pretrain.make_main_samples(x, train_idx, target_idx)
    val_x, val_y = pretrain.make_main_samples(x, val_idx, target_idx)
    test_x, test_y = pretrain.make_main_samples(x, test_idx, target_idx)

    # Reset before fine-tuning so B and C receive the same new output-layer initialization.
    common.set_seed(fold_seed + 100000)
    finetune_model = neural.build_model(
        "itransformer", x.shape[1], len(target_idx), x.shape[2], repos_dir
    )
    loaded_keys = pretrain.load_compatible_state(
        finetune_model, pretrain_model.state_dict()
    )
    finetune_model, finetune_details = pretrain.train_model(
        finetune_model,
        train_x,
        train_y,
        val_x,
        val_y,
        device,
        args.batch_size,
        args.finetune_epochs,
        args.patience,
        lr=5e-4,
        weight_decay=1e-4,
    )
    prediction = pretrain.predict(finetune_model, test_x, device, args.batch_size)
    metrics = common.regression_metrics(test_y, prediction)
    details = {
        "condition": condition,
        "n_pretrain_subjects": len(pretrain_idx),
        "n_pretrain_samples": len(pretrain_x),
        "loaded_pretrain_keys": loaded_keys,
        "pretrain_val_RMSE": pretrain_details["val_RMSE"],
        "finetune_val_RMSE": finetune_details["val_RMSE"],
        **metrics,
    }
    return details, test_y, prediction


def make_report(table: pd.DataFrame, output: Path, seed: int) -> None:
    wide = table.pivot(index=["fold", "test_subject"], columns="condition")
    paired = pd.DataFrame(
        {
            "fold": wide.index.get_level_values("fold"),
            "participant_id": wide.index.get_level_values("test_subject"),
            "B_MAE": wide["MAE"]["B_complete_only"].to_numpy(),
            "B_RMSE": wide["RMSE"]["B_complete_only"].to_numpy(),
            "B_R2": wide["R2"]["B_complete_only"].to_numpy(),
            "C_MAE": wide["MAE"]["C_complete_plus_incomplete"].to_numpy(),
            "C_RMSE": wide["RMSE"]["C_complete_plus_incomplete"].to_numpy(),
            "C_R2": wide["R2"]["C_complete_plus_incomplete"].to_numpy(),
        }
    )
    paired["delta_MAE_B_minus_C"] = paired["B_MAE"] - paired["C_MAE"]
    paired.to_csv(output.with_name(output.stem + "_paired.csv"), index=False, encoding="utf-8-sig")

    delta = paired["delta_MAE_B_minus_C"].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_mean_ci(delta, seed)
    lines = ["# 91 Matched incomplete-TE pretraining control", ""]
    for label, prefix in [("B complete-only", "B"), ("C complete + incomplete", "C")]:
        lines.append(
            f"- {label}: MAE={paired[f'{prefix}_MAE'].mean():.6f} +/- "
            f"{paired[f'{prefix}_MAE'].std(ddof=1):.6f}; "
            f"RMSE={paired[f'{prefix}_RMSE'].mean():.6f}; R2={paired[f'{prefix}_R2'].mean():.6f}"
        )
    lines.extend(
        [
            f"- Paired delta MAE (B-C): {np.mean(delta):.6f}",
            f"- Participant bootstrap 95% CI: [{ci_low:.6f}, {ci_high:.6f}]",
            f"- C wins: {np.mean(delta > 0) * 100:.1f}%",
            f"- Two-sided paired Wilcoxon p: {wilcoxon_pvalue(delta):.6g}",
            "",
            "Positive delta means that adding incomplete-TE participants reduced MAE.",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched B/C masked-TE pretraining ablation.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--incomplete-matrix", default="data/processed/60_model_matrix_incomplete_te.npz")
    parser.add_argument("--exclude-subject-prefix", nargs="*", default=["43_", "48_"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-context", type=int, default=2)
    parser.add_argument("--max-folds", type=int, default=1, help="Use 1 for smoke test; 0 for all folds.")
    parser.add_argument("--output-stem", default="91_pretraining_matched_control")
    args = parser.parse_args()

    config = common.load_config(args.config)
    matrix_path = Path(args.incomplete_matrix)
    if not matrix_path.is_absolute():
        matrix_path = common.ROOT / matrix_path
    matrix = np.load(matrix_path, allow_pickle=True)
    raw_all = matrix["x_raw"].astype(np.float32)
    observed_all = matrix["observed_mask"].astype(bool)
    subjects_all = matrix["subjects"].astype(str)
    complete_all = matrix["complete7_mask"].astype(bool)
    te_values = matrix["te_values"].astype(int)

    keep = pretrain.filter_subjects(subjects_all, args.exclude_subject_prefix)
    raw = raw_all[keep]
    observed_mask = observed_all[keep]
    subjects = subjects_all[keep]
    complete_mask = complete_all[keep]
    incomplete_idx = np.where(~complete_mask)[0]
    seed = int(config["random_seed"])
    folds = pretrain.loso_folds(subjects, complete_mask, seed)
    if args.max_folds > 0:
        folds = folds[: args.max_folds]

    if int(complete_mask.sum()) != 49 or len(incomplete_idx) != 17:
        raise ValueError(
            f"Expected 49 complete and 17 incomplete participants; got "
            f"{int(complete_mask.sum())} and {len(incomplete_idx)}."
        )

    target_idx = common.te_indices(te_values, config["fixed_target_te"])
    repos_dir = common.resolve_path(config, "external_repos_dir")
    output_dir = common.resolve_path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = importlib.import_module("torch").device(args.device)
    rows = []

    for fold, _, train_idx, val_idx, test_idx in folds:
        if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
            raise RuntimeError(f"Fold {fold} contains participant leakage.")

        # Both conditions use the scaler estimated from complete outer-fold training participants only.
        x, mean, std = pretrain.nan_standardize_from_train(raw, observed_mask, train_idx)
        condition_indices = {
            "B_complete_only": train_idx,
            "C_complete_plus_incomplete": np.concatenate([train_idx, incomplete_idx]).astype(int),
        }
        for condition, pretrain_idx in condition_indices.items():
            if set(pretrain_idx) & set(val_idx) or set(pretrain_idx) & set(test_idx):
                raise RuntimeError(f"Fold {fold} {condition} leaks validation/test participants.")
            details, true, prediction = fit_condition(
                condition,
                x,
                observed_mask,
                pretrain_idx,
                train_idx,
                val_idx,
                test_idx,
                target_idx,
                repos_dir,
                device,
                args,
                seed + fold,
            )
            if true.shape != (1, 2, raw.shape[2]) or prediction.shape != true.shape:
                raise RuntimeError(f"Unexpected output shape in fold {fold}: {true.shape}, {prediction.shape}")
            rows.append(
                {
                    "fold": fold,
                    "test_subject": subjects[test_idx[0]],
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "n_test": len(test_idx),
                    "scaler_subjects": len(train_idx),
                    "scaler_mean_checksum": float(np.sum(mean, dtype=np.float64)),
                    "scaler_std_checksum": float(np.sum(std, dtype=np.float64)),
                    **details,
                }
            )
            print(
                f"fold={fold} {condition} test={subjects[test_idx[0]]} "
                f"MAE={details['MAE']:.6f}",
                flush=True,
            )

    suffix = f"_smoke{len(folds)}" if args.max_folds > 0 else "_loso49"
    output = output_dir / f"{args.output_stem}{suffix}.csv"
    table = pd.DataFrame(rows)
    table.to_csv(output, index=False, encoding="utf-8-sig")
    make_report(table, output, seed)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
