from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd


common = importlib.import_module("teprediction.common")
neural_module = importlib.import_module("teprediction.models")


BASE_MODELS = {
    "itransformer": "iTransformer",
    "timesnet": "TimesNet",
    "dlinear": "DLinear",
}


def make_inner_split(indices: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    n_val = max(5, int(round(0.2 * len(shuffled))))
    return shuffled[n_val:], shuffled[:n_val]


def filter_subjects(subjects: np.ndarray, exclude_prefixes: list[str]) -> np.ndarray:
    keep = np.ones(len(subjects), dtype=bool)
    for prefix in exclude_prefixes:
        prefix = prefix.strip()
        if prefix:
            keep &= ~np.char.startswith(subjects.astype(str), prefix)
    return np.where(keep)[0]


def run_model(
    args: argparse.Namespace,
    raw: np.ndarray,
    metadata: pd.DataFrame,
    repos_dir: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    input_idx: np.ndarray,
    target_idx: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    settings = neural_module.TrainSettings(
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=seed,
        device=args.device,
    )
    true, pred, group_details = neural_module.run_neural(
        args.model,
        "full",
        raw,
        metadata,
        train_idx,
        val_idx,
        test_idx,
        input_idx,
        target_idx,
        repos_dir,
        settings,
    )
    return (
        true,
        pred,
        float(group_details["val_RMSE"].mean()),
        float(group_details["training_seconds"].sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="5x task: leave-one-TE-out for four base neural models only."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(BASE_MODELS),
        help="Base neural model. Ridge is intentionally not included in this 5x task.",
    )
    parser.add_argument("--mode", required=True, choices=["fixed", "loso"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--target-te", nargs="*", type=int, default=None)
    parser.add_argument("--exclude-subject-prefix", nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = common.load_config(args.config)
    matrix = common.load_matrix(config)
    raw_all = matrix["x_raw"].astype(np.float32)
    subjects_all = matrix["subjects"].astype(str)
    te_values = matrix["te_values"].astype(int)
    metadata = pd.read_csv(common.resolve_path(config, "metadata_file"))
    repos_dir = common.resolve_path(config, "external_repos_dir")
    output_dir = common.resolve_path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_seed = int(config["random_seed"])

    keep_idx = filter_subjects(subjects_all, args.exclude_subject_prefix)
    raw = raw_all[keep_idx]
    subjects = subjects_all[keep_idx]
    target_tes = [int(te) for te in (args.target_te or te_values)]

    if args.mode == "fixed":
        split_labels_all = common.load_split_labels(config, subjects_all)
        split_labels = split_labels_all[keep_idx]
        folds = [
            (
                1,
                np.where(split_labels == "train")[0],
                np.where(split_labels == "val")[0],
                np.where(split_labels == "test")[0],
            )
        ]
    else:
        folds = []
        for fold_number, test_index in enumerate(np.arange(len(subjects)), start=1):
            outer_train = np.asarray(
                [idx for idx in range(len(subjects)) if idx != test_index], dtype=int
            )
            train_idx, val_idx = make_inner_split(outer_train, base_seed + fold_number)
            folds.append((fold_number, train_idx, val_idx, np.asarray([test_index])))

    rows: list[dict] = []
    roi_rows: list[pd.DataFrame] = []

    for target_te in target_tes:
        target_idx = common.te_indices(te_values, [target_te])
        input_te = [int(te) for te in te_values if int(te) != int(target_te)]
        input_idx = common.te_indices(te_values, input_te)

        for fold_number, train_idx, val_idx, test_idx in folds:
            true, pred, val_rmse, training_seconds = run_model(
                args,
                raw,
                metadata,
                repos_dir,
                train_idx,
                val_idx,
                test_idx,
                input_idx,
                target_idx,
                base_seed + fold_number * 100 + int(target_te),
            )
            metrics = common.regression_metrics(true, pred)
            row = {
                "task": f"5x_{args.mode}_leave_one_te_out",
                "target_te": int(target_te),
                "input_te": " ".join(str(te) for te in input_te),
                "fold": int(fold_number),
                "test_subject": subjects[test_idx[0]] if len(test_idx) == 1 else "fixed_test",
                "model": args.model,
                "model_name": BASE_MODELS[args.model],
                "grouping": "full",
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "val_RMSE": val_rmse,
                "training_seconds": training_seconds,
                **metrics,
            }
            rows.append(row)

            by_roi = common.summarize_by_roi(true, pred, metadata)
            by_roi.insert(0, "task", row["task"])
            by_roi.insert(1, "target_te", int(target_te))
            by_roi.insert(2, "fold", int(fold_number))
            by_roi.insert(3, "test_subject", row["test_subject"])
            by_roi.insert(4, "model", args.model)
            by_roi.insert(5, "model_name", BASE_MODELS[args.model])
            by_roi.insert(6, "grouping", "full")
            roi_rows.append(by_roi)

            print(
                f"{args.model}/full {args.mode} target_TE={target_te} "
                f"fold={fold_number} MAE={metrics['MAE']:.6f} RMSE={metrics['RMSE']:.6f}",
                flush=True,
            )

    result_table = pd.DataFrame(rows)
    suffix = ""
    if args.exclude_subject_prefix:
        suffix = "_exclude_" + "_".join(prefix.rstrip("_") for prefix in args.exclude_subject_prefix)
    stem = f"51_{args.mode}_leave_one_te_out_{args.model}_full{suffix}"
    result_table.to_csv(output_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    pd.concat(roi_rows, ignore_index=True).to_csv(
        output_dir / f"{stem}_by_roi.csv", index=False, encoding="utf-8-sig"
    )

    by_te = (
        result_table.groupby(["target_te", "model", "model_name", "grouping"], as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            MAE_sd=("MAE", "std"),
            RMSE=("RMSE", "mean"),
            RMSE_sd=("RMSE", "std"),
            R2=("R2", "mean"),
            R2_sd=("R2", "std"),
            n=("MAE", "size"),
        )
        .sort_values("target_te")
    )
    by_te.to_csv(output_dir / f"{stem}_by_te_summary.csv", index=False, encoding="utf-8-sig")

    means = result_table[["MAE", "RMSE", "R2"]].mean()
    stds = result_table[["MAE", "RMSE", "R2"]].std(ddof=1)
    report = [
        f"# 51 {BASE_MODELS[args.model]} full {args.mode} leave-one-TE-out",
        "",
        "- 任务：每次遮盖 1 个 TE，用其余 6 个 TE 预测该 TE。",
        "- 模型范围：基础 4 个神经模型之一，不包含 Ridge。",
        f"- 目标 TE：{target_tes}",
        f"- 受试者数：{len(subjects)}",
        f"- 排除受试者前缀：{args.exclude_subject_prefix or '无'}",
        f"- MAE：{means['MAE']:.6f} ± {stds['MAE']:.6f}",
        f"- RMSE：{means['RMSE']:.6f} ± {stds['RMSE']:.6f}",
        f"- R2：{means['R2']:.6f} ± {stds['R2']:.6f}",
    ]
    (output_dir / f"{stem}_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
