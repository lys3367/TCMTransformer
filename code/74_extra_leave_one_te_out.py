from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd


common = importlib.import_module("00_common")
ridge_module = importlib.import_module("02_run_ridge")
neural_module = importlib.import_module("03_train_neural")
extra_module = importlib.import_module("73_train_extra_models")


def make_inner_split(indices: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    n_val = max(5, int(round(0.2 * len(shuffled))))
    return shuffled[n_val:], shuffled[:n_val]


def filter_subjects(subjects: np.ndarray, exclude_prefixes: list[str]) -> np.ndarray:
    keep = np.ones(len(subjects), dtype=bool)
    for prefix in exclude_prefixes:
        prefix = prefix.strip()
        if not prefix:
            continue
        keep &= ~np.char.startswith(subjects.astype(str), prefix)
    return np.where(keep)[0]


def run_one_target(
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
    settings = extra_module.TrainSettings(
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=seed,
        device=args.device,
    )
    true, pred, details = extra_module.train_one(
        args.model,
        raw,
        train_idx,
        val_idx,
        test_idx,
        input_idx,
        target_idx,
        repos_dir,
        settings,
    )
    return true, pred, float(details["val_RMSE"]), float(details["training_seconds"])


def append_roi_rows(
    roi_rows: list[pd.DataFrame],
    true: np.ndarray,
    pred: np.ndarray,
    metadata: pd.DataFrame,
    row_info: dict,
) -> None:
    by_roi = common.summarize_by_roi(true, pred, metadata)
    for key, value in reversed(list(row_info.items())):
        by_roi.insert(0, key, value)
    roi_rows.append(by_roi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-TE-out task: mask one TE and predict it from the remaining six TEs. "
            "This is an interpolation/TE-completion auxiliary task, not the main 75-115 -> 125/135 task."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["segrnn", "mtsmixer", "mole"],
    )
    parser.add_argument(
        "--grouping", default="full", choices=["full", "by_roi", "by_feature"]
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--mode",
        default="fixed",
        choices=["fixed", "loso"],
        help="fixed uses the existing train/val/test subject split; loso leaves out one subject at a time.",
    )
    parser.add_argument(
        "--target-te",
        nargs="*",
        type=int,
        default=None,
        help="Optional target TE list. Default: all available TE values.",
    )
    parser.add_argument(
        "--exclude-subject-prefix",
        nargs="*",
        default=[],
        help="Optional subject prefixes to exclude, e.g. 43_ 48_.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
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

    if args.target_te:
        target_tes = [int(te) for te in args.target_te]
    else:
        target_tes = [int(te) for te in te_values]

    rows: list[dict] = []
    roi_rows: list[pd.DataFrame] = []
    grouping = "full"

    if args.mode == "fixed":
        split_labels_all = common.load_split_labels(config, subjects_all)
        split_labels = split_labels_all[keep_idx]
        train_idx = np.where(split_labels == "train")[0]
        val_idx = np.where(split_labels == "val")[0]
        test_idx = np.where(split_labels == "test")[0]
        folds = [(1, train_idx, val_idx, test_idx)]
    else:
        folds = []
        for fold_number, test_idx in enumerate(np.arange(len(subjects)), start=1):
            outer_train = np.asarray(
                [idx for idx in range(len(subjects)) if idx != test_idx], dtype=int
            )
            train_idx, val_idx = make_inner_split(
                outer_train, base_seed + fold_number
            )
            folds.append((fold_number, train_idx, val_idx, np.asarray([test_idx])))

    for target_te in target_tes:
        target_idx = common.te_indices(te_values, [target_te])
        input_te = [int(te) for te in te_values if int(te) != int(target_te)]
        input_idx = common.te_indices(te_values, input_te)

        for fold_number, train_idx, val_idx, test_idx in folds:
            true, pred, val_rmse, training_seconds = run_one_target(
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
                "task": f"{args.mode}_leave_one_te_out",
                "target_te": int(target_te),
                "input_te": " ".join(str(te) for te in input_te),
                "fold": int(fold_number),
                "test_subject": subjects[test_idx[0]] if len(test_idx) == 1 else "fixed_test",
                "model": args.model,
                "grouping": grouping,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "val_RMSE": val_rmse,
                "training_seconds": training_seconds,
                **metrics,
            }
            rows.append(row)
            append_roi_rows(
                roi_rows,
                true,
                pred,
                metadata,
                {
                    "task": row["task"],
                    "target_te": int(target_te),
                    "fold": int(fold_number),
                    "test_subject": row["test_subject"],
                    "model": args.model,
                    "grouping": grouping,
                },
            )
            print(
                f"{args.mode} target_TE={target_te} fold={fold_number} "
                f"MAE={metrics['MAE']:.6f} RMSE={metrics['RMSE']:.6f}",
                flush=True,
            )

    result_table = pd.DataFrame(rows)
    stem = f"74_extra_{args.mode}_leave_one_te_out_{args.model}_{grouping}"
    if args.exclude_subject_prefix:
        stem += "_exclude_" + "_".join(prefix.rstrip("_") for prefix in args.exclude_subject_prefix)
    result_table.to_csv(output_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    if roi_rows:
        pd.concat(roi_rows, ignore_index=True).to_csv(
            output_dir / f"{stem}_by_roi.csv", index=False, encoding="utf-8-sig"
        )

    by_te = (
        result_table.groupby(["target_te", "model", "grouping"], as_index=False)[
            ["MAE", "RMSE", "R2"]
        ]
        .agg(["mean", "std"])
    )
    by_te.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in by_te.columns
    ]
    by_te.to_csv(output_dir / f"{stem}_by_te_summary.csv", index=False, encoding="utf-8-sig")

    overall = result_table[["MAE", "RMSE", "R2"]].mean()
    overall_std = result_table[["MAE", "RMSE", "R2"]].std(ddof=1)
    report = [
        f"# 74 Extra Leave-one-TE-out：{args.model}/{grouping}/{args.mode}",
        "",
        f"- 任务：每次遮盖 1 个 TE，用其余 6 个 TE 预测该 TE。",
        f"- 目标 TE：{target_tes}",
        f"- 受试者数：{len(subjects)}",
        f"- 排除受试者前缀：{args.exclude_subject_prefix or '无'}",
        f"- MAE：{overall['MAE']:.6f} ± {overall_std['MAE']:.6f}",
        f"- RMSE：{overall['RMSE']:.6f} ± {overall_std['RMSE']:.6f}",
        f"- R2：{overall['R2']:.6f} ± {overall_std['R2']:.6f}",
        "",
        "该任务用于评估模型能否从已观测 TE 响应曲线中补全缺失 TE，不能直接替代主任务的 75/85/95/105/115 -> 125/135 外推评估。",
    ]
    (output_dir / f"{stem}_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
