from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch


common = importlib.import_module("teprediction.common")
neural = importlib.import_module("teprediction.models")
masked = importlib.import_module("teprediction.pretrain_mole")


MODELS = ["timesnet", "dlinear"]


def load_pretrained_state(
    target: torch.nn.Module,
    source_state: dict[str, torch.Tensor],
    model_name: str,
) -> tuple[int, int]:
    target_state = target.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    adapted = 0
    for key, value in source_state.items():
        if key not in target_state:
            continue
        target_value = target_state[key]
        if tuple(value.shape) == tuple(target_value.shape):
            compatible[key] = value
            continue
        if (
            model_name == "dlinear"
            and value.ndim >= 1
            and value.shape[0] == 1
            and target_value.shape[0] == 2
            and tuple(value.shape[1:]) == tuple(target_value.shape[1:])
        ):
            repeats = [1] * value.ndim
            repeats[0] = 2
            compatible[key] = value.repeat(*repeats)
            adapted += 1
    target_state.update(compatible)
    target.load_state_dict(target_state)
    return len(compatible), adapted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Masked-TE pretraining and main-task fine-tuning for remaining base models."
    )
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--incomplete-matrix",
        default="data/processed/model_matrix_incomplete_te.npz",
    )
    parser.add_argument("--mode", choices=["fixed", "loso"], default="fixed")
    parser.add_argument("--exclude-subject-prefix", nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-context", type=int, default=2)
    parser.add_argument("--loso-max-folds", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = common.load_config(args.config)
    matrix_path = Path(args.incomplete_matrix)
    if not matrix_path.is_absolute():
        matrix_path = common.ROOT / matrix_path
    matrix = np.load(matrix_path, allow_pickle=True)
    raw_all = matrix["x_raw"].astype(np.float32)
    observed_all = matrix["observed_mask"].astype(bool)
    subjects_all = matrix["subjects"].astype(str)
    te_values = matrix["te_values"].astype(int)
    complete_all = matrix["complete7_mask"].astype(bool)
    metadata = pd.read_csv(common.resolve_path(config, "metadata_file"))
    repos_dir = common.resolve_path(config, "external_repos_dir")
    output_dir = common.resolve_path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["random_seed"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    keep = masked.filter_subjects(subjects_all, args.exclude_subject_prefix)
    raw = raw_all[keep]
    observed_mask = observed_all[keep]
    subjects = subjects_all[keep]
    complete7_mask = complete_all[keep]
    target_idx = common.te_indices(te_values, config["fixed_target_te"])
    target_te_indices = list(range(len(te_values)))

    if args.mode == "fixed":
        folds = masked.fixed_folds(config, subjects, complete7_mask)
    else:
        folds = masked.loso_folds(subjects, complete7_mask, seed)
        if args.loso_max_folds > 0:
            folds = folds[: args.loso_max_folds]

    suffix = ""
    if args.exclude_subject_prefix:
        suffix = "_exclude_" + "_".join(
            prefix.rstrip("_") for prefix in args.exclude_subject_prefix
        )
    stem = f"masked_te_pretrain_base_{args.model}_full_{args.mode}{suffix}"
    result_path = output_dir / f"{stem}.csv"
    roi_path = output_dir / f"{stem}_by_roi.csv"
    rows: list[dict] = []
    roi_frames: list[pd.DataFrame] = []
    completed: set[int] = set()
    if args.resume and result_path.exists() and roi_path.exists():
        existing = pd.read_csv(result_path)
        rows = existing.to_dict("records")
        completed = set(existing["fold"].astype(int))
        roi_frames.append(pd.read_csv(roi_path))
    elif args.resume and (result_path.exists() or roi_path.exists()):
        print(
            "Resume needs both fold-level and ROI-level CSV files; rerun from fold 1.",
            flush=True,
        )

    for fold_number, pretrain_idx, train_idx, val_idx, test_idx in folds:
        if fold_number in completed:
            print(f"{args.model} {args.mode} fold={fold_number} already complete; skip", flush=True)
            continue
        common.set_seed(seed + fold_number)
        standardize_idx = np.unique(np.concatenate([pretrain_idx, train_idx]))
        x, _, _ = masked.nan_standardize_from_train(
            raw, observed_mask, standardize_idx
        )
        pretrain_x, pretrain_y, _ = masked.make_masked_samples(
            x,
            observed_mask,
            pretrain_idx,
            target_te_indices,
            args.min_context,
        )
        val_mask_x, val_mask_y, _ = masked.make_masked_samples(
            x, observed_mask, val_idx, target_te_indices, args.min_context
        )
        pretrain_model = neural.build_model(
            args.model,
            seq_len=len(te_values),
            pred_len=1,
            n_variables=x.shape[2],
            repos_dir=repos_dir,
        )
        pretrain_model, pretrain_details = masked.train_model(
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
        source_state = {
            key: value.detach().cpu().clone()
            for key, value in pretrain_model.state_dict().items()
        }
        del pretrain_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_x, train_y = masked.make_main_samples(x, train_idx, target_idx)
        val_x, val_y = masked.make_main_samples(x, val_idx, target_idx)
        test_x, test_y = masked.make_main_samples(x, test_idx, target_idx)
        common.set_seed(seed + 10_000 + fold_number)
        finetune_model = neural.build_model(
            args.model,
            seq_len=len(te_values),
            pred_len=len(target_idx),
            n_variables=x.shape[2],
            repos_dir=repos_dir,
        )
        loaded_keys, adapted_keys = load_pretrained_state(
            finetune_model, source_state, args.model
        )
        finetune_model, finetune_details = masked.train_model(
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
        pred = masked.predict(finetune_model, test_x, device, args.batch_size)
        result = common.regression_metrics(test_y, pred)
        test_subject = str(subjects[test_idx[0]]) if len(test_idx) == 1 else "fixed_test"
        rows.append(
            {
                "task": f"masked_te_pretrain_{args.mode}",
                "fold": fold_number,
                "test_subject": test_subject,
                "model": args.model,
                "grouping": "full",
                "n_pretrain_subjects": len(pretrain_idx),
                "n_pretrain_samples": len(pretrain_x),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "loaded_pretrain_keys": loaded_keys,
                "adapted_pretrain_keys": adapted_keys,
                "pretrain_val_RMSE": pretrain_details["val_RMSE"],
                "finetune_val_RMSE": finetune_details["val_RMSE"],
                "pretrain_seconds": pretrain_details["training_seconds"],
                "finetune_seconds": finetune_details["training_seconds"],
                **result,
            }
        )
        by_roi = common.summarize_by_roi(test_y, pred, metadata)
        by_roi.insert(0, "task", f"masked_te_pretrain_{args.mode}")
        by_roi.insert(1, "fold", fold_number)
        by_roi.insert(2, "test_subject", test_subject)
        by_roi.insert(3, "model", args.model)
        by_roi.insert(4, "grouping", "full")
        roi_frames.append(by_roi)

        pd.DataFrame(rows).sort_values("fold").to_csv(
            result_path, index=False, encoding="utf-8-sig"
        )
        pd.concat(roi_frames, ignore_index=True).sort_values(
            ["fold", "roi_id"]
        ).to_csv(roi_path, index=False, encoding="utf-8-sig")
        print(
            f"{args.model} {args.mode} fold={fold_number}/{len(folds)} "
            f"MAE={result['MAE']:.6f} RMSE={result['RMSE']:.6f} "
            f"loaded={loaded_keys} adapted={adapted_keys}",
            flush=True,
        )
        del finetune_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = pd.read_csv(result_path)
    means = table[["MAE", "RMSE", "R2"]].mean()
    stds = table[["MAE", "RMSE", "R2"]].std(ddof=1)
    report = [
        f"# 81 Masked-TE pretraining + {args.model} full ({args.mode})",
        "",
        f"- subjects: {len(subjects)}",
        f"- complete 7TE subjects: {int(complete7_mask.sum())}",
        f"- incomplete TE subjects: {int((~complete7_mask).sum())}",
        f"- excluded prefixes: {args.exclude_subject_prefix}",
        f"- MAE: {means['MAE']:.6f} +/- {stds['MAE']:.6f}",
        f"- RMSE: {means['RMSE']:.6f} +/- {stds['RMSE']:.6f}",
        f"- R2: {means['R2']:.6f} +/- {stds['R2']:.6f}",
        "",
        "Masked-TE pretraining uses observed TE values only; fine-tuning returns to TE75-115 -> TE125/135.",
    ]
    (output_dir / f"{stem}_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
