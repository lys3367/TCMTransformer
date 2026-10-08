from __future__ import annotations

import argparse
import copy
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


common = importlib.import_module("00_common")
neural = importlib.import_module("03_train_neural")
extra = importlib.import_module("73_train_extra_models")


def nan_standardize_from_train(
    raw: np.ndarray, observed_mask: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = raw[train_indices].astype(np.float64)
    train_mask = observed_mask[train_indices][:, :, None]
    train = np.where(train_mask, train, np.nan)
    mean = np.nanmean(train, axis=(0, 1), keepdims=True)
    std = np.nanstd(train, axis=(0, 1), keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std >= 1e-8), std, 1.0)
    standardized = ((raw - mean) / std).astype(np.float32)
    return standardized, mean.reshape(-1).astype(np.float32), std.reshape(-1).astype(np.float32)


def make_masked_samples(
    x: np.ndarray,
    observed_mask: np.ndarray,
    subject_indices: np.ndarray,
    target_te_indices: list[int],
    min_context: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, target_rows = [], [], []
    for subject_index in subject_indices:
        available = observed_mask[subject_index]
        for target_index in target_te_indices:
            if not available[target_index]:
                continue
            if int(available.sum()) - 1 < min_context:
                continue
            sample_x = x[subject_index].copy()
            sample_x[~available] = 0.0
            sample_x[target_index] = 0.0
            sample_x = np.nan_to_num(sample_x, nan=0.0, posinf=0.0, neginf=0.0)
            sample_y = x[subject_index, target_index : target_index + 1].copy()
            if not np.isfinite(sample_y).all():
                continue
            xs.append(sample_x)
            ys.append(sample_y)
            target_rows.append(target_index)
    if not xs:
        raise ValueError("No masked-TE samples were created.")
    return (
        np.stack(xs).astype(np.float32),
        np.stack(ys).astype(np.float32),
        np.asarray(target_rows, dtype=np.int16),
    )


def make_main_samples(
    x: np.ndarray,
    subject_indices: np.ndarray,
    target_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for subject_index in subject_indices:
        sample_x = x[subject_index].copy()
        sample_y = x[subject_index, target_idx].copy()
        sample_x[target_idx] = 0.0
        if not np.isfinite(sample_x).all() or not np.isfinite(sample_y).all():
            raise ValueError(f"Complete subject index {subject_index} contains NaN.")
        xs.append(sample_x)
        ys.append(sample_y)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> tuple[nn.Module, dict]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=min(batch_size, len(train_x)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
        batch_size=min(batch_size, len(val_x)),
        shuffle=False,
    )
    best_state = None
    best_val = np.inf
    best_epoch = 0
    wait = 0
    start = time.time()
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        se = 0.0
        n = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                se += float(torch.sum((pred - batch_y) ** 2).item())
                n += int(batch_y.numel())
        val_loss = se / max(n, 1)
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "val_RMSE": float(np.sqrt(best_val)),
        "training_seconds": time.time() - start,
    }


def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x.astype(np.float32))),
        batch_size=min(batch_size, len(x)),
        shuffle=False,
    )
    with torch.no_grad():
        for (batch_x,) in loader:
            preds.append(model(batch_x.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def load_compatible_state(target: nn.Module, source_state: dict[str, torch.Tensor]) -> int:
    target_state = target.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape)
    }
    target_state.update(compatible)
    target.load_state_dict(target_state)
    return len(compatible)


def filter_subjects(subjects: np.ndarray, exclude_prefixes: list[str]) -> np.ndarray:
    keep = np.ones(len(subjects), dtype=bool)
    for prefix in exclude_prefixes:
        prefix = prefix.strip()
        if prefix:
            keep &= ~np.char.startswith(subjects.astype(str), prefix)
    return np.where(keep)[0]


def fixed_folds(
    config: dict,
    subjects: np.ndarray,
    complete7_mask: np.ndarray,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    split = pd.read_csv(common.resolve_path(config, "split_file"))
    split_map = dict(zip(split["subject_id"].astype(str), split["split"].astype(str)))
    complete_idx = np.where(complete7_mask)[0]
    train_idx = np.asarray([i for i in complete_idx if split_map.get(str(subjects[i])) == "train"], dtype=int)
    val_idx = np.asarray([i for i in complete_idx if split_map.get(str(subjects[i])) == "val"], dtype=int)
    test_idx = np.asarray([i for i in complete_idx if split_map.get(str(subjects[i])) == "test"], dtype=int)
    incomplete_idx = np.where(~complete7_mask)[0]
    pretrain_idx = np.concatenate([train_idx, incomplete_idx]).astype(int)
    return [(1, pretrain_idx, train_idx, val_idx, test_idx)]


def loso_folds(
    subjects: np.ndarray,
    complete7_mask: np.ndarray,
    seed: int,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    complete_idx = np.where(complete7_mask)[0]
    incomplete_idx = np.where(~complete7_mask)[0]
    folds = []
    for fold_number, test_subject_index in enumerate(complete_idx, start=1):
        outer_train = complete_idx[complete_idx != test_subject_index]
        rng = np.random.default_rng(seed + fold_number)
        shuffled = rng.permutation(outer_train)
        n_val = max(5, int(round(0.2 * len(shuffled))))
        val_idx = shuffled[:n_val]
        train_idx = shuffled[n_val:]
        pretrain_idx = np.concatenate([train_idx, incomplete_idx]).astype(int)
        folds.append((fold_number, pretrain_idx, train_idx, val_idx, np.asarray([test_subject_index], dtype=int)))
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Masked-TE pretraining on incomplete TE data followed by extra model full fine-tuning."
    )
    parser.add_argument("--model", required=True, choices=["segrnn", "mtsmixer", "mole"])
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--incomplete-matrix",
        default="data/processed/60_model_matrix_incomplete_te.npz",
    )
    parser.add_argument("--mode", choices=["fixed", "loso"], default="fixed")
    parser.add_argument("--exclude-subject-prefix", nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-context", type=int, default=2)
    parser.add_argument("--loso-max-folds", type=int, default=0, help="0 means all folds.")
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

    keep = filter_subjects(subjects_all, args.exclude_subject_prefix)
    raw = raw_all[keep]
    observed_mask = observed_all[keep]
    subjects = subjects_all[keep]
    complete7_mask = complete_all[keep]

    input_target_idx = common.te_indices(te_values, config["fixed_target_te"])
    target_te_indices = list(range(len(te_values)))

    if args.mode == "fixed":
        folds = fixed_folds(config, subjects, complete7_mask)
    else:
        folds = loso_folds(subjects, complete7_mask, seed)
        if args.loso_max_folds > 0:
            folds = folds[: args.loso_max_folds]

    rows = []
    roi_rows = []
    for fold_number, pretrain_idx, train_idx, val_idx, test_idx in folds:
        standardize_idx = np.unique(np.concatenate([pretrain_idx, train_idx]))
        x, _, _ = nan_standardize_from_train(raw, observed_mask, standardize_idx)

        pretrain_x, pretrain_y, _ = make_masked_samples(
            x, observed_mask, pretrain_idx, target_te_indices, args.min_context
        )
        val_mask_x, val_mask_y, _ = make_masked_samples(
            x, observed_mask, val_idx, target_te_indices, args.min_context
        )
        pretrain_model = extra.build_extra_model(
            args.model,
            seq_len=len(te_values),
            pred_len=1,
            n_variables=x.shape[2],
            repos_dir=repos_dir,
        )
        pretrain_model, pretrain_details = train_model(
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

        train_x, train_y = make_main_samples(x, train_idx, input_target_idx)
        val_x, val_y = make_main_samples(x, val_idx, input_target_idx)
        test_x, test_y = make_main_samples(x, test_idx, input_target_idx)
        finetune_model = extra.build_extra_model(
            args.model,
            seq_len=len(te_values),
            pred_len=len(input_target_idx),
            n_variables=x.shape[2],
            repos_dir=repos_dir,
        )
        loaded_keys = load_compatible_state(
            finetune_model, pretrain_model.state_dict()
        )
        finetune_model, finetune_details = train_model(
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
        pred = predict(finetune_model, test_x, device, args.batch_size)
        result = common.regression_metrics(test_y, pred)
        rows.append(
            {
                "task": f"masked_te_pretrain_{args.mode}",
                "fold": fold_number,
                "model": args.model,
                "grouping": "full",
                "n_pretrain_subjects": len(pretrain_idx),
                "n_pretrain_samples": len(pretrain_x),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "test_subject": subjects[test_idx[0]] if len(test_idx) == 1 else "fixed_test",
                "loaded_pretrain_keys": loaded_keys,
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
        by_roi.insert(2, "test_subject", subjects[test_idx[0]] if len(test_idx) == 1 else "fixed_test")
        by_roi.insert(3, "model", args.model)
        by_roi.insert(4, "grouping", "full")
        roi_rows.append(by_roi)
        print(
            f"{args.mode} fold={fold_number} MAE={result['MAE']:.6f} "
            f"RMSE={result['RMSE']:.6f} pretrain_samples={len(pretrain_x)}",
            flush=True,
        )

    suffix = ""
    if args.exclude_subject_prefix:
        suffix = "_exclude_" + "_".join(prefix.rstrip("_") for prefix in args.exclude_subject_prefix)
    stem = f"76_masked_te_pretrain_extra_{args.model}_full_{args.mode}{suffix}"
    result_table = pd.DataFrame(rows)
    result_table.to_csv(output_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    if roi_rows:
        pd.concat(roi_rows, ignore_index=True).to_csv(
            output_dir / f"{stem}_by_roi.csv", index=False, encoding="utf-8-sig"
        )
    means = result_table[["MAE", "RMSE", "R2"]].mean()
    stds = result_table[["MAE", "RMSE", "R2"]].std(ddof=1)
    report = [
        f"# 76 Masked-TE pretraining + {args.model} full fine-tuning ({args.mode})",
        "",
        f"- 受试者数：{len(subjects)}",
        f"- 完整 7TE 受试者数：{int(complete7_mask.sum())}",
        f"- 不完整 TE 受试者数：{int((~complete7_mask).sum())}",
        f"- 排除受试者前缀：{args.exclude_subject_prefix or '无'}",
        f"- MAE：{means['MAE']:.6f} ± {stds['MAE']:.6f}",
        f"- RMSE：{means['RMSE']:.6f} ± {stds['RMSE']:.6f}",
        f"- R2：{means['R2']:.6f} ± {stds['R2']:.6f}",
        "",
        "说明：预训练阶段使用不完整 TE 数据构造 masked-TE reconstruction 样本；微调阶段回到主任务，用 75/85/95/105/115 预测 125/135。",
    ]
    (output_dir / f"{stem}_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
