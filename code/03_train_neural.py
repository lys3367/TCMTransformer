from __future__ import annotations

import argparse
import copy
import importlib
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


common = importlib.import_module("00_common")


@dataclass
class TrainSettings:
    max_epochs: int = 100
    patience: int = 12
    batch_size: int = 16
    learning_rate: float = 7e-4
    weight_decay: float = 1e-4
    seed: int = 20260623
    device: str = "cuda"


class ForecastWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, None, None, None)


def clear_official_imports(repo_paths: list[Path]) -> None:
    prefixes = ("models", "model", "layers")
    for name in list(sys.modules):
        if name in prefixes or name.startswith(tuple(prefix + "." for prefix in prefixes)):
            del sys.modules[name]
    repo_strings = {str(path) for path in repo_paths}
    sys.path[:] = [path for path in sys.path if path not in repo_strings]


def ensure_reformer_shim() -> None:
    if "reformer_pytorch" in sys.modules:
        return
    try:
        import reformer_pytorch  # noqa: F401
    except ImportError:
        shim = types.ModuleType("reformer_pytorch")

        class LSHSelfAttention(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                raise RuntimeError("This project uses FullAttention, not LSH attention.")

        shim.LSHSelfAttention = LSHSelfAttention
        sys.modules["reformer_pytorch"] = shim


def build_model(
    model_name: str,
    seq_len: int,
    pred_len: int,
    n_variables: int,
    repos_dir: Path,
) -> nn.Module:
    all_repos = [
        repos_dir / "LTSF-Linear",
        repos_dir / "PatchTST" / "PatchTST_supervised",
        repos_dir / "Time-Series-Library",
        repos_dir / "iTransformer",
    ]
    clear_official_imports(all_repos)

    class Config:
        pass

    config = Config()
    if model_name == "dlinear":
        repo = repos_dir / "LTSF-Linear"
        sys.path.insert(0, str(repo))
        from models.DLinear import Model

        config.seq_len = seq_len
        config.pred_len = pred_len
        config.individual = False
        config.enc_in = n_variables
        return Model(config)

    if model_name == "patchtst":
        repo = repos_dir / "PatchTST" / "PatchTST_supervised"
        sys.path.insert(0, str(repo))
        from models.PatchTST import Model

        config.enc_in = n_variables
        config.seq_len = seq_len
        config.pred_len = pred_len
        config.e_layers = 1
        config.n_heads = 2
        config.d_model = 16
        config.d_ff = 32
        config.dropout = 0.10
        config.fc_dropout = 0.05
        config.head_dropout = 0.05
        config.individual = False
        config.patch_len = 2
        config.stride = 1
        config.padding_patch = "end"
        config.revin = True
        config.affine = True
        config.subtract_last = False
        config.decomposition = False
        config.kernel_size = 3
        return Model(config)

    if model_name == "timesnet":
        repo = repos_dir / "Time-Series-Library"
        sys.path.insert(0, str(repo))
        ensure_reformer_shim()
        from models.TimesNet import Model

        original_torch_version = torch.__version__
        torch.__version__ = "1.6.0"
        config.task_name = "long_term_forecast"
        config.seq_len = seq_len
        config.label_len = 0
        config.pred_len = pred_len
        config.enc_in = n_variables
        config.c_out = n_variables
        config.d_model = 16
        config.d_ff = 32
        config.e_layers = 1
        config.top_k = 1
        config.num_kernels = 2
        config.embed = "fixed"
        config.freq = "h"
        config.dropout = 0.10
        try:
            return ForecastWrapper(Model(config))
        finally:
            torch.__version__ = original_torch_version

    if model_name == "itransformer":
        repo = repos_dir / "iTransformer"
        sys.path.insert(0, str(repo))
        ensure_reformer_shim()
        from model.iTransformer import Model

        config.seq_len = seq_len
        config.pred_len = pred_len
        config.output_attention = False
        config.use_norm = True
        config.d_model = 16
        config.embed = "fixed"
        config.freq = "h"
        config.dropout = 0.10
        config.class_strategy = "projection"
        config.factor = 3
        config.n_heads = 2
        config.d_ff = 32
        config.e_layers = 1
        config.activation = "gelu"
        return ForecastWrapper(Model(config))

    raise ValueError(f"Unsupported model: {model_name}")


def make_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_group(
    model_name: str,
    x: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    input_idx: np.ndarray,
    target_idx: np.ndarray,
    variable_idx: np.ndarray,
    repos_dir: Path,
    settings: TrainSettings,
) -> tuple[np.ndarray, dict]:
    common.set_seed(settings.seed)
    device = torch.device(settings.device)
    train_x = x[train_idx][:, input_idx][:, :, variable_idx]
    train_y = x[train_idx][:, target_idx][:, :, variable_idx]
    val_x = x[val_idx][:, input_idx][:, :, variable_idx]
    val_y = x[val_idx][:, target_idx][:, :, variable_idx]
    test_x = x[test_idx][:, input_idx][:, :, variable_idx]

    model = build_model(
        model_name,
        len(input_idx),
        len(target_idx),
        len(variable_idx),
        repos_dir,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    loss_function = nn.MSELoss()
    loader = make_loader(
        train_x.astype(np.float32),
        train_y.astype(np.float32),
        min(settings.batch_size, len(train_idx)),
        True,
    )

    best_state = None
    best_val = np.inf
    best_epoch = 0
    wait = 0
    start = time.time()
    epochs_run = 0
    for epoch in range(1, settings.max_epochs + 1):
        epochs_run = epoch
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(val_x.astype(np.float32)),
                    torch.from_numpy(val_y.astype(np.float32)),
                ),
                batch_size=min(settings.batch_size, len(val_idx)),
                shuffle=False,
            )
            squared_error = 0.0
            n_values = 0
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                val_prediction = model(batch_x)
                squared_error += float(
                    torch.sum((val_prediction - batch_y) ** 2).item()
                )
                n_values += int(batch_y.numel())
            val_loss = squared_error / max(n_values, 1)
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= settings.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_x.astype(np.float32))),
            batch_size=min(settings.batch_size, len(test_idx)),
            shuffle=False,
        )
        for (batch_x,) in test_loader:
            predictions.append(model(batch_x.to(device)).cpu().numpy())
    prediction = np.concatenate(predictions, axis=0).astype(np.float32)
    return prediction, {
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "val_RMSE": float(np.sqrt(best_val)),
        "training_seconds": time.time() - start,
        "n_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def run_neural(
    model_name: str,
    grouping: str,
    raw: np.ndarray,
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    input_idx: np.ndarray,
    target_idx: np.ndarray,
    repos_dir: Path,
    settings: TrainSettings,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x, _, _ = common.standardize_from_train(raw, train_idx)
    true = x[test_idx][:, target_idx, :]
    prediction = np.empty_like(true)
    group_rows = []
    groups = common.make_groups(metadata, grouping)
    for group_number, (group_name, variable_idx) in enumerate(groups, start=1):
        group_prediction, details = train_group(
            model_name,
            x,
            train_idx,
            val_idx,
            test_idx,
            input_idx,
            target_idx,
            variable_idx,
            repos_dir,
            settings,
        )
        prediction[:, :, variable_idx] = group_prediction
        group_rows.append(
            {
                "group": group_name,
                "n_variables": len(variable_idx),
                **details,
            }
        )
        print(
            f"[{group_number}/{len(groups)}] {model_name} {grouping} "
            f"{group_name}: val_RMSE={details['val_RMSE']:.6f}",
            flush=True,
        )
    return true, prediction, pd.DataFrame(group_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an official DLinear, PatchTST, TimesNet, or iTransformer model."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["dlinear", "patchtst", "timesnet", "itransformer"],
    )
    parser.add_argument(
        "--grouping", default="full", choices=["full", "by_roi", "by_feature"]
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = common.load_config(args.config)
    matrix = common.load_matrix(config)
    raw = matrix["x_raw"].astype(np.float32)
    subjects = matrix["subjects"].astype(str)
    te_values = matrix["te_values"].astype(int)
    split_labels = common.load_split_labels(config, subjects)
    metadata = pd.read_csv(common.resolve_path(config, "metadata_file"))
    repos_dir = common.resolve_path(config, "external_repos_dir")
    output_dir = common.resolve_path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model != "itransformer" and args.grouping != "full":
        print(
            "Note: grouped execution is mainly intended for iTransformer, "
            "but it is allowed for controlled ablations."
        )
    if args.model == "itransformer" and args.grouping == "full":
        print(
            "Warning: full iTransformer uses attention over all variables and may "
            "require substantial GPU memory. Start with by_roi or by_feature."
        )

    train_idx = np.where(split_labels == "train")[0]
    val_idx = np.where(split_labels == "val")[0]
    test_idx = np.where(split_labels == "test")[0]
    input_idx = common.te_indices(te_values, config["fixed_input_te"])
    target_idx = common.te_indices(te_values, config["fixed_target_te"])
    settings = TrainSettings(
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed or int(config["random_seed"]),
        device=args.device,
    )
    true, prediction, groups = run_neural(
        args.model,
        args.grouping,
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
    result = common.regression_metrics(true, prediction)
    stem = f"03_{args.model}_{args.grouping}"
    pd.DataFrame(
        [
            {
                "model": args.model,
                "grouping": args.grouping,
                "task": "fixed_target",
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "n_groups": len(groups),
                "training_seconds": groups["training_seconds"].sum(),
                **result,
            }
        ]
    ).to_csv(output_dir / f"{stem}_summary.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(
        output_dir / f"{stem}_training_groups.csv",
        index=False,
        encoding="utf-8-sig",
    )
    common.summarize_by_roi(true, prediction, metadata).to_csv(
        output_dir / f"{stem}_by_roi.csv", index=False, encoding="utf-8-sig"
    )
    print(
        f"{args.model}/{args.grouping}: MAE={result['MAE']:.6f}, "
        f"RMSE={result['RMSE']:.6f}, R2={result['R2']:.6f}"
    )


if __name__ == "__main__":
    main()
