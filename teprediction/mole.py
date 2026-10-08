from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


common = importlib.import_module("teprediction.common")


EXTRA_MODELS = ["mole"]


@dataclass
class TrainSettings:
    max_epochs: int = 100
    patience: int = 12
    batch_size: int = 16
    learning_rate: float = 7e-4
    weight_decay: float = 1e-4
    seed: int = 20260623
    device: str = "cuda"




class MarkForecastWrapper(nn.Module):
    def __init__(self, model: nn.Module, seq_len: int, n_time_features: int = 4):
        super().__init__()
        self.model = model
        base = torch.zeros(seq_len, n_time_features, dtype=torch.float32)
        base[:, 0] = torch.arange(seq_len, dtype=torch.float32)
        if n_time_features > 1 and seq_len > 1:
            base[:, 1] = torch.linspace(0.0, 1.0, seq_len)
        self.register_buffer("x_mark", base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        marks = self.x_mark.unsqueeze(0).expand(x.shape[0], -1, -1)
        return self.model(x, marks)


def clear_extra_imports(repo_paths: list[Path]) -> None:
    prefixes = ("models", "model", "layers")
    for name in list(sys.modules):
        if name in prefixes or name.startswith(tuple(prefix + "." for prefix in prefixes)):
            del sys.modules[name]
    repo_paths = repo_paths + list((Path(__file__).resolve().parent / "vendor").iterdir())
    repo_strings = {str(path) for path in repo_paths}
    sys.path[:] = [path for path in sys.path if path not in repo_strings]






def build_extra_model(
    model_name: str,
    seq_len: int,
    pred_len: int,
    n_variables: int,
    repos_dir: Path,
) -> nn.Module:
    all_repos = [
        repos_dir / "MoLE",
    ]
    clear_extra_imports(all_repos)

    class Config:
        pass

    config = Config()
    config.seq_len = seq_len
    config.pred_len = pred_len
    config.enc_in = n_variables
    config.c_out = n_variables
    config.d_model = 16
    config.d_ff = 32
    config.e_layers = 1
    config.n_heads = 2
    config.dropout = 0.10
    config.head_dropout = 0.05
    config.individual = False
    config.revin = True
    config.embed = "fixed"
    config.freq = "h"

    if model_name == "mole":
        repo = common.model_source(repos_dir, "MoLE")
        if not repo.exists():
            raise FileNotFoundError("Missing MoLE under the configured model source directory.")
        sys.path.insert(0, str(repo))
        config.num_experts = 4
        config.top_k = 2
        config.t_dim = pred_len
        model_file = repo / "models" / "MoLE_DLinear.py"
        if not model_file.is_file():
            raise FileNotFoundError("Provide the official MoLE_DLinear.py snapshot; see README.")
        spec = importlib.util.spec_from_file_location("mole_official", model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Model = module.Model
        return MarkForecastWrapper(Model(config), seq_len, n_time_features=4)

    raise ValueError(f"Unknown extra model: {model_name}")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_one(
    model_name: str,
    raw: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    input_idx: np.ndarray,
    target_idx: np.ndarray,
    repos_dir: Path,
    settings: TrainSettings,
) -> tuple[np.ndarray, np.ndarray, dict]:
    common.set_seed(settings.seed)
    x, _, _ = common.standardize_from_train(raw, train_idx)
    train_x = x[train_idx][:, input_idx, :]
    train_y = x[train_idx][:, target_idx, :]
    val_x = x[val_idx][:, input_idx, :]
    val_y = x[val_idx][:, target_idx, :]
    test_x = x[test_idx][:, input_idx, :]
    true = x[test_idx][:, target_idx, :]

    device = torch.device(settings.device)
    model = build_extra_model(
        model_name,
        len(input_idx),
        len(target_idx),
        raw.shape[2],
        repos_dir,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    loss_fn = nn.MSELoss()
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
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(val_x.astype(np.float32)).to(device))
            val_loss = float(
                loss_fn(val_pred, torch.from_numpy(val_y.astype(np.float32)).to(device)).item()
            )
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
        raise RuntimeError("No valid checkpoint was produced.")
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_x.astype(np.float32))),
            batch_size=min(settings.batch_size, len(test_idx)),
            shuffle=False,
        )
        for (batch_x,) in test_loader:
            preds.append(model(batch_x.to(device)).cpu().numpy())
    prediction = np.concatenate(preds, axis=0).astype(np.float32)
    details = {
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "val_RMSE": float(np.sqrt(best_val)),
        "training_seconds": time.time() - start,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    return true, prediction, details














