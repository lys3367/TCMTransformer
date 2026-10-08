from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else ROOT / "config.json"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    return config


def resolve_path(config: dict, key: str) -> Path:
    path = Path(config[key])
    return path if path.is_absolute() else ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_matrix(config: dict) -> dict[str, np.ndarray]:
    matrix = np.load(resolve_path(config, "matrix_file"), allow_pickle=True)
    return {key: matrix[key] for key in matrix.files}


def load_split_labels(config: dict, subjects: np.ndarray) -> np.ndarray:
    split = pd.read_csv(resolve_path(config, "split_file"))
    split_map = dict(zip(split["subject_id"].astype(str), split["split"].astype(str)))
    missing = [subject for subject in subjects.astype(str) if subject not in split_map]
    if missing:
        raise ValueError(f"Split file is missing {len(missing)} subjects.")
    return np.asarray([split_map[subject] for subject in subjects.astype(str)])


def te_indices(te_values: np.ndarray, requested: Iterable[int]) -> np.ndarray:
    indices = []
    for te in requested:
        found = np.where(te_values.astype(int) == int(te))[0]
        if len(found) != 1:
            raise ValueError(f"TE={te} was not found exactly once.")
        indices.append(int(found[0]))
    return np.asarray(indices, dtype=int)


def standardize_from_train(
    raw: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = raw[train_indices]
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-8] = 1.0
    standardized = ((raw - mean) / std).astype(np.float32)
    return standardized, mean.reshape(-1).astype(np.float32), std.reshape(-1).astype(np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    error = pred - true
    denominator = float(np.sum((true - true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": float(1.0 - np.sum(error**2) / denominator)
        if denominator > 0
        else np.nan,
    }


def summarize_by_roi(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for roi_id, group in metadata.groupby("roi_id", sort=True):
        variable_indices = group["variable_index"].to_numpy(dtype=int)
        result = regression_metrics(
            y_true[:, :, variable_indices], y_pred[:, :, variable_indices]
        )
        rows.append(
            {
                "roi_id": str(roi_id),
                "roi_label": str(group["roi_label"].iloc[0]),
                "n_variables": len(variable_indices),
                **result,
            }
        )
    return pd.DataFrame(rows)


def make_groups(metadata: pd.DataFrame, mode: str) -> list[tuple[str, np.ndarray]]:
    if mode == "full":
        return [("all_variables", metadata["variable_index"].to_numpy(dtype=int))]
    if mode == "by_roi":
        return [
            (f"roi_{key}", group["variable_index"].to_numpy(dtype=int))
            for key, group in metadata.groupby("roi_id", sort=True)
        ]
    if mode == "by_feature":
        return [
            (str(key), group["variable_index"].to_numpy(dtype=int))
            for key, group in metadata.groupby("feature", sort=True)
        ]
    raise ValueError(f"Unknown grouping mode: {mode}")
