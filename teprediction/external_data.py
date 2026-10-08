from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd


common = importlib.import_module("teprediction.common")


def load_external_matrix(csv_path: Path, metadata: pd.DataFrame, expected_te: list[int]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["ROI_ID"] = df["ROI_ID"].astype(str)
    df["feature"] = df["Model"].astype(str) + "__" + df["Metric"].astype(str)
    df["variable"] = df["feature"] + "__" + df["ROI_ID"]

    variables = metadata.sort_values("variable_index")["variable"].astype(str).to_numpy()
    subjects = np.asarray(sorted(df["Subject"].astype(str).unique()), dtype=object)
    full_index = pd.MultiIndex.from_product(
        [subjects, expected_te, variables], names=["Subject", "TE", "variable"]
    )
    series = df.set_index(["Subject", "TE", "variable"])["Mean"].reindex(full_index)
    if series.isna().any():
        missing = int(series.isna().sum())
        missing_vars = series[series.isna()].reset_index()["variable"].drop_duplicates().head(20).tolist()
        raise ValueError(
            f"External matrix contains {missing} missing values. "
            f"First missing variables: {missing_vars}"
        )
    external = series.to_numpy(dtype=np.float32).reshape(len(subjects), len(expected_te), len(variables))
    return external, subjects






