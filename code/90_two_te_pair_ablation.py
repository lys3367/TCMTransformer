from __future__ import annotations

import argparse
import importlib
import sys
import types
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_full_attention_imports() -> None:
    """Provide optional imports unused by the official FullAttention path."""
    try:
        importlib.import_module("reformer_pytorch")
    except ImportError:
        shim = types.ModuleType("reformer_pytorch")

        class LSHSelfAttention:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("This experiment uses FullAttention, not LSH attention.")

        shim.LSHSelfAttention = LSHSelfAttention
        sys.modules["reformer_pytorch"] = shim

    try:
        importlib.import_module("einops")
    except ImportError:
        shim = types.ModuleType("einops")

        def rearrange(*args, **kwargs):
            raise RuntimeError("This experiment uses FullAttention, not FlashAttention.")

        shim.rearrange = rearrange
        sys.modules["einops"] = shim


ensure_full_attention_imports()

ROOT = Path(__file__).resolve().parents[1]
common = importlib.import_module("00_common")
exporter = importlib.import_module("87_export_loso_target_predictions")

CANDIDATE_TES = [75, 85, 95, 105, 115]
TWO_TE_COMBOS = [list(pair) for pair in combinations(CANDIDATE_TES, 2)]
FOLD_COLUMNS = [
    "input_te_1",
    "input_te_2",
    "input_te",
    "te_span_ms",
    "fold",
    "subject",
    "MAE",
    "MAE_TE125",
    "MAE_TE135",
]


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=FOLD_COLUMNS)
    try:
        table = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=FOLD_COLUMNS)
    for column in FOLD_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan
    return table[FOLD_COLUMNS]


def completed_keys(table: pd.DataFrame) -> set[tuple[int, int, int]]:
    required = [
        "input_te_1",
        "input_te_2",
        "fold",
        "subject",
        "MAE",
        "MAE_TE125",
        "MAE_TE135",
    ]
    valid = table.dropna(subset=required)
    return {
        (int(row.input_te_1), int(row.input_te_2), int(row.fold))
        for row in valid.itertuples()
        if np.isfinite(float(row.MAE))
        and np.isfinite(float(row.MAE_TE125))
        and np.isfinite(float(row.MAE_TE135))
    }


def calculate_maes(
    true: np.ndarray, prediction: np.ndarray
) -> tuple[float, float, float]:
    true = np.asarray(true)
    prediction = np.asarray(prediction)
    if (
        true.shape != prediction.shape
        or true.ndim != 3
        or true.shape[0:2] != (1, 2)
    ):
        raise ValueError(
            "Expected true and prediction shape (1, 2, n_variables), "
            f"got true={true.shape}, prediction={prediction.shape}."
        )
    error = np.abs(prediction - true)
    if not np.isfinite(error).all():
        raise ValueError("Prediction error contains NaN or infinite values.")
    return (
        float(error.mean()),
        float(error[:, 0, :].mean()),
        float(error[:, 1, :].mean()),
    )


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["input_te_1", "input_te_2", "input_te", "te_span_ms"]
    for keys, group in table.groupby(group_columns, sort=False):
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "MAE_mean": group["MAE"].mean(),
                "MAE_std": group["MAE"].std(ddof=1),
                "MAE_TE125_mean": group["MAE_TE125"].mean(),
                "MAE_TE125_std": group["MAE_TE125"].std(ddof=1),
                "MAE_TE135_mean": group["MAE_TE135"].mean(),
                "MAE_TE135_std": group["MAE_TE135"].std(ddof=1),
                "count": len(group),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("MAE_mean").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    # Match script 89: same matrix, cohort, metadata, folds, and train_fold.
    config = common.load_config(args.config)
    matrix = common.load_matrix(config)
    raw_all = matrix["x_raw"].astype(np.float32)
    subjects_all = matrix["subjects"].astype(str)
    cohort = exporter.canonical_indices(
        ROOT, subjects_all, "outputs/80_extra_loso_mole_full.csv"
    )
    raw = raw_all[cohort]
    subjects = subjects_all[cohort]
    if len(subjects) != 49:
        raise ValueError(f"Expected the canonical 49 subjects, got {len(subjects)}.")

    te_values = matrix["te_values"].astype(int)
    target_idx = common.te_indices(te_values, config["fixed_target_te"])
    repos = common.resolve_path(config, "external_repos_dir")
    metadata_path = common.resolve_path(config, "metadata_file")
    metadata = (
        pd.read_csv(metadata_path)
        if metadata_path.exists()
        else exporter.metadata_from_matrix(matrix)
    )
    seed = int(config["random_seed"])
    folds = exporter.make_loso_folds(49, seed)

    output_dir = ROOT / "outputs/two_te_pair_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_path = output_dir / "90_two_te_pair_folds.csv"
    summary_path = output_dir / "90_two_te_pair_summary.csv"

    existing = load_existing(folds_path)
    done = completed_keys(existing)
    if not existing.empty:
        existing = existing[
            existing.apply(
                lambda row: (
                    int(row["input_te_1"]),
                    int(row["input_te_2"]),
                    int(row["fold"]),
                )
                in done,
                axis=1,
            )
        ].drop_duplicates(["input_te_1", "input_te_2", "fold"], keep="last")
    existing.to_csv(folds_path, index=False)

    total = len(TWO_TE_COMBOS) * len(folds)
    completed = len(done)
    print(f"Two-TE combinations: {len(TWO_TE_COMBOS)}")
    print(f"LOSO folds per combination: {len(folds)}")
    print(f"Total jobs: {total}; already completed: {completed}", flush=True)

    for pair_number, input_te in enumerate(TWO_TE_COMBOS, start=1):
        te1, te2 = input_te
        input_idx = common.te_indices(te_values, input_te)
        print(f"\n=== Pair {pair_number}/10: TE{te1}+TE{te2} ===", flush=True)

        for fold_number, test_idx in enumerate(folds, start=1):
            key = (te1, te2, fold_number)
            if key in done:
                continue

            outer_train = np.concatenate(
                [fold for index, fold in enumerate(folds) if index != fold_number - 1]
            )
            rng = np.random.default_rng(seed + fold_number)
            shuffled = rng.permutation(outer_train)
            n_validation = max(5, int(round(0.2 * len(shuffled))))
            validation_idx = shuffled[:n_validation]
            train_idx = shuffled[n_validation:]

            true, prediction, _ = exporter.train_fold(
                "itransformer",
                raw,
                metadata,
                train_idx,
                validation_idx,
                test_idx,
                input_idx,
                target_idx,
                repos,
                seed + fold_number,
                args.device,
                args.batch_size,
                args.max_epochs,
                args.patience,
            )
            mae, mae_125, mae_135 = calculate_maes(true, prediction)
            row = pd.DataFrame(
                [
                    {
                        "input_te_1": te1,
                        "input_te_2": te2,
                        "input_te": f"{te1}+{te2}",
                        "te_span_ms": te2 - te1,
                        "fold": fold_number,
                        "subject": subjects[test_idx[0]],
                        "MAE": mae,
                        "MAE_TE125": mae_125,
                        "MAE_TE135": mae_135,
                    }
                ]
            )
            existing = pd.concat([existing, row], ignore_index=True)
            existing.to_csv(folds_path, index=False)
            done.add(key)
            completed += 1
            print(
                f"pair={te1}+{te2} fold={fold_number}/49 "
                f"MAE={mae:.6f} TE125={mae_125:.6f} TE135={mae_135:.6f} "
                f"[{completed}/{total}]",
                flush=True,
            )

        summary = summarize(existing)
        summary.to_csv(summary_path, index=False)
        print(
            summary[["input_te", "MAE_mean", "MAE_std", "count"]].to_string(
                index=False
            ),
            flush=True,
        )

    print(f"\nSaved folds: {folds_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
