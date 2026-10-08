from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_CHOICES = ("itransformer", "timesnet", "mole", "dlinear")

common = importlib.import_module("00_common")
neural = importlib.import_module("03_train_neural")
extra = importlib.import_module("73_train_extra_models")


def metadata_from_matrix(matrix: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable_index": np.arange(len(matrix["variables"]), dtype=int),
            "variable": matrix["variables"].astype(str),
            "model": matrix["models"].astype(str),
            "metric": matrix["metrics"].astype(str),
            "feature": matrix["features"].astype(str),
            "roi_id": matrix["roi_ids"].astype(str),
            "roi_label": matrix["roi_labels"].astype(str),
        }
    )


def canonical_indices(project: Path, subjects: np.ndarray, cohort_source: str) -> np.ndarray:
    table = pd.read_csv(project / cohort_source)
    column = "test_subject" if "test_subject" in table.columns else "subject"
    requested = set(table[column].astype(str))
    indices = np.asarray([i for i, subject in enumerate(subjects.astype(str)) if subject in requested], dtype=int)
    if len(indices) != 49 or len(requested) != 49:
        raise ValueError(f"Expected the canonical 49-subject cohort, found indices={len(indices)}, names={len(requested)}")
    return indices


def make_loso_folds(n_subjects: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.asarray([index], dtype=int) for index in rng.permutation(n_subjects)]


def train_fold(
    model: str,
    raw: np.ndarray,
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    input_idx: np.ndarray,
    target_idx: np.ndarray,
    repos_dir: Path,
    seed: int,
    device: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if model == "mole":
        settings = extra.TrainSettings(
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        return extra.train_one(
            "mole", raw, train_idx, val_idx, test_idx, input_idx, target_idx, repos_dir, settings
        )

    settings = neural.TrainSettings(
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    true, prediction, details = neural.run_neural(
        model,
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
    return true, prediction, {
        "best_epoch": int(details["best_epoch"].max()),
        "epochs_run": int(details["epochs_run"].max()),
        "val_RMSE": float(details["val_RMSE"].mean()),
        "training_seconds": float(details["training_seconds"].sum()),
        "n_parameters": int(details["n_parameters"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export target-wise LOSO predictions for one formal model.")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cohort-source", default="outputs/80_extra_loso_mole_full.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--keep-fold-files", action="store_true")
    args = parser.parse_args()

    project = ROOT
    config = common.load_config(args.config)
    matrix = common.load_matrix(config)
    raw_all = matrix["x_raw"].astype(np.float32)
    subjects_all = matrix["subjects"].astype(str)
    cohort_idx = canonical_indices(project, subjects_all, args.cohort_source)
    raw = raw_all[cohort_idx]
    subjects = subjects_all[cohort_idx]
    te_values = matrix["te_values"].astype(int)
    input_idx = common.te_indices(te_values, config["fixed_input_te"])
    target_idx = common.te_indices(te_values, config["fixed_target_te"])
    metadata_path = common.resolve_path(config, "metadata_file")
    metadata = pd.read_csv(metadata_path) if metadata_path.exists() else metadata_from_matrix(matrix)
    repos_dir = common.resolve_path(config, "external_repos_dir")
    output_dir = project / "outputs/target_predictions"
    fold_dir = output_dir / "fold_cache" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["random_seed"])
    folds = make_loso_folds(len(raw), seed)
    rows = []
    fold_paths = []
    for fold_number, test_idx in enumerate(folds, start=1):
        fold_path = fold_dir / f"fold_{fold_number:02d}.npz"
        fold_paths.append(fold_path)
        if not fold_path.exists():
            outer_train = np.concatenate([fold for i, fold in enumerate(folds) if i != fold_number - 1])
            rng = np.random.default_rng(seed + fold_number)
            shuffled = rng.permutation(outer_train)
            n_val = max(5, int(round(0.2 * len(shuffled))))
            val_idx = shuffled[:n_val]
            train_idx = shuffled[n_val:]
            true_z, pred_z, details = train_fold(
                args.model,
                raw,
                metadata,
                train_idx,
                val_idx,
                test_idx,
                input_idx,
                target_idx,
                repos_dir,
                seed + fold_number,
                args.device,
                args.batch_size,
                args.max_epochs,
                args.patience,
            )
            _, mean, std = common.standardize_from_train(raw, train_idx)
            true_native = true_z * std[None, None, :] + mean[None, None, :]
            pred_native = pred_z * std[None, None, :] + mean[None, None, :]
            np.savez_compressed(
                fold_path,
                subject=subjects[test_idx],
                true_z=true_z.astype(np.float32),
                pred_z=pred_z.astype(np.float32),
                true_native=true_native.astype(np.float32),
                pred_native=pred_native.astype(np.float32),
                mean=mean,
                std=std,
                best_epoch=np.asarray(details["best_epoch"]),
                val_RMSE=np.asarray(details["val_RMSE"]),
            )
        cached = np.load(fold_path, allow_pickle=True)
        for target_position, te in enumerate(te_values[target_idx]):
            error = np.abs(cached["pred_z"][:, target_position] - cached["true_z"][:, target_position])
            rows.append(
                {
                    "fold": fold_number,
                    "subject": str(cached["subject"][0]),
                    "model": args.model,
                    "target_TE": int(te),
                    "MAE": float(error.mean()),
                }
            )
        print(f"{args.model} fold={fold_number}/49 subject={cached['subject'][0]}", flush=True)

    payloads = [np.load(path, allow_pickle=True) for path in fold_paths]
    subject_order = np.concatenate([item["subject"].astype(str) for item in payloads])
    true_z = np.concatenate([item["true_z"] for item in payloads], axis=0)
    pred_z = np.concatenate([item["pred_z"] for item in payloads], axis=0)
    true_native = np.concatenate([item["true_native"] for item in payloads], axis=0)
    pred_native = np.concatenate([item["pred_native"] for item in payloads], axis=0)
    prediction_path = output_dir / f"87_loso_target_predictions_{args.model}.npz"
    np.savez_compressed(
        prediction_path,
        subjects=subject_order,
        target_te=te_values[target_idx],
        true_z=true_z,
        pred_z=pred_z,
        true_native=true_native,
        pred_native=pred_native,
        variables=matrix["variables"],
        models=matrix["models"],
        metrics=matrix["metrics"],
        features=matrix["features"],
        roi_ids=matrix["roi_ids"],
        roi_labels=matrix["roi_labels"],
    )
    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / f"87_target_te_subject_mae_{args.model}.csv", index=False, encoding="utf-8-sig")
    summary = detail.groupby(["model", "target_TE"])["MAE"].agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(output_dir / f"87_target_te_summary_{args.model}.csv", index=False, encoding="utf-8-sig")
    if not args.keep_fold_files:
        shutil.rmtree(fold_dir)
    print(f"Saved predictions: {prediction_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
