from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

common = importlib.import_module("00_common")
neural = importlib.import_module("03_train_neural")
extra = importlib.import_module("73_train_extra_models")

MODEL_CHOICES = ("itransformer", "timesnet", "mole", "dlinear")


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


def load_cohort(subjects: np.ndarray, manifest: Path, expected_subjects: int = 49) -> np.ndarray:
    requested = pd.read_csv(manifest)["subject"].astype(str).tolist()
    if len(requested) != expected_subjects or len(set(requested)) != expected_subjects:
        raise ValueError(
            f"The cohort manifest must contain {expected_subjects} unique subjects."
        )
    index = {subject: i for i, subject in enumerate(subjects.astype(str))}
    missing = [subject for subject in requested if subject not in index]
    if missing:
        raise ValueError(f"Matrix is missing manifest subjects: {missing}")
    # Preserve matrix order exactly, matching the established clean-49 workflow.
    requested_set = set(requested)
    indices = np.asarray(
        [i for i, subject in enumerate(subjects.astype(str)) if subject in requested_set],
        dtype=int,
    )
    if len(indices) != expected_subjects:
        raise ValueError(
            f"Expected {expected_subjects} matrix indices, found {len(indices)}."
        )
    return indices


def make_folds(n_subjects: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.asarray([index], dtype=int) for index in rng.permutation(n_subjects)]


def split_fold(
    folds: list[np.ndarray], fold_number: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_idx = folds[fold_number - 1]
    outer_train = np.concatenate(
        [fold for i, fold in enumerate(folds) if i != fold_number - 1]
    )
    shuffled = np.random.default_rng(seed + fold_number).permutation(outer_train)
    n_val = max(5, int(round(0.2 * len(shuffled))))
    return shuffled[n_val:], shuffled[:n_val], test_idx


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
            "mole",
            raw,
            train_idx,
            val_idx,
            test_idx,
            input_idx,
            target_idx,
            repos_dir,
            settings,
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


def signature(settings: dict) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required source file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(details: dict, key: str, default: float = np.nan) -> float:
    value = details.get(key, default)
    return float(np.asarray(value).reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical clean-49 Table 1 LOSO training.")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default="data/processed/clean49_subjects.csv")
    parser.add_argument("--expected-subjects", type=int, default=49)
    parser.add_argument("--output-dir", default="outputs/table1_clean49")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Use the same CUDA-enabled Python environment for all four models."
            )

    config = common.load_config(args.config)
    matrix = common.load_matrix(config)
    raw_all = matrix["x_raw"].astype(np.float32)
    subjects_all = matrix["subjects"].astype(str)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = PROJECT / manifest
    cohort_idx = load_cohort(subjects_all, manifest, args.expected_subjects)
    raw = raw_all[cohort_idx]
    subjects = subjects_all[cohort_idx]
    te_values = matrix["te_values"].astype(int)
    input_idx = common.te_indices(te_values, config["fixed_input_te"])
    target_idx = common.te_indices(te_values, config["fixed_target_te"])
    metadata_path = common.resolve_path(config, "metadata_file")
    metadata = (
        pd.read_csv(metadata_path)
        if metadata_path.exists()
        else metadata_from_matrix(matrix)
    )
    repos_dir = common.resolve_path(config, "external_repos_dir")
    seed = int(config["random_seed"])
    folds = make_folds(len(subjects), seed)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT / output_dir
    model_dir = output_dir / args.model
    model_dir.mkdir(parents=True, exist_ok=True)

    model_sources = {
        "itransformer": repos_dir / "iTransformer/model/iTransformer.py",
        "timesnet": repos_dir / "Time-Series-Library/models/TimesNet.py",
        "dlinear": repos_dir / "LTSF-Linear/models/DLinear.py",
        "mole": repos_dir / "MoLE/models/MoLE_DLinear.py",
    }
    wrapper_source = CODE_DIR / (
        "73_train_extra_models.py" if args.model == "mole" else "03_train_neural.py"
    )
    run_settings = {
        "model": args.model,
        "cohort_subjects": subjects.tolist(),
        "random_seed": seed,
        "input_te": te_values[input_idx].tolist(),
        "target_te": te_values[target_idx].tolist(),
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "matrix_file": config["matrix_file"],
        "n_variables": int(raw.shape[2]),
        "wrapper_sha256": file_sha256(wrapper_source),
        "official_model_sha256": file_sha256(model_sources[args.model]),
    }
    run_signature = signature(run_settings)
    (model_dir / "run_settings.json").write_text(
        json.dumps({**run_settings, "signature": run_signature}, indent=2),
        encoding="utf-8",
    )

    manifest_rows = []
    metric_rows = []
    fold_paths = []
    for fold_number in range(1, len(subjects) + 1):
        train_idx, val_idx, test_idx = split_fold(folds, fold_number, seed)
        fold_seed = seed + fold_number
        fold_path = model_dir / f"fold_{fold_number:02d}.npz"
        fold_paths.append(fold_path)
        manifest_rows.append(
            {
                "fold": fold_number,
                "fold_seed": fold_seed,
                "test_subject": subjects[test_idx[0]],
                "train_subjects": ";".join(subjects[train_idx]),
                "val_subjects": ";".join(subjects[val_idx]),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
            }
        )

        if fold_path.exists():
            cached = np.load(fold_path, allow_pickle=True)
            cached_signature = str(cached["signature"].reshape(-1)[0])
            if cached_signature != run_signature:
                raise RuntimeError(
                    f"Configuration mismatch in {fold_path}: "
                    f"cached={cached_signature}, current={run_signature}."
                )
        else:
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
                fold_seed,
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
                signature=np.asarray([run_signature]),
                fold=np.asarray([fold_number]),
                fold_seed=np.asarray([fold_seed]),
                subject=subjects[test_idx],
                train_subjects=subjects[train_idx],
                val_subjects=subjects[val_idx],
                target_te=te_values[target_idx],
                true_z=true_z.astype(np.float32),
                pred_z=pred_z.astype(np.float32),
                true_native=true_native.astype(np.float32),
                pred_native=pred_native.astype(np.float32),
                mean=mean.astype(np.float32),
                std=std.astype(np.float32),
                best_epoch=np.asarray([scalar(details, "best_epoch")]),
                epochs_run=np.asarray([scalar(details, "epochs_run")]),
                val_RMSE=np.asarray([scalar(details, "val_RMSE")]),
                training_seconds=np.asarray([scalar(details, "training_seconds")]),
                n_parameters=np.asarray([scalar(details, "n_parameters")]),
            )

        cached = np.load(fold_path, allow_pickle=True)
        true_z = cached["true_z"]
        pred_z = cached["pred_z"]
        overall = common.regression_metrics(true_z, pred_z)
        target_metrics = [
            common.regression_metrics(true_z[:, i, :], pred_z[:, i, :])
            for i in range(len(target_idx))
        ]
        identity_error = abs(
            overall["MAE"] - np.mean([item["MAE"] for item in target_metrics])
        )
        if identity_error > 1e-7:
            raise RuntimeError(f"Target MAE identity failed at fold {fold_number}.")
        metric_rows.append(
            {
                "model": args.model,
                "fold": fold_number,
                "test_subject": str(cached["subject"][0]),
                "fold_seed": int(cached["fold_seed"][0]),
                "n_train": len(cached["train_subjects"]),
                "n_val": len(cached["val_subjects"]),
                "n_test": 1,
                "MAE": overall["MAE"],
                "RMSE": overall["RMSE"],
                "R2": overall["R2"],
                "MAE_TE125": target_metrics[0]["MAE"],
                "MAE_TE135": target_metrics[1]["MAE"],
                "MAE_identity_error": identity_error,
                "best_epoch": float(cached["best_epoch"][0]),
                "epochs_run": float(cached["epochs_run"][0]),
                "val_RMSE": float(cached["val_RMSE"][0]),
                "training_seconds": float(cached["training_seconds"][0]),
                "n_parameters": float(cached["n_parameters"][0]),
                "signature": run_signature,
            }
        )
        pd.DataFrame(metric_rows).to_csv(
            output_dir / f"{args.model}_fold_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"{args.model} fold={fold_number}/{len(subjects)} "
            f"subject={cached['subject'][0]} MAE={overall['MAE']:.6f}",
            flush=True,
        )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    fold_manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "fold_manifest.csv"
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path).fillna("")
        if not existing.astype(str).equals(fold_manifest.astype(str)):
            raise RuntimeError("The existing fold manifest differs from the current folds.")
    else:
        fold_manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    payloads = [np.load(path, allow_pickle=True) for path in fold_paths]
    np.savez_compressed(
        output_dir / f"{args.model}_predictions.npz",
        signature=np.asarray([run_signature]),
        subjects=np.concatenate([item["subject"].astype(str) for item in payloads]),
        target_te=te_values[target_idx],
        true_z=np.concatenate([item["true_z"] for item in payloads], axis=0),
        pred_z=np.concatenate([item["pred_z"] for item in payloads], axis=0),
        true_native=np.concatenate([item["true_native"] for item in payloads], axis=0),
        pred_native=np.concatenate([item["pred_native"] for item in payloads], axis=0),
        variables=matrix["variables"],
        models=matrix["models"],
        metrics=matrix["metrics"],
        features=matrix["features"],
        roi_ids=matrix["roi_ids"],
        roi_labels=matrix["roi_labels"],
    )
    print(f"Completed {len(subjects)}-subject LOSO: {args.model}")


if __name__ == "__main__":
    main()
