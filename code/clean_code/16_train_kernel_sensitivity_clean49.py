from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT / "code"
PRIMARY_SCRIPT = Path(__file__).with_name("01_train_table1_clean49.py")
DEFAULT_OUTPUT_DIR = PROJECT / "outputs" / "kernel_sensitivity_clean49"
DEFAULT_PRIMARY_DIR = PROJECT / "outputs" / "table1_clean49"
EXPECTED_SUBJECTS = 49
EXPECTED_VARIABLES = 11248
EXPECTED_INPUT_TE = [75, 85, 95, 105, 115]
EXPECTED_TARGET_TE = [125, 135]
EXPECTED_KERNELS = (3, 5)
MODEL_CHOICES = ("dlinear", "mole")


def import_primary_module():
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    spec = importlib.util.spec_from_file_location("kernel_sensitivity_primary", PRIMARY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import canonical primary script: {PRIMARY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


primary = import_primary_module()
common = importlib.import_module("00_common")
neural = importlib.import_module("03_train_neural")
extra = importlib.import_module("73_train_extra_models")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(subjects: np.ndarray, seed: int) -> pd.DataFrame:
    folds = primary.make_folds(len(subjects), seed)
    rows = []
    for fold in range(1, len(subjects) + 1):
        train_idx, val_idx, test_idx = primary.split_fold(folds, fold, seed)
        rows.append(
            {
                "fold": fold,
                "fold_seed": seed + fold,
                "test_subject": str(subjects[test_idx[0]]),
                "train_subjects": ";".join(subjects[train_idx].astype(str)),
                "val_subjects": ";".join(subjects[val_idx].astype(str)),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
            }
        )
    return pd.DataFrame(rows)


def assert_manifest_matches(generated: pd.DataFrame, reference_path: Path) -> pd.DataFrame:
    columns = [
        "fold",
        "fold_seed",
        "test_subject",
        "train_subjects",
        "val_subjects",
        "n_train",
        "n_val",
        "n_test",
    ]
    reference = pd.read_csv(reference_path).sort_values("fold").reset_index(drop=True)
    generated = generated.sort_values("fold").reset_index(drop=True)
    if reference.columns.tolist() != columns:
        raise ValueError(f"Unexpected canonical manifest columns: {reference.columns.tolist()}")
    if len(reference) != EXPECTED_SUBJECTS:
        raise ValueError(f"Expected {EXPECTED_SUBJECTS} canonical folds, found {len(reference)}.")
    if not reference.astype(str).equals(generated[columns].astype(str)):
        bad = reference.astype(str).ne(generated[columns].astype(str)).any(axis=1)
        raise RuntimeError(
            "Generated folds differ from canonical clean-49 folds: "
            f"{reference.loc[bad, 'fold'].astype(int).tolist()}"
        )
    return reference


def load_canonical_data(config_path: str | None, manifest_path: Path, primary_dir: Path):
    config = common.load_config(config_path)
    matrix = common.load_matrix(config)
    raw_all = matrix["x_raw"].astype(np.float32)
    subjects_all = matrix["subjects"].astype(str)
    cohort_manifest = PROJECT / "data" / "processed" / "clean49_subjects.csv"
    cohort_idx = primary.load_cohort(subjects_all, cohort_manifest, EXPECTED_SUBJECTS)
    raw = raw_all[cohort_idx]
    subjects = subjects_all[cohort_idx]
    te_values = matrix["te_values"].astype(int)
    input_idx = common.te_indices(te_values, config["fixed_input_te"])
    target_idx = common.te_indices(te_values, config["fixed_target_te"])
    if raw.shape != (EXPECTED_SUBJECTS, 7, EXPECTED_VARIABLES):
        raise ValueError(f"Unexpected clean-49 matrix shape: {raw.shape}")
    if te_values[input_idx].tolist() != EXPECTED_INPUT_TE:
        raise ValueError(f"Input TE mismatch: {te_values[input_idx].tolist()}")
    if te_values[target_idx].tolist() != EXPECTED_TARGET_TE:
        raise ValueError(f"Target TE mismatch: {te_values[target_idx].tolist()}")
    seed = int(config["random_seed"])
    reference = assert_manifest_matches(build_manifest(subjects, seed), manifest_path)
    for model in MODEL_CHOICES:
        settings = json.loads((primary_dir / model / "run_settings.json").read_text(encoding="utf-8"))
        expected = {
            "random_seed": seed,
            "input_te": EXPECTED_INPUT_TE,
            "target_te": EXPECTED_TARGET_TE,
            "batch_size": 1,
            "max_epochs": 100,
            "patience": 12,
            "n_variables": EXPECTED_VARIABLES,
        }
        for key, value in expected.items():
            if settings.get(key) != value:
                raise RuntimeError(f"Canonical {model} setting mismatch for {key}: {settings.get(key)}")
    repos_dir = common.resolve_path(config, "external_repos_dir")
    return config, raw, subjects, te_values, input_idx, target_idx, seed, repos_dir, reference


def core_model(model: nn.Module) -> nn.Module:
    return model.model if hasattr(model, "model") else model


def set_kernel(model: nn.Module, kernel: int) -> nn.Module:
    core = core_model(model)
    if not hasattr(core, "decompsition"):
        raise AttributeError(f"{type(core).__name__} has no decomposition module.")
    old = core.decompsition
    old_kernel = int(old.moving_avg.kernel_size)
    if old_kernel != 25:
        raise RuntimeError(f"Reference model kernel is {old_kernel}, expected 25.")
    core.decompsition = type(old)(kernel)
    if int(core.decompsition.moving_avg.kernel_size) != kernel:
        raise RuntimeError("Failed to install the requested sensitivity kernel.")
    return model


def build_model(model_name: str, kernel: int, seq_len: int, pred_len: int, n_variables: int, repos_dir: Path):
    if model_name == "dlinear":
        model = neural.build_model("dlinear", seq_len, pred_len, n_variables, repos_dir)
    elif model_name == "mole":
        model = extra.build_extra_model("mole", seq_len, pred_len, n_variables, repos_dir)
    else:
        raise ValueError(model_name)
    return set_kernel(model, kernel)


def cpu_sanity_check(repos_dir: Path, n_variables: int, output_path: Path) -> None:
    lines = [
        "Kernel sensitivity CPU sanity check",
        "===================================",
        f"torch={torch.__version__}",
        f"n_variables={n_variables}",
    ]
    x = torch.linspace(-1.0, 1.0, 5 * n_variables, dtype=torch.float32).reshape(
        1, 5, n_variables
    )
    for model_name in MODEL_CHOICES:
        common.set_seed(20260623)
        model = (
            neural.build_model("dlinear", 5, 2, n_variables, repos_dir)
            if model_name == "dlinear"
            else extra.build_extra_model("mole", 5, 2, n_variables, repos_dir)
        )
        model.eval()
        with torch.no_grad():
            reference_output = model(x)
        model = set_kernel(model, 25)
        with torch.no_grad():
            replaced_k25_output = model(x)
        max_difference = float(torch.max(torch.abs(reference_output - replaced_k25_output)).item())
        if max_difference != 0.0:
            raise RuntimeError(f"{model_name}: replacing k25 changed reference output by {max_difference}")
        for kernel in EXPECTED_KERNELS:
            core = core_model(model)
            core.decompsition = type(core.decompsition)(kernel)
            caught = []
            with warnings.catch_warnings(record=True) as records, torch.no_grad():
                warnings.simplefilter("always")
                residual, moving = core.decompsition(x)
                output = model(x)
                caught.extend(str(item.message) for item in records)
            checks = {
                "input_shape": tuple(x.shape),
                "moving_shape": tuple(moving.shape),
                "residual_shape": tuple(residual.shape),
                "output_shape": tuple(output.shape),
                "finite": bool(
                    torch.isfinite(moving).all()
                    and torch.isfinite(residual).all()
                    and torch.isfinite(output).all()
                ),
                "warnings": len(caught),
            }
            expected_output = (1, 2, n_variables)
            if checks["moving_shape"] != tuple(x.shape) or checks["output_shape"] != expected_output:
                raise RuntimeError(f"{model_name} k{kernel}: shape audit failed: {checks}")
            if not checks["finite"] or checks["warnings"]:
                raise RuntimeError(f"{model_name} k{kernel}: finite/warning audit failed: {checks}")
            lines.append(f"{model_name} k={kernel}: {checks}")
        lines.append(f"{model_name} reference-k25 replacement max_abs_difference={max_difference}")
        del model
        gc.collect()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def train_model(
    model_name: str,
    kernel: int,
    x: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    input_idx: np.ndarray,
    target_idx: np.ndarray,
    repos_dir: Path,
    settings,
):
    common.set_seed(settings.seed)
    train_x = x[train_idx][:, input_idx, :]
    train_y = x[train_idx][:, target_idx, :]
    val_x = x[val_idx][:, input_idx, :]
    val_y = x[val_idx][:, target_idx, :]
    test_x = x[test_idx][:, input_idx, :]
    device = torch.device(settings.device)
    model = build_model(
        model_name, kernel, len(input_idx), len(target_idx), x.shape[2], repos_dir
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x.astype(np.float32)),
            torch.from_numpy(train_y.astype(np.float32)),
        ),
        batch_size=min(settings.batch_size, len(train_idx)),
        shuffle=True,
    )
    best_state = None
    best_val = np.inf
    best_epoch = 0
    wait = 0
    epochs_run = 0
    started = time.time()
    for epoch in range(1, settings.max_epochs + 1):
        epochs_run = epoch
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            if model_name == "dlinear":
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
                    prediction = model(batch_x)
                    squared_error += float(torch.sum((prediction - batch_y) ** 2).item())
                    n_values += int(batch_y.numel())
                val_loss = squared_error / max(n_values, 1)
            else:
                val_prediction = model(torch.from_numpy(val_x.astype(np.float32)).to(device))
                val_loss = float(
                    loss_fn(
                        val_prediction,
                        torch.from_numpy(val_y.astype(np.float32)).to(device),
                    ).item()
                )
        if not np.isfinite(val_loss):
            raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}.")
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
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Prediction contains NaN or Inf.")
    details = {
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "val_RMSE": float(np.sqrt(best_val)),
        "training_seconds": time.time() - started,
        "n_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    return prediction, details, best_state


def save_checkpoint(
    path: Path,
    state: dict[str, torch.Tensor],
    metadata: dict,
    checkpoint_dtype: str,
) -> None:
    dtype = torch.float16 if checkpoint_dtype == "fp16" else torch.float32
    cpu_state = {
        key: value.detach().cpu().to(dtype=dtype) if value.is_floating_point() else value.detach().cpu()
        for key, value in state.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": cpu_state, **metadata, "checkpoint_dtype": checkpoint_dtype}, path)
    del cpu_state
    gc.collect()


def fold_metrics_from_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    true_z = z["true_z"]
    pred_z = z["pred_z"]
    overall = common.regression_metrics(true_z, pred_z)
    target = [common.regression_metrics(true_z[:, i, :], pred_z[:, i, :]) for i in range(2)]
    identity_error = abs(overall["MAE"] - np.mean([item["MAE"] for item in target]))
    if identity_error > 1e-7:
        raise RuntimeError(f"Target MAE identity failed for {path}.")
    return {
        "model": str(z["model"][0]),
        "kernel": int(z["kernel"][0]),
        "fold": int(z["fold"][0]),
        "test_subject": str(z["subject"][0]),
        "fold_seed": int(z["fold_seed"][0]),
        "n_train": len(z["train_subjects"]),
        "n_val": len(z["val_subjects"]),
        "n_test": 1,
        "seq_len": int(z["seq_len"][0]),
        "pred_len": int(z["pred_len"][0]),
        "n_variables": int(z["n_variables"][0]),
        "n_parameters": int(z["n_parameters"][0]),
        "checkpoint_path": str(z["checkpoint_path"][0]),
        "MAE": overall["MAE"],
        "RMSE": overall["RMSE"],
        "R2": overall["R2"],
        "MAE_TE125": target[0]["MAE"],
        "MAE_TE135": target[1]["MAE"],
        "MAE_identity_error": identity_error,
        "best_epoch": float(z["best_epoch"][0]),
        "epochs_run": float(z["epochs_run"][0]),
        "val_RMSE": float(z["val_RMSE"][0]),
        "training_seconds": float(z["training_seconds"][0]),
        "signature": str(z["signature"][0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean-49 DLinear/MoLE kernel sensitivity.")
    parser.add_argument("--model", choices=MODEL_CHOICES)
    parser.add_argument("--kernel", type=int, choices=EXPECTED_KERNELS)
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default="data/processed/clean49_subjects.csv")
    parser.add_argument("--primary-dir", default="outputs/table1_clean49")
    parser.add_argument("--output-dir", default="outputs/kernel_sensitivity_clean49")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--checkpoint-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--sanity-variables", type=int, default=EXPECTED_VARIABLES)
    args = parser.parse_args()

    primary_dir = resolve(args.primary_dir)
    output_dir = resolve(args.output_dir)
    manifest_path = primary_dir / "fold_manifest.csv"
    (
        config,
        raw,
        subjects,
        te_values,
        input_idx,
        target_idx,
        seed,
        repos_dir,
        reference_manifest,
    ) = load_canonical_data(args.config, manifest_path, primary_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_manifest.to_csv(output_dir / "fold_manifest.csv", index=False, encoding="utf-8-sig")

    if args.sanity_only:
        cpu_sanity_check(
            repos_dir,
            args.sanity_variables,
            output_dir / "kernel_sensitivity_cpu_sanity_check.txt",
        )
        return
    if args.model is None or args.kernel is None:
        parser.error("--model and --kernel are required unless --sanity-only is used.")
    if (args.batch_size, args.max_epochs, args.patience) != (1, 100, 12):
        raise ValueError("Canonical sensitivity settings require batch=1, epochs=100, patience=12.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    model_name = args.model
    kernel = args.kernel
    setting_dir = output_dir / f"{model_name}_k{kernel}"
    checkpoint_dir = setting_dir / "checkpoints"
    setting_dir.mkdir(parents=True, exist_ok=True)
    primary_settings = json.loads(
        (primary_dir / model_name / "run_settings.json").read_text(encoding="utf-8")
    )
    source_path = (
        repos_dir / "LTSF-Linear/models/DLinear.py"
        if model_name == "dlinear"
        else repos_dir / "MoLE/models/MoLE_DLinear.py"
    )
    adapter_path = CODE_DIR / (
        "03_train_neural.py" if model_name == "dlinear" else "73_train_extra_models.py"
    )
    run_settings = {
        "analysis": "moving-average kernel sensitivity",
        "model": model_name,
        "kernel_size": kernel,
        "seq_len": len(input_idx),
        "pred_len": len(target_idx),
        "cohort_subjects": subjects.tolist(),
        "random_seed": seed,
        "input_te": te_values[input_idx].tolist(),
        "target_te": te_values[target_idx].tolist(),
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": 7e-4,
        "weight_decay": 1e-4,
        "loss": "MSELoss",
        "optimizer": "AdamW",
        "gradient_clip_norm": 1.0,
        "checkpoint_dtype": args.checkpoint_dtype,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "matrix_file": config["matrix_file"],
        "n_variables": int(raw.shape[2]),
        "canonical_primary_signature": primary_settings["signature"],
        "canonical_fold_manifest_sha256": sha256(manifest_path),
        "canonical_model_source_sha256": primary_settings["official_model_sha256"],
        "current_model_source_sha256": sha256(source_path),
        "canonical_wrapper_sha256": primary_settings["wrapper_sha256"],
        "current_wrapper_sha256": sha256(adapter_path),
        "sensitivity_script_sha256": sha256(Path(__file__)),
        "kernel_is_only_training_hyperparameter_change": True,
    }
    if run_settings["current_model_source_sha256"] != run_settings["canonical_model_source_sha256"]:
        raise RuntimeError(
            f"{model_name}: current model source differs from the canonical saved source hash."
        )
    if model_name == "dlinear" and (
        run_settings["current_wrapper_sha256"] != run_settings["canonical_wrapper_sha256"]
    ):
        raise RuntimeError("DLinear wrapper differs from the canonical saved wrapper hash.")
    if model_name == "mole":
        run_settings["canonical_wrapper_hash_matches_current"] = (
            run_settings["current_wrapper_sha256"] == run_settings["canonical_wrapper_sha256"]
        )
        run_settings["canonical_wrapper_audit_note"] = (
            "The historical wrapper file is unavailable locally; canonical model-source hash and "
            "parameter count identify MoLE_DLinear, while the current adapter reconstructs its "
            "documented deterministic time-mark interface."
        )
    signature = primary.signature(run_settings)
    (setting_dir / "run_settings.json").write_text(
        json.dumps({**run_settings, "signature": signature}, indent=2), encoding="utf-8"
    )

    folds = primary.make_folds(len(subjects), seed)
    for fold in range(1, EXPECTED_SUBJECTS + 1):
        train_idx, val_idx, test_idx = primary.split_fold(folds, fold, seed)
        fold_seed = seed + fold
        row = reference_manifest.iloc[fold - 1]
        if str(subjects[test_idx[0]]) != str(row.test_subject) or fold_seed != int(row.fold_seed):
            raise RuntimeError(f"Fold {fold} differs from canonical manifest.")
        fold_path = setting_dir / f"fold_{fold:02d}.npz"
        checkpoint_path = checkpoint_dir / f"fold_{fold:02d}_best.pt"
        if fold_path.exists():
            cached = np.load(fold_path, allow_pickle=True)
            if str(cached["signature"][0]) != signature:
                raise RuntimeError(f"Configuration mismatch in cached {fold_path}.")
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Cached fold exists but checkpoint is missing: {checkpoint_path}")
            print(f"[{model_name} k={kernel}] fold={fold}/49 cached", flush=True)
            continue

        x, mean, std = common.standardize_from_train(raw, train_idx)
        settings_class = neural.TrainSettings if model_name == "dlinear" else extra.TrainSettings
        settings = settings_class(
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=fold_seed,
            device=args.device,
        )
        prediction, details, best_state = train_model(
            model_name,
            kernel,
            x,
            train_idx,
            val_idx,
            test_idx,
            input_idx,
            target_idx,
            repos_dir,
            settings,
        )
        true_z = x[test_idx][:, target_idx, :]
        true_native = true_z * std[None, None, :] + mean[None, None, :]
        pred_native = prediction * std[None, None, :] + mean[None, None, :]
        checkpoint_metadata = {
            "model": model_name,
            "kernel": kernel,
            "fold": fold,
            "fold_seed": fold_seed,
            "test_subject": str(subjects[test_idx[0]]),
            "seq_len": len(input_idx),
            "pred_len": len(target_idx),
            "n_variables": raw.shape[2],
            "signature": signature,
        }
        save_checkpoint(checkpoint_path, best_state, checkpoint_metadata, args.checkpoint_dtype)
        np.savez_compressed(
            fold_path,
            signature=np.asarray([signature]),
            model=np.asarray([model_name]),
            kernel=np.asarray([kernel]),
            fold=np.asarray([fold]),
            fold_seed=np.asarray([fold_seed]),
            subject=subjects[test_idx],
            train_subjects=subjects[train_idx],
            val_subjects=subjects[val_idx],
            input_te=te_values[input_idx],
            target_te=te_values[target_idx],
            seq_len=np.asarray([len(input_idx)]),
            pred_len=np.asarray([len(target_idx)]),
            n_variables=np.asarray([raw.shape[2]]),
            n_parameters=np.asarray([details["n_parameters"]]),
            checkpoint_path=np.asarray([str(checkpoint_path)]),
            true_z=true_z.astype(np.float32),
            pred_z=prediction.astype(np.float32),
            true_native=true_native.astype(np.float32),
            pred_native=pred_native.astype(np.float32),
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            best_epoch=np.asarray([details["best_epoch"]]),
            epochs_run=np.asarray([details["epochs_run"]]),
            val_RMSE=np.asarray([details["val_RMSE"]]),
            training_seconds=np.asarray([details["training_seconds"]]),
        )
        metrics = fold_metrics_from_npz(fold_path)
        print(
            f"[{model_name} k={kernel}] fold={fold}/49 "
            f"subject={metrics['test_subject']} MAE={metrics['MAE']:.6f}",
            flush=True,
        )
        del x, best_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = [fold_metrics_from_npz(setting_dir / f"fold_{fold:02d}.npz") for fold in range(1, 50)]
    pd.DataFrame(rows).to_csv(
        output_dir / f"{model_name}_k{kernel}_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Complete: {model_name} k={kernel}, 49/49 folds", flush=True)


if __name__ == "__main__":
    main()
