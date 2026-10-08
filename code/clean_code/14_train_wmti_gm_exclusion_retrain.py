from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT / "code"
PRIMARY_SCRIPT = Path(__file__).with_name("01_train_table1_clean49.py")
DEFAULT_OUTPUT_DIR = PROJECT / "outputs" / "wmti_gm_exclusion_retrain"
DEFAULT_REFERENCE_MANIFEST = PROJECT / "outputs" / "table1_clean49" / "fold_manifest.csv"

EXPECTED_SUBJECTS = 49
EXPECTED_ORIGINAL_VARIABLES = 11248
EXPECTED_EXCLUDED_VARIABLES = 984
EXPECTED_RETAINED_VARIABLES = 10264
EXPECTED_GM_ROIS = 246
EXPECTED_WM_ROIS = 50
EXPECTED_WMTI_METRICS = {"AWF", "Axonal", "Hindered_AD", "Hindered_RD"}
MODEL_CHOICES = ("itransformer", "timesnet", "mole", "dlinear")


def import_primary_module():
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    spec = importlib.util.spec_from_file_location("canonical_clean49_primary", PRIMARY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import canonical primary script: {PRIMARY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cohort_without_training(subjects: np.ndarray, manifest: Path) -> np.ndarray:
    requested = pd.read_csv(manifest)["subject"].astype(str).tolist()
    if len(requested) != EXPECTED_SUBJECTS or len(set(requested)) != EXPECTED_SUBJECTS:
        raise ValueError(f"The clean-49 manifest must contain {EXPECTED_SUBJECTS} unique subjects.")
    missing = sorted(set(requested) - set(subjects.astype(str)))
    if missing:
        raise ValueError(f"Matrix is missing clean-49 subjects: {missing}")
    requested_set = set(requested)
    return np.asarray(
        [i for i, subject in enumerate(subjects.astype(str)) if subject in requested_set], dtype=int
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_variable_masks(matrix: dict[str, np.ndarray]):
    models = matrix["models"].astype(str)
    metrics = matrix["metrics"].astype(str)
    roi_ids = matrix["roi_ids"].astype(str)
    roi_labels = matrix["roi_labels"].astype(str)
    variables = matrix["variables"].astype(str)

    if len(variables) != EXPECTED_ORIGINAL_VARIABLES:
        raise ValueError(
            f"Expected {EXPECTED_ORIGINAL_VARIABLES} original variables, found {len(variables)}."
        )

    unique_rois = pd.DataFrame({"roi_id": roi_ids, "roi_label": roi_labels}).drop_duplicates()
    gm_rois = sorted(unique_rois.loc[unique_rois["roi_id"].str.startswith("G_"), "roi_id"])
    wm_rois = sorted(unique_rois.loc[unique_rois["roi_id"].str.startswith("W_"), "roi_id"])
    if len(gm_rois) != EXPECTED_GM_ROIS or len(wm_rois) != EXPECTED_WM_ROIS:
        raise ValueError(
            f"ROI audit failed: GM={len(gm_rois)} (expected {EXPECTED_GM_ROIS}), "
            f"WM={len(wm_rois)} (expected {EXPECTED_WM_ROIS})."
        )

    is_wmti = np.char.upper(models) == "WMTI"
    is_gm = np.char.startswith(roi_ids, "G_")
    exclude = is_wmti & is_gm
    keep = ~exclude
    wmti_metrics = set(np.unique(metrics[is_wmti]).tolist())

    if wmti_metrics != EXPECTED_WMTI_METRICS:
        raise ValueError(
            f"WMTI metric audit failed: found {sorted(wmti_metrics)}, "
            f"expected {sorted(EXPECTED_WMTI_METRICS)}."
        )
    if int(exclude.sum()) != EXPECTED_EXCLUDED_VARIABLES:
        raise ValueError(
            f"Expected {EXPECTED_EXCLUDED_VARIABLES} WMTI-GM variables, found {int(exclude.sum())}."
        )
    if int(keep.sum()) != EXPECTED_RETAINED_VARIABLES:
        raise ValueError(
            f"Expected {EXPECTED_RETAINED_VARIABLES} retained variables, found {int(keep.sum())}."
        )

    metadata = pd.DataFrame(
        {
            "original_variable_index": np.arange(len(variables), dtype=int),
            "model": models,
            "metric": metrics,
            "feature": matrix["features"].astype(str),
            "roi_id": roi_ids,
            "roi_label": roi_labels,
            "variable": variables,
        }
    )
    excluded = metadata.loc[exclude].copy()
    excluded["reason"] = "WMTI metric in Brainnetome gray-matter ROI"
    retained = metadata.loc[keep].copy()
    # The canonical training pipeline indexes the current matrix through
    # `variable_index`. After filtering, these indices must be contiguous in
    # the reduced 10,264-variable matrix, while `original_variable_index`
    # preserves traceability to the original 11,248-variable matrix.
    retained.insert(0, "retained_variable_index", np.arange(len(retained), dtype=int))
    retained.insert(0, "variable_index", np.arange(len(retained), dtype=int))
    if not np.array_equal(
        retained["variable_index"].to_numpy(dtype=int),
        np.arange(EXPECTED_RETAINED_VARIABLES, dtype=int),
    ):
        raise RuntimeError("Reduced-matrix variable_index is not contiguous from 0 to 10,263.")
    return keep, exclude, retained.reset_index(drop=True), excluded.reset_index(drop=True)


def build_fold_manifest(primary, subjects: np.ndarray, seed: int) -> pd.DataFrame:
    folds = primary.make_folds(len(subjects), seed)
    rows = []
    for fold_number in range(1, len(subjects) + 1):
        train_idx, val_idx, test_idx = primary.split_fold(folds, fold_number, seed)
        rows.append(
            {
                "fold": fold_number,
                "fold_seed": seed + fold_number,
                "test_subject": str(subjects[test_idx[0]]),
                "train_subjects": ";".join(subjects[train_idx].astype(str)),
                "val_subjects": ";".join(subjects[val_idx].astype(str)),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
            }
        )
    return pd.DataFrame(rows)


def assert_manifest_matches(generated: pd.DataFrame, reference_path: Path) -> None:
    if not reference_path.exists():
        raise FileNotFoundError(f"Canonical fold manifest is missing: {reference_path}")
    reference = pd.read_csv(reference_path).fillna("")
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
    if list(reference.columns) != columns:
        raise ValueError(f"Unexpected canonical manifest columns: {reference.columns.tolist()}")
    if not reference.astype(str).equals(generated[columns].fillna("").astype(str)):
        mismatch = reference.astype(str).ne(generated[columns].astype(str)).any(axis=1)
        bad_folds = reference.loc[mismatch, "fold"].tolist()
        raise RuntimeError(
            f"Generated folds do not match the canonical clean-49 manifest. Bad folds: {bad_folds}"
        )


def audit_frozen_manifest(reference_path: Path, subjects: np.ndarray, seed: int) -> pd.DataFrame:
    if not reference_path.exists():
        raise FileNotFoundError(f"Canonical fold manifest is missing: {reference_path}")
    table = pd.read_csv(reference_path).sort_values("fold").reset_index(drop=True)
    if len(table) != EXPECTED_SUBJECTS or table["fold"].astype(int).tolist() != list(
        range(1, EXPECTED_SUBJECTS + 1)
    ):
        raise ValueError("Canonical manifest does not contain folds 1--49 exactly once.")
    if table["test_subject"].astype(str).nunique() != EXPECTED_SUBJECTS:
        raise ValueError("Canonical manifest does not hold out every clean-49 subject exactly once.")
    if set(table["test_subject"].astype(str)) != set(subjects.astype(str)):
        raise ValueError("Canonical manifest test subjects differ from the clean-49 cohort.")
    expected_seed = seed + table["fold"].astype(int)
    if not np.array_equal(table["fold_seed"].astype(int), expected_seed):
        raise ValueError("Canonical manifest fold seeds do not equal base_seed + fold_number.")
    for row in table.itertuples(index=False):
        train = str(row.train_subjects).split(";")
        val = str(row.val_subjects).split(";")
        test = [str(row.test_subject)]
        if len(train) != 38 or len(val) != 10 or len(test) != 1:
            raise ValueError(f"Fold {row.fold} is not a 38/10/1 split.")
        if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
            raise ValueError(f"Fold {row.fold} has overlapping train/validation/test subjects.")
        if set(train + val + test) != set(subjects.astype(str)):
            raise ValueError(f"Fold {row.fold} does not partition all 49 subjects.")
    return table


def write_csv_checked(table: pd.DataFrame, path: Path) -> None:
    if path.exists():
        existing = pd.read_csv(path).fillna("")
        if not existing.astype(str).equals(table.fillna("").astype(str)):
            raise RuntimeError(f"Existing audit file differs from the current analysis: {path}")
        return
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    table.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def scalar(details: dict, key: str, default: float = np.nan) -> float:
    value = details.get(key, default)
    return float(np.asarray(value).reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean-49 WMTI gray-matter exclusion retraining sensitivity analysis."
    )
    parser.add_argument("--model", choices=MODEL_CHOICES)
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default="data/processed/clean49_subjects.csv")
    parser.add_argument("--reference-fold-manifest", default=str(DEFAULT_REFERENCE_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    if not args.audit_only and args.model is None:
        parser.error("--model is required unless --audit-only is used.")
    if args.batch_size != 1 or args.max_epochs != 100 or args.patience != 12:
        raise ValueError(
            "This sensitivity analysis must retain canonical settings: "
            "batch_size=1, max_epochs=100, patience=12."
        )

    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    common = importlib.import_module("00_common")
    config = common.load_config(args.config)
    matrix = common.load_matrix(config)
    raw_all = matrix["x_raw"].astype(np.float32)
    subjects_all = matrix["subjects"].astype(str)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT / manifest_path
    cohort_idx = load_cohort_without_training(subjects_all, manifest_path)
    raw_clean49 = raw_all[cohort_idx]
    subjects = subjects_all[cohort_idx]

    keep_mask, exclude_mask, retained_meta, excluded_meta = build_variable_masks(matrix)
    raw = raw_clean49[:, :, keep_mask]
    if raw.shape != (EXPECTED_SUBJECTS, 7, EXPECTED_RETAINED_VARIABLES):
        raise ValueError(f"Restricted matrix shape audit failed: {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("Restricted matrix contains non-finite values.")

    te_values = matrix["te_values"].astype(int)
    input_idx = common.te_indices(te_values, config["fixed_input_te"])
    target_idx = common.te_indices(te_values, config["fixed_target_te"])
    seed = int(config["random_seed"])

    reference_manifest = Path(args.reference_fold_manifest)
    if not reference_manifest.is_absolute():
        reference_manifest = PROJECT / reference_manifest
    frozen_manifest = audit_frozen_manifest(reference_manifest, subjects, seed)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_checked(frozen_manifest, output_dir / "fold_manifest.csv")
    # `variable_index` is an in-memory adapter required by the canonical
    # training code. Keep the audit CSV schema stable across the preflight fix.
    write_csv_checked(
        retained_meta.drop(columns=["variable_index"]),
        output_dir / "retained_variables.csv",
    )
    write_csv_checked(excluded_meta, output_dir / "excluded_wmti_gm_variables.csv")

    preflight = {
        "analysis": "WMTI gray-matter exclusion retraining",
        "primary_script": str(PRIMARY_SCRIPT),
        "primary_script_sha256": file_sha256(PRIMARY_SCRIPT),
        "reference_fold_manifest": str(reference_manifest),
        "reference_fold_manifest_sha256": file_sha256(reference_manifest),
        "cohort_manifest": str(manifest_path),
        "cohort_manifest_sha256": file_sha256(manifest_path),
        "n_subjects": len(subjects),
        "matrix_shape_original_clean49": list(raw_clean49.shape),
        "matrix_shape_restricted_clean49": list(raw.shape),
        "n_original_variables": EXPECTED_ORIGINAL_VARIABLES,
        "n_excluded_variables": int(exclude_mask.sum()),
        "n_retained_variables": int(keep_mask.sum()),
        "wmti_metrics": sorted(EXPECTED_WMTI_METRICS),
        "n_gm_rois": EXPECTED_GM_ROIS,
        "n_wm_rois": EXPECTED_WM_ROIS,
        "input_te": te_values[input_idx].tolist(),
        "target_te": te_values[target_idx].tolist(),
        "random_seed": seed,
        "frozen_canonical_manifest_passed_internal_audit": True,
        "training_regenerates_splits_with_primary_functions_and_requires_exact_match": True,
        "n_train_per_fold": sorted(frozen_manifest["n_train"].unique().tolist()),
        "n_val_per_fold": sorted(frozen_manifest["n_val"].unique().tolist()),
        "n_test_per_fold": sorted(frozen_manifest["n_test"].unique().tolist()),
    }
    (output_dir / "preflight_audit.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )
    if args.audit_only:
        print(json.dumps(preflight, indent=2), flush=True)
        return

    primary = import_primary_module()
    canonical_cohort_idx = primary.load_cohort(subjects_all, manifest_path, EXPECTED_SUBJECTS)
    if not np.array_equal(canonical_cohort_idx, cohort_idx):
        raise RuntimeError("Local preflight cohort order differs from the canonical cohort loader.")
    folds = primary.make_folds(len(subjects), seed)
    generated_manifest = build_fold_manifest(primary, subjects, seed)
    assert_manifest_matches(generated_manifest, reference_manifest)

    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    repos_dir = primary.common.resolve_path(config, "external_repos_dir")
    model_sources = {
        "itransformer": repos_dir / "iTransformer/model/iTransformer.py",
        "timesnet": repos_dir / "Time-Series-Library/models/TimesNet.py",
        "dlinear": repos_dir / "LTSF-Linear/models/DLinear.py",
        "mole": repos_dir / "MoLE/models/MoLE_DLinear.py",
    }
    wrapper_source = CODE_DIR / (
        "73_train_extra_models.py" if args.model == "mole" else "03_train_neural.py"
    )
    primary_settings_path = PROJECT / "outputs" / "table1_clean49" / args.model / "run_settings.json"
    if not primary_settings_path.exists():
        raise FileNotFoundError(f"Primary run settings are missing: {primary_settings_path}")
    primary_settings = json.loads(primary_settings_path.read_text(encoding="utf-8"))
    for key, expected in {
        "random_seed": seed,
        "input_te": te_values[input_idx].tolist(),
        "target_te": te_values[target_idx].tolist(),
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
    }.items():
        if primary_settings.get(key) != expected:
            raise RuntimeError(
                f"Canonical setting mismatch for {key}: primary={primary_settings.get(key)}, current={expected}"
            )

    run_settings = {
        "analysis": "wmti_gm_exclusion_retrain",
        "model": args.model,
        "cohort_subjects": subjects.tolist(),
        "random_seed": seed,
        "input_te": te_values[input_idx].tolist(),
        "target_te": te_values[target_idx].tolist(),
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "matrix_file": config["matrix_file"],
        "n_original_variables": EXPECTED_ORIGINAL_VARIABLES,
        "n_excluded_variables": EXPECTED_EXCLUDED_VARIABLES,
        "n_variables": EXPECTED_RETAINED_VARIABLES,
        "variable_rule": "exclude model == WMTI and roi_id startswith G_",
        "retained_variable_sha256": hashlib.sha256(
            "\n".join(retained_meta["variable"].astype(str)).encode("utf-8")
        ).hexdigest(),
        "canonical_primary_settings_signature": primary_settings["signature"],
        "canonical_fold_manifest_sha256": file_sha256(reference_manifest),
        "canonical_primary_script_sha256": file_sha256(PRIMARY_SCRIPT),
        "wrapper_sha256": file_sha256(wrapper_source),
        "official_model_sha256": file_sha256(model_sources[args.model]),
    }
    run_signature = primary.signature(run_settings)
    model_dir = output_dir / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "run_settings.json").write_text(
        json.dumps({**run_settings, "signature": run_signature}, indent=2), encoding="utf-8"
    )

    metric_rows = []
    fold_paths = []
    for fold_number in range(1, len(subjects) + 1):
        train_idx, val_idx, test_idx = primary.split_fold(folds, fold_number, seed)
        fold_seed = seed + fold_number
        reference_row = generated_manifest.iloc[fold_number - 1]
        if (
            str(subjects[test_idx[0]]) != str(reference_row["test_subject"])
            or ";".join(subjects[train_idx]) != str(reference_row["train_subjects"])
            or ";".join(subjects[val_idx]) != str(reference_row["val_subjects"])
            or fold_seed != int(reference_row["fold_seed"])
        ):
            raise RuntimeError(f"Fold {fold_number} changed after preflight validation.")

        fold_path = model_dir / f"fold_{fold_number:02d}.npz"
        fold_paths.append(fold_path)
        if fold_path.exists():
            cached = np.load(fold_path, allow_pickle=True)
            cached_signature = str(cached["signature"].reshape(-1)[0])
            if cached_signature != run_signature:
                raise RuntimeError(
                    f"Configuration mismatch in {fold_path}: cached={cached_signature}, current={run_signature}."
                )
        else:
            true_z, pred_z, details = primary.train_fold(
                args.model,
                raw,
                retained_meta,
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
            _, mean, std = primary.common.standardize_from_train(raw, train_idx)
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
                n_variables=np.asarray([EXPECTED_RETAINED_VARIABLES]),
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
        if true_z.shape != (1, 2, EXPECTED_RETAINED_VARIABLES) or pred_z.shape != true_z.shape:
            raise RuntimeError(f"Fold {fold_number} has invalid prediction shape: {true_z.shape}, {pred_z.shape}")
        if int(cached["n_variables"][0]) != EXPECTED_RETAINED_VARIABLES:
            raise RuntimeError(f"Fold {fold_number} does not contain 10,264 variables.")
        overall = primary.common.regression_metrics(true_z, pred_z)
        target_metrics = [
            primary.common.regression_metrics(true_z[:, i, :], pred_z[:, i, :])
            for i in range(2)
        ]
        identity_error = abs(overall["MAE"] - np.mean([item["MAE"] for item in target_metrics]))
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
                "n_variables": EXPECTED_RETAINED_VARIABLES,
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
            output_dir / f"{args.model}_fold_metrics.csv", index=False, encoding="utf-8-sig"
        )
        print(
            f"{args.model} fold={fold_number}/49 subject={cached['subject'][0]} "
            f"variables={EXPECTED_RETAINED_VARIABLES} MAE={overall['MAE']:.6f}",
            flush=True,
        )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

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
        retained_original_indices=retained_meta["original_variable_index"].to_numpy(dtype=int),
        variables=retained_meta["variable"].to_numpy(dtype=str),
        models=retained_meta["model"].to_numpy(dtype=str),
        metrics=retained_meta["metric"].to_numpy(dtype=str),
        features=retained_meta["feature"].to_numpy(dtype=str),
        roi_ids=retained_meta["roi_id"].to_numpy(dtype=str),
        roi_labels=retained_meta["roi_label"].to_numpy(dtype=str),
    )
    print(f"Completed clean-49 WMTI-GM exclusion retraining: {args.model}", flush=True)


if __name__ == "__main__":
    main()
