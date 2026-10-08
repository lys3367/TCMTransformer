from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path.cwd()
CODE_DIR = Path(__file__).resolve().parent

import importlib

common = importlib.import_module("teprediction.common")
external_module = importlib.import_module("teprediction.external_data")


def load_clean_trainer():
    path = Path(__file__).with_name("train.py")
    spec = importlib.util.spec_from_file_location("clean49_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical clean-49 training and ScienceDB external testing.")
    parser.add_argument("--model", required=True, choices=["itransformer", "timesnet", "mole", "dlinear"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--fold-manifest", default="outputs/table1_clean49/fold_manifest.csv")
    parser.add_argument("--external-csv", default="data/raw/ScienceDB_BN_JHU_metric_results_interpolated_qc.csv")
    parser.add_argument("--output-dir", default="outputs_sciencedb_clean49")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    trainer = load_clean_trainer()
    config = common.load_config(args.config)
    matrix = common.load_matrix(config)
    internal_all = matrix["x_raw"].astype(np.float32)
    all_subjects = matrix["subjects"].astype(str)
    te_values = matrix["te_values"].astype(int)
    metadata = pd.read_csv(common.resolve_path(config, "metadata_file"))
    repos_dir = common.resolve_path(config, "external_repos_dir")
    input_idx = common.te_indices(te_values, config["fixed_input_te"])
    target_idx = common.te_indices(te_values, config["fixed_target_te"])

    fold_manifest = Path(args.fold_manifest)
    if not fold_manifest.is_absolute():
        fold_manifest = PROJECT / fold_manifest
    clean_subjects = pd.read_csv(fold_manifest)["test_subject"].astype(str).tolist()
    if len(clean_subjects) != 49 or len(set(clean_subjects)) != 49:
        raise ValueError("The canonical fold manifest does not define 49 unique subjects.")
    lookup = {subject: index for index, subject in enumerate(all_subjects)}
    clean_idx = np.asarray([index for index, subject in enumerate(all_subjects) if subject in set(clean_subjects)])
    if len(clean_idx) != 49 or any(subject not in lookup for subject in clean_subjects):
        raise ValueError("Clean-49 subjects are not fully represented in the internal matrix.")
    internal = internal_all[clean_idx]
    internal_subjects = all_subjects[clean_idx]

    external_csv = Path(args.external_csv)
    if not external_csv.is_absolute():
        external_csv = PROJECT / external_csv
    external, external_subjects = external_module.load_external_matrix(
        external_csv, metadata, list(map(int, te_values))
    )
    raw = np.concatenate([internal, external], axis=0)

    # External evaluation uses the complete clean-49 cohort as the development pool.
    # Ten deterministic participants are reserved for checkpoint selection; the
    # remaining 39 fit the model and its fold-specific standardization parameters.
    seed = int(config["random_seed"])
    permutation = np.random.default_rng(seed).permutation(len(internal))
    val_idx = permutation[:10]
    train_idx = permutation[10:]
    test_idx = np.arange(len(internal), len(raw))

    true, prediction, details = trainer.train_fold(
        args.model,
        raw,
        metadata,
        train_idx,
        val_idx,
        test_idx,
        input_idx,
        target_idx,
        repos_dir,
        seed,
        args.device,
        args.batch_size,
        args.max_epochs,
        args.patience,
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = common.regression_metrics(true, prediction)
    subject_rows = []
    for index, subject in enumerate(external_subjects.astype(str)):
        metrics = common.regression_metrics(true[index], prediction[index])
        mae_by_te = np.mean(np.abs(prediction[index] - true[index]), axis=1)
        subject_rows.append(
            {
                "model": args.model,
                "external_subject": subject,
                **metrics,
                "MAE_TE125": float(mae_by_te[0]),
                "MAE_TE135": float(mae_by_te[1]),
            }
        )
    pd.DataFrame(subject_rows).to_csv(
        output_dir / f"{args.model}_external_subject_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {
                "model": args.model,
                "n_internal_development": 49,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_external_test": len(test_idx),
                "MAE": overall["MAE"],
                "RMSE": overall["RMSE"],
                "R2": overall["R2"],
                "val_RMSE": float(np.asarray(details.get("val_RMSE", np.nan)).reshape(-1).mean()),
            }
        ]
    ).to_csv(output_dir / f"{args.model}_external_summary.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output_dir / f"{args.model}_external_predictions.npz",
        subjects=external_subjects.astype(str),
        target_te=te_values[target_idx],
        true_z=true.astype(np.float32),
        pred_z=prediction.astype(np.float32),
        internal_train_subjects=internal_subjects[train_idx],
        internal_val_subjects=internal_subjects[val_idx],
        variables=matrix["variables"],
        roi_ids=matrix["roi_ids"],
        roi_labels=matrix["roi_labels"],
    )
    common.summarize_by_roi(true, prediction, metadata).to_csv(
        output_dir / f"{args.model}_external_by_roi.csv", index=False, encoding="utf-8-sig"
    )
    print(
        f"{args.model}: MAE={overall['MAE']:.6f}, RMSE={overall['RMSE']:.6f}, "
        f"R2={overall['R2']:.6f}; internal train/val={len(train_idx)}/{len(val_idx)}, "
        f"external n={len(test_idx)}"
    )


if __name__ == "__main__":
    main()
