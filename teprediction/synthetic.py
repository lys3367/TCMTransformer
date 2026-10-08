"""Generate artificial ROI tables without patient data."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def generate(destination, subjects=8, seed=17):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Synthetic output must be absent or empty; refusing to overwrite.")
    if subjects < 8:
        raise ValueError("Use at least eight synthetic subjects.")
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    ids = [f"synthetic-{i:03d}" for i in range(subjects)]
    for subject in ids:
        intercept, slope = rng.normal(0, .08), rng.normal(.05, .01)
        for te in [75, 85, 95, 105, 115, 125, 135]:
            for roi in ["W_1", "G_1"]:
                for metric, baseline in [("FA", .5), ("MD", 1.0)]:
                    mean = baseline + intercept + slope * (te-75)/60 + .02*(roi == "G_1") + rng.normal(0, .005)
                    rows.append([subject, te, "DTI", metric, roi, "Synthetic "+roi, mean, .01])
    pd.DataFrame(rows, columns=["Subject", "TE", "Model", "Metric", "ROI_ID", "ROI_Name", "Mean", "Std"]).to_csv(destination/"roi_metrics.csv", index=False)
    pd.DataFrame({"subject": ids}).to_csv(destination/"cohort.csv", index=False)
    pd.DataFrame({"subject": ids, "group": ["synthetic-A" if i % 2 else "synthetic-B" for i in range(subjects)]}).to_csv(destination/"groups.csv", index=False)
    base = destination.as_posix()
    config = {"raw_csv": base+"/roi_metrics.csv", "matrix_file": base+"/processed/model_matrix_complete7te.npz",
              "split_file": base+"/processed/subject_splits.csv", "metadata_file": base+"/processed/variable_metadata.csv",
              "output_dir": base+"/audit", "external_repos_dir": "", "random_seed": 20260623,
              "expected_te": [75,85,95,105,115,125,135], "fixed_input_te": [75,85,95,105,115],
              "fixed_target_te": [125,135], "train_ratio": .7, "val_ratio": .15}
    (destination/"config.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="example")
    parser.add_argument("--subjects", type=int, default=8)
    args = parser.parse_args()
    generate(args.output, args.subjects)
    print("Generated entirely synthetic data in", args.output)
