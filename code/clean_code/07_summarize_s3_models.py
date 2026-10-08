from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
MODELS = ("itransformer", "mole", "dlinear", "timesnet")
DISPLAY = {
    "itransformer": "iTransformer",
    "mole": "MoLE",
    "dlinear": "DLinear",
    "timesnet": "TimesNet",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a four-model S3 experiment.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--expected-subjects", type=int, required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT / input_dir

    rows = []
    subject_order = None
    for model in MODELS:
        table = pd.read_csv(input_dir / f"{model}_fold_metrics.csv").sort_values("fold")
        if len(table) != args.expected_subjects or table["test_subject"].nunique() != args.expected_subjects:
            raise ValueError(f"{model}: expected {args.expected_subjects} unique folds")
        current_order = table["test_subject"].astype(str).tolist()
        if subject_order is None:
            subject_order = current_order
        elif current_order != subject_order:
            raise ValueError(f"Subject order mismatch for {model}")
        row = {"model": DISPLAY[model], "n": len(table)}
        for metric in ("MAE", "RMSE", "R2"):
            row[f"{metric}_mean"] = table[metric].mean()
            row[f"{metric}_std"] = table[metric].std(ddof=1)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("MAE_mean")
    summary.to_csv(input_dir / "s3_model_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
