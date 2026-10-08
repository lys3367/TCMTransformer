from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_INPUT_COLUMNS = {"subject", "TE", "model", "metric", "mean"}
OUTPUT_COLUMNS = ["Subject", "TE", "Model", "Metric", "ROI_ID", "ROI_Name", "Mean", "Std"]


def read_subjects(run_list: Path) -> list[str] | None:
    if not run_list.exists():
        return None
    subjects: list[str] = []
    for line in run_list.read_text(encoding="utf-8").splitlines():
        first = line.replace("\r", "").split(",", 1)[0].strip()
        if not first or first.startswith("#") or first.lower() in {"subject", "sub", "sid", "id"}:
            continue
        subjects.append(first)
    return subjects


def normalize_te(value: object) -> int | None:
    text = str(value).strip()
    if text.startswith("PA_TE"):
        text = text.replace("PA_TE", "", 1)
    elif text.startswith("TE"):
        text = text.replace("TE", "", 1)
    try:
        return int(float(text))
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect existing per-subject rs/lesion_metrics_mean.csv files into modelling format."
    )
    parser.add_argument("--root", default="/media/UG1/lys/dipy/data/MTE3_clean")
    parser.add_argument("--run-list", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    run_list = Path(args.run_list) if args.run_list else root / "run_list.csv"
    output = Path(args.output) if args.output else root / "lesion_metric_results_raw.csv"

    subjects = read_subjects(run_list)
    if subjects is None:
        files = sorted(root.glob("*/rs/lesion_metrics_mean.csv"))
    else:
        files = [root / subject / "rs" / "lesion_metrics_mean.csv" for subject in subjects]

    tables = []
    missing_files = 0
    invalid_files = 0
    dropped_na = 0

    for path in files:
        if not path.exists():
            missing_files += 1
            continue
        table = pd.read_csv(path)
        missing_columns = REQUIRED_INPUT_COLUMNS - set(table.columns)
        if missing_columns:
            invalid_files += 1
            print(f"[SKIP] {path}: missing columns {sorted(missing_columns)}")
            continue

        before = len(table)
        table = table.copy()
        table["Mean"] = pd.to_numeric(table["mean"], errors="coerce")
        table["TE"] = table["TE"].map(normalize_te)
        table = table.dropna(subset=["Mean", "TE"])
        dropped_na += before - len(table)
        if table.empty:
            print(f"[SKIP] {path}: no valid numeric mean rows")
            continue

        out = pd.DataFrame(
            {
                "Subject": table["subject"].astype(str),
                "TE": table["TE"].astype(int),
                "Model": table["model"].astype(str),
                "Metric": table["metric"].astype(str),
                "ROI_ID": "L_1",
                "ROI_Name": "Lesion_mask",
                "Mean": table["Mean"],
                "Std": pd.NA,
            }
        )
        tables.append(out[OUTPUT_COLUMNS])
        print(f"[OK] {path}: rows={len(out)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if tables:
        result = pd.concat(tables, ignore_index=True)
        result = result.sort_values(["Subject", "TE", "Model", "Metric"]).reset_index(drop=True)
    else:
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)

    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Output: {output}")
    print(f"Rows: {len(result)}")
    print(f"Subjects with rows: {result['Subject'].nunique() if len(result) else 0}")
    print(f"Missing subject files: {missing_files}")
    print(f"Invalid files: {invalid_files}")
    print(f"Dropped non-numeric/NA rows: {dropped_na}")


if __name__ == "__main__":
    main()
