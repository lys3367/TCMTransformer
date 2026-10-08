from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "outputs_sciencedb_clean49"
MODELS = ("itransformer", "mole", "dlinear", "timesnet")
DISPLAY = {"itransformer": "iTransformer", "mole": "MoLE", "dlinear": "DLinear", "timesnet": "TimesNet"}


def main() -> None:
    rows = []
    for model in MODELS:
        path = OUTPUT / f"{model}_external_summary.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing external result: {path}")
        row = pd.read_csv(path).iloc[0].to_dict()
        row["model"] = DISPLAY[model]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("MAE")
    table.to_csv(OUTPUT / "s3_sciencedb_clean49_summary.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
