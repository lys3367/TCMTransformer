from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the canonical clean-49 cohort.")
    parser.add_argument(
        "--source",
        default="outputs/80_extra_loso_mole_full.csv",
        help="Existing clean-49 LOSO file containing test_subject or subject.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/clean49_subjects.csv",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = PROJECT / source
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT / output

    table = pd.read_csv(source)
    column = "test_subject" if "test_subject" in table.columns else "subject"
    subjects = table[column].astype(str).drop_duplicates()
    if len(subjects) != 49:
        raise ValueError(f"Expected 49 unique subjects in {source}, found {len(subjects)}.")
    if subjects.str.match(r"^(43_|48_)").any():
        bad = subjects[subjects.str.match(r"^(43_|48_)")].tolist()
        raise ValueError(f"Excluded subjects remain in clean-49 source: {bad}")

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject": sorted(subjects.tolist())}).to_csv(
        output, index=False, encoding="utf-8-sig"
    )
    print(f"Saved canonical clean-49 manifest: {output}")


if __name__ == "__main__":
    main()
