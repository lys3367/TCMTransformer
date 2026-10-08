from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
EXPECTED_TE = [75, 85, 95, 105, 115, 125, 135]


def canonical_feature(model: str, metric: str) -> tuple[str, str]:
    if model == "NODDI":
        return "AMICO/NODDI", f"fit_{metric}"
    if model == "DKI" and not metric.startswith("DKI_"):
        return model, f"DKI_{metric}"
    if model == "FORECAST" and metric in {"FA", "MD"}:
        return model, metric.lower()
    if model == "WMTI" and metric in {"AD", "RD"}:
        return model, f"Hindered_{metric}"
    return model, metric


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the clean-42 lesion cohort with the primary 38 metrics.")
    parser.add_argument("--input", default="data/raw/lesion_metric_results_raw.csv")
    parser.add_argument("--output", default="data/raw/06_lesion_metric_results_clean42_38metrics.csv")
    args = parser.parse_args()
    source = Path(args.input)
    if not source.is_absolute():
        source = PROJECT / source
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {source}. Run scripts/08_prepare_lesion_csv.sh on the server first."
        )

    table = pd.read_csv(source, encoding="utf-8-sig")
    table["Subject"] = table["Subject"].astype(str)
    table = table[~table["Subject"].str.startswith(("43_", "48_"))].copy()
    canonical = [canonical_feature(str(model), str(metric)) for model, metric in zip(table["Model"], table["Metric"])]
    table["Model"] = [item[0] for item in canonical]
    table["Metric"] = [item[1] for item in canonical]
    table["feature"] = table["Model"] + "__" + table["Metric"]

    main_matrix = np.load(PROJECT / "data/processed/01_model_matrix_complete7te.npz", allow_pickle=True)
    main_features = list(dict.fromkeys(main_matrix["features"].astype(str)))
    table = table[table["feature"].isin(main_features)].copy()

    # The lesion source contains participants with incomplete TE coverage.  Match
    # the primary lesion analysis by retaining complete-7TE participants first,
    # then apply the clean-cohort exclusions above.
    te_count = table.groupby("Subject")["TE"].nunique()
    subjects = sorted(te_count[te_count == len(EXPECTED_TE)].index.astype(str))
    table = table[table["Subject"].isin(subjects)].copy()
    if len(subjects) != 42:
        coverage = te_count.value_counts().sort_index().to_dict()
        raise ValueError(
            "Expected 42 clean lesion subjects with all 7 TEs after excluding "
            f"43_ and 48_, found {len(subjects)}. TE coverage counts: {coverage}"
        )

    expected = pd.MultiIndex.from_product(
        [subjects, EXPECTED_TE, main_features], names=["Subject", "TE", "feature"]
    )
    observed = table.set_index(["Subject", "TE", "feature"])["Mean"].reindex(expected)
    missing = observed[observed.isna()].reset_index()
    if not missing.empty:
        report = PROJECT / "outputs_lesion_clean42/06_missing_clean42_metrics.csv"
        report.parent.mkdir(parents=True, exist_ok=True)
        missing.to_csv(report, index=False, encoding="utf-8-sig")
        counts = missing.groupby("feature").size().sort_values(ascending=False)
        raise ValueError(
            "The clean-42 lesion source is not complete for the primary 38 metrics. "
            f"See {report}. Missing counts: {counts.to_dict()}"
        )

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    table.drop(columns="feature").sort_values(["Subject", "TE", "Model", "Metric"]).to_csv(
        output, index=False, encoding="utf-8-sig"
    )
    subject_file = PROJECT / "data/processed_lesion/06_clean42_subjects.csv"
    subject_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject": subjects}).to_csv(
        subject_file,
        index=False,
        encoding="utf-8-sig",
    )

    config = json.loads((PROJECT / "config_lesion.json").read_text(encoding="utf-8-sig"))
    config["raw_csv"] = str(output.relative_to(PROJECT)).replace("\\", "/")
    config["matrix_file"] = "data/processed_lesion/06_model_matrix_clean42_38metrics.npz"
    config["split_file"] = "data/processed_lesion/06_subject_splits_clean42.csv"
    config["metadata_file"] = "data/processed_lesion/06_variable_metadata_clean42_38metrics.csv"
    config["output_dir"] = "outputs_lesion_clean42"
    (PROJECT / "config_lesion_clean42.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Clean lesion subjects: {len(subjects)}")
    print(f"Canonical metrics: {len(main_features)}")
    print(f"Filtered source: {output}")


if __name__ == "__main__":
    main()
