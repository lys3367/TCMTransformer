"""Paired extension statistics from de-identified subject-level MAE tables."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from teprediction.summarize import bh_fdr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV: subject,model,MAE; optional target_te for LOTO")
    parser.add_argument("--output", default="outputs/paired_statistics.csv")
    parser.add_argument("--reference", default="itransformer")
    parser.add_argument("--comparators", nargs="+", default=["timesnet", "mole", "dlinear"])
    parser.add_argument("--loto", action="store_true", help="Require seven unique targets per subject/model; average within subject before testing")
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype={"subject": str})
    required = ["subject", "model", "MAE"] + (["target_te"] if args.loto else [])
    if frame[required].isna().any().any() or not np.isfinite(frame.MAE).all():
        raise ValueError("Missing or nonfinite observations")
    keys = ["subject", "model"] + (["target_te"] if args.loto else [])
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate observations")
    if args.loto:
        targets = frame.groupby(["subject", "model"]).target_te.apply(lambda x: set(x))
        if not all(x == {75, 85, 95, 105, 115, 125, 135} for x in targets):
            raise ValueError("Each LOTO subject/model must contain the seven canonical targets")
        frame = frame.groupby(["subject", "model"], as_index=False).MAE.mean()
    wide = frame.pivot(index="subject", columns="model", values="MAE")
    needed = [args.reference] + args.comparators
    if wide[needed].isna().any().any():
        raise ValueError("Models must contain the same subjects")
    rows = []
    for model in args.comparators:
        delta = wide[model] - wide[args.reference]
        p = 1.0 if np.all(delta == 0) else float(wilcoxon(delta).pvalue)
        rows.append(dict(reference=args.reference, comparator=model, n=len(delta),
                         mean_difference=float(delta.mean()), p=p))
    result = pd.DataFrame(rows)
    result["q"] = bh_fdr(result.p.tolist())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)


if __name__ == "__main__":
    main()
