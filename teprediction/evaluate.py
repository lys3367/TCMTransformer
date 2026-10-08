"""ROI-level and optional de-identified disease-group summaries of held-out predictions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from teprediction.summarize import bh_fdr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="outputs/table1_clean49")
    parser.add_argument("--output-dir", default="outputs/roi_analysis")
    parser.add_argument("--models", nargs="+", default=["itransformer", "timesnet", "mole", "dlinear"])
    parser.add_argument("--groups", help="Optional CSV with unique subject,group columns; use pseudonyms.")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, reference = [], None
    for model in args.models:
        with np.load(Path(args.input_dir)/(model+"_predictions.npz"), allow_pickle=True) as z:
            subjects, rois = z["subjects"].astype(str), z["roi_ids"].astype(str)
            true, pred = z["true_z"], z["pred_z"]
            signature = (subjects, rois, z["target_te"], true)
            if reference is not None and not all(np.array_equal(a, b) for a, b in zip(signature, reference)):
                raise ValueError("Models must share subject order, ROI order, target TE and held-out truth.")
            reference = tuple(a.copy() for a in signature)
            if not np.isfinite(true).all() or not np.isfinite(pred).all():
                raise ValueError("Nonfinite predictions")
            error = np.abs(pred-true)
            for roi in sorted(set(rois)):
                values = error[:, :, rois == roi].mean(axis=(1, 2))
                rows.extend(dict(model=model, subject=s, roi=roi, MAE=float(v)) for s, v in zip(subjects, values))
    table = pd.DataFrame(rows)
    table.to_csv(out/"roi_subject_mae.csv", index=False)
    table.groupby(["model", "roi"]).MAE.agg(["mean", "std", "count"]).reset_index().to_csv(out/"roi_summary.csv", index=False)
    if args.groups:
        groups = pd.read_csv(args.groups, dtype=str)
        if groups.subject.duplicated().any() or groups[["subject", "group"]].isna().any().any():
            raise ValueError("Invalid group table")
        by_subject = table.groupby(["model", "subject"], as_index=False).MAE.mean().merge(groups, on="subject", how="left", validate="many_to_one")
        if by_subject.group.isna().any():
            raise ValueError("Group missing for a held-out subject")
        by_subject.groupby(["model", "group"]).MAE.agg(["mean", "std", "count"]).reset_index().to_csv(out/"group_summary.csv", index=False)
        comparisons = []
        for group, frame in by_subject.groupby("group"):
            pivot = frame.pivot(index="subject", columns="model", values="MAE")
            if "itransformer" not in pivot:
                continue
            for model in args.models:
                if model == "itransformer":
                    continue
                delta = pivot[model]-pivot.itransformer
                p = 1.0 if np.all(delta == 0) else float(wilcoxon(delta).pvalue)
                comparisons.append(dict(group=group, model=model, n=len(delta), mean_difference=float(delta.mean()), p=p))
        if comparisons:
            stats = pd.DataFrame(comparisons)
            stats["q"] = bh_fdr(stats.p.tolist())
            stats.to_csv(out/"group_pairwise_fdr.csv", index=False)
    print("Saved ROI/group summaries in", out)
