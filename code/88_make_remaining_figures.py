from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["itransformer", "timesnet", "mole", "dlinear"]
COLORS = {"itransformer": "#EF5F5F", "timesnet": "#9A72C8", "mole": "#53A8E1", "dlinear": "#FFA74E"}
LABELS = {"itransformer": "iTransformer", "timesnet": "TimesNet", "mole": "MoLE", "dlinear": "DLinear"}


def save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def load_predictions(out: Path):
    paths = {m: out / f"87_loso_target_predictions_{m}.npz" for m in MODELS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing prediction files:\n" + "\n".join(missing))
    return {m: np.load(p, allow_pickle=True) for m, p in paths.items()}


def target_plot(preds, out: Path):
    rows = []
    for m, z in preds.items():
        err = np.abs(z["pred_z"] - z["true_z"]).mean(axis=2)
        for i, sub in enumerate(z["subjects"].astype(str)):
            for j, te in enumerate(z["target_te"]):
                rows.append({"subject": sub, "model": LABELS[m], "target_TE": int(te), "MAE": float(err[i, j])})
    detail = pd.DataFrame(rows)
    detail.to_csv(out / "88_target_te_subject_mae.csv", index=False, encoding="utf-8-sig")
    summary = detail.groupby(["model", "target_TE"]).MAE.agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(out / "88_target_te_summary.csv", index=False, encoding="utf-8-sig")

    # Target-specific main-task comparison.
    # Keep model identity colors fixed and use marker shape to distinguish TE125/TE135.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(MODELS), dtype=float)
    target_tes = sorted(detail["target_TE"].unique())
    offsets = np.linspace(-0.08, 0.08, len(target_tes)) if len(target_tes) > 1 else np.array([0.0])
    markers = ["o", "s", "^", "D"]

    for mi, m in enumerate(MODELS):
        model_vals = []
        for j, te in enumerate(target_tes):
            sel = summary.loc[
                (summary["model"] == LABELS[m]) & (summary["target_TE"] == te),
                "mean",
            ]
            if sel.empty:
                raise ValueError(f"Missing summary value for model={LABELS[m]}, target_TE={te}")
            value = float(sel.iloc[0])
            model_vals.append(value)
            ax.scatter(
                x[mi] + offsets[j],
                value,
                s=58,
                marker=markers[j % len(markers)],
                color=COLORS[m],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )

        if len(model_vals) > 1:
            ax.plot(
                x[mi] + offsets[:len(model_vals)],
                model_vals,
                color=COLORS[m],
                linewidth=1.8,
                alpha=0.9,
                zorder=2,
            )

    # Legend encodes target TE by marker; model identity is encoded by x-axis label/color.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D(
            [0], [0],
            marker=markers[j % len(markers)],
            linestyle="None",
            markerfacecolor="0.35",
            markeredgecolor="white",
            markersize=7,
            label=f"TE{te} ms",
        )
        for j, te in enumerate(target_tes)
    ]

    ax.set_xticks(x, [LABELS[m] for m in MODELS])
    ax.set_ylabel("Subject-level MAE")
    ax.set_title("Target-specific LOSO error")
    ax.legend(handles=legend_handles, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out / "Figure_Supplement_Target_TE125_TE135")


def native_agreement(preds, out: Path):
    wanted = [("DTI", "FA", "FA"), ("DTI", "MD", "MD"), ("FWDTI", "FW", "FW"), ("AMICO/NODDI", "fit_NDI", "NDI")]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (model_name, metric_name, title) in zip(axes.ravel(), wanted):
        z = preds["itransformer"]
        mask = (z["models"].astype(str) == model_name) & (z["metrics"].astype(str) == metric_name)
        if not mask.any():
            ax.set_title(f"{title} (missing)"); continue
        true = z["true_native"][:, :, mask].ravel(); pred = z["pred_native"][:, :, mask].ravel()
        finite = np.isfinite(true) & np.isfinite(pred); true, pred = true[finite], pred[finite]
        ax.scatter(true, pred, s=5, alpha=.18, color=COLORS["itransformer"], rasterized=True)
        lo, hi = np.nanpercentile(np.r_[true, pred], [1, 99]); ax.plot([lo, hi], [lo, hi], "--", color="0.45", lw=1)
        mae = np.mean(np.abs(pred - true)); bias = np.mean(pred - true)
        r2 = 1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)
        rows.append({"metric": title, "n": len(true), "native_MAE": mae, "bias": bias, "R2": r2})
        ax.set_title(title); ax.set_xlabel("Ground truth"); ax.set_ylabel("Prediction")
        ax.text(.04, .96, f"MAE={mae:.4g}\nBias={bias:.4g}\nR2={r2:.3f}", transform=ax.transAxes, va="top", fontsize=9)
    fig.suptitle("Native-unit agreement: iTransformer", y=.995); fig.tight_layout(); save(fig, out / "Figure7_Native_Unit_Agreement")
    pd.DataFrame(rows).to_csv(out / "88_native_unit_agreement.csv", index=False, encoding="utf-8-sig")


def family_heatmap(preds, out: Path):
    families = {"AMICO/NODDI": "NODDI"}
    rows = []
    for family in ["DTI", "DTI_RESTORE", "FWDTI", "DKI", "WMTI", "MSDKI", "GQI", "NODDI", "FORECAST"]:
        vals = []
        for m in MODELS:
            z = preds[m]; model = z["models"].astype(str); fam = model.copy(); fam[fam == "AMICO/NODDI"] = "NODDI"
            mask = fam == family
            vals.append(float(np.abs(z["pred_z"][:, :, mask] - z["true_z"][:, :, mask]).mean()) if mask.any() else np.nan)
        rows.append([family] + vals)
    table = pd.DataFrame(rows, columns=["metric_family"] + MODELS); table.to_csv(out / "88_metric_family_mae.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8.5, 5.4)); arr = table[MODELS].to_numpy(dtype=float)
    im = ax.imshow(arr, cmap="Purples", aspect="auto"); ax.set_xticks(range(4), [LABELS[m] for m in MODELS]); ax.set_yticks(range(len(table)), table.metric_family)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]): ax.text(j, i, f"{arr[i,j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("Metric-family LOSO MAE"); fig.colorbar(im, ax=ax, label="MAE"); fig.tight_layout(); save(fig, out / "Figure8_Metric_Family_MAE")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output-dir", default="outputs/remaining_figures"); args = p.parse_args()
    out = ROOT / args.output_dir; out.mkdir(parents=True, exist_ok=True); preds = load_predictions(ROOT / "outputs/target_predictions")
    target_plot(preds, out); native_agreement(preds, out); family_heatmap(preds, out)
    print(f"Saved remaining figures and tables to {out}")


if __name__ == "__main__": main()
