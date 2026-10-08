from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "paper" / "figure_ppt" / "更改图片"
OUTPUT_DIR = SOURCE_DIR

MODELS = ["iTransformer", "MoLE", "DLinear", "TimesNet"]
COLORS = {
    "iTransformer": "#EF5F5F",
    "MoLE": "#53A8E1",
    "DLinear": "#FFA74E",
    "TimesNet": "#9A72C8",
}
GROUPS = [
    "Healthy controls",
    "Glioma",
    "Other intracranial lesions",
    "Non-neoplastic neurological diseases",
]
ABBREVIATIONS = ["HC\n(n=6)", "Glioma\n(n=21)", "OIL\n(n=11)", "NND\n(n=11)"]
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "group_mae",
    ["#F7FBFF", "#DCEAF5", "#B8D4E8", "#87B5D5", "#4E87B3"],
)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SOURCE_DIR / "S2_candidate_clean49_disease_summary.csv")
    comparisons = pd.read_csv(SOURCE_DIR / "S2_candidate_clean49_disease_wilcoxon_fdr.csv")

    expected_summary = {(group, model) for group in GROUPS for model in MODELS}
    observed_summary = set(zip(summary["disease_group"], summary["model"]))
    if observed_summary != expected_summary:
        raise ValueError("Disease-group summary is incomplete")

    expected_comparisons = {(group, model) for group in GROUPS for model in MODELS[1:]}
    observed_comparisons = set(zip(comparisons["disease_group"], comparisons["comparator"]))
    if observed_comparisons != expected_comparisons:
        raise ValueError("Disease-group comparison table is incomplete")
    return summary, comparisons


def plot_figure(summary: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
        }
    )

    lookup = summary.set_index(["disease_group", "model"])["MAE_mean"]
    matrix = np.array([[lookup.loc[(group, model)] for group in GROUPS] for model in MODELS], dtype=float)
    vmin, vmax = float(matrix.min()), float(matrix.max())

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.13, 4.5),
        dpi=300,
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )

    # Panel A: four disease groups rather than one column per participant.
    ax = axes[0]
    image = ax.imshow(matrix, cmap=HEATMAP_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(GROUPS)), ABBREVIATIONS, fontsize=8)
    ax.set_yticks(np.arange(len(MODELS)), MODELS, fontsize=8)
    ax.set_title("A  Mean performance by disease group", loc="left", fontsize=11, fontweight="bold")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    midpoint = (vmin + vmax) / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > midpoint + 0.01 else "#1F2933"
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Mean subject-level MAE", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    # Panel B is intentionally unchanged from the supplied clean-49 candidate.
    ax = axes[1]
    positions, labels = [], []
    y = 0.0
    for group, short in zip(GROUPS, ["HC", "Glioma", "OIL", "NND"]):
        for comparator in MODELS[1:]:
            row = comparisons[
                (comparisons["disease_group"] == group)
                & (comparisons["comparator"] == comparator)
            ].iloc[0]
            mean = float(row["mean_delta_MAE"])
            lower = float(row["bootstrap_CI_lower"])
            upper = float(row["bootstrap_CI_upper"])
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - lower], [upper - mean]],
                fmt="o",
                color=COLORS[comparator],
                capsize=3,
                ms=4.5,
            )
            q_value = float(row["FDR_q"])
            stars = "***" if q_value < 0.001 else "**" if q_value < 0.01 else "*" if q_value < 0.05 else "ns"
            ax.text(upper + 0.002, y, stars, va="center", fontsize=8)
            positions.append(y)
            labels.append(f"{short}: {comparator}")
            y += 0.62
        y += 0.45
    ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
    ax.set_yticks(positions, labels, fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlabel(r"Paired $\Delta$MAE (comparator - iTransformer)")
    ax.set_title("B  Paired differences with 95% bootstrap CI", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.91, wspace=0.40)
    final_svg = OUTPUT_DIR / "figure4.svg"
    descriptive_svg = OUTPUT_DIR / "Figure4_group_heatmap_clean49.svg"
    fig.savefig(final_svg, facecolor="white", bbox_inches="tight")
    fig.savefig(descriptive_svg, facecolor="white", bbox_inches="tight")

    qa_dir = ROOT / "tmp" / "qa_clean49_figures"
    qa_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(qa_dir / "Figure4_group_heatmap_clean49.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"SVG: {final_svg}")
    print(f"Backup SVG: {descriptive_svg}")


if __name__ == "__main__":
    plot_figure(*load_tables())
