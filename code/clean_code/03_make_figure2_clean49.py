from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


plt.rcParams["svg.fonttype"] = "none"


ROOT = Path(__file__).resolve().parents[2]
PREDICTION_DIR = ROOT / "outputs/table1_clean49"
OUTPUT_DIR = ROOT / "paper/figure_ppt/更改图片"
MODELS = ["iTransformer", "MoLE", "DLinear", "TimesNet"]
FILES = {
    "iTransformer": "itransformer_predictions.npz",
    "TimesNet": "timesnet_predictions.npz",
    "MoLE": "mole_predictions.npz",
    "DLinear": "dlinear_predictions.npz",
}
COLORS = {
    "iTransformer": "#EF5F5F",
    "TimesNet": "#9A72C8",
    "MoLE": "#53A8E1",
    "DLinear": "#FFA74E",
}


def subject_mae(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return np.mean(np.abs(data["pred_z"] - data["true_z"]), axis=(1, 2))


def ci95(values: np.ndarray) -> tuple[float, float]:
    half = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - half), float(values.mean() + half)


def add_bracket(ax, x1: float, x2: float, y: float, text: str) -> None:
    height = 0.008
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="#222222", lw=0.9)
    ax.text((x1 + x2) / 2, y + height + 0.002, text, ha="center", va="bottom", fontsize=9)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    values = {model: subject_mae(PREDICTION_DIR / FILES[model]) for model in MODELS}
    summary = pd.read_csv(PREDICTION_DIR / "table1_clean49_summary.csv").set_index("model")
    pairwise = pd.read_csv(PREDICTION_DIR / "table1_pairwise_wilcoxon_fdr.csv")

    complementary = {
        "Target-TE": {model: float(summary.loc[model, "MAE_mean"]) for model in MODELS},
        "Lesion": {"iTransformer": 0.073992, "TimesNet": 0.102786, "MoLE": 0.073073, "DLinear": 0.074061},
        "External": {"iTransformer": 0.211087, "TimesNet": 0.304564, "MoLE": 0.214111, "DLinear": 0.213760},
        "LOTO": {"iTransformer": 0.148022, "TimesNet": 0.169311, "MoLE": 0.182382, "DLinear": 0.182388},
        "Incomplete-TE": {"iTransformer": 0.232371, "TimesNet": 0.376879, "MoLE": 0.239228, "DLinear": 0.239744},
    }
    annotations = {
        "Target-TE": {"iTransformer": "ref.", "TimesNet": "*", "MoLE": "***", "DLinear": "***"},
        "Lesion": {"iTransformer": "ref.", "TimesNet": "***", "MoLE": "ns", "DLinear": "ns"},
        "External": {model: "d." if model != "iTransformer" else "ref." for model in MODELS},
        "LOTO": {"iTransformer": "ref.", "TimesNet": "***", "MoLE": "***", "DLinear": "***"},
        "Incomplete-TE": {"iTransformer": "ref.", "TimesNet": "***", "MoLE": "***", "DLinear": "***"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.13, 3.97), dpi=300, gridspec_kw={"width_ratios": [1.05, 1.15]})

    ax = axes[0]
    positions = np.arange(1, 5)
    violin = ax.violinplot([values[m] for m in MODELS], positions=positions, widths=0.75,
                           showmeans=False, showmedians=False, showextrema=False)
    for body, model in zip(violin["bodies"], MODELS):
        body.set_facecolor(COLORS[model])
        body.set_edgecolor(COLORS[model])
        body.set_alpha(0.25)
        body.set_linewidth(1.0)
    rng = np.random.default_rng(20260825)
    for x, model in zip(positions, MODELS):
        y = values[model]
        jitter = rng.uniform(-0.17, 0.17, len(y))
        ax.scatter(x + jitter, y, s=7, color=COLORS[model], alpha=0.24, linewidths=0)
        lower, upper = ci95(y)
        ax.errorbar(x, y.mean(), yerr=[[y.mean() - lower], [upper - y.mean()]], fmt="o",
                    color="#202020", capsize=3, ms=4.2, lw=1.0, zorder=5)
    ax.set_xticks(positions, MODELS, fontsize=8.5)
    ax.set_ylabel("MAE")
    ax.set_ylim(0.12, max(max(v) for v in values.values()) + 0.105)
    ax.grid(axis="y", color="#ECECEC", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    y0 = max(max(v) for v in values.values()) + 0.016
    primary_pairs = pairwise[pairwise["model_a"] == "iTransformer"]
    q_lookup = dict(zip(primary_pairs["model_b"], primary_pairs["fdr_q_value"]))
    for index, model in enumerate(MODELS[1:], start=2):
        q = q_lookup[model]
        stars = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"
        add_bracket(ax, 1, index, y0 + (index - 2) * 0.025, stars)
    ax.text(-0.08, 1.02, "A", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axes[1]
    tasks = list(complementary)
    matrix = np.array([[complementary[t][m] for m in MODELS] for t in tasks])
    relative = (matrix / matrix[:, [0]] - 1.0) * 100.0
    cmap = LinearSegmentedColormap.from_list("relative_mae", ["#FFFFFF", "#DBCB92", "#ED8D5A", "#806886"])
    image = ax.imshow(relative, cmap=cmap, vmin=0, vmax=205, aspect="auto")
    ax.set_xticks(range(4), MODELS, rotation=0, fontsize=8.5)
    ax.set_yticks(range(len(tasks)), tasks, fontsize=8.5)
    ax.tick_params(length=0)
    for i, task in enumerate(tasks):
        for j, model in enumerate(MODELS):
            mae = matrix[i, j]
            if j == 0:
                second = "ref."
            else:
                second = f"{relative[i, j]:+.1f}% {annotations[task][model]}"
            color = "white" if relative[i, j] > 125 else "#2B2730"
            ax.text(j, i - 0.09, f"{mae:.3f}", ha="center", va="center", fontsize=7.8, color=color)
            ax.text(j, i + 0.16, second, ha="center", va="center", fontsize=7.0, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("MAE increase vs iTransformer (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.text(-0.11, 1.02, "B", transform=ax.transAxes, fontsize=16, fontweight="bold")

    fig.subplots_adjust(left=0.07, right=0.965, bottom=0.15, top=0.91, wspace=0.28)
    stem = OUTPUT_DIR / "Figure2_clean49_updated"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    qa_dir = ROOT / "tmp/qa_clean49_figures"
    qa_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(qa_dir / "Figure2_clean49_updated.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(stem.with_suffix(".svg"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
