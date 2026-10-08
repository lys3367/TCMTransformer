from __future__ import annotations

import gzip
import re
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
PREDICTION_DIR = ROOT / "outputs/table1_clean49"
OUTPUT_DIR = ROOT / "paper/figure_ppt/更改图片"
ATLAS_DIR = ROOT.parent / "BN_JHU/MNI"
T1_PATH = ATLAS_DIR / "MNI152_T1_1mm_brain.nii.gz"
MODELS = ["iTransformer", "MoLE", "DLinear", "TimesNet"]
STATISTICAL_COMPARATORS = ["TimesNet", "MoLE", "DLinear"]
FILES = {
    "iTransformer": "itransformer_predictions.npz",
    "TimesNet": "timesnet_predictions.npz",
    "MoLE": "mole_predictions.npz",
    "DLinear": "dlinear_predictions.npz",
}
COLORS = {"iTransformer": "#EF5F5F", "TimesNet": "#9A72C8", "MoLE": "#53A8E1", "DLinear": "#FFA74E"}
DISEASE_IDS = {
    "Healthy controls": {52, 53, 54, 56, 57, 59},
    "Glioma": {10, 11, 13, 14, 16, 18, 19, 25, 28, 29, 31, 34, 37, 38, 40, 41, 42, 44, 45, 50, 51},
    "Other intracranial lesions": {3, 6, 9, 20, 24, 30, 33, 36, 39, 46, 47},
    "Non-neoplastic neurological diseases": {7, 8, 12, 17, 22, 26, 27, 32, 49, 55, 58},
}
DTYPE_FROM_CODE = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 512: np.uint16, 768: np.uint32}


def read_nifti(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        blob = handle.read()
    header = blob[:348]
    endian = "<" if struct.unpack("<i", header[:4])[0] == 348 else ">"
    dims = struct.unpack(endian + "8h", header[40:56])
    shape = tuple(int(x) for x in dims[1 : int(dims[0]) + 1])
    dtype = np.dtype(DTYPE_FROM_CODE[struct.unpack(endian + "h", header[70:72])[0]]).newbyteorder(endian)
    offset = int(struct.unpack(endian + "f", header[108:112])[0])
    return np.frombuffer(blob, dtype=dtype, offset=offset).copy().reshape(shape, order="F")


def bh_fdr(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10_000) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    return tuple(np.percentile(values[indices].mean(axis=1), [2.5, 97.5]))


def load_predictions() -> dict[str, dict[str, np.ndarray]]:
    loaded = {}
    for model in MODELS:
        z = np.load(PREDICTION_DIR / FILES[model], allow_pickle=True)
        loaded[model] = {key: z[key] for key in z.files}
    reference = loaded["iTransformer"]
    for model in MODELS[1:]:
        if not np.array_equal(reference["subjects"], loaded[model]["subjects"]):
            raise ValueError(f"Subject order differs for {model}")
        if not np.array_equal(reference["true_z"], loaded[model]["true_z"]):
            raise ValueError(f"Ground truth differs for {model}")
    return loaded


def roi_statistics(loaded: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    reference = loaded["iTransformer"]
    roi_ids = reference["roi_ids"].astype(str)
    roi_labels = reference["roi_labels"].astype(str)
    rows = []
    for model in MODELS:
        error = np.abs(loaded[model]["pred_z"] - loaded[model]["true_z"])
        truth = np.abs(loaded[model]["true_z"])
        for roi in np.unique(roi_ids):
            mask = roi_ids == roi
            rows.append({
                "model": model,
                "roi_id": roi,
                "roi_label": roi_labels[np.flatnonzero(mask)[0]],
                "MAE": float(error[:, :, mask].mean()),
                "RMAE": float(error[:, :, mask].sum() / max(truth[:, :, mask].sum(), np.finfo(float).eps)),
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "Figure3_clean49_ROI_MAE_RMAE.csv", index=False, encoding="utf-8-sig")
    return table


def roi_volume(bn: np.ndarray, jhu: np.ndarray, table: pd.DataFrame, column: str) -> np.ndarray:
    volume = np.zeros(bn.shape, dtype=float)
    for row in table.itertuples(index=False):
        prefix, number = row.roi_id.split("_", 1)
        atlas = bn if prefix == "G" else jhu
        volume[atlas == int(number)] = float(getattr(row, column))
    return volume


def plot_roi_maps(table: pd.DataFrame, metric: str) -> None:
    bn = read_nifti(ATLAS_DIR / "BN_Atlas_246_1mm.nii.gz")
    jhu = read_nifti(ATLAS_DIR / "JHU-ICBM-labels-1mm.nii.gz")
    t1 = read_nifti(T1_PATH).astype(float)
    if not (bn.shape == jhu.shape == t1.shape):
        raise ValueError(f"MNI image shape mismatch: BN={bn.shape}, JHU={jhu.shape}, T1={t1.shape}")
    volumes = {model: roi_volume(bn, jhu, table[table["model"] == model], metric) for model in MODELS}
    all_values = table[metric].to_numpy()
    vmin, vmax = np.percentile(all_values, [20, 80])
    mask = (bn > 0) | (jhu > 0)
    t1_values = t1[t1 > 0]
    t1_vmin, t1_vmax = np.percentile(t1_values, [1, 99])
    anatomy_cmap = plt.cm.gray.copy()
    anatomy_cmap.set_bad("white")
    z_count = mask.sum(axis=(0, 1))
    valid = np.where(z_count > max(50, 0.08 * z_count.max()))[0]
    slices = np.linspace(valid.min(), valid.max(), 10).astype(int)[1:-1]
    fig, axes = plt.subplots(4, len(slices), figsize=(10.2, 5.1), dpi=300)
    for r, model in enumerate(MODELS):
        for c, z in enumerate(slices):
            ax = axes[r, c]
            anatomy = np.rot90(t1[:, :, z])
            ax.set_facecolor("white")
            ax.imshow(np.ma.masked_where(anatomy <= 0, anatomy), cmap=anatomy_cmap,
                      vmin=t1_vmin, vmax=t1_vmax)
            current = np.rot90(volumes[model][:, :, z])
            ax.imshow(np.ma.masked_where(current <= 0, current), cmap="turbo", vmin=vmin, vmax=vmax, alpha=0.72)
            ax.set_axis_off()
            if r == 0:
                ax.set_title(f"z={z}", fontsize=7)
            if c == 0:
                ax.text(-0.08, 0.5, model, transform=ax.transAxes, ha="right", va="center", fontsize=8.5)
    fig.subplots_adjust(left=0.13, right=0.91, top=0.93, bottom=0.04, wspace=0.01, hspace=0.01)
    cax = fig.add_axes([0.93, 0.18, 0.015, 0.64])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap="turbo"), cax=cax)
    cb.set_label(metric if metric == "MAE" else r"RMAE = $\Sigma|e|/\Sigma|y|$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    stem = OUTPUT_DIR / f"Figure3_candidate_clean49_ROI_{metric}"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    qa_dir = ROOT / "tmp/qa_clean49_figures"
    qa_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(qa_dir / f"Figure3_candidate_clean49_ROI_{metric}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_roi_comparison(table: pd.DataFrame, metric: str) -> None:
    bn = read_nifti(ATLAS_DIR / "BN_Atlas_246_1mm.nii.gz")
    jhu = read_nifti(ATLAS_DIR / "JHU-ICBM-labels-1mm.nii.gz")
    t1 = read_nifti(T1_PATH).astype(float)
    if not (bn.shape == jhu.shape == t1.shape):
        raise ValueError(f"MNI image shape mismatch: BN={bn.shape}, JHU={jhu.shape}, T1={t1.shape}")

    model_volumes = {
        model: roi_volume(bn, jhu, table[table["model"] == model], metric)
        for model in MODELS
    }
    differences = {
        model: model_volumes[model] - model_volumes["iTransformer"]
        for model in MODELS[1:]
    }
    difference_values = np.concatenate([
        (table[table["model"] == model].sort_values("roi_id")[metric].to_numpy()
         - table[table["model"] == "iTransformer"].sort_values("roi_id")[metric].to_numpy())
        for model in MODELS[1:]
    ])
    limit = float(np.percentile(np.abs(difference_values), 85))
    mask = (bn > 0) | (jhu > 0)
    z_count = mask.sum(axis=(0, 1))
    valid = np.where(z_count > max(50, 0.08 * z_count.max()))[0]
    slices = np.linspace(valid.min(), valid.max(), 8).astype(int)[1:-1]
    t1_values = t1[t1 > 0]
    t1_vmin, t1_vmax = np.percentile(t1_values, [1, 99])
    anatomy_cmap = plt.cm.gray.copy()
    anatomy_cmap.set_bad("white")

    fig, axes = plt.subplots(3, len(slices), figsize=(9.3, 4.25), dpi=300)
    for row, model in enumerate(MODELS[1:]):
        for column, z in enumerate(slices):
            ax = axes[row, column]
            anatomy = np.rot90(t1[:, :, z])
            ax.set_facecolor("white")
            ax.imshow(np.ma.masked_where(anatomy <= 0, anatomy), cmap=anatomy_cmap,
                      vmin=t1_vmin, vmax=t1_vmax)
            current = np.rot90(differences[model][:, :, z])
            ax.imshow(
                np.ma.masked_where(~np.rot90(mask[:, :, z]), current),
                cmap="RdBu_r", vmin=-limit, vmax=limit, alpha=0.78,
            )
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"z={z}", fontsize=8)
            if column == 0:
                ax.text(-0.08, 0.5, f"{model} -\niTransformer", transform=ax.transAxes,
                        ha="right", va="center", fontsize=8.5)
    fig.subplots_adjust(left=0.16, right=0.90, top=0.90, bottom=0.08, wspace=0.01, hspace=0.02)
    cax = fig.add_axes([0.92, 0.19, 0.016, 0.62])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(-limit, limit), cmap="RdBu_r"),
        cax=cax,
    )
    cb.set_label(f"Comparator - iTransformer {metric}\n(warm: iTransformer lower)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    stem = OUTPUT_DIR / f"Figure3_clean49_T1_delta_{metric}"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    qa_dir = ROOT / "tmp/qa_clean49_figures"
    qa_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(qa_dir / f"Figure3_clean49_T1_delta_{metric}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_mae_rmae_comparison(table: pd.DataFrame) -> None:
    """Render matched MAE and RMAE difference maps for visual comparison."""
    bn = read_nifti(ATLAS_DIR / "BN_Atlas_246_1mm.nii.gz")
    jhu = read_nifti(ATLAS_DIR / "JHU-ICBM-labels-1mm.nii.gz")
    t1 = read_nifti(T1_PATH).astype(float)
    if not (bn.shape == jhu.shape == t1.shape):
        raise ValueError(f"MNI image shape mismatch: BN={bn.shape}, JHU={jhu.shape}, T1={t1.shape}")

    mask = (bn > 0) | (jhu > 0)
    z_count = mask.sum(axis=(0, 1))
    valid = np.where(z_count > max(50, 0.08 * z_count.max()))[0]
    slices = np.linspace(valid.min(), valid.max(), 8).astype(int)[1:-1]
    t1_values = t1[t1 > 0]
    t1_vmin, t1_vmax = np.percentile(t1_values, [1, 99])
    anatomy_cmap = plt.cm.gray.copy()
    anatomy_cmap.set_bad("white")

    difference_volumes: dict[str, dict[str, np.ndarray]] = {}
    limits: dict[str, float] = {}
    for metric in ("MAE", "RMAE"):
        model_volumes = {
            model: roi_volume(bn, jhu, table[table["model"] == model], metric)
            for model in MODELS
        }
        difference_volumes[metric] = {
            model: model_volumes[model] - model_volumes["iTransformer"]
            for model in MODELS[1:]
        }
        reference = table[table["model"] == "iTransformer"].set_index("roi_id")[metric]
        values = np.concatenate([
            (table[table["model"] == model].set_index("roi_id")[metric] - reference).to_numpy()
            for model in MODELS[1:]
        ])
        limits[metric] = float(np.percentile(np.abs(values), 85))

    n_slices = len(slices)
    fig, axes = plt.subplots(3, n_slices * 2, figsize=(15.4, 4.4), dpi=300)
    for panel, metric in enumerate(("MAE", "RMAE")):
        offset = panel * n_slices
        for row, model in enumerate(MODELS[1:]):
            for column, z in enumerate(slices):
                ax = axes[row, offset + column]
                anatomy = np.rot90(t1[:, :, z])
                ax.set_facecolor("white")
                ax.imshow(
                    np.ma.masked_where(anatomy <= 0, anatomy),
                    cmap=anatomy_cmap,
                    vmin=t1_vmin,
                    vmax=t1_vmax,
                )
                current = np.rot90(difference_volumes[metric][model][:, :, z])
                ax.imshow(
                    np.ma.masked_where(~np.rot90(mask[:, :, z]), current),
                    cmap="RdBu_r",
                    vmin=-limits[metric],
                    vmax=limits[metric],
                    alpha=0.78,
                )
                ax.set_axis_off()
                if row == 0:
                    ax.set_title(f"z={z}", fontsize=7.5)
                if column == 0 and panel == 0:
                    ax.text(
                        -0.08,
                        0.5,
                        f"{model} -\niTransformer",
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=8,
                    )
        center = offset + (n_slices - 1) / 2
        axes[0, int(center)].text(
            0.5 if center.is_integer() else 1.02,
            1.35,
            f"{'A' if panel == 0 else 'B'}  {metric} difference",
            transform=axes[0, int(center)].transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.095, right=0.965, top=0.82, bottom=0.13, wspace=0.01, hspace=0.02)
    for panel, metric in enumerate(("MAE", "RMAE")):
        left = 0.17 if panel == 0 else 0.61
        cax = fig.add_axes([left, 0.055, 0.30, 0.022])
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(-limits[metric], limits[metric]), cmap="RdBu_r"),
            cax=cax,
            orientation="horizontal",
        )
        label = r"Comparator - iTransformer $\Delta$MAE" if metric == "MAE" else r"Comparator - iTransformer $\Delta$RMAE"
        cb.set_label(label + "  (warm: iTransformer lower)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    stem = OUTPUT_DIR / "Figure3_clean49_MAE_vs_RMAE_delta"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    qa_dir = ROOT / "tmp/qa_clean49_figures"
    qa_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(qa_dir / "Figure3_clean49_MAE_vs_RMAE_delta.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def disease_statistics(loaded: dict[str, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subjects = loaded["iTransformer"]["subjects"].astype(str)
    lookup = {identifier: group for group, identifiers in DISEASE_IDS.items() for identifier in identifiers}
    rows = []
    for model in MODELS:
        mae = np.mean(np.abs(loaded[model]["pred_z"] - loaded[model]["true_z"]), axis=(1, 2))
        for subject, value in zip(subjects, mae):
            identifier = int(re.match(r"^(\d+)", subject).group(1))
            rows.append({"subject": subject, "subject_id": identifier, "disease_group": lookup[identifier], "model": model, "MAE": value})
    long = pd.DataFrame(rows)
    summary = long.groupby(["disease_group", "model"], as_index=False).agg(n=("MAE", "size"), MAE_mean=("MAE", "mean"), MAE_SD=("MAE", "std"))
    wide = long.pivot(index=["subject", "disease_group"], columns="model", values="MAE").reset_index()
    rng = np.random.default_rng(20260825)
    comparisons = []
    for group in DISEASE_IDS:
        current = wide[wide["disease_group"] == group]
        for comparator in STATISTICAL_COMPARATORS:
            delta = (current[comparator] - current["iTransformer"]).to_numpy()
            lower, upper = bootstrap_ci(delta, rng)
            statistic, p = wilcoxon(current[comparator], current["iTransformer"], alternative="two-sided", method="auto")
            comparisons.append({"disease_group": group, "comparator": comparator, "n": len(delta), "mean_delta_MAE": delta.mean(), "bootstrap_CI_lower": lower, "bootstrap_CI_upper": upper, "wilcoxon_statistic": statistic, "wilcoxon_p": p})
    comparisons = pd.DataFrame(comparisons)
    comparisons["FDR_q"] = bh_fdr(comparisons["wilcoxon_p"].to_numpy())
    comparisons["FDR_significant_0.05"] = comparisons["FDR_q"] < 0.05
    long.to_csv(OUTPUT_DIR / "Figure4_clean49_subject_MAE.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "S2_candidate_clean49_disease_summary.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(OUTPUT_DIR / "S2_candidate_clean49_disease_wilcoxon_fdr.csv", index=False, encoding="utf-8-sig")
    return long, summary, comparisons


def plot_disease(long: pd.DataFrame, summary: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    groups = list(DISEASE_IDS)
    abbreviations = ["HC", "Glioma", "OIL", "NND"]
    fig, axes = plt.subplots(1, 2, figsize=(10.13, 4.5), dpi=300, gridspec_kw={"width_ratios": [1.08, 1.0]})
    ax = axes[0]
    x = np.arange(len(groups))
    offsets = np.linspace(-0.27, 0.27, len(MODELS))
    rng = np.random.default_rng(20260825)
    for offset, model in zip(offsets, MODELS):
        for index, group in enumerate(groups):
            values = long[(long["model"] == model) & (long["disease_group"] == group)]["MAE"].to_numpy()
            ax.scatter(np.full(len(values), x[index] + offset) + rng.uniform(-0.025, 0.025, len(values)), values,
                       color=COLORS[model], s=10, alpha=0.25, linewidths=0)
            lower, upper = bootstrap_ci(values, rng)
            ax.errorbar(x[index] + offset, values.mean(), yerr=[[values.mean() - lower], [upper - values.mean()]],
                        fmt="o", color=COLORS[model], capsize=2.5, ms=4.5, lw=1.1,
                        label=model if index == 0 else None)
    ax.set_xticks(x, abbreviations)
    ax.set_ylabel("Subject-level MAE")
    ax.set_title("A  Performance by disease group", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    positions, labels = [], []
    y = 0.0
    for group, short in zip(groups, abbreviations):
        for comparator in MODELS[1:]:
            row = comparisons[(comparisons["disease_group"] == group) & (comparisons["comparator"] == comparator)].iloc[0]
            mean, lower, upper = row["mean_delta_MAE"], row["bootstrap_CI_lower"], row["bootstrap_CI_upper"]
            ax.errorbar(mean, y, xerr=[[mean - lower], [upper - mean]], fmt="o", color=COLORS[comparator], capsize=3, ms=4.5)
            stars = "***" if row["FDR_q"] < 0.001 else "**" if row["FDR_q"] < 0.01 else "*" if row["FDR_q"] < 0.05 else "ns"
            ax.text(upper + 0.002, y, stars, va="center", fontsize=8)
            positions.append(y)
            labels.append(f"{short}: {comparator}")
            y += 0.62
        y += 0.45
    ax.axvline(0, color="#333333", ls="--", lw=1)
    ax.set_yticks(positions, labels, fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlabel(r"Paired $\Delta$MAE (comparator - iTransformer)")
    ax.set_title("B  Paired differences with 95% bootstrap CI", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.91, wspace=0.37)
    stem = OUTPUT_DIR / "Figure4_candidate_clean49_disease"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    qa_dir = ROOT / "tmp/qa_clean49_figures"
    qa_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(qa_dir / "Figure4_candidate_clean49_disease.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = load_predictions()
    roi = roi_statistics(loaded)
    plot_roi_maps(roi, "MAE")
    plot_roi_maps(roi, "RMAE")
    plot_roi_comparison(roi, "MAE")
    plot_roi_comparison(roi, "RMAE")
    plot_mae_rmae_comparison(roi)
    long, summary, comparisons = disease_statistics(loaded)
    plot_disease(long, summary, comparisons)
    winner = roi.loc[roi.groupby("roi_id")["MAE"].idxmin()]["model"].value_counts()
    print("ROI MAE winner counts:")
    print(winner.to_string())
    print("\nDisease summary:")
    print(summary.to_string(index=False))
    print("\nDisease paired comparisons:")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
