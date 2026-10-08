from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


TE_RE = re.compile(r"acq-TE([0-9]+(?:R2)?)_dwi\.nii\.gz$")

METRICS = {
    "DTI": ["FA", "MD", "RD", "AD"],
    "DTI_RESTORE": ["FA", "MD", "RD", "AD"],
    "FWDTI": ["AD", "FA", "FW", "MD", "RD"],
    "DKI": ["DKI_AD", "DKI_AK", "DKI_FA", "DKI_MD", "DKI_MK", "DKI_RD", "DKI_RK"],
    "WMTI": ["AWF", "Axonal", "Hindered_AD", "Hindered_RD"],
    "MSDKI": ["DI", "F", "MSD", "MSK", "uFA"],
    "IVIM": ["IVIM_D", "IVIM_D_star", "IVIM_f", "IVIM_S0"],
    "GQI": ["GFA", "QA"],
    "AMICO/NODDI": ["fit_FWF", "fit_NDI", "fit_ODI"],
    "FORECAST": ["d_par", "d_perp", "fa", "md"],
}

FRACTION_METRICS = {
    "FA",
    "DKI_FA",
    "GFA",
    "uFA",
    "fa",
    "fit_FWF",
    "FW",
    "AWF",
    "F",
    "IVIM_f",
    "fit_NDI",
    "fit_ODI",
}
DIFFUSIVITY_METRICS = {
    "MD",
    "RD",
    "AD",
    "md",
    "DKI_MD",
    "DKI_AD",
    "DKI_RD",
    "Hindered_AD",
    "Hindered_RD",
    "Axonal",
    "DI",
    "MSD",
    "IVIM_D",
    "IVIM_D_star",
    "d_par",
    "d_perp",
}
KURTOSIS_METRICS = {"DKI_MK", "DKI_AK", "DKI_RK", "MSK"}
SIGNAL_METRICS = {"QA", "IVIM_S0"}


def model_dir(dwi_dir: Path, model: str) -> Path:
    if model == "AMICO/NODDI":
        return dwi_dir / "AMICO" / "NODDI"
    return dwi_dir / model


def metric_bounds(metric: str) -> tuple[float | None, float | None, str]:
    if metric in FRACTION_METRICS:
        return -0.1, 1.5, "fraction"
    if metric in DIFFUSIVITY_METRICS:
        return -0.01, 0.05, "diffusivity"
    if metric in KURTOSIS_METRICS:
        return -2.0, 5.0, "kurtosis"
    if metric in SIGNAL_METRICS:
        return 0.0, None, "signal"
    return None, None, "unbounded"


def read_roi_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            roi_id = str(row.get("ROI_ID", "")).strip()
            name = str(row.get("roi_name", row.get("ROI_Name", roi_id))).strip()
            if roi_id:
                mapping[roi_id] = name or roi_id
    return mapping


def discover_te_dirs(root: Path, include_repeat: bool) -> list[tuple[str, str, Path]]:
    rows = []
    for subject_dir in sorted(root.glob("sub-*")):
        dwi_dir = subject_dir / "dwi"
        for dwi_file in sorted(dwi_dir.glob("*_dwi.nii.gz")):
            match = TE_RE.search(dwi_file.name)
            if not match:
                continue
            te_label = match.group(1)
            if te_label.endswith("R2") and not include_repeat:
                continue
            rows.append((subject_dir.name, te_label, dwi_dir))
    return rows


def clean_metric_image(image: Path, clean_dir: Path, model: str, metric: str) -> Path:
    lower, upper, _ = metric_bounds(metric)
    clean_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    out = clean_dir / f"{safe_model}_{metric}.nii.gz"
    command = ["fslmaths", str(image), "-nan"]
    if lower is not None:
        command.extend(["-thr", str(lower)])
    if upper is not None:
        command.extend(["-uthr", str(upper)])
    command.append(str(out))
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"fslmaths failed for {image}:\n{result.stderr}")
    return out


def fslstats_mean_std(image: Path, mask: Path) -> tuple[str, str] | None:
    command = ["fslstats", str(image), "-k", str(mask), "-M", "-S"]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split()
    if len(parts) < 2:
        return None
    try:
        mean = float(parts[0])
        std = float(parts[1])
    except ValueError:
        return None
    if not math.isfinite(mean) or not math.isfinite(std):
        return None
    return parts[0], parts[1]


def format_float(value: str) -> str:
    x = float(value)
    if 0 < abs(x) < 1e-5:
        return f"{x:.10e}"
    return f"{x:.10f}"


def process_one(
    subject: str,
    te_label: str,
    metric_dir: Path,
    masks: list[Path],
    roi_names: dict[str, str],
    clean_root: Path,
) -> list[list[str]]:
    rows: list[list[str]] = []
    te_numeric = te_label.replace("R2", "")
    clean_dir = clean_root / subject / f"TE{te_label}"
    for model, metrics in METRICS.items():
        directory = model_dir(metric_dir, model)
        for metric in metrics:
            image = directory / f"{metric}.nii.gz"
            if not image.exists():
                continue
            try:
                clean_image = clean_metric_image(image, clean_dir, model, metric)
            except RuntimeError as error:
                print(f"[WARN] {error}", flush=True)
                continue
            for mask in masks:
                roi_id = mask.stem
                if roi_id.endswith(".nii"):
                    roi_id = roi_id[:-4]
                stats = fslstats_mean_std(clean_image, mask)
                if stats is None:
                    continue
                mean, std = stats
                rows.append(
                    [
                        subject,
                        te_numeric,
                        model,
                        metric,
                        roi_id,
                        roi_names.get(roi_id, roi_id),
                        format_float(mean),
                        format_float(std),
                    ]
                )
    print(f"[DONE] {subject} TE{te_label}: {len(rows)} QC rows", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ScienceDB BN/JHU ROI mean/std after voxel-level physical-bound QC. "
            "This follows the MTE3_clean QC rule: invalid non-physical voxel values are "
            "removed before ROI averaging."
        )
    )
    parser.add_argument("--root", default="/media/UG1/lys/dipy/data/ScienceDB_MTE")
    parser.add_argument("--mask-root", default="/media/UG1/lys/dipy/data/ScienceDB_MTE_clean")
    parser.add_argument("--metric-root", default="/media/UG1/lys/dipy/data/ScienceDB_MTE_clean")
    parser.add_argument("--roi-mapping", default="data/raw/BN_JHU_roi_mapping.csv")
    parser.add_argument(
        "--output",
        default="data/raw/ScienceDB_BN_JHU_metric_results_native_qc.csv",
    )
    parser.add_argument(
        "--clean-root",
        default="/media/UG1/lys/dipy/data/ScienceDB_MTE_clean_qc",
        help="Temporary/traceable QC metric maps after fslmaths -nan and physical bounds.",
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--include-repeat", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    root = Path(args.root)
    mask_root = Path(args.mask_root)
    metric_root = Path(args.metric_root)
    clean_root = Path(args.clean_root)
    output = Path(args.output)
    roi_mapping = Path(args.roi_mapping)
    if not output.is_absolute():
        output = project_root / output
    if not roi_mapping.is_absolute():
        roi_mapping = project_root / roi_mapping
    output.parent.mkdir(parents=True, exist_ok=True)

    roi_names = read_roi_mapping(roi_mapping)
    jobs = []
    for subject, te_label, _ in discover_te_dirs(root, args.include_repeat):
        mask_dir = mask_root / subject / "BN_JHU"
        masks = sorted(mask_dir.glob("W_*.nii.gz")) + sorted(mask_dir.glob("G_*.nii.gz"))
        if not masks:
            print(f"[SKIP] {subject} TE{te_label}: missing ROI masks in {mask_dir}", flush=True)
            continue
        metric_dir = metric_root / subject / f"TE{te_label}"
        if not metric_dir.exists():
            print(f"[SKIP] {subject} TE{te_label}: missing metric dir {metric_dir}", flush=True)
            continue
        jobs.append((subject, te_label, metric_dir, masks, roi_names, clean_root))

    header = ["Subject", "TE", "Model", "Metric", "ROI_ID", "ROI_Name", "Mean", "Std"]
    all_rows: list[list[str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(process_one, *job) for job in jobs]
        for future in as_completed(futures):
            all_rows.extend(future.result())

    all_rows.sort(key=lambda row: (row[0], int(row[1]), row[2], row[3], row[4]))
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(all_rows)

    print(f"All done: {output} ({len(all_rows)} rows)")
    print(f"QC metric maps: {clean_root}")
    print("Expected complete rows without repeats: 3 subjects x 8 TEs x 38 metrics x 296 ROIs = 269952")


if __name__ == "__main__":
    main()
