from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


TE_RE = re.compile(r"acq-TE([0-9]+(?:R2)?)_dwi\.nii\.gz$")


MODEL_SCRIPTS = {
    "DTI": "fit_DTI.py",
    "DTI_RESTORE": "fit_DTI_RESTORE.py",
    "FWDTI": "fit_FreeWater.py",
    "DKI": "fit_DKI.py",
    "WMTI": "fit_WMTI.py",
    "MSDKI": "fit_MSDKI.py",
    "IVIM": "fit_IVIM.py",
    "GQI": "fit_GQI.py",
    "AMICO_NODDI": "fit_NODDI.py",
    "FORECAST": "fit_FORECAST.py",
}


def discover_jobs(root: Path, work_root: Path, include_repeat: bool) -> list[dict[str, str]]:
    jobs = []
    for subject_dir in sorted(root.glob("sub-*")):
        dwi_dir = subject_dir / "dwi"
        for dwi_file in sorted(dwi_dir.glob("*_dwi.nii.gz")):
            match = TE_RE.search(dwi_file.name)
            if not match:
                continue
            te_label = match.group(1)
            if te_label.endswith("R2") and not include_repeat:
                continue
            prefix = dwi_file.name.replace("_dwi.nii.gz", "")
            bval = dwi_dir / f"{prefix}_dwi.bval"
            bvec = dwi_dir / f"{prefix}_dwi.bvec"
            mask = dwi_dir / f"{prefix}_mask.nii.gz"
            if not (bval.exists() and bvec.exists() and mask.exists()):
                jobs.append(
                    {
                        "subject": subject_dir.name,
                        "te": te_label,
                        "status": "missing_sidecar",
                        "message": f"Missing bval/bvec/mask for {dwi_file}",
                    }
                )
                continue
            jobs.append(
                {
                    "subject": subject_dir.name,
                    "te": te_label,
                    "status": "ready",
                    "source_dwi": str(dwi_file),
                    "source_bval": str(bval),
                    "source_bvec": str(bvec),
                    "source_mask": str(mask),
                    "work_directory": str(work_root / subject_dir.name / f"TE{te_label}"),
                }
            )
    return jobs


def output_exists(directory: Path, model: str) -> bool:
    checks = {
        "DTI": directory / "DTI" / "FA.nii.gz",
        "DTI_RESTORE": directory / "DTI_RESTORE" / "FA.nii.gz",
        "FWDTI": directory / "FWDTI" / "FA.nii.gz",
        "DKI": directory / "DKI" / "DKI_FA.nii.gz",
        "WMTI": directory / "WMTI" / "AWF.nii.gz",
        "MSDKI": directory / "MSDKI" / "MSD.nii.gz",
        "IVIM": directory / "IVIM" / "IVIM_f.nii.gz",
        "GQI": directory / "GQI" / "GFA.nii.gz",
        "AMICO_NODDI": directory / "AMICO" / "NODDI" / "fit_NDI.nii.gz",
        "FORECAST": directory / "FORECAST" / "fa.nii.gz",
    }
    return checks.get(model, directory / model).exists()


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def prepare_work_directory(job: dict[str, str]) -> Path:
    work_dir = Path(job["work_directory"])
    work_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy(Path(job["source_dwi"]), work_dir / "dwi.nii.gz")
    link_or_copy(Path(job["source_bval"]), work_dir / "dwi.bval")
    link_or_copy(Path(job["source_bvec"]), work_dir / "dwi.bvec")
    link_or_copy(Path(job["source_mask"]), work_dir / "mask.nii.gz")
    return work_dir


def run_one(job: dict[str, str], models: list[str], fit_dir: Path, force: bool) -> list[str]:
    lines = []
    if job["status"] != "ready":
        return [f"[SKIP] {job.get('subject')} TE{job.get('te')}: {job.get('message')}"]

    directory = prepare_work_directory(job)
    for model in models:
        script = fit_dir / MODEL_SCRIPTS[model]
        if not script.exists():
            lines.append(f"[SKIP] {job['subject']} TE{job['te']} {model}: missing {script}")
            continue
        if output_exists(directory, model) and not force:
            lines.append(f"[SKIP] {job['subject']} TE{job['te']} {model}: output exists")
            continue
        command = [
            sys.executable,
            str(script),
            str(directory),
            "dwi.nii.gz",
            "dwi.bval",
            "dwi.bvec",
            "mask.nii.gz",
        ]
        env = os.environ.copy()
        env["LC_NUMERIC"] = "C"
        result = subprocess.run(command, text=True, capture_output=True, env=env)
        if result.returncode == 0:
            lines.append(f"[DONE] {job['subject']} TE{job['te']} {model}")
        else:
            tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
            lines.append(
                f"[FAIL] {job['subject']} TE{job['te']} {model}: "
                + " | ".join(tail)
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fit_*.py metrics for ScienceDB MTE.")
    parser.add_argument("--root", default="/media/UG1/lys/dipy/data/ScienceDB_MTE")
    parser.add_argument("--work-root", default="/media/UG1/lys/dipy/data/ScienceDB_MTE_clean")
    parser.add_argument("--fit-dir", default="/media/UG1/lys/dipy")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SCRIPTS))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--include-repeat", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    work_root = Path(args.work_root)
    fit_dir = Path(args.fit_dir)
    models = [model for model in args.models if model in MODEL_SCRIPTS]
    if not models:
        raise ValueError("No valid models were requested.")

    jobs = discover_jobs(root, work_root, args.include_repeat)
    ready = [job for job in jobs if job["status"] == "ready"]
    print(f"ScienceDB subjects/jobs: {len(ready)} TE directories, models={models}", flush=True)

    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(run_one, job, models, fit_dir, args.force) for job in ready]
        for future in as_completed(futures):
            for line in future.result():
                print(line, flush=True)

    for job in jobs:
        if job["status"] != "ready":
            print(f"[SKIP] {job['subject']} TE{job['te']}: {job['message']}", flush=True)


if __name__ == "__main__":
    main()
