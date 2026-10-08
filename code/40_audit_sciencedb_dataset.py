from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


TE_RE = re.compile(r"acq-TE([0-9]+(?:R2)?)_dwi\.nii\.gz$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ScienceDB MTE DWI files.")
    parser.add_argument(
        "--root",
        default="/media/UG1/lys/dipy/data/ScienceDB_MTE",
        help="ScienceDB_MTE root directory on the server.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs_sciencedb",
        help="Output directory relative to project root, or absolute path.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[1] / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for subject_dir in sorted(root.glob("sub-*")):
        dwi_dir = subject_dir / "dwi"
        for dwi_file in sorted(dwi_dir.glob("*_dwi.nii.gz")):
            match = TE_RE.search(dwi_file.name)
            if not match:
                continue
            te_label = match.group(1)
            prefix = dwi_file.name.replace("_dwi.nii.gz", "")
            rows.append(
                {
                    "subject": subject_dir.name,
                    "te_label": te_label,
                    "te_numeric": int(te_label.replace("R2", "")),
                    "is_repeat": te_label.endswith("R2"),
                    "dwi": str(dwi_file),
                    "bval": str(dwi_dir / f"{prefix}_dwi.bval"),
                    "bvec": str(dwi_dir / f"{prefix}_dwi.bvec"),
                    "mask": str(dwi_dir / f"{prefix}_mask.nii.gz"),
                    "b0ave": str(dwi_dir / f"{prefix}_B0ave.nii.gz"),
                }
            )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise SystemExit(f"No ScienceDB DWI files found under {root}")

    for column in ["bval", "bvec", "mask", "b0ave"]:
        manifest[f"has_{column}"] = manifest[column].map(lambda value: Path(value).exists())

    anat_files = []
    for pattern in ("*T1*", "*anat*", "*mprage*", "*MPRAGE*"):
        anat_files.extend(root.glob(f"**/{pattern}"))

    manifest.to_csv(out_dir / "40_sciencedb_manifest.csv", index=False, encoding="utf-8-sig")
    te_table = (
        manifest.groupby(["te_label", "te_numeric", "is_repeat"], as_index=False)
        .agg(n_subjects=("subject", "nunique"), n_files=("dwi", "size"))
        .sort_values(["te_numeric", "is_repeat", "te_label"])
    )
    te_table.to_csv(out_dir / "40_sciencedb_te_coverage.csv", index=False, encoding="utf-8-sig")

    subject_table = (
        manifest.groupby("subject", as_index=False)
        .agg(
            n_te=("te_label", "nunique"),
            te_labels=("te_label", lambda values: ",".join(sorted(values, key=lambda x: (int(x.replace("R2", "")), x.endswith("R2"))))),
            missing_sidecars=(
                "has_bval",
                lambda values: "",  # filled below for readability
            ),
        )
        .drop(columns=["missing_sidecars"])
    )
    subject_table.to_csv(out_dir / "40_sciencedb_subject_coverage.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 40 ScienceDB 数据审计",
        "",
        f"- 数据根目录：`{root}`",
        f"- 受试者数：{manifest['subject'].nunique()}",
        f"- DWI 文件数：{len(manifest)}",
        f"- 是否找到 T1/anat/MPRAGE：{'是' if anat_files else '否'}",
        "",
        "## TE 覆盖",
        "",
    ]
    for row in te_table.itertuples(index=False):
        repeat = "，重复采集" if row.is_repeat else ""
        lines.append(f"- TE{row.te_label}{repeat}：{row.n_subjects} 名受试者")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- ScienceDB 当前只有 3 名受试者，适合作为外部可行性测试，不能作为强统计验证。",
            "- 当前没有发现 T1/anat/MPRAGE 文件；若确实没有 T1，配准应使用 B0ave 或 FA 作为 DWI 空间参考。",
            "- ScienceDB 原始 TE 为 62/72/82/92/102/112/122/132，与内部模型 75/85/95/105/115 -> 125/135 不一致，外部测试前需要做 TE 插值/轻微外推。",
        ]
    )
    report = "\n".join(lines) + "\n"
    (out_dir / "40_sciencedb_audit_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
