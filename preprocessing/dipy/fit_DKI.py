#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import time
import warnings

# DIPY imports
import dipy.reconst.dki as dki
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

# 忽略数值计算警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    # -----------------
    # Parse arguments
    # -----------------
    parser = argparse.ArgumentParser(
        description="Fit DKI (Diffusion Kurtosis Imaging) model using DIPY",
        epilog="Note: DKI requires Multi-Shell data (multiple non-zero b-values).")
    parser.add_argument("-v", "--version",
                        action="version", default=argparse.SUPPRESS,
                        version='1.0_dki',
                        help="Show program's version number and exit")
    parser.add_argument('subjectDirectory', help='A directory of study subjects.')
    parser.add_argument('dwiFile', help='Name of DWI (must be multi-shell).')
    parser.add_argument('bvalFile', help='Name of b-value.')
    parser.add_argument('bvecFile', help='Name of gradient vector.')
    parser.add_argument('maskFile', help='Name of brain mask.')

    # 可选：是否使用约束拟合（更慢但更稳健，防止负峰度）
    # parser.add_argument('--constrained', action='store_true', help='Use constrained optimization (slower)')

    args = parser.parse_args()

    subjectDirectory = args.subjectDirectory
    dwiFile = os.path.join(subjectDirectory, args.dwiFile)
    bvalFile = os.path.join(subjectDirectory, args.bvalFile)
    bvecFile = os.path.join(subjectDirectory, args.bvecFile)
    maskFile = os.path.join(subjectDirectory, args.maskFile)

    print(f"Processing DKI for: {subjectDirectory}")

    # 1. 加载数据
    print("Loading data...")
    dwi_data, dwi_affine = load_nifti(dwiFile, return_img=False)
    mask_data, mask_affine = load_nifti(maskFile, return_img=False)

    # 2. 读取梯度表
    bvals, bvecs = read_bvals_bvecs(bvalFile, bvecFile)
    gtab = gradient_table(bvals, bvecs=bvecs)

    # 3. 建立 DKI 模型
    # DKI 需要多壳数据。如果你的数据只有单壳（Single Shell），这里会报错或结果不准确。
    print("Initializing Diffusion Kurtosis Model...")

    # 标准未约束 DKI (速度快，计算标准指标首选)
    dkimodel = dki.DiffusionKurtosisModel(gtab)

    # 如果需要防止负峰度值（"黑洞"伪影），可以使用约束拟合（非常慢，通常建议先做去噪而不是强行约束）
    # dkimodel = dki.DiffusionKurtosisModel(gtab, fit_method="CLS")

    # 4. 拟合模型
    print("Fitting DKI model (this may take longer than DTI)...")
    t_start = time.time()

    # 仅在 Mask 内拟合
    dkifit = dkimodel.fit(dwi_data, mask=mask_data)

    print(f"Fitting complete in {time.time() - t_start:.2f} seconds.")

    # 5. 提取指标
    print("Extracting metrics...")

    # --- Part A: DTI-based metrics (derived from DKI fit) ---
    # DKI 模型包含扩散张量 (DT) 分量，可以计算更准确的 FA/MD
    FA = dkifit.fa
    MD = dkifit.md
    AD = dkifit.ad
    RD = dkifit.rd

    # 清理 NaN
    FA[np.isnan(FA)] = 0
    MD[np.isnan(MD)] = 0
    AD[np.isnan(AD)] = 0
    RD[np.isnan(RD)] = 0

    # --- Part B: Kurtosis metrics ---
    # MK: Mean Kurtosis (平均峰度) - 反映微结构的复杂程度
    MK = dkifit.mk(0, 3) # min=0, max=3 通常用于去除极端值，或者直接用 dkifit.mk()

    # AK: Axial Kurtosis (轴向峰度)
    AK = dkifit.ak(0, 3)

    # RK: Radial Kurtosis (径向峰度)
    RK = dkifit.rk(0, 3)

    # 6. 保存结果
    outdir = os.path.join(subjectDirectory, 'DKI')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    print(f"Saving results to {outdir}...")

    # 保存 DTI 指标
    save_nifti(os.path.join(outdir, 'DKI_FA.nii.gz'), FA.astype(np.float32), dwi_affine)
    save_nifti(os.path.join(outdir, 'DKI_MD.nii.gz'), MD.astype(np.float32), dwi_affine)
    save_nifti(os.path.join(outdir, 'DKI_AD.nii.gz'), AD.astype(np.float32), dwi_affine)
    save_nifti(os.path.join(outdir, 'DKI_RD.nii.gz'), RD.astype(np.float32), dwi_affine)

    # 保存 Kurtosis 指标
    save_nifti(os.path.join(outdir, 'DKI_MK.nii.gz'), MK.astype(np.float32), dwi_affine)
    save_nifti(os.path.join(outdir, 'DKI_AK.nii.gz'), AK.astype(np.float32), dwi_affine)
    save_nifti(os.path.join(outdir, 'DKI_RK.nii.gz'), RK.astype(np.float32), dwi_affine)

    print("Done!")

if __name__ == '__main__':
    main()