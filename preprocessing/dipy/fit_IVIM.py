#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import time
import warnings

# DIPY imports
from dipy.reconst.ivim import IvimModel
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

# -----------------------------------------------------------
# 屏蔽警告，让输出更干净
# -----------------------------------------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="dipy.reconst.ivim")
warnings.filterwarnings("ignore", category=UserWarning, module="dipy.reconst.multi_voxel")

def main():
    # -----------------
    # Parse arguments
    # -----------------
    parser = argparse.ArgumentParser(
        description="Fit IVIM model using DIPY (Fully Fixed)",
        epilog="Computes Perfusion (f, D*) and Diffusion (D) parameters.")
    parser.add_argument("-v", "--version",
                        action="version", default=argparse.SUPPRESS,
                        version='1.3_ivim_fixed',
                        help="Show program's version number and exit")
    parser.add_argument('subjectDirectory', help='A directory of study subjects.')
    parser.add_argument('dwiFile', help='Name of DWI (must contain low b-values).')
    parser.add_argument('bvalFile', help='Name of b-value.')
    parser.add_argument('bvecFile', help='Name of gradient vector.')
    parser.add_argument('maskFile', help='Name of brain mask.')

    parser.add_argument('--method', choices=['trr', 'VarPro'], default='trr',
                        help="Fitting method: 'trr' (standard) or 'VarPro' (refined).")

    args = parser.parse_args()

    subjectDirectory = args.subjectDirectory
    dwiFile = os.path.join(subjectDirectory, args.dwiFile)
    bvalFile = os.path.join(subjectDirectory, args.bvalFile)
    bvecFile = os.path.join(subjectDirectory, args.bvecFile)
    maskFile = os.path.join(subjectDirectory, args.maskFile)

    print(f"Processing IVIM for: {subjectDirectory}")

    # 1. 加载数据
    print("Loading data...")
    dwi_data, dwi_affine = load_nifti(dwiFile, return_img=False)
    mask_data, mask_affine = load_nifti(maskFile, return_img=False)

    # 2. 读取梯度表
    # 【修复点】去掉关键字参数，直接传值
    bvals, bvecs = read_bvals_bvecs(bvalFile, bvecFile)

    # 【关键】显式设置 b0_threshold=0，防止 DIPY 报错
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=0)

    # 3. 检查数据要求
    low_b_indices = np.where((bvals > 0) & (bvals < 200))[0]
    if len(low_b_indices) == 0:
        print("!!! WARNING: No low b-values (0 < b < 200) detected.")
        print("!!! The results for f and D* may be unreliable.")

    # 4. 建立 IVIM 模型
    print(f"Initializing IVIM Model (Method: {args.method})...")
    ivim_model = IvimModel(gtab, fit_method=args.method)

    # 5. 拟合模型
    print("Fitting IVIM model (warnings suppressed)...")
    t_start = time.time()

    # 在 Mask 内拟合
    ivim_fit = ivim_model.fit(dwi_data, mask=mask_data)

    print(f"Fitting complete in {time.time() - t_start:.2f} seconds.")

    # 6. 提取指标
    print("Extracting metrics (f, D*, D)...")
    # 提取参数 [S0, f, D_star, D]
    S0_map = ivim_fit.model_params[..., 0]
    f_map = ivim_fit.model_params[..., 1]
    D_star_map = ivim_fit.model_params[..., 2]
    D_map = ivim_fit.model_params[..., 3]

    # 数据清理 (去除极端的异常值)
    f_map = np.clip(f_map, 0, 1)
    D_map[D_map < 0] = 0
    D_star_map[D_star_map < 0] = 0

    # 7. 保存结果
    outdir = os.path.join(subjectDirectory, 'IVIM')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    print(f"Saving results to {outdir}...")
    save_nifti(os.path.join(outdir, 'IVIM_f.nii.gz'), f_map.astype(np.float32), dwi_affine)
    save_nifti(os.path.join(outdir, 'IVIM_D_star.nii.gz'), D_star_map.astype(np.float32), dwi_affine)
    save_nifti(os.path.join(outdir, 'IVIM_D.nii.gz'), D_map.astype(np.float32), dwi_affine)
    save_nifti(os.path.join(outdir, 'IVIM_S0.nii.gz'), S0_map.astype(np.float32), dwi_affine)

    print("Done!")

if __name__ == '__main__':
    main()