#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified for robustness and warning suppression (WMTI Model).
"""

import argparse
import os
import warnings
import dipy.reconst.dki_micro as dki_micro
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti, save_nifti

# 【修改1】屏蔽数值计算警告
# 忽略拟合过程中可能出现的 RuntimeWarning (如除以零) 和 UserWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    #-----------------
    # Parse arguments
    #-----------------
    parser = argparse.ArgumentParser(
        description="Fit WMTI model with Dipy",
        epilog="Written by Ye Wu, Modified for robustness.")
    parser.add_argument("-v", "--version",
        action="version", default=argparse.SUPPRESS,
        version='1.1',
        help="Show program's version number and exit")
    parser.add_argument(
        'subjectDirectory',
        help='A directory of study subjects.')
    parser.add_argument(
        'dwiFile',
        help='Name of DWI.')
    parser.add_argument(
        'bvalFile',
        help='Name of b-value.')
    parser.add_argument(
        'bvecFile',
        help='Name of gradiet vectory.')
    parser.add_argument(
        'maskFile',
        help='Name of brain mask.')

    args = parser.parse_args()

    subjectDirectory = args.subjectDirectory
    dwiFile = os.path.join(subjectDirectory, args.dwiFile)
    bvalFile = os.path.join(subjectDirectory, args.bvalFile)
    bvecFile = os.path.join(subjectDirectory, args.bvecFile)
    maskFile = os.path.join(subjectDirectory, args.maskFile)

    print(f"Processing WMTI for: {subjectDirectory}")

    # 【修改2】数据加载增加错误捕获
    try:
        dwi_data, dwi_affine = load_nifti(dwiFile, return_img=False)
        mask_data, mask_affine = load_nifti(maskFile, return_img=False)

        # 确保使用关键字参数 bvecs
        gtab = gradient_table(bvalFile, bvecs=bvecFile)
    except Exception as e:
        print(f"[Error] Failed to load data: {e}")
        return

    # WMTI 模型拟合
    print("Fitting WMTI model...")
    # 使用 KurtosisMicrostructureModel (WMTI 是基于 DKI 的微结构模型)
    dki_micro_model = dki_micro.KurtosisMicrostructureModel(gtab)

    # 拟合
    dki_micro_fit = dki_micro_model.fit(dwi_data, mask=mask_data)

    # 提取 WMTI 特有的指标
    AWF = dki_micro_fit.awf
    Tortuosity = dki_micro_fit.tortuosity
    Restricted = dki_micro_fit.restricted_evals
    Hindered = dki_micro_fit.hindered_evals
    Axonal = dki_micro_fit.axonal_diffusivity
    Hindered_AD = dki_micro_fit.hindered_ad
    Hindered_RD = dki_micro_fit.hindered_rd

    # 保存结果
    outdir = os.path.join(subjectDirectory, 'WMTI')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    print(f"Saving results to {outdir}...")
    save_nifti(os.path.join(outdir,'AWF.nii.gz'), AWF, dwi_affine)
    save_nifti(os.path.join(outdir,'Tortuosity.nii.gz'), Tortuosity, dwi_affine)
    save_nifti(os.path.join(outdir,'Restricted.nii.gz'), Restricted, dwi_affine)
    save_nifti(os.path.join(outdir,'Hindered.nii.gz'), Hindered, dwi_affine)
    save_nifti(os.path.join(outdir,'Axonal.nii.gz'), Axonal, dwi_affine)
    save_nifti(os.path.join(outdir,'Hindered_AD.nii.gz'), Hindered_AD, dwi_affine)
    save_nifti(os.path.join(outdir,'Hindered_RD.nii.gz'), Hindered_RD, dwi_affine)

    print("Done.")

if __name__ == '__main__':
    main()