# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Created on Fri Aug 18 17:36:02 2023
#
# @author: wuye
# """
#
# import argparse
# import os
# import dipy.reconst.fwdti as fwdti
# from dipy.core.gradients import gradient_table
# from dipy.io.image import load_nifti, save_nifti
# import numpy as np
# import pandas as pd
#
#
# def main():
#     # -----------------
#     # Parse arguments
#     # -----------------
#     parser = argparse.ArgumentParser(
#         description="Fit FreeWater model with Dipy",
#         epilog="Written by Ye Wu, dr.yewu@outlook.com.\"")
#     parser.add_argument("-v", "--version",
#                         action="version", default=argparse.SUPPRESS,
#                         version='1.0',
#                         help="Show program's version number and exit")
#     parser.add_argument(
#         'subjectDirectory',
#         help='A directory of study subjects.')
#     parser.add_argument(
#         'dwiFile',
#         help='Name of DWI.')
#     parser.add_argument(
#         'bvalFile',
#         help='Name of b-value.')
#     parser.add_argument(
#         'bvecFile',
#         help='Name of gradiet vectory.')
#     parser.add_argument(
#         'maskFile',
#         help='Name of brain mask.')
#
#     args = parser.parse_args()
#
#     subjectDirectory = args.subjectDirectory
#     dwiFile = os.path.join(subjectDirectory, args.dwiFile)
#     bvalFile = os.path.join(subjectDirectory, args.bvalFile)
#     bvecFile = os.path.join(subjectDirectory, args.bvecFile)
#     maskFile = os.path.join(subjectDirectory, args.maskFile)
#
#     dwi_data, dwi_affine = load_nifti(dwiFile, return_img=False)
#     mask_data, mask_affine = load_nifti(maskFile, return_img=False)
#     gtab = gradient_table(bvalFile, bvecFile)
#
#     fwdtimodel = fwdti.FreeWaterTensorModel(gtab)
#     fwdtifit = fwdtimodel.fit(dwi_data, mask=mask_data)
#
#     FA = fwdtifit.fa
#     MD = fwdtifit.md
#     FW = fwdtifit.f
#     RD = fwdtifit.rd
#     AD = fwdtifit.ad
#     # ADC暂时不算
#     outdir = os.path.join(subjectDirectory, 'FWDTI')
#     if not os.path.exists(outdir):
#         os.mkdir(outdir)
#
#     save_nifti(os.path.join(outdir, 'FA.nii.gz'), FA, dwi_affine)
#     save_nifti(os.path.join(outdir, 'MD.nii.gz'), MD, dwi_affine)
#     save_nifti(os.path.join(outdir, 'FW.nii.gz'), FW, dwi_affine)
#     save_nifti(os.path.join(outdir, 'RD.nii.gz'), RD, dwi_affine)
#     save_nifti(os.path.join(outdir, 'AD.nii.gz'), AD, dwi_affine)
#
#
#
# if __name__ == '__main__':
#     main()




#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import warnings
import dipy.reconst.fwdti as fwdti
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti, save_nifti

# 忽略数值计算的 RuntimeWarning，避免刷屏
warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    # -----------------
    # Parse arguments
    # -----------------
    parser = argparse.ArgumentParser(
        description="Fit FreeWater model with Dipy",
        epilog="Written by Ye Wu, modified for robustness.")
    parser.add_argument("-v", "--version",
                        action="version", default=argparse.SUPPRESS,
                        version='1.1',
                        help="Show program's version number and exit")
    parser.add_argument('subjectDirectory', help='A directory of study subjects.')
    parser.add_argument('dwiFile', help='Name of DWI.')
    parser.add_argument('bvalFile', help='Name of b-value.')
    parser.add_argument('bvecFile', help='Name of gradiet vectory.')
    parser.add_argument('maskFile', help='Name of brain mask.')

    args = parser.parse_args()

    subjectDirectory = args.subjectDirectory
    dwiFile = os.path.join(subjectDirectory, args.dwiFile)
    bvalFile = os.path.join(subjectDirectory, args.bvalFile)
    bvecFile = os.path.join(subjectDirectory, args.bvecFile)
    maskFile = os.path.join(subjectDirectory, args.maskFile)

    print(f"Processing: {subjectDirectory}")

    # 加载数据
    dwi_data, dwi_affine = load_nifti(dwiFile, return_img=False)
    mask_data, mask_affine = load_nifti(maskFile, return_img=False)

    # 【修复1】使用关键字参数传递 bvecs
    #读取bval和bvec文件
    gtab = gradient_table(bvalFile, bvecs=bvecFile)

    # 【检查】简单的b值检查，提醒用户
    unique_b = np.unique(gtab.bvals)
    # 过滤掉接近0的b值
    non_zero_b = unique_b[unique_b > 50]
    print(f"Detected b-values: {unique_b}")
    if len(non_zero_b) < 2:
        print("!!! WARNING: Free Water DTI typically requires Multi-Shell data (at least 2 non-zero b-values).")
        print("!!! Fitting on Single-Shell data is ill-posed and may cause convergence errors (maxfev/overflow).")

    # 建立模型
    # 可以在这里调整参数，例如 ftol (function tolerance) 来放宽收敛条件，但这通常治标不治本
    fwdtimodel = fwdti.FreeWaterTensorModel(gtab)

    print("Fitting model... (RuntimeWarnings are suppressed)")
    # 拟合
    fwdtifit = fwdtimodel.fit(dwi_data, mask=mask_data)

    # 提取结果
    FA = fwdtifit.fa
    MD = fwdtifit.md
    FW = fwdtifit.f
    RD = fwdtifit.rd
    AD = fwdtifit.ad

    # 输出目录
    outdir = os.path.join(subjectDirectory, 'FWDTI')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # 保存
    save_nifti(os.path.join(outdir, 'FA.nii.gz'), FA, dwi_affine)
    save_nifti(os.path.join(outdir, 'MD.nii.gz'), MD, dwi_affine)
    save_nifti(os.path.join(outdir, 'FW.nii.gz'), FW, dwi_affine)
    save_nifti(os.path.join(outdir, 'RD.nii.gz'), RD, dwi_affine)
    save_nifti(os.path.join(outdir, 'AD.nii.gz'), AD, dwi_affine)

    print(f"Done: {subjectDirectory}")

if __name__ == '__main__':
    main()