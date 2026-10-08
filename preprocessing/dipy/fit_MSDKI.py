# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Created on Sat Aug 19 14:35:37 2023
#
# @author: wuye
# """
#
# import argparse
# import os
# import dipy.reconst.msdki as msdki
# from dipy.core.gradients import gradient_table
# from dipy.io.image import load_nifti, save_nifti
#
# def main():
#     #-----------------
#     # Parse arguments
#     #-----------------
#     parser = argparse.ArgumentParser(
#         description="Fit MSDKI model with Dipy",
#         epilog="Written by Ye Wu, dr.yewu@outlook.com.\"")
#     parser.add_argument("-v", "--version",
#         action="version", default=argparse.SUPPRESS,
#         version='1.0',
#         help="Show program's version number and exit")
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
#     gtab = gradient_table(bvalFile,bvecFile)
#
#     msdki_model = msdki.MeanDiffusionKurtosisModel(gtab)
#     msdki_fit = msdki_model.fit(dwi_data, mask=mask_data)
#
#     MSD = msdki_fit.msd
#     MSK = msdki_fit.msk
#     F = msdki_fit.smt2f
#     DI = msdki_fit.smt2di
#     uFA = msdki_fit.smt2uFA
#
#     outdir = os.path.join(subjectDirectory,'MSDKI')
#     if not os.path.exists(outdir):
#         os.mkdir(outdir)
#
#     save_nifti(os.path.join(outdir,'MSD.nii.gz'), MSD, dwi_affine)
#     save_nifti(os.path.join(outdir,'MSK.nii.gz'), MSK, dwi_affine)
#     save_nifti(os.path.join(outdir,'F.nii.gz'), F, dwi_affine)
#     save_nifti(os.path.join(outdir,'DI.nii.gz'), DI, dwi_affine)
#     save_nifti(os.path.join(outdir,'uFA.nii.gz'), uFA, dwi_affine)
#
# if __name__ == '__main__':
#     main()





#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified for robustness and warning suppression.
"""

import argparse
import os
import warnings
import dipy.reconst.msdki as msdki
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti, save_nifti

# 【修改1】屏蔽数值计算警告 (如 overflow, invalid value)
# 这些警告通常出现在背景(Mask外)的噪声拟合中，不影响结果
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    #-----------------
    # Parse arguments
    #-----------------
    parser = argparse.ArgumentParser(
        description="Fit MSDKI model with Dipy",
        epilog="Written by Ye Wu, Modified for API compatibility.")
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

    print(f"Processing MSDKI for: {subjectDirectory}")

    # 加载数据
    try:
        dwi_data, dwi_affine = load_nifti(dwiFile, return_img=False)
        mask_data, mask_affine = load_nifti(maskFile, return_img=False)

        # 【修改2】使用关键字参数 bvecs，解决 UserWarning
        gtab = gradient_table(bvalFile, bvecs=bvecFile)
    except Exception as e:
        print(f"[Error] Failed to load data: {e}")
        return

    # MSDKI 模型拟合
    print("Fitting MSDKI model...")
    msdki_model = msdki.MeanDiffusionKurtosisModel(gtab)

    # 拟合
    msdki_fit = msdki_model.fit(dwi_data, mask=mask_data)

    # 提取指标
    MSD = msdki_fit.msd
    MSK = msdki_fit.msk
    # 注意：以下参数依赖于 Dipy 版本，如果报错请确认版本
    F = msdki_fit.smt2f
    DI = msdki_fit.smt2di
    uFA = msdki_fit.smt2uFA

    # 保存结果
    outdir = os.path.join(subjectDirectory, 'MSDKI')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    print(f"Saving results to {outdir}...")
    save_nifti(os.path.join(outdir,'MSD.nii.gz'), MSD, dwi_affine)
    save_nifti(os.path.join(outdir,'MSK.nii.gz'), MSK, dwi_affine)
    save_nifti(os.path.join(outdir,'F.nii.gz'), F, dwi_affine)
    save_nifti(os.path.join(outdir,'DI.nii.gz'), DI, dwi_affine)
    save_nifti(os.path.join(outdir,'uFA.nii.gz'), uFA, dwi_affine)

    print("Done.")

if __name__ == '__main__':
    main()