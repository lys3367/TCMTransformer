#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import warnings
import numpy as np

# DIPY imports
import dipy.reconst.dti as dti
import dipy.denoise.noise_estimate as ne
from dipy.reconst.dti import fractional_anisotropy
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

# -----------------------------------------------------------
# 屏蔽警告，保持输出整洁
# -----------------------------------------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    parser = argparse.ArgumentParser(
        description="Fit DTI model using RESTORE (Saves FA, MD, AD, RD for Multi-TE analysis)")
    parser.add_argument('subjectDirectory', help='A directory of study subjects.')
    parser.add_argument('dwiFile', help='Name of DWI.')
    parser.add_argument('bvalFile', help='Name of b-value.')
    parser.add_argument('bvecFile', help='Name of gradient vector.')
    parser.add_argument('maskFile', help='Name of brain mask.')

    args = parser.parse_args()

    subjectDirectory = args.subjectDirectory
    dwiFile = os.path.join(subjectDirectory, args.dwiFile)
    bvalFile = os.path.join(subjectDirectory, args.bvalFile)
    bvecFile = os.path.join(subjectDirectory, args.bvecFile)
    maskFile = os.path.join(subjectDirectory, args.maskFile)

    print(f"-> 开始处理被试: {subjectDirectory} (DTI RESTORE - FA, MD, AD, RD)")

    # =======================================================
    # 加入全局容错机制 (Try-Except)
    # =======================================================
    try:
        # 1. 加载数据
        dwi_data, dwi_affine = load_nifti(dwiFile, return_img=False)
        mask_data, mask_affine = load_nifti(maskFile, return_img=False)

        # 2. 读取梯度表 (严格使用位置参数)
        bvals, bvecs = read_bvals_bvecs(bvalFile, bvecFile)
        gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=0)

        # 3. 估计背景噪声的标准差 (Sigma)
        # 这是 RESTORE 算法识别异常值所必需的关键参数
        sigma = ne.estimate_sigma(dwi_data)

        # 4. 初始化 DTI-RESTORE 模型
        # 指定 fit_method 为 "RESTORE" 并传入估算好的 sigma
        dti_restore = dti.TensorModel(gtab, fit_method="RESTORE", sigma=sigma)

        # 5. 在 Mask 范围内拟合模型
        tenfit = dti_restore.fit(dwi_data, mask=mask_data)

        # 6. 提取四大核心微观结构指标
        FA = fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 0

        MD = tenfit.md
        AD = tenfit.ad
        RD = tenfit.rd

        # 7. 保存结果
        outdir = os.path.join(subjectDirectory, 'DTI_RESTORE')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        save_nifti(os.path.join(outdir, 'FA.nii.gz'), FA.astype(np.float32), dwi_affine)
        save_nifti(os.path.join(outdir, 'MD.nii.gz'), MD.astype(np.float32), dwi_affine)
        save_nifti(os.path.join(outdir, 'AD.nii.gz'), AD.astype(np.float32), dwi_affine)
        save_nifti(os.path.join(outdir, 'RD.nii.gz'), RD.astype(np.float32), dwi_affine)

        print(f"<- 处理完成，已保存 FA, MD, AD, RD 至: {outdir}\n")

    # 捕获所有异常并优雅退出
    except EOFError:
        print(f"   [跳过] 致命错误: 文件损坏或不完整 (EOFError)。请检查 {dwiFile}")
    except Exception as e:
        print(f"   [跳过] 处理失败: {type(e).__name__} - {str(e)}")

if __name__ == '__main__':
    main()