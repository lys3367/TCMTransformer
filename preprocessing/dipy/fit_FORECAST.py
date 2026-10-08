#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import warnings
import numpy as np

# DIPY imports
from dipy.reconst.forecast import ForecastModel
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
        description="Fit FORECAST model (Voxel-wise Scalar Metrics ONLY for Multi-TE analysis)")
    parser.add_argument('subjectDirectory', help='A directory of study subjects.')
    parser.add_argument('dwiFile', help='Name of DWI (Multi-shell data recommended).')
    parser.add_argument('bvalFile', help='Name of b-value.')
    parser.add_argument('bvecFile', help='Name of gradient vector.')
    parser.add_argument('maskFile', help='Name of brain mask.')

    args = parser.parse_args()

    subjectDirectory = args.subjectDirectory
    dwiFile = os.path.join(subjectDirectory, args.dwiFile)
    bvalFile = os.path.join(subjectDirectory, args.bvalFile)
    bvecFile = os.path.join(subjectDirectory, args.bvecFile)
    maskFile = os.path.join(subjectDirectory, args.maskFile)

    print(f"-> 开始处理被试: {subjectDirectory} (FORECAST - Scalar Metrics Only)")

    # 1. 加载数据
    dwi_data, dwi_affine = load_nifti(dwiFile, return_img=False)
    mask_data, mask_affine = load_nifti(maskFile, return_img=False)

    # 2. 读取梯度表 (严格使用位置参数避免报错)
    bvals, bvecs = read_bvals_bvecs(bvalFile, bvecFile)
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=0)

    # 3. 初始化 FORECAST 模型
    # 完全采用原代码中的参数: sh_order_max=6, dec_alg="CSD"
    fm = ForecastModel(gtab, sh_order_max=4, dec_alg="CSD")

    # 4. 拟合模型
    # 在 Mask 内拟合模型以节省时间
    f_fit = fm.fit(dwi_data, mask=mask_data)

    # 5. 提取具有 TE 分析意义的交叉不变张量指标 (排除 ODF)
    # 提取平行扩散率, 垂直扩散率, 各向异性分数, 平均扩散率
    d_par = f_fit.dpar
    d_perp = f_fit.dperp
    fa = f_fit.fractional_anisotropy()
    md = f_fit.mean_diffusivity()

    # 6. 保存结果
    outdir = os.path.join(subjectDirectory, 'FORECAST')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 将提取的标量指标保存为 NIfTI 格式
    save_nifti(os.path.join(outdir, 'd_par.nii.gz'), d_par.astype(np.float32), dwi_affine)
    save_nifti(os.path.join(outdir, 'd_perp.nii.gz'), d_perp.astype(np.float32), dwi_affine)
    save_nifti(os.path.join(outdir, 'fa.nii.gz'), fa.astype(np.float32), dwi_affine)
    save_nifti(os.path.join(outdir, 'md.nii.gz'), md.astype(np.float32), dwi_affine)

    print(f"<- 处理完成，已保存标量指标 (d_par, d_perp, fa, md) 至: {outdir}\n")

if __name__ == '__main__':
    main()