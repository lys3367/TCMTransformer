#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import time
import warnings

# DIPY imports
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.direction import peaks_from_model
from dipy.data import default_sphere
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

# -----------------------------------------------------------
# 屏蔽警告，保持输出整洁
# -----------------------------------------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # -----------------
    # Parse arguments
    # -----------------
    parser = argparse.ArgumentParser(
        description="Fit GQI (Generalized Q-Sampling Imaging) model using DIPY",
        epilog="Calculates GFA and correct quantitative QA (Quantitative Anisotropy).")
    parser.add_argument('subjectDirectory', help='A directory of study subjects.')
    parser.add_argument('dwiFile', help='Name of DWI.')
    parser.add_argument('bvalFile', help='Name of b-value.')
    parser.add_argument('bvecFile', help='Name of gradient vector.')
    parser.add_argument('maskFile', help='Name of brain mask.')

    # GQI 特有参数: sampling_length
    # 默认设为 3.0，如果你研究的主要是体内水分子扩散，1.2 也是常被推荐的比例值
    parser.add_argument('--sampling_length', type=float, default=3.0,
                        help='Diffusion sampling length (default: 3.0).')

    args = parser.parse_args()

    subjectDirectory = args.subjectDirectory
    dwiFile = os.path.join(subjectDirectory, args.dwiFile)
    bvalFile = os.path.join(subjectDirectory, args.bvalFile)
    bvecFile = os.path.join(subjectDirectory, args.bvecFile)
    maskFile = os.path.join(subjectDirectory, args.maskFile)

    print(f"-> 开始处理被试: {subjectDirectory} (GQI)")

    # =======================================================
    # 加入全局容错机制 (Try-Except) 以防批量处理时中断
    # =======================================================
    try:
        # 1. 加载数据
        dwi_data, dwi_affine = load_nifti(dwiFile, return_img=False)
        mask_data, mask_affine = load_nifti(maskFile, return_img=False)

        # 2. 读取梯度表
        bvals, bvecs = read_bvals_bvecs(bvalFile, bvecFile)

        # 【重要】确保 bvecs 是单位向量，避免 GQI 对梯度模长敏感导致的计算偏差
        bvecs_norm = np.linalg.norm(bvecs, axis=1)
        bvecs_norm[bvecs_norm == 0] = 1  # 避免除以0
        bvecs = bvecs / bvecs_norm[:, None]

        gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=0)

        # 3. 建立 GQI 模型
        gq_model = GeneralizedQSamplingModel(gtab, sampling_length=args.sampling_length)

        # 4. 拟合模型并提取峰值
        # 【核心修正】：normalize_peaks 必须为 False 才能获得具有物理意义的定量 QA
        gq_peaks = peaks_from_model(
            model=gq_model,
            data=dwi_data,
            sphere=default_sphere,
            relative_peak_threshold=0.5,
            min_separation_angle=25,
            mask=mask_data,
            return_odf=False,
            normalize_peaks=False
        )

        # 5. 提取指标
        # GFA (Generalized Fractional Anisotropy) - 广义各向异性分数
        GFA = gq_peaks.gfa

        # QA (Quantitative Anisotropy) - 定量各向异性
        # 取最主峰 (第 0 个峰) 的定量各向异性 (QA0)
        QA = gq_peaks.peak_values[..., 0]

        # 6. 保存结果
        outdir = os.path.join(subjectDirectory, 'GQI')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        save_nifti(os.path.join(outdir, 'GFA.nii.gz'), GFA.astype(np.float32), dwi_affine)
        save_nifti(os.path.join(outdir, 'QA.nii.gz'), QA.astype(np.float32), dwi_affine)

        print(f"<- 处理完成，已保存 GFA 和 QA 至: {outdir}\n")

    # 捕获异常并优雅退出
    except EOFError:
        print(f"   [跳过] 致命错误: 文件损坏或不完整 (EOFError)。请检查 {dwiFile}")
    except Exception as e:
        print(f"   [跳过] 处理失败: {type(e).__name__} - {str(e)}")

if __name__ == '__main__':
    main()