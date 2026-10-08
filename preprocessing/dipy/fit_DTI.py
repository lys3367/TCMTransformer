# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
# import argparse
# import os
# import numpy as np
# import warnings
# import dipy.reconst.dti as dti
# from dipy.core.gradients import gradient_table
# from dipy.io.gradients import read_bvals_bvecs
# from dipy.io.image import load_nifti, save_nifti
# from dipy.reconst.dti import fractional_anisotropy, color_fa
#
# # 忽略数值计算的 RuntimeWarning (如除以0产生的NaN)
# warnings.filterwarnings("ignore", category=RuntimeWarning)
#
# def main():
#     # -----------------
#     # Parse arguments
#     # -----------------
#     parser = argparse.ArgumentParser(
#         description="Fit DTI model with metrics from reconst_dti.py (FA, MD, RGB, Evecs)",
#         epilog="Modified to match dipy.reconst.dti output metrics.")
#     parser.add_argument("-v", "--version",
#                         action="version", default=argparse.SUPPRESS,
#                         version='1.0_dti_full',
#                         help="Show program's version number and exit")
#     parser.add_argument('subjectDirectory', help='A directory of study subjects.')
#     parser.add_argument('dwiFile', help='Name of DWI.')
#     parser.add_argument('bvalFile', help='Name of b-value.')
#     parser.add_argument('bvecFile', help='Name of gradient vector.')
#     parser.add_argument('maskFile', help='Name of brain mask.')
#
#     args = parser.parse_args()
#
#     subjectDirectory = args.subjectDirectory
#     dwiFile = os.path.join(subjectDirectory, args.dwiFile)
#     bvalFile = os.path.join(subjectDirectory, args.bvalFile)
#     bvecFile = os.path.join(subjectDirectory, args.bvecFile)
#     maskFile = os.path.join(subjectDirectory, args.maskFile)
#
#     print(f"Processing DTI (Standard Metrics) for: {subjectDirectory}")
#
#     # 1. 加载数据
#     dwi_data, dwi_affine = load_nifti(dwiFile, return_img=False)
#     mask_data, mask_affine = load_nifti(maskFile, return_img=False)
#
#     # 2. 读取梯度表
#     bvals, bvecs = read_bvals_bvecs(bvalFile, bvecFile)
#     gtab = gradient_table(bvals, bvecs=bvecs)
#
#     # 3. 拟合 DTI 模型
#     print("Fitting Tensor Model...")
#     tenmodel = dti.TensorModel(gtab)
#     tenfit = tenmodel.fit(dwi_data, mask=mask_data)
#
#     # 4. 计算指标 (Strictly following reconst_dti.py metrics)
#     print("Computing metrics: FA, MD, RGB, Eigenvectors...")
#
#     # --- FA (Fractional Anisotropy) ---
#     FA = fractional_anisotropy(tenfit.evals)
#     FA[np.isnan(FA)] = 0
#     # 确保 FA 在 0-1 之间，这对于生成 RGB 图至关重要
#     FA = np.clip(FA, 0, 1)
#
#     # --- MD (Mean Diffusivity) ---
#     MD = dti.mean_diffusivity(tenfit.evals)
#     # 或者 MD = tenfit.md
#
#     # --- RGB (Color FA) ---
#     # reconst_dti.py 中将 RGB 图转换为 0-255 的 uint8 格式保存
#     RGB = color_fa(FA, tenfit.evecs)
#     RGB_uint8 = np.array(255 * RGB, 'uint8')
#
#     # --- Eigenvectors (特征向量) ---
#     # 保存特征向量用于后续分析或可视化
#     Evecs = tenfit.evecs.astype(np.float32)
#
#     # 5. 保存结果
#     outdir = os.path.join(subjectDirectory, 'DTI_Metrics')
#     if not os.path.exists(outdir):
#         os.mkdir(outdir)
#
#     print(f"Saving files to {outdir}...")
#
#     # 使用简洁的文件名，但包含所有 reconst_dti.py 的输出类型
#     save_nifti(os.path.join(outdir, 'FA.nii.gz'), FA.astype(np.float32), dwi_affine)
#     save_nifti(os.path.join(outdir, 'MD.nii.gz'), MD.astype(np.float32), dwi_affine)
#
#     # 保存 RGB (Color FA)
#     save_nifti(os.path.join(outdir, 'RGB.nii.gz'), RGB_uint8, dwi_affine)
#
#     # 保存 Eigenvectors (4D image)
#     save_nifti(os.path.join(outdir, 'Evecs.nii.gz'), Evecs, dwi_affine)
#
#     print(f"Done! Output files: FA, MD, RGB, Evecs saved in {outdir}")
#
# if __name__ == '__main__':
#     main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import warnings
import numpy as np

# DIPY imports
import dipy.reconst.dti as dti
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
        description="Fit DTI model (Saves FA, MD, AD, RD for Multi-TE analysis)")
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

    print(f"-> 开始处理被试: {subjectDirectory} (DTI - FA, MD, AD, RD)")

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

        # 3. 初始化 DTI 模型
        # 使用官方推荐的加权最小二乘法 (WLS) 进行拟合
        tenmodel = dti.TensorModel(gtab, fit_method="WLS")

        # 4. 在 Mask 范围内拟合模型
        tenfit = tenmodel.fit(dwi_data, mask=mask_data)

        # 5. 提取四大核心微观结构指标
        # 计算 FA 并将背景区域的 NaN (非数字) 替换为 0
        FA = fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 0

        # 提取 MD, AD, RD
        MD = tenfit.md
        AD = tenfit.ad
        RD = tenfit.rd

        # 6. 保存结果
        outdir = os.path.join(subjectDirectory, 'DTI')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # 将四个指标保存为 NIfTI 格式
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