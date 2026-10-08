# Table 1 clean-49 重训代码

本目录只负责论文主任务的 canonical clean-49 LOSO 重训。四个正式模型为：

- iTransformer
- TimesNet
- MoLE
- DLinear

同一次训练同时保存 overall 和 TE125/TE135 预测，因此 Table 1 与 Supplementary Table S1 必须由同一批结果生成。

## 文件

- `00_build_clean49_manifest.py`：从现有 clean-49 MoLE LOSO 文件提取并冻结 49 名受试者名单。
- `01_train_table1_clean49.py`：按统一 folds、seed、标准化和训练参数重训一个模型，保存每折预测。
- `02_summarize_table1_s1.py`：检查四模型 folds 完全一致，同时生成 Table 1、S1 和配对显著性结果。
- `run_table1_clean49.sh`：服务器上一键顺序运行四个模型并保存日志。

## 服务器运行

```bash
cd /media/UG1/lys/dipy/BN_JHU296
conda activate mte
chmod +x code/clean_code/run_table1_clean49.sh
screen -S table1_clean49
GPU_ID=0 BS=1 bash code/clean_code/run_table1_clean49.sh
```

运行前必须激活 `mte` 环境。脚本只调用当前环境的 `python`，并在训练前打印解释器路径、NumPy、Pandas、PyTorch 版本和 CUDA 状态。四个模型必须使用同一个虚拟环境。

上传到服务器后必须确认没有残留旧解释器：

```bash
grep -nE "PYTHON_TORCH|/opt/fsl/bin/python" code/clean_code/run_table1_clean49.sh
```

正常情况下该命令没有输出。日志中的 `Python:` 必须指向 `/home/lys/.conda/envs/mte/bin/python`，否则不要继续训练。

按 `Ctrl+A`、再按 `D` 退出 screen 而不中止训练。重新进入：

```bash
screen -r table1_clean49
```

## 输出

全部新结果写入 `outputs/table1_clean49/`，不会覆盖历史结果：

- `fold_manifest.csv`：49 折受试者划分。
- `<model>/fold_XX.npz`：每折标准化空间及原始单位预测。
- `<model>_fold_metrics.csv`：每折 overall、TE125、TE135 指标。
- `<model>_predictions.npz`：模型的 49 折合并预测。
- `table1_clean49_summary.csv`：Table 1 数值。
- `s1_target_specific_summary.csv`：Supplementary Table S1 数值。
- `table1_pairwise_wilcoxon_fdr.csv`：Table 1 的受试者级配对检验。
- `table1_s1_consistency_check.csv`：overall 与两个 target MAE 平均关系检查。

脚本支持断点续跑：已完成且配置签名一致的 fold 会直接读取；配置不一致时会报错，不会静默混合结果。

## 两张 GPU 并行运行

先在普通终端生成一次 cohort manifest：

```bash
cd /media/UG1/lys/dipy/BN_JHU296
conda activate mte
python code/clean_code/00_build_clean49_manifest.py
```

GPU 0 运行耗时最长的 iTransformer：

```bash
screen -dmS table1_gpu0 bash -lc 'source ~/.bashrc && conda activate mte && cd /media/UG1/lys/dipy/BN_JHU296 && GPU_ID=0 MODELS="itransformer" BUILD_MANIFEST=0 SUMMARIZE=0 BS=1 bash code/clean_code/run_table1_clean49.sh'
```

GPU 1 顺序运行 TimesNet、MoLE 和 DLinear：

```bash
screen -dmS table1_gpu1 bash -lc 'source ~/.bashrc && conda activate mte && cd /media/UG1/lys/dipy/BN_JHU296 && GPU_ID=1 MODELS="timesnet mole dlinear" BUILD_MANIFEST=0 SUMMARIZE=0 BS=1 bash code/clean_code/run_table1_clean49.sh'
```

两个 screen 都结束后只运行一次汇总：

```bash
cd /media/UG1/lys/dipy/BN_JHU296
conda activate mte
python code/clean_code/02_summarize_table1_s1.py
```

## WMTI gray-matter exclusion 重训敏感性分析

该补充分析不改动 canonical Table 1 训练代码或结果。它从输入和目标中同时删除 4 个 WMTI metrics × 246 个 Brainnetome gray-matter ROI，共 984 个变量，保留每个 TE 的 10,264 个变量，并严格复用 clean-49 的 49 折 train/validation/test subjects 和 fold seed。

新增文件：

- `14_train_wmti_gm_exclusion_retrain.py`：变量审计、canonical fold 核对、四模型逐折重训和断点续跑；
- `15_summarize_wmti_gm_exclusion_retrain.py`：验证 196 个 fold 文件并生成汇总、原实验变化和 Wilcoxon/BH-FDR；
- `run_wmti_gm_exclusion_retrain.sh`：`mte` 环境下的服务器入口和日志记录。

所有结果独立写入 `outputs/wmti_gm_exclusion_retrain/`，不会覆盖 `outputs/table1_clean49/`。运行前可先执行不使用 GPU 的审计：

```bash
cd /media/UG1/lys/dipy/BN_JHU296
conda activate mte
AUDIT_ONLY=1 bash code/clean_code/run_wmti_gm_exclusion_retrain.sh
```

两张 GPU 推荐分配为 MoLE 独占一张卡，其余三个模型使用另一张卡：

```bash
conda activate mte
screen -dmS wmti_gm_gpu0 bash -c 'cd /media/UG1/lys/dipy/BN_JHU296 && GPU_ID=0 MODELS="mole" SUMMARIZE=0 bash code/clean_code/run_wmti_gm_exclusion_retrain.sh'
screen -dmS wmti_gm_gpu1 bash -c 'cd /media/UG1/lys/dipy/BN_JHU296 && GPU_ID=1 MODELS="itransformer timesnet dlinear" SUMMARIZE=0 bash code/clean_code/run_wmti_gm_exclusion_retrain.sh'
```

两个 screen 全部结束后运行：

```bash
cd /media/UG1/lys/dipy/BN_JHU296
conda activate mte
python code/clean_code/15_summarize_wmti_gm_exclusion_retrain.py
```

最终 `wmti_gm_exclusion_retrain_audit.txt` 只有在四模型 196 个 fold 均存在、每折张量均为 `(1, 2, 10264)`，且所有受试者划分和 seed 与主实验一致时才会生成。

## DLinear/MoLE moving-average kernel 正式敏感性分析

该分析只改变 DLinear 和 MoLE-DLinear 的 moving-average kernel：新增 `k=3` 和 `k=5` 的完整 clean-49 LOSO；reference `k=25` 直接读取 `outputs/table1_clean49/`，不重新训练。第三方模型文件保持不变。

新增文件：

- `16_train_kernel_sensitivity_clean49.py`：canonical fold 预审计、CPU forward sanity check、逐折训练、checkpoint 和断点续跑；
- `17_summarize_kernel_sensitivity_clean49.py`：合并 k=3/5/25、Wilcoxon/BH-FDR、四模型排名和最终 audit/report；
- `run_kernel_sensitivity_clean49_2gpu.sh`：GPU0 运行 MoLE/DLinear k=3，GPU1 运行 MoLE/DLinear k=5，全部成功后自动汇总。

运行时必须先激活 `mte`，不使用 `/opt/fsl/bin/python`：

```bash
cd /media/UG1/lys/dipy/BN_JHU296
conda activate mte
chmod +x code/clean_code/run_kernel_sensitivity_clean49_2gpu.sh
screen -dmS kernel_sensitivity_clean49 bash -c 'cd /media/UG1/lys/dipy/BN_JHU296 && GPU0=0 GPU1=1 CHECKPOINT_DTYPE=fp16 bash code/clean_code/run_kernel_sensitivity_clean49_2gpu.sh'
```

脚本会先检查 CUDA、`nvidia-smi`、主机内存和活动 GPU compute process，再用完整 `(1,5,11248)` CPU 输入检查 DLinear/MoLE 的 k=3/5 输出。发现活动 GPU 任务时默认终止；只有人工确认可共存后才可设置 `ALLOW_BUSY_GPU=1`。

MoLE 每折约 506,205,040 个参数。默认在训练和预测完成后保存完整 FP16 state dict，预计 98 个 MoLE checkpoint 共约 99 GB；`CHECKPOINT_DTYPE=fp32` 约需 198 GB。序列化精度不参与训练或结果计算，预测仍由最佳 FP32 训练状态生成。

正式结果写入 `outputs/kernel_sensitivity_clean49/`，不会覆盖 `outputs/table1_clean49/`。最终汇总器只有在四个新增配置各 49 折、全部 checkpoint、prediction shape、subject split 和 fold seed 均通过时才生成 `kernel_sensitivity_audit.txt` 与 `kernel_sensitivity_report.md`。
