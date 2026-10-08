# 最终开源整理记录

日期：2026-10-08。只修改独立开源副本；原始研究代码、影像结果和论文文件不变。

## 公开边界

从已完成拟合和ROI聚合的 multi-TE 指标表开始，提供正式四模型预测和论文数值扩展分析。
不承诺由原始dMRI一键生成论文输入。所有命令与数据契约见README，重命名对应见RENAMING.csv。

## 删除或不公开

| 内容 | 原因 |
|---|---|
| preprocessing、MRI拟合/配准、直接mask/重采样候选脚本 | 位于ROI表输入边界上游；有独立研究副本，不混入公开预测入口 |
| 第三方整仓的训练脚本、数据加载、无关模型及示例 | 只保留三模型实际导入闭包9个文件和许可证 |
| MoLE第三方源码 | 许可证不明确，提供来源和准确文件SHA256；自有适配器保留 |
| Ridge、PatchTST、SegRNN、SparseTSF、MTS-Mixer、modified iTransformer等分支 | 非最终四模型报告对象；删除入口和残余模型构造分支 |
| 编号shell封装和服务器环境导出 | 硬编码机器路径且重复；替换为统一CLI和pyproject依赖 |
| 从历史结果反推cohort的脚本 | 公开复现要求显式提供获授权的匿名manifest，不依赖旧结果文件 |
| 私有名单、患者表、原始预测、完整日志及内部写作说明 | 不需要公开，避免隐私与未发布内容扩散 |
| 旧绘图布局与论文文件 | 公开数值汇总，避免硬编码版面、私有病种文件和额外图像依赖 |
| WMTI仅重算旧评估分支及非最终候选基线 | 保留正式排除变量重训和kernel敏感性，去除重复历史路径 |
| 多份过时环境和哈希清单 | 公开依赖仅一份；重命名后的包装代码哈希不再冒充历史原件 |

## 实现变化

26个原有模块按功能迁移、更新动态import和文件加载路径。神经模型训练函数、训练折标准化
和正式49/42拆分算法保留；为小样本非法拆分增加明确报错。第三方核心源码保持字节一致。
移除旧模型构造，MoLE从候选搜索改成准确加载正式MoLE_DLinear.py。
路径由工作目录与显式配置决定；正式病灶预测直接复用同一输入审计和训练器，不再捆绑上游
重采样CSV转换。ROI/疾病数值汇总替换依赖论文图像布局的旧绘图程序。

## 49人FA atlas结论

**作者确认：正式clean-49的49人均使用FWDTI FA配准。**确认来自本次2026-10-08对话。
按照作者要求，不连接服务器、不重新执行配准。本轮新增配准执行数为0；独立逐例影像验证
未执行。没有生成或伪造“49/49重跑通过”报告，也没有覆盖原始图谱或结果。

## 仍无法确认或未完成的范围

- 两份内部QC报告的生成程序，以及原始dMRI完整预处理的历史执行记录。
- MoLE精确历史upstream commit与明确可再分发许可证。
- 公共现代CPU环境与历史PyTorch1.11.0+cu115 GPU环境的逐位一致性。
- 全部论文扩展在真实数据上的重新训练/复现；本轮验证范围以VALIDATION.md为准。
- 原始私有cohort重命名后若改变排序，可能改变随机折映射；要复现原fold，应保持受试者顺序
  或对照原manifest审计，不应声称任意匿名化命名必然生成相同折。

所有真实数据路径和队列信息由使用者提供。代码公开不等于公开患者数据许可。

## 最终建议公开文件树

共 52 个文件（前一公开版647个，减少 595 个）。下列即开源副本的完整建议上传集合；构建缓存、虚拟环境和本地测试输出不在其中。

```text
TCMTransformer/
  .gitignore
  LICENSE
  README.md
  THIRD_PARTY_NOTICES.md
  config.json
  docs/RELEASE_AUDIT.md
  docs/RENAMING.csv
  docs/VALIDATION.md
  pyproject.toml
  teprediction/__init__.py
  teprediction/__main__.py
  teprediction/common.py
  teprediction/evaluate.py
  teprediction/export_predictions.py
  teprediction/external.py
  teprediction/external_data.py
  teprediction/input_te_count.py
  teprediction/interpolate.py
  teprediction/kernel_sensitivity.py
  teprediction/leave_one_te.py
  teprediction/leave_one_te_mole.py
  teprediction/linear_reference.py
  teprediction/matched_pretraining.py
  teprediction/metric_errors.py
  teprediction/models.py
  teprediction/mole.py
  teprediction/native_units.py
  teprediction/prepare.py
  teprediction/prepare_incomplete.py
  teprediction/pretrain_baselines.py
  teprediction/pretrain_itransformer.py
  teprediction/pretrain_mole.py
  teprediction/summarize.py
  teprediction/summarize_kernel.py
  teprediction/summarize_wmti.py
  teprediction/synthetic.py
  teprediction/te_pairs.py
  teprediction/train.py
  teprediction/vendor/LTSF-Linear/LICENSE
  teprediction/vendor/LTSF-Linear/models/DLinear.py
  teprediction/vendor/Time-Series-Library/LICENSE
  teprediction/vendor/Time-Series-Library/layers/Conv_Blocks.py
  teprediction/vendor/Time-Series-Library/layers/Embed.py
  teprediction/vendor/Time-Series-Library/models/TimesNet.py
  teprediction/vendor/iTransformer/LICENSE
  teprediction/vendor/iTransformer/layers/Embed.py
  teprediction/vendor/iTransformer/layers/SelfAttention_Family.py
  teprediction/vendor/iTransformer/layers/Transformer_EncDec.py
  teprediction/vendor/iTransformer/model/iTransformer.py
  teprediction/vendor/iTransformer/utils/masking.py
  teprediction/wmti_sensitivity.py
  tests/smoke.py
```

测试结果见 [VALIDATION.md](VALIDATION.md)。自有代码采用作者选择的MIT许可证；第三方保留各自许可证。
