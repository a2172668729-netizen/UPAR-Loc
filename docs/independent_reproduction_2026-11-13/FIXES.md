# 独立复现中的代码修订

上游提交 `2efcfed6e53de28e5f6e5420dea214e8769f3eb8`。仅修改本目录下的 Git 克隆。完整前后差异见 [reproduction.patch](audit/reproduction.patch)。

| 文件 | 原因及修改 | 对算法与指标的影响 |
| --- | --- | --- |
| `reproduce.py` | B2 训练打开了进程级 TF32，原 `all` 后续阶段沿用该设置。现在各阶段统一以关闭 matmul 和 cuDNN TF32 开始，B2 训练内部仍启用原有 matmul TF32。 | 固定特征提取、无标签对齐和目标推理的 FP32 数值设置。原 `all` 与本次结果可能存在舍入差异。训练超参、损失和分阶段协议一致。 |
| `src/raw_pipeline.py` | 增加已完成图像计数和日志，写入元数据前核对总数。 | 无计算变化。 |
| `src/reviewer_training.py` | 划分断言覆盖全部三组交集，校验 Train-Fit 候选行号与地点，保存候选池，并记录挖掘与训练进度。 | 保留原划分、随机种子、采样和训练逻辑。 |
| `src/research_pipeline.py` | 保留 PCA 投影、Cayley 前 OP、伪配对索引、损失轨迹，以及读真值前保存的完整排名，并记录明确的历史暴露评测标记。 | 增加验收证据。排序和评分逻辑一致。 |

安装期间 NVIDIA 索引下载停滞。重试使用本轮从 PyTorch 下载的原版本 CUDA wheels，并从 PyPI 解析其余依赖。仓库依赖文件保留原样。
