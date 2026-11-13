# 从头复现：University-1652 + DINOv3

## 目标与输入

唯一需要用户自行取得的**数据与权重**：

1. University-1652 原始数据集（按官方布局解压）。
2. DINOv3 ViT-B/16 官方预训练权重：`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`。

还需要仓库源码依赖 [VFM-Loc](https://github.com/DingLei14/VFM-Loc) 的公开实现（Git 提交 `162d885d25f0215025d7fe9f0a1c70f64e7ba119`），**无需下载旧研究团队的中间特征、OP 矩阵或重排序权重**。不包含第三方数据、预训练权重，需遵守上游各自许可。

目录布局：

```text
University-1652/
  train/
    drone/0001/*.jpg
    satellite/0001/*.jpg
  test/
    query_drone/xxxx/*.jpg
    gallery_satellite/xxxx/*.jpg
```

输入目录采用字典序排序，保存图片顺序 JSON，防止后来特征与标签错位。训练池来自官方 Train：701 个地点、约 37,854 个 drone 查询；目标 Test 包含 37,855 个查询、951 张卫星图库图片。实际数量以解析和输出校验为准。

## 一次性完整运行

```bash
git clone https://github.com/DingLei14/VFM-Loc.git upstream/VFM-Loc
git -C upstream/VFM-Loc checkout 162d885d25f0215025d7fe9f0a1c70f64e7ba119
python3.10 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-reproduction.txt
.venv/bin/python reproduce.py all \
  --dataset /absolute/path/University-1652 \
  --weights /absolute/path/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --upstream upstream/VFM-Loc \
  --output outputs/reproduction
```

默认用 CUDA，如果 GPU 显存不足，可减少特征提取批大小 `--batch-size 4`。训练 batch 由 `configs/from_scratch.json` 中的 `training.batch_size` 单独控制。允许 CPU 执行结构自检，但实际 DINOv3 和完整 600 步优化/重排序训练建议使用 GPU。运行已有输出目录时特征提取默认拒绝覆盖；确认后加 `--overwrite` 重新提取。

也可以按照独立审稿人的数据隔离要求分段运行。所有命令都要沿用相同的 `--dataset`、`--output` 和 `--config`；提取阶段还需传 `--weights`、`--upstream`，训练阶段同样要传 `--upstream`：

```bash
.venv/bin/python reproduce.py extract_source --dataset ... --weights ... --upstream upstream/VFM-Loc
.venv/bin/python reproduce.py align_source --dataset ...
.venv/bin/python reproduce.py train --dataset ... --upstream upstream/VFM-Loc
.venv/bin/python reproduce.py extract_target --dataset ... --weights ... --upstream upstream/VFM-Loc
.venv/bin/python reproduce.py align_target --dataset ...
.venv/bin/python reproduce.py evaluate --dataset ...
```

推荐使用 `all`，它严格按上述顺序执行：**源域训练及模型选型完成后才读取 Test 图像**。`align` 不读取任何地点标签；`train` 仅在官方 Train 上使用地点 ID；`evaluate` 在所有排名形成后才加载目标 Test 的地点 ID 计算指标。

## 四个计算阶段

**阶段 1：DINOv3 特征。** 复用指定版本 VFM-Loc 的 `DinoV3Embedder` 与官方 Albumentations 预处理规范：224×224、ImageNet 标准化、双视角相同预处理。按 Round 7 的实际实现采用**两次不同精度的前向**：在 FP32 下提取 GeM/R-MAC 五区域描述符，再用 FP16 autocast 提取 `196×768` 个 Patch token。区域特征保存为 float32，Patch 保存为 float16。源域和 Test 各产生查询、图库两组缓存，使用磁盘映射避免把全部 Patch 一次放入内存。

**阶段 2：无标签全局对齐。** 将五区域特征输入 `src/op_core.py`：每个视角独立 PCA-512，Whitening 指数 0.55/0.10；CSLS 生成伪配对，逐图库候选限额 55，初始拟合后再进行 4 次 OP 更新，随后进行 600 步 Cayley 排序优化（align=0）。源域与目标域分别**独立拟合**；目标域 Test 的全部无标签特征允许参与拟合，因此这是 *transductive* 适配。此阶段不访问任何标签或测试指标。输出单独的全局描述符，并校验来源和输入图像顺序。

**阶段 3：源域重排序训练。** 使用 Train 的地点 ID，先用 `RandomState(20261110)` 将 701 个地点分为 561 Train-Fit / 140 Val，再用 `RandomState(20261111)` 分出 70 个 Val-Selection / 70 个 Val-Confirmation 地点。所有 701 张源域卫星图组成检索图库。先用**仅在 561 个 Train-Fit 地点上拟合的官方 PCA-256/监督 OP**挖掘困难负例，再按每个 epoch 重新采样 **1 正 + 3 困难负例 + 2 随机负例**。负例 ID 不同、FP16 Patch 视觉组不同，且不与正例相同视觉组或地点。训练负例只来自 Train-Fit 卫星图库。该训练使用源域身份标签，不碰 Test。使用双向 Patch Cross-Attention，在源域分别训练 seeds 11/22/33，batch=16、12 epoch、AdamW（lr=1e-4，weight_decay=.01）、CosineAnnealingLR、BCEWithLogitsLoss 和梯度裁剪 5。每个 epoch 仅在 Val-Selection 上搜索 λ∈{0,.2,.4,.6,.8,1,1.2}，同时选择 seed/epoch/λ；最后对固定的 Val-Confirmation 做一次结果报告。这一阶段不接触官方 Test。

**阶段 4：目标域一次性评测。** 载入源域选出的 checkpoint，在目标域无标签全局检索 Top-20 候选上运行双向 Patch Cross-Attention，按照源域 Val-Selection 冻结的融合系数 λ 重排候选身份。对于全局分数相同的候选使用**显式完整图库排名**，不通过添加 epsilon 假装分数有差异。全局排名与融合排名都固定后，才解析 Test 文件夹中的地点标签，计算全部查询上的 R@1 / R@5 / R@10 / R@20 / R@50 / mAP。

输出文件（均写在 `--output` 中）：

```text
features/source_query/{regions.npy,patches.npy,image_paths.json,metadata.json}
features/source_gallery/{...}
features/target_query/{...}
features/target_gallery/{...}
source_descriptors.pt  source_descriptors.json
target_descriptors.pt  target_descriptors.json
reranker/source_split.json
reranker/best.pt
reranker/train_report.json
final_metrics.json
```

> 本次仅通过 Windows/WSL 文件读取核对了原实验实现，并据此修改开源代码；**没有启动任何训练、测试或命令**。修订细节和待验项目见 [审稿人静态审计记录](REVIEWER_STATIC_AUDIT_2026-11-13.md)。

## 复现协议边界

- **不能承诺历史 90.2681% / 94.0511%。** 旧冻结资产与本新增训练链并非已经验证的同一份训练过程；此前实验记录不包含足以证明每一处负例采样、优化器状态、训练停点与 Tensor 核算完全一致的完整来源。本版本默认参数根据源域协议制定，必须把实际跑出的成绩单独记录。
- 原先的 90.27% 来自历史已多次观察的 Test，且曾经历训练过程和选型审计，不能写成新盲测提升。目前配置按 Round 7 原始冻结方案设置 cap=55，源域对 checkpoint/seed/λ 重新选型；历史 J2 的 seed33 / epoch7 / λ=1.0 是旧结果，**不能强制写入这次输出**。历史参数选型曾暴露 Test，不能声称此前的选择全过程严格盲测。
- 单元测试只验证实现路径、排序语义及负例约束；**不验证第三方权重是否匹配、不证明模型训练必然收敛，也不证明完整指标。**
- 本项目保留旧冻结资产入口 `evaluate_retrieval.py` 以追溯历史指标。两条运行路径的资产、配置和输出必须分开，不得复用历史缓存冒充从头重训。
- 严格意义上的论文级精确数值复现，还需在独立机器上完整检查官方 DINOv3 前向数值、重排序训练源码的每项超参数和原始数据读取；至少应报告代码提交、原始目录结构、权重 SHA-256、随机种子、依赖版本、CUDA 与硬件信息。
