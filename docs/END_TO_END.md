# 从零独立复现：University-1652 + DINOv3 权重

## 审稿人需要准备什么？

只需要以下两个**外部数据资产**：

1. [University-1652](https://github.com/layumi/University1652-Baseline) 原始图片，具有 Train/Test 官方目录结构。
2. Meta 官方 **DINOv3 ViT-B/16 LVD1689M** checkpoint：`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`。

本仓库已经提供：DINOv3 ViT-B/16 **推理架构源码**、图像预处理和五区域 GeM 特征提取、无标签 PCA/OP 与 Cayley 排序优化、源域监督的 PCA-256/OP 困难负例挖掘、B2 双向 Cross-Attention 训练、融合与评测。因此**不需要克隆 VFM-Loc 或下载任何旧实验的特征、OP 旋转矩阵、B2 权重**。

DINOv3 代码来自 Meta 官方仓库的固定 [2025-08-14 提交](https://github.com/facebookresearch/dinov3/commit/d51437b5e494881a204e71ee8e8a17e471fdcada)，位于 [third_party/dinov3](../third_party/dinov3)，许可见 [DINOv3 License](../third_party/dinov3/LICENSE.md)。VFM-Loc 的相关算法在本仓库中独立实现，并保留原论文引用；没有原样复制 VFM-Loc 无明确许可证的实现。

## 数据集应当长什么样？

```text
University-1652/
  train/
    drone/<地点ID>/*.jpeg
    satellite/<地点ID>/*.jpg
  test/
    query_drone/<地点ID>/*.jpeg
    gallery_satellite/<地点ID>/*.jpg
```

查询/图库分别排序并记录 `image_paths.json`，避免文件行序混淆。Train 数据约 37,854 张无人机图和 701 张卫星图；Test 为 37,855 条查询、951 张图库。若目录数量与预期不符，检查数据集是否完整解压。

## 完整安装与运行

在 Linux/WSL2、Python 3.10 环境中操作；GPU 推荐 CUDA 12.8 / PyTorch 2.7.1。使用**新的空目录**，不要把旧研究缓存放入输出目录：

```bash
git clone https://github.com/a2172668729-netizen/cvgl-round7-j2.git
cd cvgl-round7-j2
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements-reproduction.txt

.venv/bin/python reproduce.py all \
  --dataset /absolute/path/University-1652 \
  --weights /absolute/path/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --output outputs/reproduction
```

不需要 `--upstream`，不需要联网下载 DINOv3 代码。因本仓库目前为 Private，独立用户需提前获得仓库的 GitHub 访问权限。使用公共数据和公开权重时请遵守各自许可。

RTX 5060 8GB 可能需要将特征提取批量降为 `--batch-size 4`。此参数只影响 DINOv3 提取，B2 训练 batch 在 [configs/from_scratch.json](../configs/from_scratch.json) 中管理；源域/目标域 PCA、OP 和 B2 模型参数不随之改变。

## 从头复现的六个阶段

```text
University-1652 Train 原始图像
      ↓  内置 DINOv3 FP32 R-MAC 区域特征 + FP16 Patch
源域全局无标签 PCA-512 / 伪配对 OP / Cayley 优化
      ↓
源域 561 Train-Fit + 70 Val-Selection + 70 Val-Confirmation
      ↓  原始源域 Train-Fit ID 监督 PCA-256/OP 挖难负例
重新训练 B2、源域选定 seed / epoch / λ、Confirmation 仅验一次
      ↓  冻结模型选择
University-1652 Test 原始图像
      ↓  内置 DINOv3 提取 & 目标域无标签 PCA/OP
全局 Top-20 + 双向 Patch Cross-Attention + 冻结 λ
      ↓  所有排名形成后读取 Test 标签
R@1、R@5、R@10、R@20、R@50、mAP
```

### 特征提取

严格使用 224×224、OpenCV `INTER_LINEAR_EXACT` 缩放及 ImageNet mean/std。区域和 Patch 分别前向：**FP32** 区域 GeM/R-MAC 五区域，每个 768 维；**FP16 autocast** 的 196×768 Patch token。全部通过冻结 DINOv3 ViT-B/16 得到，不重新训练 DINOv3。

### 无标签对齐

无人机与卫星各自 PCA-512，Whitening 指数分别为 0.55 / 0.10；CSLS k=100 产生伪配对，图库上限 cap=55，四轮 OP 更新，再进行 600 步 Cayley 正交排序优化（align=0）。两域分开拟合，只使用相应域的图像特征，不调用标签或验证器。目标域属于 **transductive target-label-free** 设置。

### B2 重排序训练

地点按原始实验两阶段划分：先用 `RandomState(20261110)` 从 701 个 Train 地点选出 561 Train-Fit / 140 Val；再使用 `RandomState(20261111)` 分割为 70 Val-Selection / 70 Val-Confirmation。

源域监督阶段仅在 **561 个 Train-Fit 地点**拟合 PCA-256 与真实 ID 的 OP，获得困难负例排序候选；每条查询每轮抽取 **1 正 + 3 困难负例 + 2 随机负例**，并以冻结 Patch 的视觉哈希排除重复和冲突 ID。B2 使用双向、2 层、4 头 Cross-Attention、AdamW、BCEWithLogitsLoss、CosineAnnealingLR、12 epoch、batch=16，按原 Round 7 seeds 11/22/33 完整训练。

每个 epoch 在 **Val-Selection** 上从 λ={0, 0.2, ..., 1.2} 选择融合系数，并按 R@1、mAP 冻结最佳 seed/epoch/λ；此后才对 **Val-Confirmation** 做一次报告。不得利用 Test 指标反向选模型。

### 目标域评测

完成训练并冻结选型后，才提取官方 Test 图片的特征并在其**无标签分布**上拟合目标对齐。全局检索产生 Top-20，用已训练 B2 计算匹配分数，依据源域冻结的 λ 重排候选身份，显式处理全局分数并列排名。**所有排名形成之后**再解析目标地点 ID 计算指标。

## 分阶段运行

如果需要检查某一步，只用相同的 `--dataset`、`--output` 和 `--config`。只有提取阶段需 `--weights`，不再有 `--upstream`：

```bash
.venv/bin/python reproduce.py extract_source --dataset ... --weights ...
.venv/bin/python reproduce.py align_source --dataset ...
.venv/bin/python reproduce.py train --dataset ...
.venv/bin/python reproduce.py extract_target --dataset ... --weights ...
.venv/bin/python reproduce.py align_target --dataset ...
.venv/bin/python reproduce.py evaluate --dataset ...
```

输出目录预期有 `features/{source_query,source_gallery,target_query,target_gallery}`、`source_descriptors.pt`、`target_descriptors.pt`、`reranker/best.pt`、`reranker/train_report.json`、`final_metrics.json`。输出默认拒绝覆盖已有特征；明确重提取请加 `--overwrite`。

## 研究与验收边界

**截至 2026-11-13，这份自包含代码尚未在真实数据上重新执行端到端测试**；本次仅执行 GitHub 源码编辑和静态核对，没有运行训练、Python、CLI、测试或 CI。不能把此前冻结特征/权重的 J2 90.2681% R@1 / 94.0511% mAP 当作新流水线已经重现的成绩。

原 Round 7 的历史 J2 使用 seed=33、epoch=7、λ=1.0；本仓库从零训练会重新按源域 Val-Selection 选择，未必选中相同的配置。历史 Test 曾多次暴露，因此即使再次得到同一数字，也仍应标为 **Historically Exposed Test Compatibility Evaluation — Not a Blind Test**。此外，本系统是 **source-supervised, target-label-free**，并非全部 training-free。

复现提交和协议对照可查阅 [此前审稿人静态审计](REVIEWER_STATIC_AUDIT_2026-11-13.md)，并在运行后补充环境版本、权重 SHA-256、完整日志、模型选型信息及最新生成的指标文件。
