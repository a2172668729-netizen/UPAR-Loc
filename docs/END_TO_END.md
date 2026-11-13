# UPAR-Loc 复现指南

本指南描述目前仓库中唯一推荐的完整流程：从 **University-1652 原图 + DINOv3 ViT-B/16 官方权重**独立完成训练和评测。无需预先生成特征、OP 矩阵或重排序权重，也无需额外克隆 VFM-Loc。

## 准备

推荐 Linux / WSL2、Python 3.10，GPU 使用支持 CUDA 的 PyTorch。已在 Windows 11 + WSL2 Ubuntu 22.04、RTX 5060、PyTorch 2.7.1+cu128 环境下完成端到端复现。

数据集目录：

```text
University-1652/
  train/
    drone/<地点ID>/*.jpeg
    satellite/<地点ID>/*.jpg
  test/
    query_drone/<地点ID>/*.jpeg
    gallery_satellite/<地点ID>/*.jpg
```

Train 包含约 37,854 张无人机图和 701 张卫星图；Test 包含 37,855 张无人机查询和 951 张卫星图库图像。请自行下载 Meta DINOv3 ViT-B/16 权重 `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`。

## 安装与运行

```bash
git clone https://github.com/a2172668729-netizen/UPAR-Loc.git
cd UPAR-Loc

python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements-reproduction.txt

.venv/bin/python reproduce.py all \
  --dataset /absolute/path/University-1652 \
  --weights /absolute/path/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --output outputs/reproduction \
  --batch-size 4 --workers 2 --device cuda
```

`--batch-size` 控制特征提取的批量大小；重排序训练的批量大小由 `configs/from_scratch.json` 控制。若显存允许，可提高提取批量。

## 流程与数据隔离

1. **提取源域特征：** 冻结 DINOv3 ViT-B/16；输入尺寸 224×224。分别提取 FP32 的五区域 GeM 特征（每个 768 维）和 FP16 的 196×768 Patch 特征。
2. **无标签全局对齐：** 无人机和卫星分别进行 PCA-512；白化指数为 0.55 / 0.10；CSLS k=100 产生伪配对，单张卫星图最多接收 55 个伪配对，更新 OP 4 轮，再执行 600 步 Cayley 排序优化。源域、目标域各自拟合。
3. **源域监督训练：** 将官方 Train 的 701 个地点按固定随机种子拆分为 561 个 Train-Fit、70 个 Val-Selection 和 70 个 Val-Confirmation。仅在 Train-Fit 的真实配对上拟合 PCA-256/OP 以挖掘困难负样本；每条查询每个 epoch 抽取 1 正 + 3 困难负例 + 2 随机负例。使用双向 2 层、4 头 Patch Cross-Attention，AdamW、BCEWithLogitsLoss、CosineAnnealingLR，batch 16，每个种子训练 12 个 epoch，随机种子为 11、22、33。
4. **选取模型：** 每个 epoch 只在源域 Val-Selection 检验融合权重 λ∈{0, 0.2, ..., 1.2}，按 Recall@1、AP 选定 seed / epoch / λ；冻结后在 Val-Confirmation 上做一次确认。
5. **目标域检索：** 冻结模型选择后提取 Test 图像特征，在 Test **无标签图像集合**上拟合 PCA/OP；全局检索得到 Top-20，使用训练好的 Patch 网络重排序。只有最终排名生成后才读取 Test 地点 ID 计算指标。

最终目标域不使用真实地点配对关系做训练、模型选择或 OP 拟合。由于目标对齐使用整个目标查询集和图库，属于 **transductive**；由于重排序在官方 Train 标签上学习，属于 **source-supervised**。

## 分阶段运行

需要单独检查中间结果时，可沿相同 `--dataset`、`--output` 和 `--config` 参数运行：

```bash
.venv/bin/python reproduce.py extract_source --dataset ... --weights ...
.venv/bin/python reproduce.py align_source --dataset ...
.venv/bin/python reproduce.py train --dataset ...
.venv/bin/python reproduce.py extract_target --dataset ... --weights ...
.venv/bin/python reproduce.py align_target --dataset ...
.venv/bin/python reproduce.py evaluate --dataset ...
```

输出主要包含 `features/`、`source_descriptors.pt`、`target_descriptors.pt`、`reranker/best.pt`、`reranker/train_report.json` 与 `final_metrics.json`。默认不覆盖已有特征；若明确需要重新提取，可加 `--overwrite`。单独运行训练或评测阶段前，应先生成对应上游产物。

## 已验证结果

2026-11-13 在全新克隆、独立 Python 虚拟环境及原始图像与权重上完成端到端运行，并独立重算指标：

| University-1652 D2S | Recall@1 | AP |
| --- | ---: | ---: |
| 无标签全局检索 | 81.6775% | 88.7374% |
| 完整 UPAR-Loc | **89.3224%** | **93.3904%** |

源域选择得到 **seed=22、epoch=7、λ=1.2**。结果包含 37,855 条查询和完整 951 张图库；独立评分与正式评分一致。实际复现不保证每台机器的浮点结果完全相同。

这里的 AP 是与 VFM-Loc 一致的评分定义。University-1652 Test 在研究过程中曾被多次查看，这些数字不构成全新盲测；也尚未验证跨数据集泛化。

## 来源

- [VFM-Loc 原论文](https://doi.org/10.1109/TGRS.2026.3748024)及[官方源码](https://github.com/DingLei14/VFM-Loc)
- [University-1652](https://github.com/layumi/University1652-Baseline)
- [Meta DINOv3](https://github.com/facebookresearch/dinov3)；仓库中附带推理所需的源码及 [Meta 许可证](../third_party/dinov3/LICENSE.md)
