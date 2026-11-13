# UPAR-Loc

**Unlabeled Pairing and Attention-based Reranking for Cross-View Geo-Localization**

UPAR-Loc 用无人机图像检索对应的卫星地点，重点解决两个问题：如何在**没有目标地点真实配对标签**的情况下对齐不同视角的特征，以及如何利用局部结构信息纠正全局检索中的排序错误。

本项目基于 [VFM-Loc](https://github.com/DingLei14/VFM-Loc) 和冻结的 DINOv3 ViT-B/16 特征，提供从原始图像、预训练权重到训练与评测的完整实现。

## 方法

**1. 无标签跨视角对齐（Unlabeled Pairing）**

无人机与卫星特征分别进行 PCA-512 和不同强度的白化，然后使用 CSLS 挖掘伪配对，通过迭代正交 Procrustes（OP）与无标签排序优化建立跨视角映射。目标域对齐只使用**无标签的查询图像与卫星图库**，不需要真实地点对应关系。由于拟合时使用目标域整批未标注图像，这是 *transductive* 设置。

**2. 基于注意力的局部重排序（Attention-based Reranking）**

先通过全局相似度检索 Top-20 卫星候选，再对无人机与卫星的 DINOv3 Patch 特征进行**双向 Cross-Attention** 匹配，融合局部与全局分数，调整候选顺序。

重排序网络在 University-1652 **官方 Train** 的地点 ID 上监督训练：每条查询使用 1 个正样本、3 个困难负样本和 2 个随机负样本。困难负样本挖掘使用仅在源域训练地点拟合的有标签 PCA/OP；这一步**不使用目标 Test 标签**。整个方法应表述为 **source-supervised, transductive target-label-free**，并非完全无监督或全流程 training-free。

## 实验结果

University-1652，**Drone → Satellite（D2S）**，官方 Test：37,855 张查询图像、951 张卫星图库图像。

| 方法 | Recall@1 ↑ | AP ↑ |
| --- | ---: | ---: |
| VFM-Loc 论文报告（DINOv3） | 74.42% | 83.18% |
| UPAR-Loc：无标签全局对齐 | 81.68% | 88.74% |
| **UPAR-Loc：对齐 + Patch 重排序** | **89.32%** | **93.39%** |

以上 UPAR-Loc 数字来自 **2026-11-13 的完整独立复现**：从原始图像提取特征、拟合源域和目标域对齐，并以 3 个随机种子重新训练重排序网络，最终独立重算评测指标。重排序相比本次全局检索基线提升 **7.65 个百分点 Recall@1、4.65 个百分点 AP**。

AP 使用与 VFM-Loc 一致的评分定义。原论文公开评测实现使用目标评测地点标签拟合 OP，而 UPAR-Loc 不使用目标域真实配对标签，但增加了源域监督训练。因此表格是**同一基准测试上的结果对比**，不表示两者使用完全相同的监督条件。此前研究曾多次观察此 Test 结果，当前成绩不应称作独立盲测；跨数据集泛化尚待评估。

## 从零复现

只需自行准备：

- [University-1652](https://github.com/layumi/University1652-Baseline) 原始数据集；
- Meta DINOv3 ViT-B/16 预训练权重（文件名：`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`）。

仓库包含所需 DINOv3 推理实现，无须克隆 VFM-Loc 或下载我们的实验缓存和训练权重。

```bash
git clone https://github.com/a2172668729-netizen/UPAR-Loc.git
cd UPAR-Loc

python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements-reproduction.txt

.venv/bin/python reproduce.py all \
  --dataset /path/to/University-1652 \
  --weights /path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --output outputs/reproduction \
  --batch-size 4 --workers 2 --device cuda
```

程序依次完成：**源域特征与对齐 → 重排序训练与源域选型 → 目标域无标签对齐 → 完整 D2S 评测**。最终指标保存在 `outputs/reproduction/final_metrics.json`。硬件、随机性与计算设置可能使实际数值出现差异。

完整的数据目录、方法参数、训练协议与分阶段命令见 [复现说明](docs/END_TO_END.md)。

## 致谢与引用

UPAR-Loc 建立在 [VFM-Loc](https://doi.org/10.1109/TGRS.2026.3748024)、[DINOv3](https://github.com/facebookresearch/dinov3) 和 [University-1652](https://github.com/layumi/University1652-Baseline) 之上。第三方 DINOv3 代码及许可证见 [third_party/dinov3](third_party/dinov3)。
