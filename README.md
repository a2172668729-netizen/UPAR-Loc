# UPAR-Loc

**Unlabeled Pairing and Attention-based Reranking for Cross-View Geo-Localization**

UPAR-Loc 基于 [VFM-Loc](https://github.com/DingLei14/VFM-Loc) 和 DINOv3，研究无人机到卫星图像的跨视角地理定位。我们改进了两个环节：**不依赖目标地点真实配对的特征对齐**，以及**利用 Patch 对应关系的候选重排序**。

## 方法

**Unlabeled Alignment.** 对无人机和卫星特征分别进行 PCA 与不同强度的 whitening，借助 CSLS 挖掘伪配对，再通过迭代 Orthogonal Procrustes（OP）和排序优化学习跨视角映射。

**Attention-based Reranking.** 全局检索得到 Top-20 卫星候选后，以双向 Cross-Attention 匹配查询与候选的 DINOv3 Patch 特征，并融合局部和全局相似度重新排序。

重排序器使用 University-1652 官方 Train 的配对标签训练；目标 Test 仅用于无标签对齐与最终评测，不使用其真实配对信息拟合模型。整体设置为 **source-supervised, transductive target-label-free**。

## 实验结果

University-1652 **Drone → Satellite**，官方 Test（37,855 queries / 951 gallery）：

| 方法 | Recall@1 ↑ | AP ↑ |
| --- | ---: | ---: |
| VFM-Loc（DINOv3，论文报告） | 74.42% | 83.18% |
| UPAR-Loc：无标签全局对齐 | 81.68% | 88.74% |
| **UPAR-Loc：对齐 + Patch 重排序** | **89.32%** | **93.39%** |

UPAR-Loc 的结果来自 **2026-11-13 从原始图像与预训练权重开始的独立复现**，包括重新训练重排序器和独立重算指标。加入重排序后，Recall@1 提升 **7.65** 个百分点，AP 提升 **4.65** 个百分点。

表中 AP 沿用 VFM-Loc 的评分实现。原论文与 UPAR-Loc 使用的监督条件不同：前者的公开实现使用目标评测地点标签拟合 OP；后者使用源域标签训练重排序器。详见[独立复现报告](docs/independent_reproduction_2026-11-13/REPRODUCTION_REPORT.md)。

## 复现

准备 [University-1652](https://github.com/layumi/University1652-Baseline) 原始数据集和 [Meta DINOv3 ViT-B/16](https://github.com/facebookresearch/dinov3) 预训练权重（`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`）。本仓库已包含运行所需的模型实现，无需另行克隆 VFM-Loc，也无需预先下载特征、对齐矩阵或重排序权重。

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

命令将依次完成特征提取、无标签对齐、源域训练与模型选择、目标域检索和重排序。结果保存在 `outputs/reproduction/final_metrics.json`。数据目录、完整参数和分阶段命令见[复现指南](docs/END_TO_END.md)。

## 致谢

本项目基于 [VFM-Loc](https://doi.org/10.1109/TGRS.2026.3748024)、[DINOv3](https://github.com/facebookresearch/dinov3) 和 [University-1652](https://github.com/layumi/University1652-Baseline)。第三方 DINOv3 代码及许可证见 [third_party/dinov3](third_party/dinov3)。
