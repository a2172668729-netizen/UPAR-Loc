# UPAR-Loc 复现指南

从 **University-1652 原始图像**和 **DINOv3 ViT-B/16 预训练权重**开始，即可独立完成对齐、训练与 D2S 评测。无需预先准备任何特征、OP 矩阵或重排序权重。

## 环境与数据

推荐 Python 3.10、Linux / WSL2 和 CUDA GPU。本项目已在 Windows 11 + WSL2 Ubuntu 22.04、RTX 5060、PyTorch 2.7.1+cu128 上完成全流程复现。

数据集目录应包含：

```text
University-1652/
  train/
    drone/<location_id>/*.jpeg
    satellite/<location_id>/*.jpg
  test/
    query_drone/<location_id>/*.jpeg
    gallery_satellite/<location_id>/*.jpg
```

DINOv3 权重文件：`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`。仓库已经包含所需的 DINOv3 推理代码。

## 一键运行

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

运行结束后查看 `outputs/reproduction/final_metrics.json`。`--batch-size` 控制图像特征提取的 batch size；重排序训练参数由 `configs/from_scratch.json` 控制。

## 方法与训练设置

1. **特征提取：** 冻结 DINOv3 ViT-B/16，输入 224×224；提取五区域 GeM 特征（5×768，FP32）和 Patch 特征（196×768，FP16）。
2. **全局对齐：** 两视角分别进行 PCA-512 和 whitening（无人机 0.55、卫星 0.10）；用 CSLS（k=100）构建伪配对，卫星侧容量上限为 55。迭代更新 OP 4 轮，再进行 600 步 Cayley 排序优化。Train 与 Test 的无标签对齐分别拟合。
3. **重排序训练：** 将官方 Train 的 701 个地点拆为 561 个训练、70 个模型选择、70 个确认地点。训练时每条查询采样 1 个正例、3 个困难负例和 2 个随机负例；困难负例使用只在训练地点上拟合的有标签 PCA-256/OP 挖掘。模型为双向两层、四头 Patch Cross-Attention，使用 AdamW、BCEWithLogitsLoss、CosineAnnealingLR，batch size 16；随机种子 11、22、33 各训练 12 个 epoch。
4. **模型选择：** 只用源域模型选择集决定 seed、epoch 和融合系数 λ（0 到 1.2，步长 0.2），随后在确认集上复核。
5. **目标域评测：** 使用整批未标注 Test 图像拟合目标域 PCA/OP；全局检索后重排 Top-20，最终才读取 Test 地点 ID 计算指标。

重排序器在源域有监督训练，目标域没有配对标签参与学习或参数选择；由于目标对齐使用整批未标注 Test 图像，属于 **source-supervised, transductive target-label-free** 设置。

## 分阶段执行

```bash
.venv/bin/python reproduce.py extract_source --dataset ... --weights ...
.venv/bin/python reproduce.py align_source --dataset ...
.venv/bin/python reproduce.py train --dataset ...
.venv/bin/python reproduce.py extract_target --dataset ... --weights ...
.venv/bin/python reproduce.py align_target --dataset ...
.venv/bin/python reproduce.py evaluate --dataset ...
```

分阶段运行时保持 `--dataset`、`--output` 和 `--config` 一致。主要输出包括 `features/`、`source_descriptors.pt`、`target_descriptors.pt`、`reranker/best.pt`、`reranker/train_report.json` 和 `final_metrics.json`。默认保留已有特征，重新提取时可使用 `--overwrite`。

## 已验证结果

2026-11-13 在全新克隆和独立虚拟环境中，从原图与官方权重重新提取特征、训练三个随机种子并独立重算指标。

| University-1652 D2S | Recall@1 | AP |
| --- | ---: | ---: |
| 无标签全局对齐 | 81.6775% | 88.7374% |
| **UPAR-Loc 完整方法** | **89.3224%** | **93.3904%** |

选出的配置为 **seed 22、epoch 7、λ=1.2**；完整评测包含 37,855 条查询和 951 张图库。AP 使用与 VFM-Loc 一致的计算方式。历史实验曾达到 90.2681% / 94.0511%，但从零复现没有精确重现该数值，差异分析和完整日志保留在[独立复现报告](independent_reproduction_2026-11-13/REPRODUCTION_REPORT.md)。

[正式指标](independent_reproduction_2026-11-13/results/final_metrics.json) · [独立评分](independent_reproduction_2026-11-13/results/independent_metrics.json) · [环境与依赖](independent_reproduction_2026-11-13/environment.txt)

## 参考

[VFM-Loc](https://github.com/DingLei14/VFM-Loc) · [University-1652](https://github.com/layumi/University1652-Baseline) · [Meta DINOv3](https://github.com/facebookresearch/dinov3) · [第三方代码许可证](../third_party/dinov3/LICENSE.md)
