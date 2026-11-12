# 跨视角地理定位：无标签对齐与局部重排序

本项目研究 University-1652 **无人机图像 → 卫星图像** 的跨视角检索。使用冻结的 **DINOv3 ViT-B/16**，经过目标域**无标签全局特征对齐**，再使用在源域身份标签上训练的**双向 Patch Cross-Attention 重排序器**重新排列 Top-20 检索候选。

仓库包含两条相互独立的路径：

- **从原始图像完整运行**：输入 University-1652 数据集与 DINOv3 预训练权重，自行提取特征、拟合 PCA/OP、在源域训练重排序模型、最后测试。入口 [`reproduce.py`](reproduce.py)，协议见 [完整运行说明](docs/END_TO_END.md)。
- **已有冻结资产的历史评测**：使用之前生成的描述符、Patch、重排序 checkpoint 复核历史实验。入口 [`evaluate_retrieval.py`](evaluate_retrieval.py)，协议见 [冻结资产复评说明](docs/REPRODUCTION.md)。

## 从数据集与 DINOv3 权重出发

准备好 [University-1652 官方目录](https://github.com/layumi/University1652-Baseline) 和 DINOv3 ViT-B/16 checkpoint。官方 VFM-Loc 源码为**公开代码依赖**（无需任何额外训练权重），以下命令下载指定版本：

```bash
git clone https://github.com/DingLei14/VFM-Loc.git upstream/VFM-Loc
git -C upstream/VFM-Loc checkout 162d885d25f0215025d7fe9f0a1c70f64e7ba119
python3.10 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-reproduction.txt
```

完整运行命令：

```bash
.venv/bin/python reproduce.py all \
  --dataset /path/to/University-1652 \
  --weights /path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --upstream upstream/VFM-Loc \
  --output outputs/reproduction
```

内部执行：原图特征提取 → 源域/目标域无标签 PCA+OP → 源域有监督重排序训练与模型选型 → 对 Test 一次性评测。也可执行 `extract`、`align`、`train`、`evaluate` 四个阶段，按需检查每一阶段输出。默认参数在 [`configs/from_scratch.json`](configs/from_scratch.json)。

**请不要将上述全新训练流程等同于已确认复现历史的 90.2681%**：原先的全部训练源码、采样细节和产物尚未独立核对；当前新流程是依据可取得的源域协议重新整理的实现，尚未在原始数据和 GPU 上跑完。每个阶段的日志和结果会保存到指定 output 目录。

## 实验方法

| 模块 | 具体作用 | 使用标签情况 |
| --- | --- | --- |
| DINOv3 特征提取 | 五区域 GeM/R-MAC 与 196 个 Patch token | 不使用地点标签 |
| 全局无标签对齐 | 独立 PCA-512、CSLS 伪配对、OP、600 步 Cayley 优化 | 目标域和源域拟合均不读标签 |
| 双向 Patch Cross-Attention | 对初排前 20 张卫星图重新排序 | **源域训练使用**地点标签 |
| University-1652 Test | Recall@1 / Recall@5 / mAP 等 | 只在生成排名后计算指标 |

这里的“无标签”特指**目标域特征对齐**；整个系统是*源域监督训练 + 目标域无标签适配*，不能称为完全无监督或全流程 training-free。

## 历史冻结资产记录（仅作对照）

| 系统 | Recall@1 | mAP |
| --- | ---: | ---: |
| 全局检索基线 | 81.7541% | 88.7828% |
| 全局检索 + 局部重排序 | **90.2681%** | **94.0511%** |

这些是**此前冻结资产的历史 Test 兼容评测指标**，不是新代码重跑出来的结果。Test 已在研究过程中多次暴露，不属于独立盲测。仓库现已使用确定性同分排名修复历史排序边界问题，新结果必须从代码实测得出，不预填上述成绩。

## 目录导航

| 文件 | 用途 |
| --- | --- |
| [`reproduce.py`](reproduce.py) | 完整流水线的命令行入口 |
| [`src/raw_pipeline.py`](src/raw_pipeline.py) | 从图片生成区域、Patch 特征以及严格的图片顺序记录 |
| [`src/research_pipeline.py`](src/research_pipeline.py) | 无标签拟合、困难负例、训练与最终评测 |
| [`src/op_core.py`](src/op_core.py) | PCA/CSLS/OP/Cayley 算法 |
| [`src/patch_attention.py`](src/patch_attention.py) | 双向 Cross-Attention 模型 |
| [`src/correct_reranking.py`](src/correct_reranking.py) | 修复后的候选身份重排、同分确定性排名 |
| [`evaluate_retrieval.py`](evaluate_retrieval.py) | 旧冻结资产复评入口 |
| [`docs/END_TO_END.md`](docs/END_TO_END.md) | 从头运行协议、硬件与结果限制 |

## 代码自检

```bash
.venv/bin/python -m compileall -q reproduce.py evaluate_retrieval.py src tests
.venv/bin/python tests/test_reranking.py
.venv/bin/python -m unittest discover -s tests -p test_portability.py
.venv/bin/python -m unittest discover -s tests -p test_reproduction.py
```

仓库在 GitHub Actions 上设置了 CPU 单元测试；CPU 测试不代表 DINOv3 权重加载、CUDA 训练与全量数据实验已经验收。

详细说明与无法从旧记录证明的部分见 [完整运行说明](docs/END_TO_END.md)。
