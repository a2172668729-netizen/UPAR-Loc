# 跨视角地理定位：无标签特征对齐与局部重排序

本仓库整理 University-1652 数据集 **Drone → Satellite（无人机图像检索卫星图）** 的冻结评测代码。方法由两个阶段组成：

1. **全局检索基线**：使用 DINOv3 提取多区域特征，通过目标域无标签 PCA／正交对齐得到全局描述符，对卫星图库进行初始排序。
2. **局部特征重排序**：使用在源域身份标签上训练的**双向 Patch Cross-Attention** 模型，对每张无人机查询的前 20 个候选卫星图重新排序。

本仓库的目标是**检查已有实验代码并复核冻结资产的检索指标**，不是从原始图像开始的一键训练项目。原始数据、预提取特征、局部模型权重与训练脚本并未全部包含在这里。

## 已有实验记录（历史测试结果）

| 方法 | Recall@1 | mAP |
| --- | ---: | ---: |
| 全局检索基线 | 81.7541% | 88.7828% |
| 加入局部重排序后 | **90.2681%** | **94.0511%** |
| 历史提升 | **+8.5141 个百分点** | **+5.2683 个百分点** |

**以上数字来自修复前的冻结历史评测记录，并非当前代码重新运行后的成绩。** 本仓库已经修复相同分数的候选无法按目标顺序交换的问题，同时把候选选择与指标排序改为确定性规则。修复可能改变少量含并列分数的排名，因此必须重新执行全量评测后才能写入新版本指标。

这些成绩来自**研究过程中已经多次观察过的 University-1652 Test**，不能称为独立盲测，也不能直接证明跨数据集泛化。目标域对齐不读取身份标签，但局部重排序模型在源域使用监督训练，所以整个方法不应宣传为“完全无监督”或“无需训练”。

## 运行环境

历史环境：WSL2 Ubuntu 22.04、Python 3.10.12、PyTorch 2.7.1+cu128、RTX 5060。

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
```

## 运行现有资产

先从可信渠道取得冻结资产，再复制 [资产路径示例](configs/assets.example.json) 为 `configs/assets.local.json`，填写实际位置。支持绝对路径、空格路径、相对于配置文件目录的路径；旧配置中的历史字段名也可以继续读取。

```bash
# 校验文件 SHA-256；不启动模型推理
.venv/bin/python evaluate_retrieval.py --assets configs/assets.local.json --check-assets

# 仅用前 4 条查询进行无标签推理检查
.venv/bin/python evaluate_retrieval.py --assets configs/assets.local.json --queries 4

# 完整重算 37,855 × 20 = 757,100 对图像的局部重排序
.venv/bin/python evaluate_retrieval.py --assets configs/assets.local.json
```

`--mode cached` 仅用于回放原先的局部分数缓存；当同分排序修复导致候选顺序发生变化时会主动报错，应使用默认的完整推理模式。历史 `--require-exact` 选项只比较**旧版指标**，新算法不保证与历史数字逐位相等，不建议用于修复后的验收。

运行结果会输出 JSON，方法名统一为 `global_retrieval`（全局检索）和 `patch_reranking`（局部重排序）。

## 代码位置

| 文件 | 用途 |
| --- | --- |
| [evaluate_retrieval.py](evaluate_retrieval.py) | 推荐评测入口：资产校验、推理、完整指标 |
| [src/patch_attention.py](src/patch_attention.py) | 双向 Patch Cross-Attention 网络 |
| [src/correct_reranking.py](src/correct_reranking.py) | 分数融合、候选重排、**同分时的显式排名** |
| [src/op_core.py](src/op_core.py) | 无标签 PCA／OP 对齐方法实现（非全流程运行入口） |
| [configs/experiment.json](configs/experiment.json) | 冻结参数、哈希与历史指标 |
| [docs/REPRODUCTION.md](docs/REPRODUCTION.md) | 实验协议、外部资产、限制与复核步骤 |

旧文件名、旧资产键与旧缓存内部字段仅为历史兼容保留。阅读和调用请使用上表中的新入口与名称。

## 自动测试

```bash
.venv/bin/python tests/test_reranking.py
.venv/bin/python -m unittest discover -s tests -p test_portability.py
```

测试覆盖排列错误的回归、随机置换、分数守恒、候选外分数不变、相同分数与 Top-20 边界、评测中显式排名的使用。**单元测试通过不等于外部资产已经完成独立复核。**

详细复核步骤与科学结论的适用范围见 [复现说明](docs/REPRODUCTION.md)。
