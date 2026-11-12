# 实验协议与复核说明

> 本文仅说明**旧冻结资产评测**。现已新增从 University-1652 原图和 DINOv3 权重出发的完整训练流水线，参见 [从头复现说明](END_TO_END.md)。两条流程使用不同配置和产物，历史指标不能视作新流水线的运行结果。

## 1. 本项目在做什么

数据集是 University-1652 的 Drone → Satellite 检索任务。对每张无人机查询图片，从 951 张卫星图库图片中找到对应地点。历史完整评测包含 **37,855 条查询、951 张图库图像**。

现有方法分两阶段：

- **全局检索基线**：冻结的 DINOv3 ViT-B/16 提取 GeM／R-MAC 多区域视觉描述，分别对无人机视角和卫星视角进行 PCA-512 处理，进行目标域无标签的伪配对与正交 Procrustes（OP）对齐，再形成全局检索分数。
- **局部特征重排序**：对初排前 20 名候选逐一计算双向 Patch Cross-Attention 相似度，并与全局分数融合，以确定新的候选顺序。

局部模型在源域身份标签上训练，全局对齐使用目标域无标签特征。项目的合理表述是**源域有监督、目标域无标签适配的跨视角检索**，而非整个方法完全无监督。

## 2. 需要自行准备的外部资产

仓库不包含数据集图片、全量 patch 特征、冻结模型权重或训练脚本。运行完整评测必须取得与 [配置文件](../configs/experiment.json) SHA-256 一致的文件。

| 配置字段 | 内容 | 用途 |
| --- | --- | --- |
| `global_descriptors` | 37,855×512 的查询描述符、951×512 的图库描述符 | 两种运行模式 |
| `patch_checkpoint` | 双向 Cross-Attention 模型权重（历史记录 seed 33 / epoch 7） | 直接推理 |
| `query_patches` | float16，37,855×196×768 的查询 patch 特征 | 直接推理 |
| `gallery_patches` | float16，951×196×768 的图库 patch 特征 | 直接推理 |
| `evaluation_bundle` | 与上述查询和图库严格同顺序的真实地点标签 | 全量指标计算 |
| `patch_score_cache` | 历史缓存的候选顺序与局部模型输出 | 仅缓存回放 |

完整直接推理需要约 12 GiB 外部资产。资产由仓库使用者通过有权访问的数据集和实验记录取得，不能仅凭官方数据集与 DINOv3 权重恢复冻结的局部模型 checkpoint。

将 [assets.example.json](../configs/assets.example.json) 复制为 `configs/assets.local.json`。填入文件路径；相对路径按**该 JSON 所在目录**解析，绝对路径也可用。为了兼容现有机器上的配置，程序仍接受旧的 `v5_descriptors`、`b2_checkpoint`、`b2_scores` 字段，但推荐改用新名称。

所有实际使用的资产在加载前接受 SHA-256 校验。校验能够证明资产字节与记录一致，**不能独立证明资产上游制作流程没有使用测试标签**。

## 3. 修复后的重排序与并列分数

历史代码通过一种“重排候选身份”的方式保留分数：计算局部融合分数以决定目标名次，然后把前 20 名原始全局分数从高到低重新分配给目标候选。候选外图库的分数完全不变，前 20 名原始分数的多重集也不变。

这种设计有一个必须明确的边界：**若两个候选原本分数相同，仅交换两个相同分数不能实际改变它们的排序。** 当第 20 名与未选中的图库图片同分时，也可能出现预期候选掉出 Top-20 的情况。

当前代码采用**显式排名**解决，不对分数加入 epsilon，也不破坏原始分数多重集：

1. 全局检索选择 Top-20 时，同分按 gallery index 升序决定顺序。
2. 局部融合先确定目标候选顺序，再按原设计分配原始全局分数。
3. 最终排名按**分数降序**排列；同分时，按**融合目标候选顺序**优先，未入选的图像再按 gallery index 升序排列。候选与非候选在 Top-20 边界同分时优先保留候选身份。
4. 评测函数直接接受这一显式排名；**不得重新对输出的分数矩阵调用普通 `argsort` 并忽略同分规则**。

代码接口 `fuse_reorder_ranked(...)` 返回 `(score_matrix, gallery_ranking)`。旧版 `fuse_reorder(...)` 只返回分数，保留用于历史兼容，但在同分场景下**不足以完全表达目标排序**。

这次修改改变了排序协议。旧的缓存可能与新候选顺序不一致，此时回放会报错；不能偷偷复用错位的局部分数。

## 4. 完整运行与输出

在 Ubuntu / WSL2、具有兼容 PyTorch/CUDA 的环境中：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
cp configs/assets.example.json configs/assets.local.json
# 编辑 assets.local.json 中的真实路径
.venv/bin/python evaluate_retrieval.py --assets configs/assets.local.json --check-assets
.venv/bin/python evaluate_retrieval.py --assets configs/assets.local.json --queries 4
.venv/bin/python evaluate_retrieval.py --assets configs/assets.local.json
```

完整评测采用整个图库与所有正样本，输出 Recall@1 / @5 / @10 / @20 / @50 和 mAP。输出中的 `global_retrieval` 是全局检索基线，`patch_reranking` 是局部重排序融合系统。查询标签仅在完整融合排名生成后用于计算指标；不参与此评测入口中的候选选择或局部分数计算。

`--mode cached` 仅适合在缓存依赖哈希与候选顺序都匹配时做历史回放。默认 `fresh` 将重新推理全部 **757,100 对候选**。如果输入资产与历史记录不匹配，程序会拒绝运行，而不是静默使用另一批数据。

测试命令：

```bash
.venv/bin/python tests/test_reranking.py
.venv/bin/python -m unittest discover -s tests -p test_portability.py
```

## 5. 历史数字、评价边界与尚缺证据

以下是**旧版冻结评测**记录，保留作历史比较，并非修复后的新成绩：

| 方法 | Recall@1 | mAP |
| --- | ---: | ---: |
| 全局检索基线 | 81.7541% | 88.7828% |
| 全局检索 + 局部重排序 | 90.2681% | 94.0511% |

原始历史报告保存在 [reference_run.json](reference_run.json)，其中仍有旧实验代号，用于核对原日志，不再作为对外方法名。新配置中该记录使用 `historical_metrics` 字段；程序会显示新结果与旧结果的差值。

由于新的确定性同分规则可能改变指标，**不能把旧数字作为新版本已经通过的门槛**。历史 `--require-exact` 选项只检查“是否与旧数字完全相同”，不适合用作修复后正确性证明。

实验科学边界：

- University-1652 Test 在历史研究中已多次被查看；结果不能称作独立盲测。
- 源域训练的具体数据隔离、训练阶段的标签使用，以及冻结全局描述符的生成过程，需要独立审查其完整源码与产物来源。当前评测入口只证明自身不以测试标签生成排名。
- 当前仓库是冻结资产的评测快照，不含原始图像→特征→训练→对齐→评测的一整套完全独立运行工程。
- 目前未提供跨数据集泛化结果，也不据此作相关结论。

上游项目包括 [VFM-Loc](https://github.com/DingLei14/VFM-Loc) 和 DINOv3；其源码、权重、数据许可须分别遵守。此仓库中的历史文件及别名仅用于复查与兼容。
