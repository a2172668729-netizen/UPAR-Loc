# 独立复现静态审计记录（2026-11-13）

本次目标：假设审稿人只持有 **University-1652 原始图片、DINOv3 ViT-B/16 权重和 GitHub 仓库**，检查其能否独立重建 J2（V5 全局检索 + B2 局部重排序）。本次为**静态检查与代码修订**，没有运行任何代码、模型、终端命令、训练或测试；提交信息使用 `[skip ci]`，避免自动触发 CI。

## 只读检查到的本机材料

- WSL2 数据集目录存在：`/home/allen/datasets/University-1652`（具有 `train` 和 `test`）。
- DINOv3 权重文件存在：`/home/allen/cvgl-work/VFM-Loc/pretrained_weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`；文件元数据大小 **342,860,279 bytes**。本次没有读取完整权重或计算 SHA-256。
- 真实历史源码：`/home/allen/cvgl-work/validated_research_round7`、`rerank-vfmloc/round4`、`final_round7_j2_20261112`；均只读。
- 历史验证 `verification_j2_fresh_20261113/full_forward.json` 记录了全量前向 J2 R@1 **90.268129%** / mAP **94.051117%**；`portable_j2_validation_20261113/full_fresh.json` 也保存了相同数字。这些是**现有输出文件**，本次没有重新执行。它们不能证明“从原图到训练”已经独立复现。

## 从原始源码确认并修正的问题

| 模块 | 原始 Round 7 实际做法 | 之前 GitHub 的问题 | 现有代码的静态修订 |
| --- | --- | --- | --- |
| DINOv3 特征 | **FP32** 提取五区域 R-MAC；再单独使用 **FP16 autocast** 提取 196×768 Patch | 把 FP16 Patch 直接池化成区域描述符 | 恢复双前向精度协议，分开缓存 |
| Train 地点划分 | `RandomState(20261110)` 先 561/140，`RandomState(20261111)` 再 70/70 | 单次打乱 561/70/70，训练地点不同 | 恢复两次地点划分 |
| B2 难负例 | 在 **561 个 Train-Fit 地点**拟合官方监督 PCA-256/OP 挖 top-10，再按 Patch 视觉组过滤 | 只用新 V5 向量，且在全部 Gallery 中直接取 5 个 hard | 恢复源域 Oracle 特征挖掘，限制训练负例到 Train-Fit |
| 负例采样 | 每轮 `3 hard + 2 random`，排除相同地点与视觉组 | 固定 5 hard，不逐轮换 | 恢复逐 epoch 分组采样 |
| 训练损失与优化 | BCEWithLogitsLoss、AdamW、CosineAnnealingLR、batch16、clip5、weight decay .01、12 epoch | CrossEntropy、batch8、clip1、weight decay 1e-4、无 Cosine | 恢复原参数与优化目标 |
| 模型选择 | seeds 11/22/33，每 epoch 在 Val-Selection 搜索 λ=0…1.2；Confirmation 最后只验一次 | 固定 seed33 和 λ=0.6 | 改为源域独立选 seed、epoch、λ 并写 checkpoint |
| 目标 Test | 冻结 B2 后，用目标无标签 PCA/OP，最后读取 Test 标签评分 | 全流程先提取目标图片再训练源域 | `reproduce.py all` 改为源域完成后才处理目标图片 |
| FP32 B2 打分 | 历史训练和推理使用 FP32 Patch MHA | 自动混精参与 B2 推理 | 不再启用自动混精 |

修订的主要文件：`src/raw_pipeline.py`、`src/research_pipeline.py`、`src/reviewer_training.py`、`reproduce.py`、`configs/from_scratch.json`、`docs/END_TO_END.md` 和 synthetic 测试样例。

## 仍然需要真实运行才能确认的事情

本次没有进行数值验证，**不能声明独立复现 PASS**。开机后由审稿人实际运行时，应至少核对：

1. 全新环境中 VFM-Loc / DINOv3 权重加载成功；两次前向产生的特征与独立历史实现具有可解释的数值偏差。
2. 源域训练的 **561/70/70 地点清单**、来源行序和原始 Round 4 索引一致；不同来源的数据缓存不得混用。
3. 新训练器在 seeds 11/22/33 的 12 epoch 全部正常收敛，且负例采样无重复、跨地点错误索引或源域验证污染。
4. 源域确认结果、最终选出的 checkpoint/epoch/λ 可追溯；**不得用 Test 指标重新挑选**。
5. Test 全部 37,855 查询、951 张图库的排名和 mAP 可独立重算；报告 R@1、mAP、计算环境、权重 SHA-256、运行日志和代码提交哈希。
6. 如果这次从头复现与历史 90.27% 不一致，必须以这次真实输出为准，不把旧结果当成自动验收条件。

## 方法和评测边界

- J2 的目标域对齐是 **transductive target-label-free**：使用目标域无标签分布拟合 PCA/OP；目标真实地点 ID 只在最后评分阶段读取。
- B2 在 University-1652 **官方 Train 的源域地点 ID 上有监督训练**，因此完整系统不可宣传为 training-free 或全流程无监督。
- 冻结版本的 `cap=55` 对应 Round 7 历史配置，但更早阶段发生过 Test 信息暴露与方法选型争议。这次并非重新获得盲测资格。其与此前源域选型 `cap=50` 的实验协议不能混淆。
- 原 J2 冻结记录的 **seed33 / epoch7 / λ=1.0** 是历史选型结果。新的完整流水线会重新在源域选型；不得强制把这三个数字写入此次运行输出。
- 这里的“静态修复”仅表示源码已按照可查到的原始实现修订，**没有运行或验收**。上一次 CPU CI 通过发生在这些修订之前，不能替代对修订后代码的验证。

## 后续修订：不再依赖 VFM-Loc 原仓库运行

在上述静态审计后，进一步把 DINOv3 ViT-B/16 推理所需的 **Meta 官方代码子集**（固定 2025-08-14 版本）及其完整许可证整理到 `third_party/dinov3/`。新增 `src/dinov3_backbone.py`，严格从本地官方 ViT-B/16 checkpoint 加载权重；`src/raw_pipeline.py` 现在独立完成预处理、FP32 五区域特征及 FP16 Patch 提取。源域监督 PCA-256/OP 则在 `src/source_oracle.py` 中独立实现。入口 `reproduce.py` **不再需要 `--upstream`**，只需 `--dataset` 和 `--weights`。详见 [完整复现说明](END_TO_END.md)。

VFM-Loc 上游公开仓库未提供明确许可，因此本次没有搬运其源码；相关数学算法自行实现并保留引用。Meta DINOv3 源码的许可与版权在 `third_party/dinov3/LICENSE.md` 中保留。

**本次后续整合仍仅进行源码编辑与静态检查，没有运行 Python、测试、GPU 前向或训练。** 因此旧 CI 通过结果不代表改造后代码已经验收。
