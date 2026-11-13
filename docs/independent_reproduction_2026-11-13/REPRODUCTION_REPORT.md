# University-1652 D2S 从零独立复现与验收

Windows 11 / WSL2 Ubuntu 22.04 · 上游提交 `2efcfed6e53de28e5f6e5420dea214e8769f3eb8`

**Historically Exposed Test Compatibility Evaluation — Not a Blind Test**

## 验收结论

从原始 University-1652 图像和指定 DINOv3 权重出发，独立完成了特征提取、源域与目标域对齐、B2 三个随机种子共 36 个 epoch 的训练和正式 Test 评测。无标签全局对齐实测 **81.6775% / 88.7374%**，加入 B2 后实测 **89.3224% / 93.3904%**，提升 **+7.6450 / +4.6530** 个百分点，指标顺序为 Recall@1 / mAP。与历史 J2 的差值为 -0.9457 / -0.6607 个百分点。 [正式指标](results/final_metrics.json) · [独立重算](results/independent_metrics.json)

第三方可以在取得私有仓库访问权限后，用该提交、本报告附带的本地修订和两个原始资产执行同一流程。端到端运行与评分审计通过。历史精确数值复现判定为 **FAIL**，判断依据是两项指标都应匹配给定历史值的四位小数精度。源域验证采用传导式全局对齐，Test 的历史暴露限制了泛化结论。 [本地修订](FIXES.md)

## 结果对照

| 方法 | 本次 R@1 / mAP | 历史 R@1 / mAP | 本次减历史（百分点） |
| --- | --- | --- | --- |
| VFM-Loc 论文 DINOv3 | 论文值 | 74.42% / 83.18% | — |
| 无标签全局对齐（J0） | 81.6775% / 88.7374% | 81.7541% / 88.7828% | -0.0766 / -0.0454 |
| 全局对齐 + 双向 Patch 重排序（J2） | 89.3224% / 93.3904% | 90.2681% / 94.0511% | -0.9457 / -0.6607 |

两种新排名均包含 37,855 行及完整 951 图库。Top-1 分别命中 30,919 和 33,813 条查询。B2 将 3,738 条查询从错排改为首位命中，同时将 844 条原本首位命中的查询移出首位。独立 NumPy 校验了每行是完整排列、每条查询恰有一个正例、重排序只改变 Top-20 内部顺序，并按正例排名倒数重算 mAP。University-1652 基准脚本另有梯形积分版本，同一批排名按该版本分别为 85.2074% 和 91.3564%。表中沿用仓库和 VFM-Loc 作者实现的 AP 定义。 [原论文](https://arxiv.org/html/2603.13855v2) · [VFM-Loc 评分实现](https://github.com/DingLei14/VFM-Loc/blob/main/vfm_loc/engine.py) · [University-1652 基准评分实现](https://github.com/layumi/University1652-Baseline/blob/master/evaluate_gpu.py)

## 阶段验收

| 阶段 | 结论 | 运行证据 |
| --- | --- | --- |
| 新克隆、新环境、CUDA | PASS | 上游 detached HEAD，PyTorch 2.7.1+cu128、CUDA 12.8、RTX 5060，`pip check` 通过。 [环境](environment.txt) |
| 数据集与地点划分 | PASS | Train 37,854 Drone / 701 Satellite，Test 37,855 Drone / 951 Satellite。源域地点 561 / 70 / 70 互斥，查询 30,294 / 3,780 / 3,780。 [清点](audit/inventory.json) |
| 单元与集成测试 | PASS | 13 个 unittest 和 13 个仓库排序脚本检查。 [单元与集成](logs/unit_integration.log) · [排序](logs/reranking_direct.log) |
| 最小真实图像测试 | PASS | 32 张源域原图，188 个权重键严格加载，CUDA 双前向与磁盘数组一致，batch 16 实际完成 5 次更新，峰值分配 1,895.7 MiB。 [真实图像日志](logs/real_image_smoke.log) |
| Train 全量原图提取 | PASS | 所有指定源域图像生成 FP32 区域 [N,5,768] 及 FP16 Patch [N,196,768]，独立扫描数值均有限。 [主日志](logs/reproduce-all.log) |
| 源域 PCA-512 / OP / Cayley | PASS | cap 55、初始解和四次更新共五次 SVD，伪对数量 [34238, 35821, 36245, 36384, 36421]，Cayley 600 步，最大正交误差 0.000225。 [源域对齐](results/source_descriptors.json) |
| Train-Fit 挖掘与 B2 训练 | PASS | 独立复核全部 30,294 条查询及负例池，seed 11/22/33 各 12 epoch，batch 16。实际 epoch 耗时 231.7–245.6 秒。 [训练报告](results/reranker/train_report.json) |
| 源域选型与确认 | PASS | 冻结 seed **22**、epoch **7**、λ **1.2**。Selection **94.2328% / 96.9978%**，Confirmation **96.3757% / 98.0745%**。 [训练报告](results/reranker/train_report.json) |
| Test 特征与无标签对齐 | PASS | 源域确认完成后读取目标原图，目标 OP 五次解的伪对数量 [35793, 35786, 36025, 36191, 36272]，Cayley 600 步，最大正交误差 0.000193。 [目标域对齐](results/target_descriptors.json) |
| Top-20 全量重排序与复核 | PASS | 最终评测和独立验证退出码均为 0。代码 SHA-256 在运行前后相同。 [独立重算](results/independent_metrics.json) · [代码校验](logs/source-check.log) |

## 标签与数据边界

特征提取保存文件顺序，拟合函数接收特征张量，路径中地点 ID 由训练或评分入口单独解析。B2 训练和选型的读取链路只指向源域生成物。目标特征由正式运行在源域最佳 checkpoint 和确认报告写完后首次创建。生成全局和融合排名后才解析 Test 地点 ID。复核确认 Train 与 Test 地点集合互斥、Train-Fit 候选均来自 561 个训练地点、正例地点一致、负例地点与正例不同。源域卫星图按原始文件哈希发现两个重复组，均位于各自同一分组内。Train 与 Test 卫星图的原始文件哈希交集为 0。该检查未度量跨地点的近重复图像。 [数据清点](audit/inventory.json) · [评分复核](results/independent_metrics.json)

源域无标签 PCA-512/OP 使用官方 Train 的全部三组图像，Selection 和 Confirmation 都用该对齐表示及完整源域图库。因此两组的无标签图像参与全局对齐，源域监督 PCA-256/OP、负例池与 B2 梯度更新限于 Train-Fit。Confirmation 查询真值只在冻结后的确认评分使用。Selection 和 Confirmation 均在完整源域图库上评分，包含其他源域分组的图库及其评分标签。数据集和指定权重只读访问。运行和验证均使用独立虚拟环境，DINOv3 推理模块来自克隆内的 Meta 源码。 [模型加载日志](logs/real_image_smoke.log)

## 历史差异定位

全局对齐相对历史差值为 -0.0766 / -0.0454 个百分点。全局阶段独立于 B2 checkpoint 和融合权重，因而这部分差值已将排查范围缩小到特征预处理和前向精度、PCA / OP / Cayley 数值计算及排名并列。最终一轮目标 OP 有 36,272 个伪对，评分后用真值诊断的伪对精度为 82.9152%。当前具备新特征、PCA 基底、伪对、旋转矩阵和排名的完整证据。缺少与同一批历史逐阶段张量的配对对照，无法把该差值定量归因到其中某一步。

本次源域重新选出的组合与历史 seed 33、epoch 7、λ=1.0 有差异。历史 B2 相对 J0 的提升为 +8.5140 / +5.2683 个百分点，本次为 +7.6450 / +4.6530 个百分点。J2 与历史的差值还可能受到源域难负例的排序与采样、重新训练的权重、冻结 λ 及候选并列的影响。独立评分与正式结果一致，排除了本次报告计算或抄录错误。FP32/TF32 阶段设置在预检查期间固定，且完整正式配置保持原样。历史值仅在正式模型选型结束后用于结果比较。 [修订](FIXES.md) · [训练报告](results/reranker/train_report.json) · [评分复核](results/independent_metrics.json)

## 复现命令与交付

```bash
cd /home/allen/cvgl-work/independent_reproduction_20261113/repo
../.venv/bin/python reproduce.py all \
  --dataset /home/allen/datasets/University-1652 \
  --weights /home/allen/cvgl-work/VFM-Loc/pretrained_weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --output /home/allen/cvgl-work/independent_reproduction_20261113/results --batch-size 4 --workers 2 --device cuda
```

正式运行起始 2026-11-13T12:26:12+08:00，结束 2026-11-13T15:20:26+08:00。执行时使用空输出目录。可复核启动环境、门禁与后处理脚本均位于 `audit/`。日志、全部新特征、OP、36 个训练 checkpoint、最优 checkpoint 与排名留存在本目录。原研究目录、数据、系统配置和 GitHub 远程均无修改。本次没有 Git 提交或推送。

- [环境与权重 SHA-256](environment.txt)
- [代码修订说明](FIXES.md) · [完整 diff](audit/reproduction.patch)
- [正式命令](logs/formal-command.log) · [主日志](logs/reproduce-all.log)
- [最终指标](results/final_metrics.json) · [独立校验](results/independent_metrics.json)
