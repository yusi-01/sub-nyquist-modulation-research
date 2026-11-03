# GATE-3D Blind Projection 训练前门禁与启动记录

## 研究边界

本轮仅训练一个模型：冻结 GATE-3C 盲支持估计与双槽测量子空间投影、1024 点正交 IFFT 后，输入随机初始化的原生 Res1dNet31(16,26)。训练源 16900、Validation 源 3380、Development 源 3380，每源 11 档 Full component SNR。训练 seed 271060、batch 512、200 epoch 上限，原 SHELL BCEWithLogits、标签平滑、Adam、5 epoch warmup + cosine，依据 Validation 逐路 ModAcc 选择 checkpoint。Final Test 不参与。

## 数据与代码固定记录

服务器工作目录仅在 `zh_3220241128` 账号下：`/home/zh_3220241128/SN/SHELL_original_multi_snr_20261029`。正式配对数据目录为 `runs/shell_dual_paired_16900_seed271060_20261101/data_v2_paired`，`dataset_manifest.json` SHA-256 为 `51449a16a42f38175580657cfbd33de5ed179ca7df1b5304455817faae38bc95`。不生成新信号源，三个 split 由原 H5 与逐行身份核对过的盲支持缓存读取。

| 对象 | SHA-256 |
|---|---|
| 冻结 GATE-3C `gate3c_physical.py` | `c0d5b17749cee8cf1b27e6d3c9dc6f82128b409384a538f4c591de7ed16e7951` |
| `cache_support.py` | `35dff0472588c174ada628ea1dd3e7513d7ee7d1bdbc6e8b9c5fc37c8246012b` |
| `blind_input.py` | `e249ccf6cbab70b25167256c61e549d1cf06062cab266e368f2ee3b5a1df754e` |
| `train_blind_projection.py` | `7e9ea51799ddc27dec0b84f778d3a17debe480502b6ffbfbea6a121665c2ca1d` |
| Train 盲支持缓存 | `19d13883f56cab99f618899771596a32da6dd1ee5eace31a8f6f38ff60759f5d` |
| Validation 盲支持缓存 | `90dc6b137567c8e214a8641eba2b033b1b88e1f7d9a28a8474e6959945a93cbd` |
| Development 盲支持缓存 | `c9b4d961a57d5c6f1fcca5bb3ac9ee1bfddf597ab9a430e4347b7acda61d808a` |

Train 缓存有 185900 个视图，Validation 和 Development 各 37180 个视图；缓存仅含行号、`source_uid`、`view_uid` 和两个盲估计 slot。训练读取时逐行核对缓存与 H5 的源、视图身份。真实 slot、标签、SNR 和 Full 独有观测均不进入投影选择。

## P0 数值与资源门禁

正式 H5 batch 512 检查通过：MCS 输入 `[512,16,1024]`、标签 `[512,26]`、盲支持 `[512,2]`；投影输出 `[512,16,1024]`，数值有限；Res1dNet31 共 6,320,026 个参数，输出 `[512,26]`；原 BCE 损失为有限值 `5.5954122543`；完整反向梯度有限，未执行 `optimizer.step`。抽样重算的冻结 GATE-3C 盲支持与缓存一致，GPU 投影与原 NumPy 实现的复数最大绝对误差不超过 `2.98×10⁻⁷`，相对 L2 约 `2.7×10⁻⁷`。最终分类输入仅在投影和 IFFT 之后作一次物理 IQ/8 缩放，无逐样本归一化。

单批前后向 GPU 峰值：已分配 5.2283 GiB、保留 5.4316 GiB。启动前服务器每卡可用显存约 48.66 GiB，可用 RAM 约 113 GiB，磁盘可用约 232 GiB。正式输出目录在启动前不存在，未覆盖旧实验。以上门禁 **PASS**。

## 启动与当前状态

2026-11-03 在物理 GPU0 启动唯一批准的训练，PID `4025246`。输出位于 `runs/gate3d_blind_projection_20261103/train_blind_seed271060`，启动日志为 `runs/gate3d_blind_projection_20261103/train_blind_seed271060.launch.log`。运行配置和 `history.csv`、Validation 最佳 checkpoint 已开始写入。复查时进程正常，完成第 9 个 epoch；GPU0 已用约 6054 MiB，系统可用 RAM 约 110 GiB，磁盘可用 232 GiB。前期 Validation 指标逐步上升；不得据此改变冻结训练参数。

本记录只证明启动和门禁通过，不代表模型最终有效。完成后仅用 Validation 选定 checkpoint，在相同 Development 源及噪声视图上与冻结 Raw MCS16、Full80 配对评价，按 `source_uid` 成组 Bootstrap，完整报告 11 档 SNR、13 类指标、支持错误传播和高 SNR 退化。结果为单训练 seed 的 Development 证据。

