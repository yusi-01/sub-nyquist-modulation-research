# GATE-3B Phase80 单种子 Development 结果与研究裁决

日期：2026-11-03（北京时间）。本轮仅训练固定 Phase80 零填充频域表示 + 原 Res1dNet31(80,26)，seed 271060。Phase80 数学上等价于把 8 路 MCS IQ 按真实 Coset 放回 40960 点网格、其余位置填零，再作正交 FFT。**它不是反混叠、Full 恢复或相位补偿方法。**

## 1. 训练与数据审计

正式 v2 Train/Validation/Development 分别为 16900/3380/3380 个独立源，每源 11 档 Full component SNR。门禁已验证 batch512 编码及完整 CNN 前后向、标签两路 26 维语义和梯度有限；显存峰值已分配 5.419 GiB、保留 5.619 GiB。正式训练共完成 **200 epoch**，仅按 Validation 逐路 ModAcc 选 checkpoint。最佳为 **epoch 32**：Validation 逐路 ModAcc **8.081%**、PairExact **0.726%**、Macro-F1 **0.0525**、损失 **0.26948**。最佳 epoch 的训练损失及完整 200 轮曲线见 `raw/history.csv`；第 200 轮训练/Validation 损失分别为 **0.24447/0.28271**，Validation ModAcc **7.374%**。最佳 checkpoint、配置、初始化指纹、逐视图预测和评价 JSON 均保存在服务器原输出目录；本地 `raw/` 也留存原件。

四组冻结 Development 预测按 `source_uid, view_uid, noise_seed, SNR, slot_1/2, true_1/2` 逐行完全一致：**37180 视图、3380 独立源、每档 3380 源**。预测标签和 correct/pair 标志逐行重算一致；没有读取 Final Test。数据配对门禁 **PASS**。机器可读证据见 `analysis/GATE3B质量门禁与摘要.json`。

## 2. 主要结果

下表为 −10 至 0 dB 的六档等权平均；Macro-F1 为逐档 Macro-F1 的平均。完整 11 档数据见 `analysis/四模型逐SNR指标.csv`。

|模型|低 SNR PairExact|低 SNR 逐路 ModAcc|低 SNR Macro-F1|
|---|---:|---:|---:|
|Full80|57.002%|75.202%|0.7458|
|MCS16|25.330%|49.827%|0.4872|
|MCS-mask80|23.718%|48.075%|0.4655|
|**Phase80**|**0.518%**|**7.747%**|**0.0473**|

主要同参数量对照 Phase80−mask80：低 SNR PairExact **−23.20 个百分点**，按 source_uid 对 11 视图成组 bootstrap 的 95% 区间 **[−24.16,−22.23]**；逐路 ModAcc **−40.33 个百分点**，区间 **[−41.20,−39.46]**。对 MCS16 的低 SNR PairExact 差为 **−24.81 个百分点**，区间 **[−25.79,−23.87]**。全 11 档 Phase80 逐路 ModAcc **7.649%**、PairExact **0.487%**、Macro-F1 **0.0486**；−10 与 +10 dB 逐路 ModAcc 分别为 **7.825%/7.544%**，没有出现随 SNR 上升的正常增益。逐档四格计数（包含 D_plus 与 D_minus）和成组区间见 `analysis/三组比较逐SNR配对四格.csv`、`analysis/三组比较源级成组区间.csv`。例如 −10 dB 的 Phase80/mask80 同正确 1、Phase 独正确 18、mask 独正确 246、同错误 3115；0 dB 分别为 9/5/1309/2057。

## 3. 类别与混淆审计

Phase80 出现严重类别塌缩。全 11 档中 **ASK8、QAM16、QAM256 的预测频数为 0**；APSK16、QPSK、BPSK 分别占全部预测的 **27.31%、25.90%、18.34%**。低 SNR 的 QPSK/PSK8/PSK16 Recall 分别为 **27.98%/0.35%/0.32%**，mask80 对应 **23.43%/20.58%/46.51%**。QPSK→PSK16/PSK8 次数由 mask80 的 **1198/665** 降到 Phase80 的 **7/12**，但这是 PSK16/PSK8 几乎不被预测造成的方向交换，**不能称为 QPSK 混淆改善**。Phase80 的 PSK16→QPSK 为 883/3120，PSK8→QPSK 为 825/3120。QAM16 Recall **0**，QAM64 **5.19%**；两类主要被预测成 APSK16、QPSK、BPSK 等。完整 13 类每档 Recall/Precision/预测频数、13×13 混淆矩阵及低高 SNR 汇总均见 `analysis/`，未择优呈报。

## 4. GATE-3B 裁决

**A. Phase80 是否优于 mask80：否，明显劣于。B. 是否优于 MCS16：否。C. QPSK→PSK16/PSK8 是否改善：不能这样判定；错误方向随类别塌缩改变。D. 是否导致严重退化：是，三个类别完全不被预测，PSK8/PSK16 Recall 近零。E. 当前频域表示是否有继续研究的实际价值：作为负结果和表示敏感性证据有价值，但按预注册停止条件不晋升为论文方法，不自动调整编码或重训。**

本实验支持的结论仅是：在当前单 seed、相同 8 路观测、相同 80 通道 SHELL 与冻结训练协议下，这一**零填充频域表示失败**，CNN 对表示组织高度敏感。它不能证明频域表示普遍无效，也不能把失败唯一归因于 Coset 几何、测量相干度或网络架构中的某一因素；更不能证明反混叠或相位补偿机制。Development 已用于裁决，结果不是跨 seed 稳定性或 Final Test 论文显著性结论。本轮停止，不启动第二编码、额外 seed 或其他训练。

