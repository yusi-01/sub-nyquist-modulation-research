# 原生 SHELL-MCS 单种子 Development 结果与 Full 启动

日期：2026-11-02（北京时间）。范围：已冻结 v2 配对双信号数据，训练源 16900、Validation 3380、Development 3380；每源 11 档 Full component SNR −10:2:+10 dB。只运行 seed 271060、`Res1dNet31(16,26)`，原 BCEWithLogits、标签平滑、Adam、warmup+cosine、batch512 和 Validation 逐路准确率选点。Final Test 未读取。

## MCS 结果与门禁

200 epoch 完成，最佳 Validation checkpoint 位于 epoch **188**。选点规则核对：200 行 history 的最高 Validation 逐路 ModAcc 对应 epoch 188；没有使用 Development 选点。最佳 epoch 的训练损失 **0.08854**，Validation 损失 **0.13562**；Validation 逐路 ModAcc **60.73%**，PairExact **38.47%**，Macro-F1 **0.5983**。同一冻结权重在独立 Development 37180 个视图（3380 个独立源）得到逐路 ModAcc **60.72%**、PairExact **38.21%**、Macro-F1 **0.5982**，评价损失 **0.13604**。逐视图预测已保存在服务器本任务目录 `train_mcs_seed271060/development_predictions.csv`，行数 37180、唯一 source_uid 数 3380。

Development 13 类 Recall：APSK16 73.51%、APSK32 61.78%、APSK64 57.29%、ASK8 93.55%、BPSK 98.25%、OQPSK 94.21%、PSK16 45.72%、PSK8 36.94%、QAM128 92.67%、QAM16 41.03%、QAM256 **13.34%**、QAM64 37.01%、QPSK 44.09%。不存在近零 Recall，但 QAM256 是明确弱类；不因此修改训练方案。训练/Validation 损失存在差距，已如实保留。历史 58.22%/35.27% 基于另一批源和旧生成器，**不能把本次差值解释为模型方法提升**。

数据 P0、Full/MCS 逐点配对缓存、batch512 前后向与 MCS 正常性门禁均 PASS。启动 Full 前复查：4 卡各约 48.7 GiB 空闲、可用 RAM 115 GiB、磁盘 155 GiB。已按冻结顺序启动 `Res1dNet31(80,26)` Full 单 seed271060 训练，独立随机初始化、相同 200 epoch/batch/损失/优化/选点规则；服务器进程 PID **2366891**。Full 训练刚开始，尚无完整 epoch 或 Development 结果。需待 Full 完成后按 source_uid 对两臂预测做逐 SNR 与配对统计，本报告不是最终论文显著性结论。

