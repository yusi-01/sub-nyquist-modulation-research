# Seed 271060：Full80 / MCS-mask80 同架构控制

状态：**Development Test 审计完成，单训练种子**。

- 两臂模型均为 Res1dNet311、80输入通道、3,885,965 参数，初始化状态一致。
- 11档 Full component SNR 为 −10 至 +10 dB，步长2 dB；每档 2,600 个 view。
- checkpoint 仅由 Validation 选择：Full epoch 166，MCS-mask epoch 162。
- 低 SNR Full 优势和高 SNR MCS-mask 小幅反超均只属于单种子经验结果，不能据此断言物理信息损失。
- 类别塌缩定义与原报告的一处文字修正见当前复验报告：原 MCS16 的 PSK8 有9个 `<26` 预测项，其中6项为零预测。

`protocol/` 保存数据 manifest、训练配置、映射门禁和代码哈希；`results/` 保存中文审计和小型指标表。逐样本预测 CSV、原始 H5 和 checkpoint 保存在原始产物目录，未纳入本次 Git 归档。

`code/04_evaluate_control.py` 是原产物中的代码快照；11档 v5 评价时服务器实际脚本为 `code/04_evaluate_control_v5_server.py`，SHA-256 `7e749152a128479b78a7d70893bd61c1ebe99b18c201c1e0b9f146964b0ca2bc`。两版并存以保留演变记录。
