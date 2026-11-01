# 双信号 Full80/MCS-mask80 单种子开发对照

2026-11-01；历史 Coset `[16,11,1,0,27,24,37,31]`。仅 Development 探索，未读取 Final Test。主报告见 [中文结果](results/双信号Full80_MCSmask80_单种子开发报告_20261101.md)。

## 协议

- 冻结 v2 物理内核；独立源 train/Validation/Development 为 1690/338/338，11 档 −10 至 +10 dB。生成顺序 169 种有序调制组合精确均衡，频率排序后不再精确均衡。
- Full 域一次加噪并保存同源 Full/MCS 观测。三 split 的源与噪声种子隔离，抽检逐点配对误差为零。真实 slot 和 η 仅在评价后关联。
- 相同 `DualV6(80,13)` 网络与 seed 271070；Full80 对比固定 16 活动通道 MCS-mask80。各 200 epoch，最佳 checkpoint 仅按 Validation PairExact、Macro-F1 并列规则选择。
- 服务器保存原始 H5、训练日志与最佳 checkpoint。本仓库只收录代码、QA 摘要、结果报告、CSV/图与压缩逐样本证据；不收录大型原始数据或模型权重。

## 状态与解释边界

两臂完成训练，按同源键核对 3718 条 Development 预测。Full/MCS PairExact 为 2.58%/1.75%，绝对准确率偏低；源级配对区间跨零，η 分层非单调。该单种子结果不支持将理想 Gram 噪声放大直接解释为 CNN 调制错误，也不足以晋升结构感知网络。详见报告及 `results/`。

## 可复核文件

- `code/`：数据门禁、训练/评价、逐源配对分析代码。
- `protocol/gate.json`：三 split 门禁摘要；`protocol/training_config_*.json`：冻结训练配置。
- `results/`：中文报告、历史曲线、逐源预测和 η 分层表；`artifacts/analysis_export_20261101.tar.gz` 为服务器非模型原始导出包，SHA-256 `1c40868898b600110ef951e66e20137bb56c125aa20ad5b96740efa75796f093`。
