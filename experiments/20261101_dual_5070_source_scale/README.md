# 双信号 MCS-mask80：1690→5070 训练源最小控制

本实验只增加独立训练源：保留旧 train H5 的 1690 源及全部 11 档噪声视图，追加独立命名空间的 3380 源。原 Validation 与 Development H5 原样复用。模型仍为 DualV6(80,13)，seed 271070、200 epoch、batch256、Adam lr1e−3、weight decay1e−4、label smoothing0.1；只由 Validation PairExact、Macro-F1 并列规则选择 checkpoint。Final Test 未读取。

2026-11-01 数据生成、唯一模型的200 epoch训练与统一评价已完成。训练前后的四 split 门禁均为 PASS，H5 SHA-256 完全一致。新模型 Validation/Development 的逐路ModAcc、PairExact和Macro-F1均优于旧模型，但 Development APSK32 Recall仅0.70%，泛化失败尚未解决，当前不直接扩至16900源。详见 [`results/双信号MCSmask80_1690至5070源最小控制_结果审计_20261101.md`](results/双信号MCSmask80_1690至5070源最小控制_结果审计_20261101.md)。本目录 `code/` 保存追加生成、QA、训练、评价及新噪声诊断代码。

服务器保留大型 H5、checkpoint、逐 epoch 日志；GitHub 保存代码、哈希门禁和结果摘要。不得用 Development 选择 checkpoint 或追加训练方案。
