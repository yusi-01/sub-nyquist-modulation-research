# 双信号 MCS-mask80：1690→5070 训练源最小控制

本实验只增加独立训练源：保留旧 train H5 的 1690 源及全部 11 档噪声视图，追加独立命名空间的 3380 源。原 Validation 与 Development H5 原样复用。模型仍为 DualV6(80,13)，seed 271070、200 epoch、batch256、Adam lr1e−3、weight decay1e−4、label smoothing0.1；只由 Validation PairExact、Macro-F1 并列规则选择 checkpoint。Final Test 未读取。

2026-11-01 数据生成已完成。`protocol/dual5070_pretrain_gate.json` 对四个 split 的 UID、干净源哈希、噪声 seed、Coset 和 Full/MCS 映射给出 PASS。旧 H5 未改写。训练正在独立服务器目录 `runs/dual_full_mcs_dev_20261101/train_mask_5070_seed271070_20261101` 执行；结果完成后补入 `results/`。本目录 `code/` 保存追加生成、QA、训练、评价及新噪声诊断代码。

服务器保留大型 H5、checkpoint、逐 epoch 日志；GitHub 保存代码、哈希门禁和结果摘要。不得用 Development 选择 checkpoint 或追加训练方案。
