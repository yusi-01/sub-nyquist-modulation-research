# Seed 271061 / 271062 复验

状态：**服务器串行训练中**。四臂顺序为 271061 Full80、271061 MCS-mask80、271062 Full80、271062 MCS-mask80。训练结果和三种子统计尚未产生。

训练服务器当前输出：`/home/zh_3220241128/SN/SHELL_original_multi_snr_20261029/runs/single_full_mcs_mask_multiseed_20261031_v3/`。

`code/run_serial_training.sh` 是本轮启动脚本；`results/三种子复验_执行状态与机制预案_20261031.md` 记录判定标准及下一阶段最小 Coset 机制实验。完成训练后，应把每个 seed 的 checkpoint 哈希、训练配置、最佳 Validation epoch、11档 Development 指标、逐类混淆和配对统计加入此版本，再更新状态。
