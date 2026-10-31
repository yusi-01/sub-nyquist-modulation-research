# Seed 271061 / 271062 复验

状态：**服务器训练中**。seed 271061 Full80 已完成；seed 271061 MCS-mask80 在 GPU0，seed 271062 Full80 在 GPU1，seed 271062 MCS-mask80 在 GPU2。训练结果尚未统一评价，三种子统计未产生。

训练服务器当前输出：`/home/zh_3220241128/SN/SHELL_original_multi_snr_20261029/runs/single_full_mcs_mask_multiseed_20261031_v3/`。

`code/run_serial_training.sh` 是初始队列脚本；`code/launch_seed271062_parallel.sh` 是用户要求加速后的两臂并行启动脚本。`results/三种子复验_执行状态与机制预案_20261031.md` 记录判定标准及下一阶段最小 Coset 机制实验。完成训练后，应把每个 seed 的 checkpoint 哈希、训练配置、最佳 Validation epoch、11档 Development 指标、逐类混淆和配对统计加入此版本，再更新状态。
