# Seed 271061 / 271062 复验

状态：**四个新训练与11档 Development 评价均完成**。seed 271060、271061、271062 的完整对照见 [`results/单信号_Full_MCS-mask_三种子复验研究报告_20261031.md`](results/单信号_Full_MCS-mask_三种子复验研究报告_20261031.md)。低 SNR Full 优势与高 SNR mask 反超方向均在三个种子中复现；高 SNR 反超幅度和类别混淆方向波动显著。

训练服务器当前输出：`/home/zh_3220241128/SN/SHELL_original_multi_snr_20261029/runs/single_full_mcs_mask_multiseed_20261031_v3/`。

`code/run_serial_training.sh` 是初始队列脚本；`code/launch_seed271062_parallel.sh` 是两臂并行启动脚本；`code/evaluate_seed.sh` 和 `code/analyze_three_seeds.py` 产生评价和三种子统计。`artifacts/` 含新增两种子的压缩逐样本预测与训练元数据；原种子的压缩预测在前一版本 `artifacts/`。`results/` 含全部逐类指标、混淆、配对 CI、曲线及研究报告。服务器继续保留原 H5 和 checkpoint。
