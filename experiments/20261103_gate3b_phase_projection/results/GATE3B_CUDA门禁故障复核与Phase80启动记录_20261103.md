# GATE-3B CUDA 门禁故障复核与 Phase80 启动记录

日期：2026-11-03（北京时间）。本轮仅修复独立显存门禁的调用顺序，冻结编码和训练脚本均未修改。训练只启动 Phase80、seed 271060 一个任务；Final Test 未读取。

## 故障定位

先前的独立门禁通过 `CUDA_VISIBLE_DEVICES=0` 运行，在创建任何 CUDA 张量之前执行 `torch.cuda.reset_peak_memory_stats(torch.device('cuda:0'))`，于 `torch/cuda/memory.py:374` 调入 `_cuda_resetPeakMemoryStats(device)` 时抛出 `RuntimeError: Invalid device argument`。实际实参为 `device(type='cuda', index=0)`，当时 `torch.cuda.is_initialized()` 为 `False`。

服务器 Python 3.12.2、PyTorch 2.7.0+cu126、`torch.version.cuda=12.6`。该进程 `CUDA_VISIBLE_DEVICES=0`、`torch.cuda.is_available()=True`、`device_count()=1`、`current_device()=0`；宿主机 `nvidia-smi` 列出 4 张 NVIDIA RTX A6000。独立进程先在逻辑 `cuda:0` 创建张量、完成简单计算并同步后，向 `reset_peak_memory_stats` 分别传入 `torch.device('cuda:0')`、整数 `0` 和省略参数均成功，随后可读 `max_memory_allocated`、`max_memory_reserved`。另一独立进程在初始化前传入原实参，复现完全相同的堆栈。因此直接原因是**显存峰值重置早于 CUDA 上下文初始化**，不是物理与逻辑 GPU 编号错配，也不是 CUDA 张量计算故障。

独立门禁的最小修复 diff（仅调用顺序，不涉及训练脚本）：

```diff
 device = torch.device('cuda:0')
+torch.cuda.set_device(0)
+torch.empty(1, device=device)
+torch.cuda.synchronize()
 torch.cuda.reset_peak_memory_stats(device)
```

## 正式 H5 单批门禁

在本账号的冻结正式 v2 数据 `runs/shell_dual_paired_16900_seed271060_20261101/data_v2_paired` 上，通过原训练脚本的 `Views` 读取：Train/Validation/Development 分别为 16900/3380/3380 独立源，每源 11 视图。与已冻结 mask80 的 `config.json` 核对，数据目录、Coset、seed 及 batch 均一致。原 16 实通道输入 `[512,16,1024]`，双路类别索引 0–12，每个 26 维 BCE 目标恰有两个 1 且对应各自 13 维输出段。

在 `CUDA_VISIBLE_DEVICES=0` 的逻辑 `cuda:0` 上，PhaseProjection 输出 `[512,80,1024]`、`float32`、全部有限，且编码器无可训练参数；Res1dNet31(80,26) 有 6,324,122 个参数，输出 `[512,26]` 且有限。按冻结的 0.026 标签平滑及 BCEWithLogits 计算损失 `4.594619274139404`；完整反向后的所有参数梯度均存在且有限。**未执行 `optimizer.step()`。**实测峰值已分配显存 **5.419 GiB**、峰值保留显存 **5.619 GiB**，无 CUDA 设备错误或 OOM。门禁裁决：**PASS**。

启动前四卡各空闲约 48.66 GiB，系统可用 RAM 约 115 GiB，磁盘可用约 233 GiB。冻结编码 SHA-256 `5416ee7e5e99f8388185391699b1ddb17b396638f640666193f804c070622675`，训练脚本 SHA-256 `bcf7786c16c4c347b92c0ed12078154ad16ec2cfb408761e6cd5e742363ebd2e`，均与 GATE-3B 冻结报告一致。训练输出目录在启动前不存在。

## 唯一训练启动

按冻结命令使用 GPU 0 启动 Phase80、Res1dNet31(80,26)、seed 271060、batch512、200 epoch；训练数据、Validation 选模和 Development 评价规则均由冻结脚本执行。启动 PID **3398735**，输出目录 `runs/gate3b_phase_projection_20261103/train_phase80_seed271060`，日志 `runs/gate3b_phase_projection_20261103/train_phase80_seed271060.launch.log`。启动后 PID 存活、`config.json` 和 `initialization.json` 已写入；GPU 0 使用约 6.14 GiB，RAM 可用约 113 GiB，磁盘可用约 233 GiB。启动时尚未完成首个 epoch，分类收益仍为 **UNKNOWN**。后续只按 Validation 选最佳权重，并用同源 Development 与冻结 mask80、MCS16、Full80 做完整配对评价；本记录不预报结果。

