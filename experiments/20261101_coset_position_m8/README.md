# M=8 Coset 位置归因（2026-11-01）

状态：**六个候选模型训练与 11 档 Development 评价完成，正式结果已审计**。本实验使用原 Full H5，保持 Fs=2 GHz、L=40、M=8、同一源/噪声/标签、同一80通道 Res1dNet311。仅改变8个采样位置；候选值始终映射到固定的基线16个活动输入通道，控制第一层非零位置与有效 fan-in。

本目录在查看候选的任何 Development 准确率前提交：[`protocol/实验合同与自检_20261101.md`](protocol/实验合同与自检_20261101.md) 记录假设、候选、门禁和停止条件；`protocol/coset_preflight.json` 记录单音/FFT校准和数据无关的矩阵相关性；`protocol/mapping_gates/` 记录三 split 的抽样映射检验；`code/` 保存训练、评价、QA 与启动脚本。旧版三种子基线见 [`../20261031_single_full_mcs_mask_multiseed/`](../20261031_single_full_mcs_mask_multiseed/)。

训练启动后新增 `code/evaluate_when_complete.sh` 和 `code/analyze_position_results.py`：等待六个训练配置完成后，用 GPU3 顺序评价同一11档 Development，并生成同源配对统计。它们不改变已冻结的候选、模型或训练设置。

预注册位置：基线 `[16,11,1,0,27,24,37,31]`，较低最大相干度 `[9,8,6,1,25,21,36,27]`，较高最大相干度 `[14,13,4,0,22,17,36,35]`。三个训练种子为271060–271062；每个替代位置各训练一次，共6个新模型。只用 Validation 选 checkpoint，Development 仅分析，Final Test 保持未读。

正式结论见 [`results/单信号_M8_Coset位置归因_三种子结果审计_20261101.md`](results/单信号_M8_Coset位置归因_三种子结果审计_20261101.md)。两种位置在三个重点低 SNR 档都得到正向 ModAcc 差，但较高相干度候选的收益更大；高 SNR 差异及类别塌缩随种子变化。`results/` 保存完整的三种子表格，`artifacts/position_analysis_export_20261101.tar.gz` 保存六臂训练元数据和逐源预测压缩包，不包含 H5 原始数据或大型 checkpoint。服务器原始 checkpoint 仍保留在用户账号目录下。
