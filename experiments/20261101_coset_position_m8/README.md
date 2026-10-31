# M=8 Coset 位置归因（2026-11-01）

状态：**候选已冻结，服务器训练启动中**。本实验使用原 Full H5，保持 Fs=2 GHz、L=40、M=8、同一源/噪声/标签、同一80通道 Res1dNet311。仅改变8个采样位置；候选值始终映射到固定的基线16个活动输入通道，控制第一层非零位置与有效 fan-in。

本目录在查看候选的任何 Development 准确率前提交：[`protocol/实验合同与自检_20261101.md`](protocol/实验合同与自检_20261101.md) 记录假设、候选、门禁和停止条件；`protocol/coset_preflight.json` 记录单音/FFT校准和数据无关的矩阵相关性；`protocol/mapping_gates/` 记录三 split 的抽样映射检验；`code/` 保存训练、评价、QA 与启动脚本。旧版三种子基线见 [`../20261031_single_full_mcs_mask_multiseed/`](../20261031_single_full_mcs_mask_multiseed/)。

训练启动后新增 `code/evaluate_when_complete.sh` 和 `code/analyze_position_results.py`：等待六个训练配置完成后，用 GPU3 顺序评价同一11档 Development，并生成同源配对统计。它们不改变已冻结的候选、模型或训练设置。

预注册位置：基线 `[16,11,1,0,27,24,37,31]`，较低最大相干度 `[9,8,6,1,25,21,36,27]`，较高最大相干度 `[14,13,4,0,22,17,36,35]`。三个训练种子为271060–271062；每个替代位置各训练一次，共6个新模型。只用 Validation 选 checkpoint，Development 仅分析，Final Test 保持未读。
