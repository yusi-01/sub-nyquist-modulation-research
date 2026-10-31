# Sub-Nyquist 调制识别研究归档

本仓库存放可复核的研究协议、代码版本、质量门禁、指标摘要和中文分析。原始 H5 数据及模型 checkpoint 保存在训练服务器和本地备份中；仓库中的 manifest 与 SHA-256 用于定位和核验它们。

## 当前进度

|版本|状态|入口|
|---|---|---|
|2026-10-30/31 Single Full80 vs MCS-mask80，seed 271060|Development 审计完成|[实验索引](experiments/20261031_single_full_mcs_mask_seed271060/README.md)|
|seed 271061、271062 复验|训练与11档 Development 评价完成；三种子报告已归档|[实验索引](experiments/20261031_single_full_mcs_mask_multiseed/README.md)|

## 文件原则

- `experiments/` 中每轮实验独立保存协议、代码快照、输入/输出哈希和结果摘要，不覆盖历史版本。
- Development Test 用于研究分析；Final Test 不参与 checkpoint 选择或本轮复验。
- 不提交密码、私钥、原始大体积 H5 或未经大文件存储方案管理的 checkpoint；三种子 Development 预测以压缩包保留。
- 服务器路径仅作索引，实际研究结论须以对应 manifest、预测文件和代码哈希共同核验。

详见[文件导航与保留规则](docs/研究文件导航与归档规则_20261031.md)。
