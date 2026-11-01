# 双信号 SHELL Full/MCS 正式执行合同

日期：2026-11-01。状态：**方案冻结；历史实物核心门禁 PASS；新配对数据 P0 未执行；训练未启动。**本文件是本轮唯一执行合同。训练 seed 固定 271060，仅一轮单种子 Development 实验。禁止 Final Test、额外模型、额外 seed 和历史结果驱动调参。

## 本任务持续要求（2026-11-01 用户补充）

服务器地址 `10.8.18.11:22`，仅以 `zh_3220241128` 账号访问，任何操作均限制在该账号目录或本任务临时路径，不触碰其他账号。用户后一处写作 `zh_3202241128`，与明确提供的登录账号不同；执行时以实际登录账号 `zh_3220241128` 的 HOME 为硬边界。每次服务器作业前检查 GPU 显存、可用 RAM 与磁盘；清理仅针对本任务确认无用的临时文件，原始数据、checkpoint、日志和有复现价值的结果保留。重要结果整理为中文报告并上传 GitHub；上传前核对内容与 Git 状态。提出或运行具体方案前回顾研究目标、冻结合同、代码语义、数据/资源预算和停止条件，发现不合理处先修正。不得把密码写入代码、报告、Git 或日志。

## 一、历史 SHELL 有界实物审计裁决

补充认证后已进入 `/home/zh_3220241128`，只读核查服务器原目录 `/home/zh_3220241128/SN/SHELL_original_multi_snr_20261029`。下列状态以实际文件检查为准。

|检查|状态|证据与界限|
|---|---|---|
|三 H5 SHA-256、行数、标签实物|PASS|train/Validation/Development SHA-256 依次为 `17d245dd589d87518364903f3caa0b081ac2c317136a8cda1ea2015f5e4d1cdf`、`dfd03f0633ea3c74a50ab18bd7749c23c8d858b9c392b265a0ff2f6447d56b434`、`bb56556c66989bf698c79a5681039effa8163d9926f0a70fbac61aa407dd12893`；行数 185900/37180/37180，均为 1024×16；每行恰有两个非零槽标签、值域 0–13。|
|训练日志—最佳 checkpoint—评价 JSON 对应|PASS|日志记载 185900/37180 行、Validation 最佳 0.5823292093 于 epoch 151；评价 JSON 明确指向 `best_checkpoint_mr.pt` 和 Development H5。权重 SHA-256 `63c65902f7217ad899bdce63a6e1f354190df1c8e290eb0d6bbe774f9c781c20`，JSON SHA-256 `ad03a04e939287204e12048ac2d0aca4b41270a1dbb04cafb787b9215d13b535`。|
|冻结权重复评 0.582154 / 0.352663|PASS|原 `evaluate_labeled.py` 对指定 Development H5 与冻结 `best_checkpoint_mr.pt` 重评 37180 行，逐路 `0.582154384077461`，PairExact `0.3526627218934911`，与历史 JSON 完全一致。|
|train/Validation/Development 源 UID、clean hash、噪声 seed 零交集|PASS|三 H5 分别有 16900/3380/3380 个唯一 UID 与同数唯一 clean hash；噪声 seed 各有 185900/37180/37180 个唯一值。三个 split 两两 UID、clean hash、noise seed 交集全部为 0。|
|日志、manifest、种子与 clean Full 重建|PASS（抽样）|服务器 `dataset_config.json` 固定 master seed 271050、generator version、16900/3380/3380 及 11 档；历史生成代码 SHA-256 已记录。按生成器的调度和 clean source 函数，对各 split 首/中/末共 9 个源重建，UID、clean Full hash、槽和标签均匹配。尚未全量重建 23660 个 clean Full，不能称全量逐样本再生成 PASS。没有从含噪 MCS 伪造哈希。|
|双路频率升序标签|PASS（抽样）|三 H5 每行恰两非零槽；按生成器重建的 9 个源，槽与频率升序标签一致。全量标签—波形重建仍未执行。|
|16 通道 IQ 顺序及缩放|PASS（抽样/代码）|生成器按 Coset 顺序交错写 I/Q；首个训练源 11 档 X 与相同代码重生成的数组逐元素相同。生成器 `shell_raw_from_physical` 与旧 loader 固定 min/max 的代数组合为物理 IQ/8。|
|最新 v2 物理合同一致性|PASS（核心常数），UNKNOWN（全量实物）|旧/新核心均列 Fs 2 GHz、L40、M8、相同 Coset、24 槽和 13 类，旧源抽样可重建；生成器版本、随机流与 H5 schema 不同，全部噪声/波形未逐点再生核验，不能把旧 H5 当新版 v2 严格配对数据。|

**裁决：A 历史三 split 在 UID、clean hash 与噪声 seed 层面已实物证明零交集（PASS）；全量波形再生未做。B 58.22%/35.27% 升级为可复评的历史独立 Development 证据，仍不是本轮严格配对 Full/MCS 或 Final Test 结论。C 历史 clean Full 可由已保存的代码、配置和 seed 确定性重建，9 源抽样 PASS；但旧 H5 没有完整 Full noisy 视图，若要用于严格配对需全量逐视图重建与核验。按本轮冻结决定，新实验仍用 v2 生成器，历史审计至此停止。**

## 二、冻结模型与协议

MCS 为原生 `Res1dNet31(16,26)`；Full 为 `Res1dNet31(80,26)`，80 通道按 40 个 Full 时间位置的 IQ 展平，完整保留每源 40960 个复数采样点。两者仅首层输入通道数不同，主干、池化与 26 维分类层保持原 SHELL 定义。`DualV6` 仅是 Res1dNet31 输出 reshape 包装，不列为主模型。两臂从 seed 271060 随机初始化，不加载历史权重。

两路标签按实际子带频率升序，24 槽 `Y` 非零处提取类别后按原 `double_one_hot` 得到 26 维目标；两路调制允许同类。损失精确沿用旧 `BCEWithLogitsLoss` 与 `target*(1-0.026)+0.026/26`，并按旧训练脚本的输出维度和 reduction 执行。Adam lr 0.001、weight decay 0.0001；batch 512；200 epoch；5 epoch warmup 后 cosine；AMP 使用旧流程。checkpoint 仅按 Validation **逐路 accuracy** 最大选择，相等时沿用旧脚本的覆盖规则；PairExact、Macro-F1、Recall 只报告不参与选点。M5 独立训练和评价，绝不将真实 slot 输入调制网络。

统一尺度：两臂都喂入物理 IQ/8；MCS 按冻结 Coset `[16,11,1,0,27,24,37,31]` 抽取，顺序为 `[I0,Q0,...,I7,Q7]`；Full 保留 40 个位置的 IQ。可用兼容旧 SHELL 的仿射 X 编码与旧 loader 解码，但必须以数值门禁证明等于 `/8`，禁止二次缩放或逐样本归一化。

## 三、严格配对数据合同

采用**已冻结并通过既有 QA 的 v2 正式物理生成器** `v2_formal_core.py`，明确固定本次代码 SHA-256、配置 SHA-256、`GENERATOR_VERSION` 和 11 档 Full component SNR，不再使用旧 `original_shell_multi_snr` 名称的生成器来拼接数据。若服务器存在与冻结代码一致的完整历史种子/元数据，可先只读验证历史重建；无论其结果如何，本轮新配对数据依本段 v2 生成，避免依赖未证实的历史 Full。

Train 16900、Validation 3380、Development 3380 个互异双信号源；每源 11 档 `−10:2:+10 dB` Full component SNR。两个不同随机子带、13 类调制、169 种生成顺序组合配额分别为 100/20/20；频率升序重排后的组合分布另作审计，不预设它仍严格均衡。Full 域每个源每档只实现一次复 AWGN，MCS 逐点从同一 noisy Full 按 `40n+Coset[m]` 抽取；同源、同噪声 seed、同标签、同观察时长。Full noisy 波形可由 clean Full 加命名 noise seed 按需确定性重建；不要求同时把全部 Full 与 MCS 展开载入 RAM。Final Test 不生成、不读取、不参与选模。

每源保存 source_uid、component UID/hash、clean Full hash、两个槽/标签、11 档 view_uid/SNR/noise_seed、Full/MCS 配对校验值。逐源预测保存两臂预测和真值、槽、SNR、源/视图身份，供以后计算 Full 正确而 MCS 错误的样本。服务器新输出目录必须与历史实验分开，原始 checkpoint/H5/日志不可覆盖。

## 四、P0 与资源门禁

生成前核算实际可用磁盘、RAM、GPU 与分块缓存峰值；按 16900/3380/3380 ×11 × Full 40960 complex float32 计算最坏显式 Full 存储需求，选择紧凑 clean Full + seed 和按需/分块读取。先用独立小规模 QA 数据检查代码和模型前向、单 batch 峰值显存与读写吞吐；这只是门禁，不是小规模模型训练或规模搜索。

P0 必须全部 PASS：三个 split 的 source_uid、component/clean Full hash、noise seed 集合零交集；每源 11 档元数据齐全且 169 组合覆盖；频率升序标签与源级波形一致；Full/MCS 逐点精确配对；两臂输入均为物理 IQ/8；float32 数值范围与有限值、无 int16 回绕；生成器/配置/H5 manifest 哈希记录完整；随机初始化与前向输出尺寸 26 正确。任一失败停止，不训练、不改物理参数。

## 五、获准后的单 seed 执行与开发评价

P0 全通过后先训练 MCS，再检查 Validation 逐路 ModAcc、PairExact、Macro-F1、13 类 Recall、训练/验证损失与最佳 epoch。若数据、损失、标签、选点和优化运行正常，即使未达到历史 58.22%，也不擅自调参；随后训练 Full。两臂使用同一数据身份、噪声、训练规则与 seed 271060，各自随机初始化。

Development 仅在各自 Validation 选点完成后评价一次，报告 11 档 Full/MCS 逐路 ModAcc、Macro-F1、PairExact、13 类 Recall、训练/Validation 损失与最佳 epoch，并保存逐源预测。配对差、Full 对 MCS 错误的计数与不确定性按独立 source_uid 成组计算，不把 11 档视图当 11 个独立源。单 seed 结果标注为 Development，不能宣称最终论文显著性。若两臂独立源成绩及关键类 Recall 支持稳定判别，才具备下一阶段研究低 SNR 额外混淆的条件；本合同不授权自动开展该下一阶段。

## 六、当前执行决策

历史核心审计：**PASS**（上述范围）。历史 Full 重建可能性：**PASS（抽样）**，全量严格配对仍 **UNKNOWN**。新 v2 配对数据 P0：**UNKNOWN，未生成/未执行**。资源与前向门禁：**UNKNOWN，未执行**。可信双信号基线训练启动条件：**未满足**。当前无新配对实验结果可判断是否已具备研究双信号低 SNR 额外混淆的条件。

