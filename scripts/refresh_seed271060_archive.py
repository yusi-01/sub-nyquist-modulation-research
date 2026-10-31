"""从本地审计产物更新小型、可复核的 Git 归档文件。"""
from pathlib import Path
from shutil import copy2

repo = Path(__file__).resolve().parents[1]
workspace = repo.parent
deliverables = workspace / "deliverables"
control = deliverables / "single_full_mcs_mask_control_20261030_20261031_093802"
paired = deliverables / "single_full_mcs_paired_20261030_20261030_170342"
base = repo / "experiments" / "20261031_single_full_mcs_mask_seed271060"
multiseed = repo / "experiments" / "20261031_single_full_mcs_mask_multiseed"

items = [
    (deliverables / "研究文件导航与归档规则_20261031.md", repo / "docs" / "研究文件导航与归档规则_20261031.md"),
    (deliverables / "单信号_Full_MCS-mask_三种子复验_执行状态与机制预案_20261031.md", multiseed / "results" / "三种子复验_执行状态与机制预案_20261031.md"),
    (deliverables / "single_full_mcs_mask_multiseed_20261031" / "run_serial_training.sh", multiseed / "code" / "run_serial_training.sh"),
    (deliverables / "single_full_mcs_mask_multiseed_20261031" / "launch_seed271062_parallel.sh", multiseed / "code" / "launch_seed271062_parallel.sh"),
    (control / "单信号_Full_MCS-mask_同架构控制_结果审计_20261031.md", base / "results" / "单种子结果审计.md"),
    (control / "审计_分区汇总.csv", base / "results" / "审计_分区汇总.csv"),
    (control / "审计_关键类别指标.csv", base / "results" / "审计_关键类别指标.csv"),
    (control / "审计_关键双向混淆.csv", base / "results" / "审计_关键双向混淆.csv"),
    (control / "审计_类别塌缩.csv", base / "results" / "审计_类别塌缩.csv"),
    (control / "审计_逐SNR配对检验.csv", base / "results" / "审计_逐SNR配对检验.csv"),
    (control / "审计_逐SNR三模型指标.csv", base / "results" / "审计_逐SNR三模型指标.csv"),
    (control / "审计_子带位置指标.csv", base / "results" / "审计_子带位置指标.csv"),
    (control / "development_11snr.manifest.json", base / "protocol" / "development_11snr.manifest.json"),
    (control / "mapping_gate.json", base / "protocol" / "mapping_gate.json"),
    (control / "code_sha256.txt", base / "protocol" / "code_sha256.txt"),
    (control / "mcs_mask_model" / "training_config.json", base / "protocol" / "mask_training_config.json"),
    (control / "mcs_mask_model" / "best_checkpoint_metadata.json", base / "protocol" / "mask_best_checkpoint_metadata.json"),
    (paired / "full_data" / "manifest.json", base / "protocol" / "full_data_manifest.json"),
    (paired / "full_model" / "training_config.json", base / "protocol" / "full_training_config.json"),
    (paired / "full_model" / "best_checkpoint_metadata.json", base / "protocol" / "full_best_checkpoint_metadata.json"),
    (paired / "code_snapshot" / "02_train_full.py", base / "code" / "02_train_full.py"),
    (control / "code_snapshot" / "03_train_mcs_mask.py", base / "code" / "03_train_mcs_mask.py"),
    (control / "code_snapshot" / "04_evaluate_control.py", base / "code" / "04_evaluate_control.py"),
    (control / "code_snapshot" / "mask_common.py", base / "code" / "mask_common.py"),
    (control / "audit_mask_results.py", base / "code" / "audit_mask_results.py"),
]

for src, dest in items:
    if not src.is_file():
        raise FileNotFoundError(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    copy2(src, dest)
    print(dest.relative_to(repo))
