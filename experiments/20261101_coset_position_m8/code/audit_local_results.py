"""对下载的位置实验结果执行本地一致性审计。"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent / "server_export"

for seed in (271060, 271061, 271062):
    for arm in ("lower_coherence", "higher_coherence"):
        run = ROOT / f"seed_{seed}" / arm
        config = json.loads((run / "training_config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "evaluation_11snr" / "control_metrics.json").read_text(encoding="utf-8"))
        history = json.loads((run / "history.json").read_text(encoding="utf-8"))["history"]
        pred = pd.read_csv(run / "evaluation_11snr" / "control_predictions.csv")
        assert len(history) == 200
        assert config["checkpoint_sha256"] == metrics["checkpoints"]["mcs_mask"]
        assert config["parameters"] == 3885965
        assert len(metrics["per_snr"]) == 11
        assert config["best_epoch"] >= 1 and config["best_epoch"] <= 200
        print(seed, arm, "PASS", "epoch", config["best_epoch"],
              "val", round(config["best_validation_accuracy"], 5),
              "rows", len(pred), "checkpoint", config["checkpoint_sha256"][:12])

analysis = ROOT / "three_seed_analysis"
for name, expected in (("逐种子逐SNR指标.csv", 99), ("逐源配对差.csv", 66)):
    frame = pd.read_csv(analysis / name)
    assert len(frame) == expected, (name, len(frame))
    print(name, len(frame), "PASS")
