"""将冻结 Development 逐 SNR 结果绘成论文讨论用静态图。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ARMS = ["full", "mcs", "mask"]
NAMES = {"full": "Full80", "mcs": "MCS16", "mask": "MCS-mask80"}
COLORS = {"full": "#1f77b4", "mcs": "#d95f02", "mask": "#6a3d9a"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10,
                         "svg.fonttype": "none"})
    scores = pd.read_csv(args.analysis / "三模型逐SNR指标.csv")
    classes = pd.read_csv(args.analysis / "三模型逐SNR逐类指标.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    for arm in ARMS:
        s = scores[scores.arm.eq(arm)].sort_values("snr_db")
        q = classes[classes.arm.eq(arm) & classes.true_class.eq("QPSK")].sort_values("snr_db")
        axes[0].plot(s.snr_db, s.pair_exact * 100, marker="o", lw=1.7, ms=3.5,
                     color=COLORS[arm], label=NAMES[arm])
        axes[1].plot(s.snr_db, s.modacc * 100, marker="o", lw=1.7, ms=3.5,
                     color=COLORS[arm], label=NAMES[arm])
        axes[2].plot(q.snr_db, q.recall * 100, marker="o", lw=1.7, ms=3.5,
                     color=COLORS[arm], label=NAMES[arm])
    for ax, title in zip(axes, ["双路全对率", "逐路调制准确率", "QPSK 召回率"]):
        ax.set_title(title)
        ax.set_xlabel("Full component SNR (dB)")
        ax.set_ylabel("百分比 (%)")
        ax.set_xticks(list(range(-10, 11, 4)))
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(args.analysis / "三模型逐SNR与QPSK对照.png", dpi=240)
    fig.savefig(args.analysis / "三模型逐SNR与QPSK对照.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
