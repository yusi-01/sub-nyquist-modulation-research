"""绘制 GATE-3D 冻结 Development 的完整逐 SNR 对照曲线。"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

results = Path(__file__).resolve().parents[1] / "results"
data = pd.read_csv(results / "三模型逐SNR指标.csv")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
for ax, (metric, title) in zip(axes, (("pair_exact", "PairExact"),
                                      ("modacc", "Component ModAcc"),
                                      ("macro_f1", "Macro-F1"))):
    for arm, label, color in (("raw", "Raw MCS16", "#335c9f"),
                              ("blind", "Blind Projection", "#c15b2a"),
                              ("full", "Full80", "#45936b")):
        rows = data[data.arm.eq(arm)].sort_values("snr_db")
        ax.plot(rows.snr_db, rows[metric], marker="o", markersize=3, linewidth=1.7,
                color=color, label=label)
    ax.axvspan(-10, 0, color="#e8eef8", alpha=.5, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Full component SNR (dB)")
    ax.set_xticks(list(range(-10, 11, 4)))
    ax.grid(alpha=.25)
axes[0].set_ylabel("Fraction")
axes[0].legend(frameon=False, loc="lower right", fontsize=8)
fig.suptitle("GATE-3D: frozen single-seed Development comparison", y=1.03)
fig.tight_layout()
fig.savefig(results / "GATE3D_11档SNR三模型性能曲线.png", bbox_inches="tight")
fig.savefig(results / "GATE3D_11档SNR三模型性能曲线.svg", bbox_inches="tight")
plt.close(fig)

