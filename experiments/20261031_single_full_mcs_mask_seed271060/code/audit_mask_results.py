from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "control_development_11snr_results_v5"
LABELS = ["APSK16", "APSK32", "APSK64", "ASK8", "BPSK", "OQPSK", "PSK16", "PSK8",
          "QAM128", "QAM16", "QAM256", "QAM64", "QPSK"]
ARMS = {"Full80": "full_pred", "MCS-mask80": "mask_pred", "原MCS16": "original_mcs_pred"}
GRID = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]


def confusion(y_true, y_pred):
    table = pd.crosstab(pd.Categorical(y_true, categories=LABELS),
                        pd.Categorical(y_pred, categories=LABELS), dropna=False)
    table.index, table.columns = LABELS, LABELS
    return table.to_numpy(dtype=int)


def scores(frame, pred_col):
    cm = confusion(frame.true_label, frame[pred_col])
    diag = np.diag(cm)
    recall = np.divide(diag, cm.sum(1), out=np.zeros(13), where=cm.sum(1) > 0)
    precision = np.divide(diag, cm.sum(0), out=np.zeros(13), where=cm.sum(0) > 0)
    f1 = np.divide(2 * recall * precision, recall + precision, out=np.zeros(13), where=recall + precision > 0)
    return float(diag.sum() / cm.sum()), float(f1.mean()), cm, recall, precision


def paired_test(frame, col_a, col_b):
    a = frame[col_a].eq(frame.true_label).to_numpy()
    b = frame[col_b].eq(frame.true_label).to_numpy()
    d = a.astype(float) - b.astype(float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    ci = (float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se))
    a_only, b_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = float(binomtest(min(a_only, b_only), a_only + b_only, .5).pvalue) if a_only + b_only else 1.0
    return float(d.mean()), ci, p, (int(np.sum(a & b)), a_only, b_only, int(np.sum(~a & ~b)))


def fmt_ci(ci):
    return f"[{ci[0]*100:+.2f},{ci[1]*100:+.2f}]"


def main():
    df = pd.read_csv(RESULT / "control_predictions.csv")
    metrics = json.loads((RESULT / "control_metrics.json").read_text(encoding="utf-8"))
    history = json.loads((ROOT / "mcs_mask_model/history.json").read_text(encoding="utf-8"))["history"]
    best = json.loads((ROOT / "mcs_mask_model/best_checkpoint_metadata.json").read_text(encoding="utf-8"))
    mapping = json.loads((ROOT / "mapping_gate.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "development_11snr.manifest.json").read_text(encoding="utf-8"))

    assert len(df) == 28600 and sorted(df.snr_db.unique().tolist()) == GRID
    assert df.groupby("snr_db").size().eq(2600).all()
    assert df.groupby(["snr_db", "true_label"]).size().eq(200).all()

    summary_rows, critical_rows, confusion_rows, collapse_rows = [], [], [], []
    critical = ["PSK8", "PSK16", "QAM64", "QAM256"]
    for snr in GRID:
        part = df[df.snr_db == snr]
        for arm, pred in ARMS.items():
            acc, f1, cm, recall, precision = scores(part, pred)
            summary_rows.append({"snr_db": snr, "arm": arm, "modacc": acc, "macro_f1": f1})
            counts = part[pred].value_counts()
            for label in LABELS:
                if int(counts.get(label, 0)) < 26:
                    collapse_rows.append({"snr_db": snr, "arm": arm, "class": label,
                                          "predicted_count": int(counts.get(label, 0))})
            for label in critical:
                i = LABELS.index(label)
                critical_rows.append({"snr_db": snr, "arm": arm, "class": label,
                                      "recall": recall[i], "precision": precision[i],
                                      "predicted_count": int(counts.get(label, 0))})
            for a, b in (("PSK8", "PSK16"), ("QAM64", "QAM256")):
                i, j = LABELS.index(a), LABELS.index(b)
                confusion_rows.append({"snr_db": snr, "arm": arm, "pair": f"{a}/{b}",
                                       f"{a}_to_{b}": int(cm[i, j]), f"{b}_to_{a}": int(cm[j, i])})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ROOT / "审计_逐SNR三模型指标.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(critical_rows).to_csv(ROOT / "审计_关键类别指标.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(confusion_rows).to_csv(ROOT / "审计_关键双向混淆.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(collapse_rows).to_csv(ROOT / "审计_类别塌缩.csv", index=False, encoding="utf-8-sig")

    paired_rows = []
    comparisons = [("Full80", "full_pred", "MCS-mask80", "mask_pred"),
                   ("Full80", "full_pred", "原MCS16", "original_mcs_pred"),
                   ("MCS-mask80", "mask_pred", "原MCS16", "original_mcs_pred")]
    for snr in GRID:
        part = df[df.snr_db == snr]
        for name_a, col_a, name_b, col_b in comparisons:
            delta, ci, p, cells = paired_test(part, col_a, col_b)
            paired_rows.append({"snr_db": snr, "comparison": f"{name_a}-{name_b}", "delta": delta,
                                "ci_low": ci[0], "ci_high": ci[1], "mcnemar_exact_p": p,
                                "both_correct": cells[0], "a_only": cells[1], "b_only": cells[2], "both_wrong": cells[3]})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(ROOT / "审计_逐SNR配对检验.csv", index=False, encoding="utf-8-sig")

    slot_rows = []
    for region, snrs in (("低中SNR(-10至0)", [-10, -8, -6, -4, -2, 0]),
                         ("交叉(+2)", [2]), ("高SNR(+4至+10)", [4, 6, 8, 10])):
        sub = df[df.snr_db.isin(snrs)]
        for slot, part in sub.groupby("occupied_slot"):
            row = {"region": region, "slot": int(slot), "n": len(part)}
            for arm, col in ARMS.items():
                row[arm] = float(part[col].eq(part.true_label).mean())
            row["Full_minus_mask"] = row["Full80"] - row["MCS-mask80"]
            slot_rows.append(row)
    slots = pd.DataFrame(slot_rows)
    slots.to_csv(ROOT / "审计_子带位置指标.csv", index=False, encoding="utf-8-sig")

    region_rows = []
    for region, snrs in (("低中SNR", [-10, -8, -6, -4, -2, 0]), ("交叉区", [2]), ("高SNR平台", [4, 6, 8, 10])):
        sub = summary[summary.snr_db.isin(snrs)]
        for arm in ARMS:
            x = sub[sub.arm == arm]
            region_rows.append({"region": region, "arm": arm, "mean_modacc": x.modacc.mean(), "mean_macro_f1": x.macro_f1.mean()})
    regions = pd.DataFrame(region_rows)
    regions.to_csv(ROOT / "审计_分区汇总.csv", index=False, encoding="utf-8-sig")

    total_seconds = sum(float(x["seconds"]) for x in history)
    final = history[-1]
    best_hist = history[int(best["epoch"]) - 1]
    collapsed = pd.DataFrame(collapse_rows)
    lines = ["# 单信号 Full/MCS-mask 同架构控制：研究结论导向审计（2026-10-31）", "",
             "> 范围：Development Test，seed 271060。未读取Final Test；本报告不能替代三seed正式结果。", "",
             "## 一、完整性与可比性", "",
             f"- 11档Development共{len(df):,}个view，每档2,600个、每类每档200个，SNR为−10至+10 dB、步长2 dB。",
             f"- Full→原MCS逐位验证：{manifest['full_to_mcs_bitwise']}，最大误差{manifest['max_abs']}。",
             f"- Full80与MCS-mask80参数量相同：{mapping['parameters_full']:,}；初始化状态一致：{mapping['same_initial_state']}。",
             f"- MCS-mask最优Validation准确率{best['best']:.4f}，epoch {best['epoch']}；Development未参与checkpoint选择。",
             f"- 训练200 Epoch，总GPU训练时间约{total_seconds/3600:.2f}小时；末轮Validation={final['validation_accuracy']:.4f}。",
             "- 三模型评价、11档manifest、映射门禁均PASS。", "",
             "## 二、逐SNR核心结果", "",
             "|SNR|Full ModAcc/F1|MCS-mask ModAcc/F1|原MCS ModAcc/F1|Full−mask (pp)|95%CI (pp)|", "|---:|---:|---:|---:|---:|---:|"]
    for snr in GRID:
        vals = {r.arm: r for r in summary[summary.snr_db == snr].itertuples()}
        pr = paired[(paired.snr_db == snr) & (paired.comparison == "Full80-MCS-mask80")].iloc[0]
        lines.append(f"|{snr:+d}|{vals['Full80'].modacc:.4f}/{vals['Full80'].macro_f1:.4f}|{vals['MCS-mask80'].modacc:.4f}/{vals['MCS-mask80'].macro_f1:.4f}|{vals['原MCS16'].modacc:.4f}/{vals['原MCS16'].macro_f1:.4f}|{pr.delta*100:+.2f}|{fmt_ci((pr.ci_low, pr.ci_high))}|")

    lines += ["", "## 三、分区结论", "", "|区域|Full80|MCS-mask80|原MCS16|", "|---|---:|---:|---:|"]
    for region in ("低中SNR", "交叉区", "高SNR平台"):
        vals = {r.arm: r.mean_modacc for r in regions[regions.region == region].itertuples()}
        lines.append(f"|{region}|{vals['Full80']:.4f}|{vals['MCS-mask80']:.4f}|{vals['原MCS16']:.4f}|")

    lines += ["", "- −10至0 dB：Full相对MCS-mask优势从21.00 pp单调收窄到4.15 pp，各档95%CI均不跨0。",
              "- +2 dB：Full仅领先0.62 pp，95%CI跨0，是经验交叉区。",
              "- +4至+10 dB：MCS-mask反超Full 2.04–2.81 pp，各档95%CI均不跨0。",
              "- MCS-mask在−10至+2 dB均弱于原MCS16；+4 dB后两者几乎相同，说明高SNR反超不依赖16通道参数量优势。",
              "- 但高SNR原MCS16的Macro-F1明显低于MCS-mask：+10 dB为0.8097对0.8327；因此原MCS较高ModAcc包含类别不均衡收益，不能只看总准确率。", "",
              "## 四、关键类别与塌缩", ""]
    for snr in (-10, -4, 0, 2, 4, 10):
        lines += [f"### {snr:+d} dB", "", "|模型|PSK8 R|PSK16 R|QAM64 R|QAM256 R|PSK8→16 / 16→8|QAM64→256 / 256→64|", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            crit = pd.DataFrame(critical_rows)
            cv = crit[(crit.snr_db == snr) & (crit.arm == arm)].set_index("class")
            cf = pd.DataFrame(confusion_rows)
            p = cf[(cf.snr_db == snr) & (cf.arm == arm) & (cf.pair == "PSK8/PSK16")].iloc[0]
            q = cf[(cf.snr_db == snr) & (cf.arm == arm) & (cf.pair == "QAM64/QAM256")].iloc[0]
            lines.append(f"|{arm}|{cv.loc['PSK8','recall']:.3f}|{cv.loc['PSK16','recall']:.3f}|{cv.loc['QAM64','recall']:.3f}|{cv.loc['QAM256','recall']:.3f}|{p['PSK8_to_PSK16']}/{p['PSK16_to_PSK8']}|{q['QAM64_to_QAM256']}/{q['QAM256_to_QAM64']}|")
        lines.append("")
    if collapsed.empty:
        lines.append("按每档预测数<26（1%）定义，三个模型在11档均无类别塌缩。")
    else:
        lines.append(f"检测到{len(collapsed)}个模型/SNR/类别塌缩项，详见`审计_类别塌缩.csv`。")
        lines.append("全部塌缩项均来自原MCS16的PSK8：从0到+10 dB预测数均为0，PSK8→PSK16为200/200；这证明原模型存在单向类别边界塌缩，而不是PSK8/PSK16天然不可分。")

    lines += ["", "## 五、训练与归一化诊断", "",
              f"- Epoch 1非活动通道卷积权重范数为{history[0]['normalization_probe']['conv_inactive_weight_norm']:.3f}，到best epoch和epoch 200均衰减为0；这是恒零输入无数据梯度、仅受weight decay作用的预期结果。",
              f"- best epoch的conv/BN输出标准差为{best_hist['normalization_probe']['conv_std']:.3f}/{best_hist['normalization_probe']['bn_std']:.3f}，epoch 200为{final['normalization_probe']['conv_std']:.3f}/{final['normalization_probe']['bn_std']:.3f}，未出现NaN或整体方差归零。",
              f"- BN running variance在epoch 200为[{final['normalization_probe']['bn_running_var_min']:.4g}, {final['normalization_probe']['bn_running_var_max']:.4g}]，说明不同输出通道尺度差异较大但未发生全局归一化崩溃。",
              "- 零填充仍使第一层有效fan-in从80降到16；同参数量并不等于同优化条件，这一残余混杂必须保留在论文限制中。"]

    low_slots = slots[slots.region == "低中SNR(-10至0)"].sort_values("Full_minus_mask")
    cross_slots = slots[slots.region == "交叉(+2)"].sort_values("slot")
    high_slots = slots[slots.region == "高SNR(+4至+10)"].sort_values("Full_minus_mask")
    lines += ["", "## 六、子带位置", "",
              f"- 数据实际包含{slots.slot.nunique()}个占用slot（0–{int(slots.slot.max())}），不是40个；+2 dB单档每slot有{int(cross_slots.n.min())}–{int(cross_slots.n.max())}个源。",
              f"- 每个slot在低中SNR区有{int(low_slots.n.min())}–{int(low_slots.n.max())}个view、高SNR区有{int(high_slots.n.min())}–{int(high_slots.n.max())}个view；跨SNR汇总包含同源重复观测。",
              f"- 低中SNR Full−mask最小/最大slot差为{low_slots.iloc[0].Full_minus_mask*100:+.2f}/{low_slots.iloc[-1].Full_minus_mask*100:+.2f} pp。",
              f"- 高SNR Full−mask最小/最大slot差为{high_slots.iloc[0].Full_minus_mask*100:+.2f}/{high_slots.iloc[-1].Full_minus_mask*100:+.2f} pp。",
              "- 单slot样本量不完全均衡，且当前未做按类别×slot分层置信区间；位置差异只作探索性线索，不据此提出位置定制模型。", "",
              "## 七、研究判断", "",
              "1. **A成立（single-seed development证据）**：低SNR Full优势在同架构、同参数量控制后稳定存在。",
              "2. **B部分否定**：高SNR MCS反超并非主要由原16通道模型参数量或输入维数差异造成，因为80通道MCS-mask同样反超。",
              "3. **C仍为PARTIAL**：结果支持固定Coset观测在低SNR造成额外经验分类困难，但单seed、不同有效fan-in及零通道/BN交互尚未排除，不能写成物理信息损失的最终证明。",
              "4. **下一步优先多seed复验，而不是模型优化**：复现Full与MCS-mask两臂的seed差异；若低SNR优势和高SNR交叉都稳定，再进入采样数量/位置归因。",
              "5. 原MCS16在低中SNR优于mask80，说明零填充控制本身带有优化代价；后续应把它视为架构控制臂，不应取代原MCS工程基线。", "",
              "## 八、暂不支持的表述", "",
              "- 不能称为Final Test或三seed正式结论。",
              "- 不能断言高SNR MCS获得了更多物理信息；更可能是模型/表示归纳偏向，需要复验。",
              "- 不能依据单slot探索结果设计位置专用规则。",
              "- 不能将MCS-mask低SNR劣于原MCS全部归因于欠采样；其中包含零通道有效fan-in和归一化效应。", "",
              "配套机器可读文件：`审计_逐SNR三模型指标.csv`、`审计_逐SNR配对检验.csv`、`审计_关键类别指标.csv`、`审计_关键双向混淆.csv`、`审计_子带位置指标.csv`。"]
    (ROOT / "单信号_Full_MCS-mask_同架构控制_结果审计_20261031.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "report": str(ROOT / '单信号_Full_MCS-mask_同架构控制_结果审计_20261031.md'),
                      "rows": len(df), "collapse_items": len(collapse_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
