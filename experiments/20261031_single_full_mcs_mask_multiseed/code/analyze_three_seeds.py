"""三训练种子在同一 Development 源上的配对评价。"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "single_full_mcs_mask_control_20261030_20261031_093802"
NEW = ROOT / "server_export"
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)
LABELS = ["APSK16", "APSK32", "APSK64", "ASK8", "BPSK", "OQPSK", "PSK16", "PSK8", "QAM128", "QAM16", "QAM256", "QAM64", "QPSK"]
SNRS = list(range(-10, 11, 2))
SEEDS = (271060, 271061, 271062)
KEYS = ["source_uid", "view_uid", "noise_seed", "snr_db", "occupied_slot", "true_id", "true_label"]


def metrics(y, pred):
    yt = pd.Categorical(y, categories=LABELS)
    yp = pd.Categorical(pred, categories=LABELS)
    cm = pd.crosstab(yt, yp, dropna=False).to_numpy(dtype=int)
    d = np.diag(cm)
    recall = np.divide(d, cm.sum(1), out=np.zeros(13), where=cm.sum(1) > 0)
    precision = np.divide(d, cm.sum(0), out=np.zeros(13), where=cm.sum(0) > 0)
    f1 = np.divide(2 * recall * precision, recall + precision, out=np.zeros(13), where=(recall + precision) > 0)
    return cm, recall, precision, float(d.sum() / cm.sum()), float(f1.mean())


frames = {}
for seed in SEEDS:
    path = (OLD / "control_development_11snr_results_v5" / "control_predictions.csv" if seed == 271060
            else NEW / f"seed_{seed}" / "development_11snr" / "control_predictions.csv")
    df = pd.read_csv(path, usecols=KEYS + ["full_pred", "mask_pred"], dtype={"source_uid": str, "view_uid": str, "noise_seed": str})
    df = df.sort_values("view_uid").reset_index(drop=True)
    assert len(df) == 28600 and sorted(df.snr_db.unique().tolist()) == SNRS
    assert df.groupby("snr_db").size().eq(2600).all()
    assert df.groupby(["snr_db", "true_label"]).size().eq(200).all()
    assert not df.view_uid.duplicated().any()
    if frames:
        assert df[KEYS].equals(frames[271060][KEYS]), f"seed {seed} 的测试身份不同"
    frames[seed] = df

seed_rows, class_rows, cm_rows, pair_rows, collapse_rows = [], [], [], [], []
paired_vectors = {}
rng = np.random.default_rng(20261031)
for seed, df in frames.items():
    for snr in SNRS:
        part = df[df.snr_db == snr]
        assert part.source_uid.nunique() == 2600
        y = part.true_label.to_numpy()
        for arm, col in (("Full80", "full_pred"), ("MCS-mask80", "mask_pred")):
            pred = part[col].to_numpy()
            cm, recall, precision, acc, f1 = metrics(y, pred)
            seed_rows.append(dict(seed=seed, snr_db=snr, arm=arm, modacc=acc, macro_f1=f1))
            counts = cm.sum(0)
            for i, label in enumerate(LABELS):
                class_rows.append(dict(seed=seed, snr_db=snr, arm=arm, label=label,
                                       recall=recall[i], precision=precision[i], predicted_count=int(counts[i])))
                if counts[i] < 26:
                    collapse_rows.append(dict(seed=seed, snr_db=snr, arm=arm, label=label,
                                              predicted_count=int(counts[i]), zero_prediction=bool(counts[i] == 0)))
                for j, predicted in enumerate(LABELS):
                    cm_rows.append(dict(seed=seed, snr_db=snr, arm=arm, true_label=label,
                                        predicted_label=predicted, count=int(cm[i, j])))
        a = (part.full_pred == part.true_label).to_numpy()
        b = (part.mask_pred == part.true_label).to_numpy()
        d = a.astype(np.int8) - b.astype(np.int8)
        paired_vectors[(seed, snr)] = d
        boot = d[rng.integers(0, len(d), size=(3000, len(d)))].mean(axis=1)
        pair_rows.append(dict(seed=seed, snr_db=snr, delta=float(d.mean()),
                              ci_low=float(np.quantile(boot, .025)), ci_high=float(np.quantile(boot, .975)),
                              both_correct=int((a & b).sum()), full_only=int((a & ~b).sum()),
                              mask_only=int((~a & b).sum()), both_wrong=int((~a & ~b).sum())))

seed_metrics = pd.DataFrame(seed_rows)
classes = pd.DataFrame(class_rows)
confusion = pd.DataFrame(cm_rows)
paired = pd.DataFrame(pair_rows)
collapse = pd.DataFrame(collapse_rows)
for name, frame in (("逐种子逐SNR指标", seed_metrics), ("逐类指标", classes), ("完整混淆矩阵长表", confusion),
                    ("逐源配对差", paired), ("类别塌缩项", collapse)):
    frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")

agg = seed_metrics.groupby(["snr_db", "arm"])[["modacc", "macro_f1"]].agg(["mean", "std", "min", "max"])
agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
agg.reset_index().to_csv(OUT / "三种子均值标准差范围.csv", index=False, encoding="utf-8-sig")

paired_agg = paired.groupby("snr_db")["delta"].agg(["mean", "std", "min", "max"]).reset_index()
joint_rows = []
for snr in SNRS:
    ds = np.stack([paired_vectors[(seed, snr)] for seed in SEEDS])
    mean_by_source = ds.mean(0)
    boot = mean_by_source[rng.integers(0, len(mean_by_source), size=(3000, len(mean_by_source)))].mean(axis=1)
    joint_rows.append((snr, float(np.quantile(boot, .025)), float(np.quantile(boot, .975))))
joint = pd.DataFrame(joint_rows, columns=["snr_db", "fixed_3seed_source_ci_low", "fixed_3seed_source_ci_high"])
paired_agg.merge(joint, on="snr_db").to_csv(OUT / "三种子配对差汇总.csv", index=False, encoding="utf-8-sig")

pair_labels = (("PSK8", "PSK16"), ("QAM64", "QAM256"))
rows = []
for seed in SEEDS:
    for snr in SNRS:
        for arm in ("Full80", "MCS-mask80"):
            sub = confusion[(confusion.seed == seed) & (confusion.snr_db == snr) & (confusion.arm == arm)]
            for left, right in pair_labels:
                fwd = int(sub[(sub.true_label == left) & (sub.predicted_label == right)]["count"].iloc[0])
                rev = int(sub[(sub.true_label == right) & (sub.predicted_label == left)]["count"].iloc[0])
                rows.append(dict(seed=seed, snr_db=snr, arm=arm, pair=f"{left}/{right}",
                                 forward=fwd, reverse=rev, direction=fwd - rev))
pd.DataFrame(rows).to_csv(OUT / "关键双向混淆.csv", index=False, encoding="utf-8-sig")

history_rows = []
for seed in SEEDS:
    for arm in ("full", "mask"):
        d = ((ROOT.parent / "single_full_mcs_paired_20261030_20261030_170342" / "full_model") if seed == 271060 and arm == "full"
             else OLD / "mcs_mask_model" if seed == 271060 else NEW / f"seed_{seed}" / f"{arm}_model")
        cfg = json.loads((d / "training_config.json").read_text(encoding="utf-8"))
        best = json.loads((d / "best_checkpoint_metadata.json").read_text(encoding="utf-8"))
        hist = json.loads((d / "history.json").read_text(encoding="utf-8"))["history"]
        assert len(hist) == 200 and cfg["seed"] == seed and cfg["parameters"] == 3885965
        for row in hist:
            history_rows.append(dict(seed=seed, arm=arm, epoch=row["epoch"], train_loss=row["train_loss"],
                                     validation_accuracy=row["validation_accuracy"], best_epoch=best["epoch"],
                                     best_validation_accuracy=best["best"]))
pd.DataFrame(history_rows).to_csv(OUT / "训练曲线.csv", index=False, encoding="utf-8-sig")

print("身份门禁 PASS；三种子评价已写入", OUT)
print(paired.pivot(index="snr_db", columns="seed", values="delta").round(4).to_string())
print("最佳epoch：", [(r.seed, r.arm, int(r.best_epoch)) for r in pd.DataFrame(history_rows).drop_duplicates(["seed", "arm"]).itertuples()])
