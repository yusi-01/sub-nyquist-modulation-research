"""冻结三个 Development 预测后执行 GATE-3A 同源配对分析。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["APSK16", "APSK32", "APSK64", "ASK8", "BPSK", "OQPSK", "PSK16", "PSK8", "QAM128", "QAM16", "QAM256", "QAM64", "QPSK"]
SNRS = list(range(-10, 11, 2))
KEY = ["source_uid", "view_uid", "noise_seed", "full_component_snr_db", "slot_1", "slot_2", "true_1", "true_2"]
ARMS = ("full", "mcs", "mask")
PAIRS = (("full", "mcs"), ("full", "mask"), ("mcs", "mask"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_check(paths: dict[str, Path]) -> tuple[pd.DataFrame, dict]:
    opts = {"dtype": {"source_uid": str, "view_uid": str, "noise_seed": str}}
    frames = {name: pd.read_csv(path, **opts) for name, path in paths.items()}
    reference = frames["full"]
    assert reference.shape == (37180, 13)
    assert reference.source_uid.nunique() == 3380 and reference.view_uid.is_unique
    assert reference.groupby("source_uid").size().eq(11).all()
    assert reference.groupby("full_component_snr_db").source_uid.nunique().reindex(SNRS).eq(3380).all()
    assert reference.slot_1.lt(reference.slot_2).all()
    assert set(reference.true_1) | set(reference.true_2) == set(LABELS)
    assert reference[KEY].duplicated().sum() == 0
    out = reference[KEY].copy()
    for name, frame in frames.items():
        assert frame.shape == reference.shape and frame[KEY].equals(reference[KEY]), f"{name} 的身份/真值与 Full 不符"
        assert set(frame.pred_1) | set(frame.pred_2) <= set(LABELS)
        for pos in (1, 2):
            assert frame[f"correct_{pos}"].eq(frame[f"true_{pos}"].eq(frame[f"pred_{pos}"]).astype(int)).all()
        assert frame.pair_exact.eq(frame.correct_1 & frame.correct_2).all()
        for field in ("pred_1", "pred_2", "correct_1", "correct_2", "pair_exact"):
            out[f"{name}_{field}"] = frame[field].to_numpy()
    out["snr_db"] = out.full_component_snr_db.astype(int)
    qa = {"gate": "PASS", "rows": len(out), "sources": 3380,
          "per_snr_sources": {str(s): int(out.loc[out.snr_db.eq(s), "source_uid"].nunique()) for s in SNRS},
          "prediction_sha256": {name: sha256(path) for name, path in paths.items()}}
    return out, qa


def grouped_ci(pivot: pd.DataFrame, bootstrap_index: np.ndarray, selected: list[int]) -> tuple[float, float, float]:
    source_values = pivot[selected].mean(axis=1).to_numpy()
    boot = source_values[bootstrap_index].mean(axis=1)
    return float(source_values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def analyze(d: pd.DataFrame, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(271060)
    bootstrap_index = rng.integers(0, 3380, size=(2000, 3380), dtype=np.int32)
    score_rows, class_rows, confusion_rows, pair_rows, direction_rows = [], [], [], [], []
    for snr in SNRS:
        t = d[d.snr_db.eq(snr)]
        truth = pd.concat((t.true_1, t.true_2), ignore_index=True)
        for arm in ARMS:
            pred = pd.concat((t[f"{arm}_pred_1"], t[f"{arm}_pred_2"]), ignore_index=True)
            matrix = pd.crosstab(truth, pred).reindex(index=LABELS, columns=LABELS, fill_value=0)
            f1_values = []
            for label in LABELS:
                support = int(matrix.loc[label].sum())
                predicted = int(matrix[label].sum())
                tp = int(matrix.loc[label, label])
                recall = tp / support if support else np.nan
                precision = tp / predicted if predicted else np.nan
                f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
                f1_values.append(f1)
                class_rows.append({"snr_db": snr, "arm": arm, "true_class": label,
                                   "support": support, "predicted_count": predicted,
                                   "tp": tp, "recall": recall, "precision": precision, "f1": f1})
                for target in LABELS:
                    count = int(matrix.loc[label, target])
                    confusion_rows.append({"snr_db": snr, "arm": arm, "true_class": label,
                                           "pred_class": target, "count": count,
                                           "row_fraction": count / support if support else np.nan})
            score_rows.append({"snr_db": snr, "arm": arm, "independent_sources": 3380,
                               "modacc": (t[f"{arm}_correct_1"].sum() + t[f"{arm}_correct_2"].sum()) / (2 * len(t)),
                               "pair_exact": t[f"{arm}_pair_exact"].mean(),
                               "macro_f1": float(np.mean(f1_values))})
            for label in ("QPSK", "QAM16", "QAM64", "QAM256", "PSK8", "PSK16"):
                for target in LABELS:
                    if target == label:
                        continue
                    direction_rows.append({"snr_db": snr, "arm": arm, "true_class": label,
                                           "pred_class": target, "count": int(matrix.loc[label, target]),
                                           "support": int(matrix.loc[label].sum())})
        for a, b in PAIRS:
            ap, bp = t[f"{a}_pair_exact"].to_numpy(), t[f"{b}_pair_exact"].to_numpy()
            pair_rows.append({"snr_db": snr, "comparison": f"{a}-{b}", "independent_sources": 3380,
                              "both_correct": int(((ap == 1) & (bp == 1)).sum()),
                              "d_plus": int(((ap == 1) & (bp == 0)).sum()),
                              "d_minus": int(((ap == 0) & (bp == 1)).sum()),
                              "both_wrong": int(((ap == 0) & (bp == 0)).sum())})
    pd.DataFrame(score_rows).to_csv(output / "三模型逐SNR指标.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "三模型逐SNR逐类指标.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output / "三模型完整13x13混淆矩阵.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output / "三组比较逐SNR配对四格.csv", index=False)
    pd.DataFrame(direction_rows).to_csv(output / "重点类别混淆方向.csv", index=False)
    interval_rows = []
    for a, b in PAIRS:
        pair_pivot = d.assign(delta=d[f"{a}_pair_exact"] - d[f"{b}_pair_exact"]).pivot(
            index="source_uid", columns="snr_db", values="delta").reindex(columns=SNRS)
        mod_pivot = d.assign(delta=(d[f"{a}_correct_1"] + d[f"{a}_correct_2"] -
                                    d[f"{b}_correct_1"] - d[f"{b}_correct_2"]) / 2).pivot(
            index="source_uid", columns="snr_db", values="delta").reindex(columns=SNRS)
        assert pair_pivot.shape == mod_pivot.shape == (3380, 11)
        assert not pair_pivot.isna().any().any() and not mod_pivot.isna().any().any()
        for group_name, selected in [("all", SNRS), ("low_-10_to_0", SNRS[:6]), ("high_2_to_10", SNRS[6:])]:
            pair, pair_lo, pair_hi = grouped_ci(pair_pivot, bootstrap_index, selected)
            mod, mod_lo, mod_hi = grouped_ci(mod_pivot, bootstrap_index, selected)
            interval_rows.append({"comparison": f"{a}-{b}", "snr_range": group_name,
                                  "independent_sources": 3380, "pair_delta": pair,
                                  "pair_ci_low": pair_lo, "pair_ci_high": pair_hi,
                                  "modacc_delta": mod, "modacc_ci_low": mod_lo, "modacc_ci_high": mod_hi})
    pd.DataFrame(interval_rows).to_csv(output / "三组比较源级成组区间.csv", index=False)
    collapse = []
    for arm in ARMS:
        predictions = pd.concat((d[f"{arm}_pred_1"], d[f"{arm}_pred_2"]), ignore_index=True)
        for label in LABELS:
            collapse.append({"arm": arm, "class": label, "predicted_count": int(predictions.eq(label).sum()),
                             "predicted_fraction": float(predictions.eq(label).mean())})
    pd.DataFrame(collapse).to_csv(output / "类别预测覆盖与塌缩检查.csv", index=False)
    low = d[d.snr_db.le(0)]
    emphasis = {}
    for arm in ARMS:
        for target in ("PSK16", "PSK8", "OQPSK"):
            count = sum(int((low[f"true_{pos}"].eq("QPSK") & low[f"{arm}_pred_{pos}"].eq(target)).sum())
                        for pos in (1, 2))
            emphasis[f"{arm}_QPSK_to_{target}_low_views"] = count
    return {"low_snr_focus": emphasis, "bootstrap_resamples": 2000,
            "interval_unit": "source_uid", "checkpoint_selection": "Validation only"}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (*ARMS, "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ARMS}
    d, qa = load_and_check(paths)
    summary = analyze(d, args.out)
    qa.update(summary)
    (args.out / "GATE3A质量门禁与摘要.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

