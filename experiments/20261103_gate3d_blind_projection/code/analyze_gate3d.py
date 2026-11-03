"""对冻结的 GATE-3D、Raw MCS16 和 Full80 做源级配对审计。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1] / "results"
BASE = ROOT / "20261102_gate2_shell_confusion" / "results" / "raw"
SUPPORT = ROOT / "20261103_gate3c_blind_support" / "results" / "formal_development_blind" / "逐源视图摘要.csv"
LABELS = ["APSK16", "APSK32", "APSK64", "ASK8", "BPSK", "OQPSK", "PSK16", "PSK8", "QAM128", "QAM16", "QAM256", "QAM64", "QPSK"]
SNRS = list(range(-10, 11, 2))
KEY = ["source_uid", "view_uid", "noise_seed", "full_component_snr_db", "slot_1", "slot_2", "true_1", "true_2"]
ARMS = ("raw", "blind", "full")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load() -> tuple[pd.DataFrame, dict]:
    paths = {"raw": BASE / "mcs_development_predictions.csv",
             "blind": HERE / "development_predictions.csv",
             "full": BASE / "full_development_predictions.csv"}
    opt = {"dtype": {"source_uid": str, "view_uid": str, "noise_seed": str}}
    frames = {arm: pd.read_csv(path, **opt) for arm, path in paths.items()}
    ref = frames["raw"]
    assert ref.shape == (37180, 13)
    assert ref.source_uid.nunique() == 3380 and ref.view_uid.is_unique
    assert ref.groupby("source_uid").size().eq(11).all()
    assert ref.groupby("full_component_snr_db").source_uid.nunique().reindex(SNRS).eq(3380).all()
    assert ref.slot_1.lt(ref.slot_2).all()
    assert set(ref.true_1) | set(ref.true_2) == set(LABELS)
    merged = ref[KEY].copy()
    for arm, frame in frames.items():
        assert frame.shape == ref.shape and frame[KEY].equals(ref[KEY]), f"{arm} 身份或真值不匹配"
        assert set(frame.pred_1) | set(frame.pred_2) <= set(LABELS)
        for pos in (1, 2):
            assert frame[f"correct_{pos}"].eq(frame[f"true_{pos}"].eq(frame[f"pred_{pos}"]).astype(int)).all()
            merged[f"{arm}_pred_{pos}"] = frame[f"pred_{pos}"].to_numpy()
            merged[f"{arm}_correct_{pos}"] = frame[f"correct_{pos}"].to_numpy()
        assert frame.pair_exact.eq(frame.correct_1 & frame.correct_2).all()
        merged[f"{arm}_pair_exact"] = frame.pair_exact.to_numpy()
    sup = pd.read_csv(SUPPORT, **opt)
    sup_key = ["source_uid", "view_uid", "noise_seed", "slot_1", "slot_2"]
    assert sup.shape[0] == len(ref) and sup[sup_key].equals(ref[sup_key])
    assert np.array_equal(sup.snr_db.to_numpy(), ref.full_component_snr_db.to_numpy())
    same_set = ((sup.est_1.eq(sup.slot_1) & sup.est_2.eq(sup.slot_2)) |
                (sup.est_1.eq(sup.slot_2) & sup.est_2.eq(sup.slot_1)))
    assert sup.support_exact.eq(same_set.astype(int)).all()
    merged["support_exact"] = sup.support_exact.to_numpy()
    merged["est_1"] = sup.est_1.to_numpy()
    merged["est_2"] = sup.est_2.to_numpy()
    merged["snr_db"] = merged.full_component_snr_db.astype(int)
    meta = {"status": "PASS", "rows": len(merged), "sources": ref.source_uid.nunique(),
            "views_per_source": 11, "sources_per_snr": {str(s): 3380 for s in SNRS},
            "prediction_sha256": {arm: digest(path) for arm, path in paths.items()},
            "support_sha256": digest(SUPPORT)}
    return merged, meta


def metric_table(t: pd.DataFrame, arm: str) -> tuple[dict, pd.DataFrame]:
    truth = pd.concat((t.true_1, t.true_2), ignore_index=True)
    pred = pd.concat((t[f"{arm}_pred_1"], t[f"{arm}_pred_2"]), ignore_index=True)
    matrix = pd.crosstab(truth, pred).reindex(index=LABELS, columns=LABELS, fill_value=0)
    f1 = []
    for label in LABELS:
        tp = int(matrix.loc[label, label])
        support = int(matrix.loc[label].sum())
        predicted = int(matrix[label].sum())
        f1.append(2 * tp / (support + predicted) if support + predicted else 0.0)
    return {"independent_sources": int(t.source_uid.nunique()),
            "views": len(t),
            "modacc": float((t[f"{arm}_correct_1"].sum() + t[f"{arm}_correct_2"].sum()) / (2 * len(t))),
            "pair_exact": float(t[f"{arm}_pair_exact"].mean()),
            "macro_f1": float(np.mean(f1))}, matrix


def ci(pivot: pd.DataFrame, indices: np.ndarray, snrs: list[int]) -> tuple[float, float, float]:
    source_means = pivot[snrs].mean(axis=1).to_numpy()
    boot = source_means[indices].mean(axis=1)
    return float(source_means.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    d, meta = load()
    history = pd.read_csv(HERE / "history.csv")
    assert len(history) == 200 and history.epoch.tolist() == list(range(1, 201))
    best = json.loads((HERE / "best_validation.json").read_text(encoding="utf-8"))
    dev = json.loads((HERE / "development.json").read_text(encoding="utf-8"))
    best_row = history.loc[history.validation_component_modacc.idxmax()]
    assert int(best_row.epoch) == best["epoch"]
    assert abs(best_row.validation_component_modacc - best["metrics"]["component_modacc"]) < 1e-12
    assert abs(best_row.validation_loss - best["metrics"]["loss"]) < 1e-12
    actual_dev, _ = metric_table(d, "blind")
    for field in ("component_modacc", "pair_exact", "macro_f1"):
        key = "modacc" if field == "component_modacc" else field
        assert abs(actual_dev[key] - dev[field]) < 1e-12
    meta.update({"history_sha256": digest(HERE / "history.csv"),
                 "checkpoint_sha256": digest(HERE / "best_checkpoint_mr.pt"),
                 "best_epoch": int(best_row.epoch), "best_validation": best["metrics"],
                 "development": dev, "bootstrap_seed": 271060, "bootstrap_replicates": 2000})

    scores, classes, confusions, pairs, support_rows, directions = [], [], [], [], [], []
    for snr in SNRS:
        t = d[d.snr_db.eq(snr)]
        for arm in ARMS:
            row, matrix = metric_table(t, arm)
            scores.append({"snr_db": snr, "arm": arm, **row})
            for label in LABELS:
                support = int(matrix.loc[label].sum())
                predicted = int(matrix[label].sum())
                tp = int(matrix.loc[label, label])
                classes.append({"snr_db": snr, "arm": arm, "true_class": label,
                                "support": support, "predicted_count": predicted, "tp": tp,
                                "recall": tp / support if support else np.nan,
                                "precision": tp / predicted if predicted else np.nan,
                                "f1": 2 * tp / (support + predicted) if support + predicted else 0.0})
                for target in LABELS:
                    count = int(matrix.loc[label, target])
                    confusions.append({"snr_db": snr, "arm": arm, "true_class": label,
                                       "pred_class": target, "count": count,
                                       "row_fraction": count / support if support else np.nan})
                    if (label, target) in (("QPSK", "PSK16"), ("QPSK", "PSK8"),
                                           ("PSK8", "PSK16"), ("PSK16", "PSK8"),
                                           ("QAM16", "APSK64"), ("QAM64", "APSK64"),
                                           ("QAM256", "QAM64")):
                        directions.append({"snr_db": snr, "arm": arm, "true_class": label,
                                           "pred_class": target, "count": count, "support": support})
        for a, b in (("blind", "raw"), ("full", "blind")):
            ap, bp = t[f"{a}_pair_exact"].to_numpy(), t[f"{b}_pair_exact"].to_numpy()
            pairs.append({"snr_db": snr, "comparison": f"{a}-{b}", "independent_sources": 3380,
                          "both_correct": int(((ap == 1) & (bp == 1)).sum()),
                          "d_plus": int(((ap == 1) & (bp == 0)).sum()),
                          "d_minus": int(((ap == 0) & (bp == 1)).sum()),
                          "both_wrong": int(((ap == 0) & (bp == 0)).sum())})
        for exact, group in t.groupby("support_exact"):
            row, _ = metric_table(group, "blind")
            support_rows.append({"snr_db": snr, "support_exact": int(exact), **row})

    ranges = {"all": SNRS, "low_-10_to_0": SNRS[:6], "high_2_to_10": SNRS[6:]}
    aggregates, aggregate_classes = [], []
    for name, selected in ranges.items():
        t = d[d.snr_db.isin(selected)]
        for arm in ARMS:
            row, matrix = metric_table(t, arm)
            aggregates.append({"snr_range": name, "arm": arm, **row})
            for label in LABELS:
                support = int(matrix.loc[label].sum())
                predicted = int(matrix[label].sum())
                tp = int(matrix.loc[label, label])
                aggregate_classes.append({"snr_range": name, "arm": arm, "true_class": label,
                                          "support": support, "predicted_count": predicted, "tp": tp,
                                          "recall": tp / support if support else np.nan,
                                          "precision": tp / predicted if predicted else np.nan,
                                          "f1": 2 * tp / (support + predicted) if support + predicted else 0.0})

    rng = np.random.default_rng(271060)
    indices = rng.integers(0, 3380, size=(2000, 3380), dtype=np.int32)
    intervals = []
    for a, b in (("blind", "raw"), ("full", "blind")):
        pair_pivot = d.assign(delta=d[f"{a}_pair_exact"] - d[f"{b}_pair_exact"]).pivot(
            index="source_uid", columns="snr_db", values="delta").reindex(columns=SNRS)
        mod_pivot = d.assign(delta=(d[f"{a}_correct_1"] + d[f"{a}_correct_2"] -
                                    d[f"{b}_correct_1"] - d[f"{b}_correct_2"]) / 2).pivot(
            index="source_uid", columns="snr_db", values="delta").reindex(columns=SNRS)
        assert pair_pivot.shape == mod_pivot.shape == (3380, 11)
        assert not pair_pivot.isna().any().any() and not mod_pivot.isna().any().any()
        for name, selected in {**{str(s): [s] for s in SNRS}, **ranges}.items():
            pair, plo, phi = ci(pair_pivot, indices, selected)
            mod, mlo, mhi = ci(mod_pivot, indices, selected)
            intervals.append({"comparison": f"{a}-{b}", "snr_range": name,
                              "independent_sources": 3380, "pair_delta": pair,
                              "pair_ci_low": plo, "pair_ci_high": phi,
                              "modacc_delta": mod, "modacc_ci_low": mlo, "modacc_ci_high": mhi})
    for name, selected in ranges.items():
        t = d[d.snr_db.isin(selected)]
        for exact, group in t.groupby("support_exact"):
            row, _ = metric_table(group, "blind")
            support_rows.append({"snr_db": name, "support_exact": int(exact), **row})

    wrong = d[d.support_exact.eq(0)].copy()
    wrong["spacing"] = wrong.slot_2 - wrong.slot_1
    wrong["mod_pair"] = wrong.true_1 + "+" + wrong.true_2
    keep = ["source_uid", "view_uid", "noise_seed", "snr_db", "slot_1", "slot_2", "spacing",
            "true_1", "true_2", "mod_pair", "est_1", "est_2", "raw_pair_exact", "blind_pair_exact",
            "full_pair_exact", "raw_correct_1", "raw_correct_2", "blind_correct_1", "blind_correct_2"]
    wrong[keep].to_csv(HERE / "支持估计错误逐源视图.csv", index=False)
    d.to_csv(HERE / "Raw_Blind_Full逐源配对预测.csv", index=False)
    for filename, rows in (("三模型逐SNR指标.csv", scores), ("三模型逐SNR逐类指标.csv", classes),
                           ("三模型完整13x13混淆矩阵.csv", confusions), ("重点类别混淆方向.csv", directions),
                           ("三模型低高SNR汇总.csv", aggregates), ("三模型低高SNR逐类汇总.csv", aggregate_classes),
                           ("Raw_Blind逐SNR配对四格.csv", pairs), ("源级成组Bootstrap区间.csv", intervals),
                           ("支持正确错误条件分类指标.csv", support_rows)):
        pd.DataFrame(rows).to_csv(HERE / filename, index=False)
    (HERE / "GATE3D_审计摘要.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"qa": meta["status"], "best_epoch": meta["best_epoch"],
                      "development": actual_dev, "wrong_support": len(wrong)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

