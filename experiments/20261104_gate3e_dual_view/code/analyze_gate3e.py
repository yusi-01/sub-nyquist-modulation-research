"""审计 GATE-3E 冻结预测，并生成同源配对的 Development 指标。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[1] / "results"
ROOT = Path(__file__).resolve().parents[2]
RAW = HERE / "raw"
BASE = ROOT / "20261102_gate2_shell_confusion" / "results" / "raw"
BLIND = ROOT / "20261103_gate3d_blind_projection" / "results" / "development_predictions.csv"
SUPPORT = ROOT / "20261103_gate3c_blind_support" / "results" / "formal_development_blind" / "逐源视图摘要.csv"
LABELS = ["APSK16", "APSK32", "APSK64", "ASK8", "BPSK", "OQPSK", "PSK16", "PSK8", "QAM128", "QAM16", "QAM256", "QAM64", "QPSK"]
SNRS = list(range(-10, 11, 2))
KEY = ["source_uid", "view_uid", "noise_seed", "full_component_snr_db", "slot_1", "slot_2", "true_1", "true_2"]
ARMS = ["raw", "blind", "full", "shared_raw", "shared_projection", "fused"]
RANGES = {"low_-10_to_0": SNRS[:6], "high_2_to_10": SNRS[6:], "all_11": SNRS}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_predictions(frame: pd.DataFrame, prefix: str) -> None:
    for pos in (1, 2):
        assert frame[f"{prefix}_correct_{pos}"].eq(frame[f"true_{pos}"].eq(frame[f"{prefix}_pred_{pos}"]).astype(int)).all()
    assert frame[f"{prefix}_pair_exact"].eq(frame[f"{prefix}_correct_1"] & frame[f"{prefix}_correct_2"]).all()


def load() -> tuple[pd.DataFrame, dict, dict]:
    paths = {"raw": BASE / "mcs_development_predictions.csv", "blind": BLIND,
             "full": BASE / "full_development_predictions.csv"}
    opt = {"dtype": {"source_uid": str, "view_uid": str, "noise_seed": str}}
    frames = {name: pd.read_csv(path, **opt) for name, path in paths.items()}
    gate = pd.read_csv(RAW / "development_predictions.csv", **opt)
    ref = frames["raw"]
    assert ref.shape == (37180, 13) and gate.shape[0] == 37180
    assert ref.view_uid.is_unique and ref.source_uid.nunique() == 3380
    assert ref.groupby("source_uid").size().eq(11).all()
    assert ref.groupby("full_component_snr_db").source_uid.nunique().reindex(SNRS).eq(3380).all()
    assert ref.slot_1.lt(ref.slot_2).all()
    assert set(ref.true_1) | set(ref.true_2) == set(LABELS)
    assert gate.row_index.eq(np.arange(37180)).all() and gate[KEY].equals(ref[KEY])
    merged = ref[KEY].copy()
    for arm, frame in frames.items():
        assert frame[KEY].equals(ref[KEY]), f"{arm} 身份不一致"
        for pos in (1, 2):
            assert frame[f"correct_{pos}"].eq(frame[f"true_{pos}"].eq(frame[f"pred_{pos}"]).astype(int)).all()
            merged[f"{arm}_pred_{pos}"] = frame[f"pred_{pos}"].to_numpy()
            merged[f"{arm}_correct_{pos}"] = frame[f"correct_{pos}"].to_numpy()
        assert frame.pair_exact.eq(frame.correct_1 & frame.correct_2).all()
        merged[f"{arm}_pair_exact"] = frame.pair_exact.to_numpy()
    for branch, prefix in (("shared_raw", "raw"), ("shared_projection", "projection"), ("fused", "fused")):
        for pos in (1, 2):
            merged[f"{branch}_pred_{pos}"] = gate[f"{prefix}_pred_{pos}"].to_numpy()
            merged[f"{branch}_correct_{pos}"] = gate[f"{prefix}_correct_{pos}"].to_numpy()
        merged[f"{branch}_pair_exact"] = gate[f"{prefix}_pair_exact"].to_numpy()
        assert_predictions(merged, branch)
    sup = pd.read_csv(SUPPORT, **opt)
    sup_key = ["source_uid", "view_uid", "noise_seed", "slot_1", "slot_2"]
    assert sup.shape[0] == 37180 and sup[sup_key].equals(ref[sup_key])
    assert np.array_equal(sup.snr_db.to_numpy(), ref.full_component_snr_db.to_numpy())
    assert np.array_equal(gate[["est_1", "est_2"]].to_numpy(), sup[["est_1", "est_2"]].to_numpy())
    exact = ((sup.est_1.eq(sup.slot_1) & sup.est_2.eq(sup.slot_2)) |
             (sup.est_1.eq(sup.slot_2) & sup.est_2.eq(sup.slot_1))).astype(int)
    assert sup.support_exact.eq(exact).all()
    merged["support_exact"] = exact.to_numpy()
    merged["est_1"] = gate.est_1.to_numpy()
    merged["est_2"] = gate.est_2.to_numpy()
    merged["alpha"] = gate.alpha.to_numpy()
    merged["snr_db"] = merged.full_component_snr_db.astype(int)

    npz = np.load(RAW / "development_branch_logits.npz")
    assert set(npz.files) == {"row_index", "raw_logits", "projection_logits", "fused_logits", "alpha", "estimated_support"}
    assert np.array_equal(npz["row_index"], np.arange(37180))
    assert np.array_equal(npz["estimated_support"], gate[["est_1", "est_2"]].to_numpy())
    assert np.allclose(npz["alpha"], gate.alpha.to_numpy(), atol=5e-8, rtol=0)
    assert np.isfinite(npz["alpha"]).all() and np.all((npz["alpha"] > 0) & (npz["alpha"] <= .5))
    for name, branch in (("raw_logits", "shared_raw"), ("projection_logits", "shared_projection"), ("fused_logits", "fused")):
        x = npz[name]
        assert x.shape == (37180, 26) and np.isfinite(x).all()
        pred = np.stack((x[:, :13].argmax(1), x[:, 13:].argmax(1)), axis=1)
        expected = np.column_stack([merged[f"{branch}_pred_{pos}"].map({v: i for i, v in enumerate(LABELS)}).to_numpy() for pos in (1, 2)])
        assert np.array_equal(pred, expected)
    alpha = npz["alpha"][:, None]
    assert np.allclose(npz["fused_logits"], (1 - alpha) * npz["raw_logits"] + alpha * npz["projection_logits"], atol=2e-5, rtol=2e-5)

    history = pd.read_csv(RAW / "history.csv")
    best = json.loads((RAW / "best_validation.json").read_text(encoding="utf-8"))
    dev = json.loads((RAW / "development.json").read_text(encoding="utf-8"))
    assert history.epoch.tolist() == list(range(1, 201))
    assert np.isfinite(history.select_dtypes(include="number").to_numpy()).all()
    # 训练脚本以 >= 选点，因此并列最大值采用最后一轮。
    best_epoch = int(history.loc[history.validation_component_modacc.eq(history.validation_component_modacc.max()), "epoch"].iloc[-1])
    assert best_epoch == best["epoch"]
    row = history.loc[history.epoch.eq(best_epoch)].iloc[0]
    assert abs(row.validation_component_modacc - best["metrics"]["component_modacc"]) < 1e-12
    assert abs(row.validation_loss - best["metrics"]["loss"]) < 1e-12
    measured = score(merged, "fused")[0]
    for key, field in (("modacc", "component_modacc"), ("pair_exact", "pair_exact"), ("macro_f1", "macro_f1")):
        assert abs(measured[key] - dev[field]) < 1e-12
    meta = {"status": "PASS", "rows": 37180, "sources": 3380, "views_per_source": 11,
            "best_epoch": best_epoch, "best_validation": best, "development": dev,
            "sha256": {str(p.relative_to(ROOT)): sha(p) for p in [*paths.values(), SUPPORT, *(RAW / n for n in ["config.json", "history.csv", "best_validation.json", "best_checkpoint_mr.pt", "development.json", "development_predictions.csv", "development_branch_logits.npz"])]},
            "bootstrap_seed": 271060, "bootstrap_replicates": 2000}
    return merged, meta, dict(npz)


def score(t: pd.DataFrame, arm: str) -> tuple[dict, np.ndarray]:
    true = np.concatenate([t.true_1.to_numpy(), t.true_2.to_numpy()])
    pred = np.concatenate([t[f"{arm}_pred_1"].to_numpy(), t[f"{arm}_pred_2"].to_numpy()])
    mat = pd.crosstab(pd.Categorical(true, categories=LABELS), pd.Categorical(pred, categories=LABELS), dropna=False).to_numpy()
    support = mat.sum(1)
    predicted = mat.sum(0)
    f1 = np.divide(2 * np.diag(mat), support + predicted, out=np.zeros(13, dtype=float), where=(support + predicted) > 0)
    return {"sources": t.source_uid.nunique(), "views": len(t),
            "modacc": (t[f"{arm}_correct_1"].sum() + t[f"{arm}_correct_2"].sum()) / (2 * len(t)),
            "pair_exact": t[f"{arm}_pair_exact"].mean(), "macro_f1": f1.mean()}, mat


def main() -> None:
    d, meta, logits = load()
    d.to_csv(HERE / "六模型同源逐视图配对预测.csv", index=False)
    scores, classes, confusions, directions, aggregates, pairs, intervals, branches, margins, alpha_rows, support_rows = ([] for _ in range(11))
    ranges = {**{str(s): [s] for s in SNRS}, **RANGES}
    for name, selected in ranges.items():
        t = d[d.snr_db.isin(selected)]
        for arm in ARMS:
            sc, mat = score(t, arm)
            (scores if name in {str(s) for s in SNRS} else aggregates).append({"snr_range": name, "arm": arm, **sc})
            for i, label in enumerate(LABELS):
                n, p, tp = int(mat[i].sum()), int(mat[:, i].sum()), int(mat[i, i])
                classes.append({"snr_range": name, "arm": arm, "true_class": label, "support": n,
                                "predicted_count": p, "tp": tp, "recall": tp/n if n else np.nan,
                                "precision": tp/p if p else np.nan, "f1": 2*tp/(n+p) if n+p else 0.0})
                if name in {str(s) for s in SNRS}:
                    for j, target in enumerate(LABELS):
                        confusions.append({"snr_db": int(name), "arm": arm, "true_class": label,
                                           "pred_class": target, "count": int(mat[i, j]),
                                           "row_fraction": mat[i, j]/n if n else np.nan})
                for target in ("PSK16", "PSK8", "APSK64", "QAM64"):
                    if (label, target) in (("QPSK", "PSK16"), ("QPSK", "PSK8"), ("PSK8", "PSK16"),
                                           ("PSK16", "PSK8"), ("QAM16", "APSK64"), ("QAM64", "APSK64"),
                                           ("QAM256", "QAM64")):
                        directions.append({"snr_range": name, "arm": arm, "true_class": label,
                                           "pred_class": target, "count": int(mat[i, LABELS.index(target)]), "support": n})
        for a, b in (("fused", "raw"), ("fused", "blind"), ("fused", "shared_raw"), ("fused", "shared_projection"),
                     ("shared_raw", "shared_projection")):
            ap, bp = t[f"{a}_pair_exact"].to_numpy(), t[f"{b}_pair_exact"].to_numpy()
            pairs.append({"snr_range": name, "comparison": f"{a}-{b}", "sources": t.source_uid.nunique(),
                          "both_correct": int(((ap == 1) & (bp == 1)).sum()),
                          "d_plus": int(((ap == 1) & (bp == 0)).sum()),
                          "d_minus": int(((ap == 0) & (bp == 1)).sum()),
                          "both_wrong": int(((ap == 0) & (bp == 0)).sum()),
                          "net_pp": 100 * (ap.mean() - bp.mean())})
    rng = np.random.default_rng(271060)
    indices = rng.integers(0, 3380, size=(2000, 3380), dtype=np.int32)
    for a, b in (("fused", "raw"), ("fused", "blind"), ("fused", "shared_raw"), ("fused", "shared_projection")):
        for metric in ("pair_exact", "modacc"):
            if metric == "pair_exact":
                value = d[f"{a}_pair_exact"] - d[f"{b}_pair_exact"]
            else:
                value = (d[f"{a}_correct_1"] + d[f"{a}_correct_2"] - d[f"{b}_correct_1"] - d[f"{b}_correct_2"]) / 2
            pivot = d.assign(delta=value).pivot(index="source_uid", columns="snr_db", values="delta").reindex(columns=SNRS)
            assert pivot.shape == (3380, 11) and not pivot.isna().any().any()
            for name, selected in ranges.items():
                source = pivot[selected].mean(axis=1).to_numpy()
                boot = source[indices].mean(axis=1)
                intervals.append({"snr_range": name, "comparison": f"{a}-{b}", "metric": metric,
                                  "net_pp": 100 * source.mean(), "ci95_lo_pp": 100 * np.quantile(boot, .025),
                                  "ci95_hi_pp": 100 * np.quantile(boot, .975), "sources": 3380})
    indices_ = np.arange(len(d))
    y = np.column_stack([d[f"true_{pos}"].map({v: i for i, v in enumerate(LABELS)}).to_numpy() for pos in (1, 2)])
    for snr in SNRS:
        t = d[d.snr_db.eq(snr)]
        idx = indices_[d.snr_db.eq(snr).to_numpy()]
        alpha_rows.append({"snr_db": snr, "sources": 3380, "alpha_mean": t.alpha.mean(),
                           "alpha_p05": t.alpha.quantile(.05), "alpha_median": t.alpha.median(),
                           "alpha_p95": t.alpha.quantile(.95), "alpha_max": t.alpha.max()})
        for exact, g in t.groupby("support_exact"):
            for arm in ("shared_raw", "shared_projection", "fused"):
                sc, _ = score(g, arm)
                support_rows.append({"snr_db": snr, "support_exact": int(exact), "arm": arm,
                                     "alpha_mean": g.alpha.mean(), **sc})
        for arm, key in (("shared_raw", "raw_logits"), ("shared_projection", "projection_logits"), ("fused", "fused_logits")):
            x = logits[key][idx]
            norms = np.linalg.norm(x, axis=1)
            truth = y[idx]
            for pos in (0, 1):
                sl = x[:, pos*13:(pos+1)*13].copy()
                good = sl[np.arange(len(sl)), truth[:, pos]].copy()
                sl[np.arange(len(sl)), truth[:, pos]] = -np.inf
                margin = good - sl.max(1)
                margins.append({"snr_db": snr, "arm": arm, "position": pos+1,
                                "logit_l2_mean": norms.mean(), "logit_l2_median": np.median(norms),
                                "true_minus_max_wrong_margin_mean": margin.mean(),
                                "true_minus_max_wrong_margin_median": np.median(margin),
                                "margin_positive_fraction": (margin > 0).mean()})
        r, p, f = (t[f"{arm}_pair_exact"].to_numpy() for arm in ("shared_raw", "shared_projection", "fused"))
        branches.append({"snr_db": snr, "sources": 3380,
                         "raw_correct_projection_wrong": int(((r == 1) & (p == 0)).sum()),
                         "raw_wrong_projection_correct": int(((r == 0) & (p == 1)).sum()),
                         "both_correct": int(((r == 1) & (p == 1)).sum()),
                         "both_wrong": int(((r == 0) & (p == 0)).sum()),
                         "fused_corrects_raw": int(((f == 1) & (r == 0)).sum()),
                         "fused_damages_raw": int(((f == 0) & (r == 1)).sum()),
                         "fused_corrects_projection": int(((f == 1) & (p == 0)).sum()),
                         "fused_damages_projection": int(((f == 0) & (p == 1)).sum())})
    outputs = {"六模型逐SNR指标.csv": scores, "六模型低高SNR汇总.csv": aggregates,
               "六模型逐SNR逐类指标.csv": classes, "六模型完整13x13混淆矩阵.csv": confusions,
               "重点类别混淆方向.csv": directions, "配对PairExact四格.csv": pairs,
               "源级成组Bootstrap区间.csv": intervals, "共享双分支与融合纠错.csv": branches,
               "alpha逐SNR分布.csv": alpha_rows, "分支logit范数与margin.csv": margins,
               "支持正确错误条件指标.csv": support_rows}
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(HERE / name, index=False, encoding="utf-8-sig")
    (HERE / "GATE3E_审计摘要.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS", json.dumps({"best_epoch": meta["best_epoch"], "development": meta["development"],
                              "summary": pd.DataFrame(aggregates).to_dict(orient="records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

