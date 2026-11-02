"""GATE-2 冻结预测的配对、混淆与双槽几何分析；不读取波形或训练模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

LABELS = ["APSK16", "APSK32", "APSK64", "ASK8", "BPSK", "OQPSK", "PSK16", "PSK8", "QAM128", "QAM16", "QAM256", "QAM64", "QPSK"]
SNRS = list(range(-10, 11, 2))
KEY = ["source_uid", "view_uid", "noise_seed", "full_component_snr_db", "slot_1", "slot_2", "true_1", "true_2"]
RHO_BINS = [-0.001, 0.2, 0.4, 0.601]
RHO_NAMES = ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6]"]
SPACING_BINS = [0, 4, 8, 12, 23]
SPACING_NAMES = ["1–4", "5–8", "9–12", "13–23"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ci(values: np.ndarray, sample_indices: np.ndarray) -> tuple[float, float]:
    means = values[sample_indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def check_manifest(pred: pd.DataFrame, manifest_path: Path) -> None:
    rows = [json.loads(line) for line in manifest_path.open(encoding="utf-8")]
    assert len(rows) == 3380
    assert pred.shape[0] == 37180 and pred.source_uid.nunique() == 3380
    assert pred.view_uid.is_unique
    assert pred.groupby("source_uid").size().eq(11).all()
    assert pred.groupby("full_component_snr_db").size().reindex(SNRS).eq(3380).all()
    assert pred.slot_1.lt(pred.slot_2).all()
    assert set(pred.true_1) | set(pred.true_2) == set(LABELS)
    assert set(pred.pred_1) | set(pred.pred_2) <= set(LABELS)
    by_uid = pred.set_index(["source_uid", "view_uid"])
    for row in rows:
        uid = row["source_uid"]
        slots = row["occupied_slots"]
        labels = [LABELS[k] for k in row["modulation_labels"]]
        assert len(row["views"]) == 11
        for view in row["views"]:
            p = by_uid.loc[(uid, view["view_uid"])]
            assert str(p.noise_seed) == str(view["noise_seed"])
            assert int(p.full_component_snr_db) == int(view["target_full_component_snr_db"])
            assert [int(p.slot_1), int(p.slot_2)] == slots
            assert [p.true_1, p.true_2] == labels


def prepare(full_path: Path, mcs_path: Path, manifest_path: Path, geom_path: Path) -> tuple[pd.DataFrame, dict]:
    opts = {"dtype": {"source_uid": str, "view_uid": str, "noise_seed": str}}
    f = pd.read_csv(full_path, **opts)
    m = pd.read_csv(mcs_path, **opts)
    assert f.shape == m.shape == (37180, 13)
    assert f[KEY].equals(m[KEY]), "Full/MCS 身份、SNR、slot 或真标签不一致"
    for arm, a in [("full", f), ("mcs", m)]:
        check_manifest(a, manifest_path)
        for k in (1, 2):
            assert (a[f"correct_{k}"] == (a[f"true_{k}"] == a[f"pred_{k}"]).astype(int)).all(), arm
        assert (a.pair_exact == (a.correct_1 & a.correct_2)).all(), arm
    d = f[KEY].copy()
    for arm, a in [("full", f), ("mcs", m)]:
        for col in ["pred_1", "pred_2", "correct_1", "correct_2", "pair_exact"]:
            d[f"{arm}_{col}"] = a[col].to_numpy()
    d["snr"] = d.full_component_snr_db.astype(int)
    d["mod_pair"] = d.true_1 + "/" + d.true_2
    d["same_class"] = d.true_1.eq(d.true_2)
    d["pair_case"] = np.select(
        [(d.full_pair_exact == 1) & (d.mcs_pair_exact == 1),
         (d.full_pair_exact == 1) & (d.mcs_pair_exact == 0),
         (d.full_pair_exact == 0) & (d.mcs_pair_exact == 1)],
        ["both_correct", "d_plus", "d_minus"], default="both_wrong")
    d["pair_delta"] = d.full_pair_exact - d.mcs_pair_exact
    d["mod_delta"] = (d.full_correct_1 + d.full_correct_2 - d.mcs_correct_1 - d.mcs_correct_2) / 2
    g = pd.read_csv(geom_path).query("arm == 'baseline'").copy()
    assert len(g) == 276 and not g.duplicated(["slot_i", "slot_j"]).any()
    assert g.rho_abs.max() < 0.581 and g.eta.max() < 1.509
    d = d.merge(g[["slot_i", "slot_j", "spacing_slots", "spacing_mhz", "rho_abs", "eta"]],
                left_on=["slot_1", "slot_2"], right_on=["slot_i", "slot_j"], validate="many_to_one")
    assert len(d) == 37180 and d.rho_abs.notna().all()
    d["rho_bin"] = pd.cut(d.rho_abs, RHO_BINS, labels=RHO_NAMES, right=False)
    d["spacing_bin"] = pd.cut(d.spacing_slots, SPACING_BINS, labels=SPACING_NAMES)
    assert d.rho_bin.notna().all() and d.spacing_bin.notna().all()
    source = d[d.snr == -10].sort_values("source_uid")
    assert source.source_uid.nunique() == 3380
    max_within_spacing_range = float(g.groupby("spacing_slots").rho_abs.agg(lambda x: x.max() - x.min()).max())
    qa = {"rows": len(d), "sources": d.source_uid.nunique(), "snrs": SNRS,
          "per_snr_sources": d.groupby("snr").source_uid.nunique().to_dict(),
          "sha256": {"full_predictions": sha256(full_path), "mcs_predictions": sha256(mcs_path),
                     "manifest": sha256(manifest_path), "geometry": sha256(geom_path)},
          "geometry_pairs": len(g), "geometry_rho_max": float(g.rho_abs.max()),
          "geometry_eta_max": float(g.eta.max()),
          "max_rho_range_within_exact_spacing": max_within_spacing_range}
    return d, qa


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("full", "mcs", "manifest", "geometry", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d, qa = prepare(a.full, a.mcs, a.manifest, a.geometry)
    d.to_csv(a.out / "逐视图配对与几何.csv", index=False)
    (a.out / "质量门禁.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")

    # 相同 3380 个源跨 11 档重采样，避免把多档噪声视图当成独立源。
    rng = np.random.default_rng(271060)
    base = d.pivot(index="source_uid", columns="snr", values="pair_delta").reindex(columns=SNRS)
    mod = d.pivot(index="source_uid", columns="snr", values="mod_delta").reindex(columns=SNRS)
    assert base.shape == mod.shape == (3380, 11) and not base.isna().any().any()
    ix = rng.integers(0, 3380, size=(2000, 3380), dtype=np.int32)
    snr_rows = []
    for j, snr in enumerate(SNRS):
        t = d[d.snr == snr]
        counts = t.pair_case.value_counts()
        row = {"snr_db": snr, "independent_sources": 3380, **{k: int(counts.get(k, 0)) for k in
                 ["both_correct", "d_plus", "d_minus", "both_wrong"]}}
        for arm in ("full", "mcs"):
            row[f"{arm}_pair_exact"] = t[f"{arm}_pair_exact"].mean()
            row[f"{arm}_modacc"] = (t[f"{arm}_correct_1"].sum() + t[f"{arm}_correct_2"].sum()) / (2 * len(t))
        row["pair_net_delta"] = row["d_plus"] / 3380 - row["d_minus"] / 3380
        row["modacc_net_delta"] = row["full_modacc"] - row["mcs_modacc"]
        row["pair_ci_low"], row["pair_ci_high"] = ci(base.iloc[:, j].to_numpy(), ix)
        row["modacc_ci_low"], row["modacc_ci_high"] = ci(mod.iloc[:, j].to_numpy(), ix)
        snr_rows.append(row)
    pd.DataFrame(snr_rows).to_csv(a.out / "逐SNR配对四格与源级区间.csv", index=False)

    range_rows = []
    for title, selected in [("all", SNRS), ("low_-10_to_0", SNRS[:6]), ("high_2_to_10", SNRS[6:])]:
        bp = base[selected].mean(axis=1).to_numpy()
        bm = mod[selected].mean(axis=1).to_numpy()
        t = d[d.snr.isin(selected)]
        c = t.pair_case.value_counts()
        range_rows.append({"range": title, "independent_sources": 3380, "views": len(t),
                           **{k: int(c.get(k, 0)) for k in ["both_correct", "d_plus", "d_minus", "both_wrong"]},
                           "pair_delta": bp.mean(), "pair_ci_low": ci(bp, ix)[0], "pair_ci_high": ci(bp, ix)[1],
                           "mod_delta": bm.mean(), "mod_ci_low": ci(bm, ix)[0], "mod_ci_high": ci(bm, ix)[1]})
    pd.DataFrame(range_rows).to_csv(a.out / "低高SNR源级配对区间.csv", index=False)

    class_rows, confusion_rows, error_rows, direction_rows, pair_case_class_rows = [], [], [], [], []
    for snr in SNRS:
        t = d[d.snr == snr]
        for arm in ("full", "mcs"):
            truths = pd.concat([t.true_1, t.true_2], ignore_index=True)
            preds = pd.concat([t[f"{arm}_pred_1"], t[f"{arm}_pred_2"]], ignore_index=True)
            mat = pd.crosstab(truths, preds).reindex(index=LABELS, columns=LABELS, fill_value=0)
            for true in LABELS:
                tp, support = int(mat.loc[true, true]), int(mat.loc[true].sum())
                predicted = int(mat[true].sum())
                recall = tp / support if support else np.nan
                precision = tp / predicted if predicted else np.nan
                f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
                class_rows.append({"snr_db": snr, "arm": arm, "true_class": true, "support": support,
                                   "predicted_count": predicted, "tp": tp, "recall": recall,
                                   "precision": precision, "f1": f1})
                for pred in LABELS:
                    confusion_rows.append({"snr_db": snr, "arm": arm, "true_class": true,
                                           "pred_class": pred, "count": int(mat.loc[true, pred]),
                                           "row_fraction": int(mat.loc[true, pred]) / support if support else np.nan})
        for true in LABELS:
            out = {"snr_db": snr, "true_class": true, "support": 0,
                   "full_only_component": 0, "mcs_only_component": 0,
                   "both_correct_component": 0, "both_wrong_component": 0}
            for k in (1, 2):
                z = t[t[f"true_{k}"] == true]
                fc, mc = z[f"full_correct_{k}"].eq(1), z[f"mcs_correct_{k}"].eq(1)
                out["support"] += len(z)
                out["full_only_component"] += int((fc & ~mc).sum())
                out["mcs_only_component"] += int((~fc & mc).sum())
                out["both_correct_component"] += int((fc & mc).sum())
                out["both_wrong_component"] += int((~fc & ~mc).sum())
                for case, mask, pred_arm in [("full_only", fc & ~mc, "mcs"),
                                             ("mcs_only", ~fc & mc, "full"),
                                             ("both_wrong_mcs", ~fc & ~mc, "mcs"),
                                             ("both_wrong_full", ~fc & ~mc, "full")]:
                    vc = z.loc[mask, f"{pred_arm}_pred_{k}"].value_counts()
                    for pred in LABELS:
                        direction_rows.append({"snr_db": snr, "true_class": true, "case": case,
                                               "pred_class": pred, "count": int(vc.get(pred, 0))})
            out["component_net_delta"] = (out["full_only_component"] - out["mcs_only_component"]) / out["support"]
            error_rows.append(out)
            for pair_case in ("d_plus", "d_minus", "both_wrong"):
                w = t[t.pair_case == pair_case]
                for k in (1, 2):
                    v = w[w[f"true_{k}"] == true]
                    pair_case_class_rows.append({"snr_db": snr, "pair_case": pair_case,
                                                 "true_class": true, "output_position": k,
                                                 "component_occurrences": len(v),
                                                 "full_component_wrong": int(v[f"full_correct_{k}"].eq(0).sum()),
                                                 "mcs_component_wrong": int(v[f"mcs_correct_{k}"].eq(0).sum())})
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(a.out / "逐SNR逐类Recall_Precision_F1.csv", index=False)
    confusion_df = pd.DataFrame(confusion_rows)
    confusion_df.to_csv(a.out / "完整13x13混淆矩阵长表.csv", index=False)
    aggregate_class_rows = []
    for title, selected in [("all", SNRS), ("low", SNRS[:6]), ("high", SNRS[6:])]:
        for arm in ("full", "mcs"):
            mat = confusion_df[(confusion_df.snr_db.isin(selected)) & (confusion_df.arm == arm)]
            mat = mat.pivot_table(index="true_class", columns="pred_class", values="count", aggfunc="sum", fill_value=0)
            for true in LABELS:
                tp, support, predicted = int(mat.loc[true, true]), int(mat.loc[true].sum()), int(mat[true].sum())
                aggregate_class_rows.append({"range": title, "arm": arm, "true_class": true,
                                             "support": support, "tp": tp, "precision": tp / predicted if predicted else np.nan,
                                             "recall": tp / support if support else np.nan,
                                             "f1": 2 * tp / (support + predicted) if support + predicted else 0.0})
    pd.DataFrame(aggregate_class_rows).to_csv(a.out / "总体低高SNR逐类指标.csv", index=False)
    pd.DataFrame(error_rows).to_csv(a.out / "逐SNR逐类配对错误四格.csv", index=False)
    pd.DataFrame(direction_rows).to_csv(a.out / "重点混淆方向全类长表.csv", index=False)
    pd.DataFrame(pair_case_class_rows).to_csv(a.out / "逐SNR_Pair四格内逐类错误.csv", index=False)
    class_df.groupby(["snr_db", "arm"]).f1.mean().rename("macro_f1").to_csv(a.out / "逐SNR_MacroF1.csv")

    # 同类/异类与输出位置按每个 SNR 分层，分母均为独立 source_uid。
    sub_rows = []
    for snr in SNRS:
        t = d[d.snr == snr]
        for same, z in t.groupby("same_class"):
            c = z.pair_case.value_counts()
            sub_rows.append({"snr_db": snr, "same_class": bool(same), "independent_sources": len(z),
                             "full_pair_exact": z.full_pair_exact.mean(), "mcs_pair_exact": z.mcs_pair_exact.mean(),
                             "d_plus": int(c.get("d_plus", 0)), "d_minus": int(c.get("d_minus", 0)),
                             "pair_net_delta": z.pair_delta.mean()})
    pd.DataFrame(sub_rows).to_csv(a.out / "同类异类逐SNR_PairExact.csv", index=False)
    pos_rows = []
    for snr in SNRS:
        t = d[d.snr == snr]
        for k in (1, 2):
            fc, mc = t[f"full_correct_{k}"].eq(1), t[f"mcs_correct_{k}"].eq(1)
            pos_rows.append({"snr_db": snr, "output_position": k, "independent_sources": len(t),
                             "full_modacc": fc.mean(), "mcs_modacc": mc.mean(),
                             "full_only": int((fc & ~mc).sum()), "mcs_only": int((~fc & mc).sum()),
                             "net_delta": fc.mean() - mc.mean()})
    pd.DataFrame(pos_rows).to_csv(a.out / "两个输出位置逐SNR准确率.csv", index=False)
    pos_ci_rows = []
    for arm in ("full", "mcs"):
        v = d.assign(pos_delta=d[f"{arm}_correct_1"] - d[f"{arm}_correct_2"])
        v = v.groupby("source_uid").pos_delta.mean().sort_index().to_numpy()
        lo, hi = ci(v, ix)
        pos_ci_rows.append({"arm": arm, "independent_sources": len(v),
                            "position_1_minus_2": v.mean(), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(pos_ci_rows).to_csv(a.out / "输出位置源级差异区间.csv", index=False)

    # 源级汇总可直接复核几何关联，11 个视图只贡献一个源记录。
    source_cols = ["source_uid", "slot_1", "slot_2", "true_1", "true_2", "mod_pair", "same_class",
                   "spacing_slots", "spacing_bin", "rho_abs", "rho_bin", "eta"]
    sources = d[d.snr == -10][source_cols].sort_values("source_uid").reset_index(drop=True)
    geometry_ranges = [("all", SNRS), ("low", SNRS[:6]), ("high", SNRS[6:])]
    geometry_ranges += [(f"snr_{snr}", [snr]) for snr in SNRS]
    for title, selected in geometry_ranges:
        x = d[d.snr.isin(selected)].groupby("source_uid")[["pair_delta", "mod_delta"]].mean()
        sources[f"{title}_pair_delta"] = sources.source_uid.map(x.pair_delta)
        sources[f"{title}_mod_delta"] = sources.source_uid.map(x.mod_delta)
    sources.to_csv(a.out / "逐源配对差与双槽几何.csv", index=False)

    geom_rows = []
    for title, selected in [("all", SNRS), ("low", SNRS[:6]), ("high", SNRS[6:])]:
        z = d[d.snr.isin(selected)]
        for kind in ("rho_bin", "spacing_bin"):
            for group, t in z.groupby(kind, observed=True):
                s = t.groupby("source_uid")[["pair_delta", "mod_delta"]].mean()
                c = t.pair_case.value_counts()
                v = s.pair_delta.to_numpy()
                bx = rng.integers(0, len(v), size=(1000, len(v)), dtype=np.int32)
                lo, hi = ci(v, bx)
                geom_rows.append({"range": title, "stratifier": kind, "bin": str(group),
                                  "independent_sources": len(s), "views": len(t),
                                  "full_pair_exact": t.full_pair_exact.mean(), "mcs_pair_exact": t.mcs_pair_exact.mean(),
                                  "d_plus": int(c.get("d_plus", 0)), "d_minus": int(c.get("d_minus", 0)),
                                  "pair_net_delta": s.pair_delta.mean(), "mod_net_delta": s.mod_delta.mean(),
                                  "pair_ci_low": lo, "pair_ci_high": hi})
    pd.DataFrame(geom_rows).to_csv(a.out / "相关性和间距预固定分层.csv", index=False)

    # SNR 已在源内平均，调制有序组合和粗间距固定效应控制可观测混杂。
    regress = []
    for title in ("all", "low", "high"):
        for outcome in ("pair", "mod"):
            fit = smf.ols(f"{title}_{outcome}_delta ~ rho_abs + C(mod_pair) + C(spacing_bin)", data=sources).fit(cov_type="HC3")
            q = fit.conf_int().loc["rho_abs"]
            regress.append({"range": title, "outcome": outcome, "sources": len(sources),
                            "rho_coefficient_per_0p1": fit.params["rho_abs"] / 10,
                            "ci_low": q.iloc[0] / 10, "ci_high": q.iloc[1] / 10,
                            "p_value_descriptive": fit.pvalues["rho_abs"], "r_squared": fit.rsquared,
                            "controls": "ordered modulation pair fixed effects + coarse spacing bin"})
    pd.DataFrame(regress).to_csv(a.out / "控制调制组合和粗间距后的相关性关联.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    snr_table = pd.read_csv(a.out / "逐SNR配对四格与源级区间.csv")
    axes[0].plot(snr_table.snr_db, snr_table.pair_net_delta * 100, marker="o", label="PairExact")
    axes[0].fill_between(snr_table.snr_db.to_numpy(), snr_table.pair_ci_low.to_numpy() * 100,
                         snr_table.pair_ci_high.to_numpy() * 100, alpha=.2)
    axes[0].plot(snr_table.snr_db, snr_table.modacc_net_delta * 100, marker="s", label="ModAcc")
    axes[0].set(xlabel="Full component SNR (dB)", ylabel="Full - MCS (percentage points)")
    axes[0].legend(); axes[0].grid(alpha=.25)
    low = pd.read_csv(a.out / "逐SNR逐类配对错误四格.csv").query("snr_db <= 0")
    low = low.groupby("true_class")[["full_only_component", "mcs_only_component"]].sum().reindex(LABELS)
    axes[1].barh(np.arange(13) - .18, low.full_only_component, height=.36, label="Full correct, MCS wrong")
    axes[1].barh(np.arange(13) + .18, low.mcs_only_component, height=.36, label="MCS correct, Full wrong")
    axes[1].set_yticks(np.arange(13), LABELS); axes[1].invert_yaxis()
    axes[1].set(xlabel="Component-view count, SNR <= 0 dB"); axes[1].legend(fontsize=8)
    fig.savefig(a.out / "低SNR配对差与类别方向.png", dpi=170)
    fig.savefig(a.out / "低SNR配对差与类别方向.svg")
    plt.close(fig)
    print(json.dumps({"gate": "PASS", **qa, "results": str(a.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

