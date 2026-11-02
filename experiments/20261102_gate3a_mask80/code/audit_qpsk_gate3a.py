"""GATE-3A 训练前有界核查：仅统计冻结 Development 的 QPSK 错分。"""
from pathlib import Path
import argparse
import json

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    d = pd.read_csv(args.paired, dtype={"source_uid": str, "view_uid": str, "noise_seed": str})
    assert len(d) == 37180 and d.source_uid.nunique() == 3380
    rows = []
    for pos in (1, 2):
        z = d[d[f"true_{pos}"] == "QPSK"].copy()
        z = z.rename(columns={f"slot_{pos}": "true_slot", f"mcs_pred_{pos}": "mcs_pred",
                              f"full_pred_{pos}": "full_pred", f"full_correct_{pos}": "full_correct"})
        z["output_position"] = pos
        rows.append(z[["source_uid", "view_uid", "snr", "output_position", "true_slot",
                       "mcs_pred", "full_pred", "full_correct"]])
    q = pd.concat(rows, ignore_index=True)
    assert len(q) == 5720 and q.true_slot.between(0, 23).all()
    q["mcs_to_psk8_or_16"] = q.mcs_pred.isin(["PSK8", "PSK16"])
    q["full_correct_mcs_to_psk8_or_16"] = q.mcs_to_psk8_or_16 & q.full_correct.eq(1)
    out = q.groupby(["snr", "output_position", "true_slot", "mcs_pred"], as_index=False).agg(
        occurrences=("source_uid", "size"),
        unique_sources=("source_uid", "nunique"),
        full_correct_count=("full_correct", "sum"),
    )
    out.to_csv(args.out / "QPSK错分_逐SNR位置真实slot.csv", index=False)
    by_snr_pos = q.groupby(["snr", "output_position"], as_index=False).agg(
        qpsk_components=("source_uid", "size"),
        mcs_psk8_or_16=("mcs_to_psk8_or_16", "sum"),
        full_correct_mcs_psk8_or_16=("full_correct_mcs_to_psk8_or_16", "sum"),
    )
    by_snr_pos.to_csv(args.out / "QPSK错分_逐SNR输出位置.csv", index=False)
    by_slot = q.groupby("true_slot", as_index=False).agg(
        qpsk_components=("source_uid", "size"),
        unique_sources=("source_uid", "nunique"),
        mcs_psk8_or_16=("mcs_to_psk8_or_16", "sum"),
        full_correct_mcs_psk8_or_16=("full_correct_mcs_to_psk8_or_16", "sum"),
    )
    by_slot["mcs_confusion_rate"] = by_slot.mcs_psk8_or_16 / by_slot.qpsk_components
    by_slot.to_csv(args.out / "QPSK错分_逐真实slot.csv", index=False)
    summary = {
        "sources": int(d.source_uid.nunique()), "views": len(d), "qpsk_component_views": len(q),
        "mcs_qpsk_to_psk16": int(q.mcs_pred.eq("PSK16").sum()),
        "mcs_qpsk_to_psk8": int(q.mcs_pred.eq("PSK8").sum()),
        "mcs_qpsk_to_oqpsk": int(q.mcs_pred.eq("OQPSK").sum()),
        "full_correct_mcs_qpsk_to_psk8_or_16": int(q.full_correct_mcs_to_psk8_or_16.sum()),
        "slot_support_min": int(by_slot.qpsk_components.min()),
        "slot_support_max": int(by_slot.qpsk_components.max()),
    }
    (args.out / "QPSK有限核查摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

