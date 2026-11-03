"""冻结 Coset 的 Oracle 投影与双槽 SOMP；分类器不参与。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

N, L, Q, M = 40960, 40, 1024, 8
C = np.array([16, 11, 1, 0, 27, 24, 37, 31])
SLOTS = (np.arange(24) - 12) % L  # 由 -575 至 +575 MHz 中心在 q=512 映射。
FREQ = np.arange(Q)
D = np.exp(2j * np.pi * C[:, None] * FREQ / N)
A = np.exp(2j * np.pi * C[:, None] * SLOTS[None, :] / L) / np.sqrt(M)
LO, HI = 235, 790  # 20 Mbaud、RRC 0.35：中心 512 ± 276.48 bin，向外取整。


def complex_mcs(iq: np.ndarray) -> np.ndarray:
    """冻结 H5 的 [n,I0,Q0,...] 转为八路复 IQ/8。"""
    return (iq[:, 0::2].astype(np.float64) + 1j * iq[:, 1::2].astype(np.float64)).T / 8


def projection(y: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """给定支持的正交投影；QR 避免显式逆，输出仍是八路正交 FFT。"""
    u, _ = np.linalg.qr(A[:, slots])
    v = D.conj() * y
    return D * (u @ (u.conj().T @ v))


def blind_somp(y: np.ndarray) -> tuple[np.ndarray, int]:
    """仅从 MCS 频域观测估计无序双槽；不接收真值、SNR 或 Full。"""
    v = D[:, LO:HI].conj() * y[:, LO:HI]
    first = int(np.argmax(np.sum(np.abs(A.conj().T @ v) ** 2, axis=1)))
    residue = v - A[:, first, None] * (A[:, first].conj() @ v)[None, :]
    scores = np.sum(np.abs(A.conj().T @ residue) ** 2, axis=1)
    scores[first] = -1
    second = int(np.argmax(scores))
    return np.array([first, second], dtype=int), first


def run(path: Path, out: Path, limit: int | None, oracle_only: bool) -> None:
    out.mkdir(parents=True, exist_ok=False)
    rows, source_rows = [], []
    with h5py.File(path, "r") as h:
        count, views = h["mcs_iq"].shape[:2]
        assert views == 11 and h["clean_full_iq"].shape == (count, N, 2)
        count = min(count, limit) if limit else count
        for i in range(count):
            full = h["clean_full_iq"][i]
            slots = h["occupied_slots"][i].astype(int)
            assert 0 <= slots[0] < slots[1] < 24
            clean = complex_mcs(full.reshape(Q, L, 2)[:, C, :].reshape(Q, 16))
            yc = np.fft.fft(clean, axis=1, norm="ortho")
            oracle_clean = projection(yc, slots)
            distortion = float(np.sum(np.abs(oracle_clean - yc) ** 2) / np.sum(np.abs(yc) ** 2))
            gram = A[:, slots].conj().T @ A[:, slots]
            eigen = np.linalg.eigvalsh(gram)
            uid = h["source_uid"][i].decode()
            labels = h["modulation_labels"][i].tolist()
            source_rows.append(dict(source_uid=uid, slot_1=int(slots[0]), slot_2=int(slots[1]), spacing=int(slots[1]-slots[0]), mod_1=labels[0], mod_2=labels[1], clean_distortion=distortion, gram_eig_min=float(eigen[0]), gram_eig_max=float(eigen[1]), gram_condition=float(eigen[1]/eigen[0])))
            for j in range(views):
                raw = complex_mcs(h["mcs_iq"][i, j]); y = np.fft.fft(raw, axis=1, norm="ortho")
                oracle = projection(y, slots)
                raw_noise = float(np.sum(np.abs(y-yc)**2))
                clean_power = float(np.sum(np.abs(yc)**2))
                oe = float(np.sum(np.abs(oracle-yc)**2))
                on = float(np.sum(np.abs(oracle-oracle_clean)**2))
                row=dict(source_uid=uid, view_uid=h["view_uid"][i,j].decode(), noise_seed=int(h["noise_seed"][i,j]), snr_db=int(h["target_full_component_snr_db"][i,j]), slot_1=int(slots[0]), slot_2=int(slots[1]), spacing=int(slots[1]-slots[0]), mod_1=labels[0], mod_2=labels[1], clean_distortion=distortion, clean_power=clean_power, raw_noise_power=raw_noise, oracle_noise_power=on, oracle_error=oe, oracle_gain_db=10*np.log10(raw_noise/oe), oracle_noise_ratio=on/raw_noise)
                if not oracle_only:
                    est, first = blind_somp(y)
                    blind = projection(y, est)
                    wrong = np.array([s for s in range(24) if s not in slots][-2:])
                    bad = projection(y, wrong)
                    overlap = len(set(est) & set(slots))
                    row.update(est_1=int(est[0]),est_2=int(est[1]),first_correct=int(first in slots),second_completes=int(first in slots and overlap==2),support_overlap=overlap,support_exact=int(overlap==2),blind_error=float(np.sum(np.abs(blind-yc)**2)),wrong_error=float(np.sum(np.abs(bad-yc)**2)))
                    row.update(blind_gain_db=10*np.log10(raw_noise/row['blind_error']),wrong_gain_db=10*np.log10(raw_noise/row['wrong_error']))
                rows.append(row)
    df=pd.DataFrame(rows); sf=pd.DataFrame(source_rows)
    assert not df.duplicated(["view_uid"]).any() and df.groupby("source_uid").size().eq(11).all()
    df.to_csv(out/"逐源视图摘要.csv",index=False); sf.to_csv(out/"源级几何与失真.csv",index=False)
    spec=dict(sources=("source_uid","nunique"),views=("view_uid","size"),oracle_gain_db=("oracle_gain_db","mean"),oracle_noise_ratio=("oracle_noise_ratio","mean"),clean_distortion=("clean_distortion","mean"))
    if not oracle_only:
        spec.update(support_exact=("support_exact","mean"),at_least_one=("support_overlap",lambda x: np.mean(x>=1)),first_correct=("first_correct","mean"),second_completes=("second_completes","mean"),blind_gain_db=("blind_gain_db","mean"),wrong_gain_db=("wrong_gain_db","mean"))
    agg=df.groupby("snr_db").agg(**spec).reset_index()
    agg.to_csv(out/"逐SNR支持估计与质量.csv",index=False)
    summary={"mode":"oracle" if oracle_only else "blind","sources":count,"views":len(df),"coset":C.tolist(),"slot_to_slice":SLOTS.tolist(),"somp_q_range":[LO,HI-1],"max_clean_distortion":float(sf.clean_distortion.max()),"mean_clean_distortion":float(sf.clean_distortion.mean()),"mean_oracle_gain_db":float(df.oracle_gain_db.mean()),"mean_oracle_noise_ratio":float(df.oracle_noise_ratio.mean())}
    if not oracle_only: summary.update(mean_blind_gain_db=float(df.blind_gain_db.mean()),low_support_exact=float(df[df.snr_db<=0].support_exact.mean()))
    (out/"门禁摘要.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--h5",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--limit",type=int); ap.add_argument("--oracle-only",action="store_true")
    args=ap.parse_args(); run(args.h5,args.out,args.limit,args.oracle_only)

