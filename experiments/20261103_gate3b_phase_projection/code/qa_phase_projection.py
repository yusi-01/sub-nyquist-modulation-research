"""GATE-3B 固定相位编码的解析、合成及冻结 H5 数值门禁。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from phase_projection import COSET, FS, L, M, N, Q, PhaseProjection, encode_numpy, phase_numpy, reconstruct_mcs_numpy


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def iq(x: np.ndarray) -> np.ndarray:
    out = np.empty((Q, 2 * M), dtype=np.float32)
    out[:, 0::2], out[:, 1::2] = x.real, x.imag
    return out


def complex_encoded(z: np.ndarray) -> np.ndarray:
    return z[0::2].astype(np.float64) + 1j * z[1::2].astype(np.float64)


def synthetic_gate() -> dict:
    assert FS == 2_000_000_000 and N == 40960 and L == 40 and M == 8 and Q == 1024
    assert COSET.tolist() == [16, 11, 1, 0, 27, 24, 37, 31]
    b = phase_numpy()
    unitary_error, rank_min, rank_max = 0.0, M, M
    for q in (0, 1, Q // 2, Q - 1):
        a = b[:, q, :]
        sv = np.linalg.svd(a, compute_uv=False)
        unitary_error = max(unitary_error, float(np.max(np.abs(a @ a.conj().T - np.eye(M)))))
        rank_min = min(rank_min, int(np.linalg.matrix_rank(a)))
        rank_max = max(rank_max, int(np.linalg.matrix_rank(a)))
        assert np.max(np.abs(sv - 1)) < 1e-12
    assert unitary_error < 1e-12 and rank_min == rank_max == M

    # 频率中心是半片索引，q=512 时应与历史 A_C 正号相位逐列一致。
    centers = -575e6 + 50e6 * np.arange(24)
    k_half = centers / (FS / L)
    p = np.mod((k_half - 0.5).astype(int), L)
    old_a = np.exp(2j * np.pi * COSET[:, None] * k_half[None, :] / L) / np.sqrt(M)
    phase_error = float(np.max(np.abs(b[:, Q // 2, p] * np.sqrt(L / M) - old_a)))
    assert phase_error < 1e-12

    rng = np.random.default_rng(271060)
    full = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
    sampled = full.reshape(Q, L)[:, COSET].T.astype(np.complex128)
    full_fft = np.fft.fft(full.astype(np.complex128), norm="ortho").reshape(L, Q)
    native_fft = np.fft.fft(sampled, axis=-1, norm="ortho")
    from_full = np.einsum("mqp,pq->mq", b, full_fft, optimize=True)
    alias_error = float(np.max(np.abs(from_full - native_fft)))
    assert alias_error < 1e-11
    raw_iq = iq(sampled.T)
    encoded = encode_numpy(raw_iq)
    recovered = reconstruct_mcs_numpy(encoded)
    recovery_max_abs = float(np.max(np.abs(recovered - raw_iq)))
    recovery_rel = float(np.linalg.norm(recovered - raw_iq) / np.linalg.norm(raw_iq))
    assert recovery_max_abs < 2e-6 and recovery_rel < 2e-7
    z = complex_encoded(encoded)
    input_energy = float(np.sum(np.abs(sampled / 8.0) ** 2))
    output_energy = float(np.sum(np.abs(z) ** 2))
    energy_rel = abs(output_energy / input_energy - 1)
    assert energy_rel < 1e-7
    native_back = np.fft.ifft(native_fft, axis=-1, norm="ortho")
    ifft_error = float(np.max(np.abs(native_back - sampled)))
    assert ifft_error < 1e-12
    with torch.no_grad():
        tensor = torch.from_numpy(raw_iq.T[None].copy())
        model = PhaseProjection()
        assert sum(p.numel() for p in model.parameters()) == 0
        torch_out = model(tensor)[0].numpy()
    torch_rel = float(np.linalg.norm(torch_out - encoded) / np.linalg.norm(encoded))
    assert torch_rel < 2e-5 and torch_out.shape == (80, Q)
    return {"rank_each_q_complex": M, "right_nullity_complex": L - M,
            "nonzero_singular_values": "全部为 1", "row_unitarity_max_abs": unitary_error,
            "historical_A_C_phase_max_abs": phase_error, "full_to_mcs_dft_max_abs": alias_error,
            "fft_ifft_max_abs": ifft_error, "recovery_max_abs": recovery_max_abs,
            "recovery_relative_l2": recovery_rel, "energy_relative_error": energy_rel,
            "torch_numpy_relative_l2": torch_rel, "trainable_parameters": 0}


def h5_gate(data_root: Path, source_count: int) -> dict:
    output = {}
    for split, expected in (("train", 16900), ("validation", 3380), ("development_test", 3380)):
        path = data_root / "dual" / split / "dataset.h5"
        rows = []
        with h5py.File(path, "r") as h:
            assert h["mcs_iq"].shape == (expected, 11, Q, 2 * M)
            assert h["clean_full_iq"].shape == (expected, N, 2)
            indices = np.unique(np.linspace(0, expected - 1, source_count, dtype=int))
            for i in indices:
                clean_full = h["clean_full_iq"][int(i)]
                clean_mcs = clean_full.reshape(Q, L, 2)[:, COSET, :].reshape(Q, 2 * M)
                for j in range(11):
                    noisy = h["mcs_iq"][int(i), j]
                    encoded = encode_numpy(noisy)
                    clean_encoded = encode_numpy(clean_mcs)
                    recovered = reconstruct_mcs_numpy(encoded)
                    raw_error = float(np.max(np.abs(recovered - noisy)))
                    assert raw_error < 2e-6
                    assert np.isfinite(encoded).all()
                    noise_iq = noisy - clean_mcs
                    noise_z = complex_encoded(encoded) - complex_encoded(clean_encoded)
                    noise_mcs = noise_iq[:, 0::2] + 1j * noise_iq[:, 1::2]
                    noise_in = float(np.mean(np.abs(noise_mcs / 8.0) ** 2))
                    noise_out = float(np.mean(np.abs(noise_z) ** 2))
                    total_z = complex_encoded(encoded)
                    clean_z = complex_encoded(clean_encoded)
                    # 等距投影的 40 个候选平均噪声能量为 8 路输入均值的 1/5。
                    ratio = noise_out / noise_in
                    assert abs(ratio - M / L) < 2e-6
                    rows.append({"snr_db": int(h["target_full_component_snr_db"][int(i), j]),
                                 "recovery_max_abs": raw_error, "noise_ratio": ratio,
                                 "output_abs_max": float(np.max(np.abs(encoded))),
                                 "output_rms": float(np.sqrt(np.mean(encoded.astype(np.float64) ** 2))),
                                 "slice_total_power": np.mean(np.abs(total_z) ** 2, axis=1),
                                 "slice_clean_power": np.mean(np.abs(clean_z) ** 2, axis=1),
                                 "slice_noise_power": np.mean(np.abs(noise_z) ** 2, axis=1)})
        by_snr = {}
        for snr in range(-10, 11, 2):
            subset = [r for r in rows if r["snr_db"] == snr]
            assert len(subset) == len(indices)
            by_snr[str(snr)] = {"views": len(subset),
                                "max_recovery_abs": max(r["recovery_max_abs"] for r in subset),
                                "mean_noise_energy_ratio": float(np.mean([r["noise_ratio"] for r in subset])),
                                "mean_output_rms": float(np.mean([r["output_rms"] for r in subset])),
                                "max_output_abs": max(r["output_abs_max"] for r in subset),
                                "candidate_total_power": np.mean([r["slice_total_power"] for r in subset], axis=0).tolist(),
                                "candidate_clean_power": np.mean([r["slice_clean_power"] for r in subset], axis=0).tolist(),
                                "candidate_noise_power": np.mean([r["slice_noise_power"] for r in subset], axis=0).tolist()}
        output[split] = {"sources_sampled": len(indices), "views_checked": len(rows),
                         "per_snr": by_snr, "dataset_path": str(path)}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--source-count", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = {"synthetic": synthetic_gate(), "real_h5": None,
              "data_access": "模型编码仅使用 MCS IQ；QA 单独读取 clean Full 以测噪声，不送入编码器",
              "implementation_sha256": sha256(Path(__file__).with_name("phase_projection.py"))}
    if args.data:
        record["real_h5"] = h5_gate(args.data, args.source_count)
    record["gate"] = "PASS" if args.data else "UNKNOWN"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gate": record["gate"], "synthetic": record["synthetic"],
                      "real_splits": list(record["real_h5"] or {})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
