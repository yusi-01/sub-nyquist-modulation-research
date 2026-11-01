"""从冻结 clean Full 与噪声 seed 分块重建 Full，逐视图核对 MCS。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

COSET = [16, 11, 1, 0, 27, 24, 37, 31]


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    assert args.data.resolve() != args.out.resolve()
    args.out.mkdir(parents=True)
    report = {}
    for split, count in (("train", 16900), ("validation", 3380), ("development_test", 3380)):
        source = args.data / "dual" / split / "dataset.h5"
        target = args.out / (split + ".h5")
        with h5py.File(source, "r") as h, h5py.File(target, "w") as o:
            assert h["clean_full_iq"].shape == (count, 40960, 2)
            assert h["mcs_iq"].shape == (count, 11, 1024, 16)
            z = o.create_dataset("noisy_full_iq", (count, 11, 40960, 2), dtype="f4", chunks=(1, 1, 40960, 2), compression="lzf")
            for i in range(count):
                clean_iq = h["clean_full_iq"][i]
                clean = clean_iq[:, 0] + 1j * clean_iq[:, 1]
                for j in range(11):
                    seed = int(h["noise_seed"][i, j])
                    power = float(h["target_noise_power_full"][i, j])
                    rng = np.random.default_rng(seed)
                    sigma = np.sqrt(power / 2.0)
                    noise = (rng.normal(0, sigma, 40960) + 1j * rng.normal(0, sigma, 40960)).astype(np.complex64)
                    full = clean + noise
                    iq = np.empty((40960, 2), np.float32)
                    iq[:, 0], iq[:, 1] = full.real, full.imag
                    mcs = full.reshape(1024, 40)[:, COSET]
                    mcs_iq = np.empty((1024, 16), np.float32)
                    mcs_iq[:, 0::2], mcs_iq[:, 1::2] = mcs.real, mcs.imag
                    assert np.array_equal(mcs_iq, h["mcs_iq"][i, j]), (split, i, j)
                    z[i, j] = iq
                if i % 500 == 0:
                    print("CACHE_PROGRESS", split, i, count, flush=True)
        report[split] = {"sources": count, "views": count * 11, "sha256": file_sha256(target), "paired_exact": True}
        print("CACHE_DONE", split, report[split], flush=True)
    (args.out / "cache_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

