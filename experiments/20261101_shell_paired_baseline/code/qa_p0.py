"""16900/3380/3380 双信号紧凑 v2 数据的只读 P0。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

SPLITS = {"train": 16900, "validation": 3380, "development_test": 3380}
GRID = np.arange(-10, 11, 2)
COSET = [16, 11, 1, 0, 27, 24, 37, 31]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--code", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    sys.path.insert(0, str(args.code))
    import v2_formal_core as core
    cfg = json.loads((args.data / "config_snapshot.json").read_text())
    assert cfg["tasks"] == ["dual"] and cfg["splits"] == list(SPLITS)
    assert cfg["master_seed"] == 271050 and cfg["physical"]["Coset"] == COSET
    assert cfg["full_component_snr_grid_db"] == GRID.tolist()
    assert cfg["store_noisy_full"] is False
    sets = {}; results = {}
    for split, count in SPLITS.items():
        path = args.data / "dual" / split / "dataset.h5"
        with h5py.File(path) as h:
            assert h.attrs["generator_version"] == core.GENERATOR_VERSION
            assert h["source_uid"].shape == (count,) and h["clean_full_iq"].shape == (count, 40960, 2)
            assert h["mcs_iq"].shape == (count, 11, 1024, 16)
            uid = set(map(bytes, h["source_uid"][:]))
            clean_hash = set(map(bytes, h["clean_full_hash"][:]))
            noise = set(map(int, h["noise_seed"][:].reshape(-1)))
            component = set(map(bytes, h["component_uid"][:].reshape(-1)))
            assert (len(uid), len(clean_hash), len(noise), len(component)) == (count, count, count * 11, count * 2)
            sets[split] = (uid, clean_hash, noise, component)
            labels = h["modulation_labels"][:].astype(int)
            slots = h["occupied_slots"][:]
            assert np.all(slots[:, 0] < slots[:, 1]) and np.all((labels >= 0) & (labels < 13))
            assert np.allclose(h["target_full_component_snr_db"][:], GRID[None, :])
            schedule = core.split_schedule("dual", split, count, 271050)["mods_component_order"].astype(int)
            before = np.bincount(schedule[:, 0] * 13 + schedule[:, 1], minlength=169)
            after = np.bincount(labels[:, 0] * 13 + labels[:, 1], minlength=169)
            assert before.min() == before.max() == count // 169
            assert np.all(np.isfinite(h["component_power_full"][:]))
            assert np.max(np.abs(h["component_power_full"][:] - 1)) < 1e-4
            for i in (0, count // 2, count - 1):
                source = core.generate_clean_source("dual", split, i, 271050, schedule[i], core.split_schedule("dual", split, count, 271050)["slots_component_order"][i])
                assert source["source_uid"] == h["source_uid"][i].decode()
                assert source["clean_full_hash"] == h["clean_full_hash"][i].decode()
                assert np.array_equal(source["occupied_slots"], slots[i])
                assert np.array_equal(source["modulation_labels"], labels[i])
                for j in (0, 5, 10):
                    view = core.generate_view(source, "dual", split, 271050, float(GRID[j] + 20))
                    assert int(view["noise_seed"]) == int(h["noise_seed"][i, j])
                    assert np.array_equal(view["mcs_iq"], h["mcs_iq"][i, j])
                    assert np.all(np.isfinite(h["mcs_iq"][i, j]))
            results[split] = {"sources": count, "views": count * 11, "h5_sha256": sha(path), "ordered_combo_min": int(after.min()), "ordered_combo_max": int(after.max()), "ordered_combo_missing": int(np.sum(after == 0)), "rebuild_sample_pass": True}
    overlap = {}
    for a, b in (("train", "validation"), ("train", "development_test"), ("validation", "development_test")):
        counts = [len(x & y) for x, y in zip(sets[a], sets[b])]
        assert counts == [0, 0, 0, 0], (a, b, counts)
        overlap[a + "__" + b] = counts
    report = {"status": "PASS", "results": results, "split_overlap_uid_clean_noise_component": overlap, "generator_sha256": sha(args.code / "v2_generate_formal_dataset.py"), "core_sha256": sha(args.code / "v2_formal_core.py")}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

