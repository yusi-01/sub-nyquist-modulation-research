"""位置对照输入映射门禁；仅读取现有 Full H5 的少量样本。"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

BASE = [16, 11, 1, 0, 27, 24, 37, 31]
DEST = np.array([v for c in BASE for v in (2*c, 2*c+1)], dtype=int)


def check(path, coset):
    source = np.array([v for c in coset for v in (2*c, 2*c+1)], dtype=int)
    with h5py.File(path, "r", swmr=True) as h:
        assert h["X"].shape[1:] == (80, 1024)
        indices = sorted(set([0, 1, len(h["Y"])//2, len(h["Y"])-1]))
        for idx in indices:
            x = h["X"][idx].astype(np.float32)
            mapped = np.zeros_like(x)
            mapped[DEST] = x[source]
            assert np.array_equal(mapped[DEST], x[source])
            assert np.count_nonzero(np.delete(mapped, DEST, axis=0)) == 0
            assert np.isfinite(mapped).all()
            if coset == BASE:
                expected = np.zeros_like(x)
                expected[DEST] = x[DEST]
                assert np.array_equal(mapped, expected)
    return {"samples_checked": len(indices), "shape": [80, 1024], "bitwise": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--preflight", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    assert preflight["status"] == "PASS" and preflight["candidate_search"]["selection_uses_test_accuracy"] is False
    args.output.mkdir(parents=True, exist_ok=False)
    for name, arm in preflight["arms"].items():
        coset = arm["coset_ordered"]
        result = {"status": "PASS", "name": name, "coset_ordered": coset,
                  "source_channels": [v for c in coset for v in (2*c, 2*c+1)],
                  "destination_channels": DEST.tolist(),
                  "splits": {split: check(args.data_root / f"{split}.h5", coset)
                             for split in ("train", "validation", "development_test")}}
        (args.output / f"mapping_gate_{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name, "PASS", flush=True)


if __name__ == "__main__":
    main()
