from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

COSET = np.asarray([16, 11, 1, 0, 27, 24, 37, 31], dtype=np.int64)
ACTIVE_CHANNELS = np.asarray([v for p in COSET for v in (2 * p, 2 * p + 1)], dtype=np.int64)
LABELS = ["APSK16", "APSK32", "APSK64", "ASK8", "BPSK", "OQPSK", "PSK16",
          "PSK8", "QAM128", "QAM16", "QAM256", "QAM64", "QPSK"]
TRAIN_GRID = np.arange(-10.0, 12.0, 2.0, dtype=np.float32)
DEV_GRID = np.asarray([-10.0, -6.0, -2.0, 2.0, 10.0], dtype=np.float32)


def install_paths(project_root: Path):
    sys.path.insert(0, str(project_root / "experiment"))
    sys.path.insert(0, str(project_root / "shell"))
    import v2_formal_core as core
    return core


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def apply_mask(x80: np.ndarray) -> np.ndarray:
    output = np.zeros_like(x80, dtype=np.float32)
    output[ACTIVE_CHANNELS] = x80[ACTIVE_CHANNELS]
    return np.ascontiguousarray(output)


def extract_mcs16(x80: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x80.reshape(40, 2, 1024)[COSET].reshape(16, 1024), dtype=np.float32)


def normalize_original_mcs(raw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((((raw + 9348.0) / (8191.0 + 9348.0)) * 2.0 - 1.0).T, dtype=np.float32)


def selected_original_indices(full_snr: np.ndarray, original_snr: np.ndarray) -> np.ndarray:
    # Full development 只有五档，原 MCS development 仍为每源十一档。
    if len(full_snr) == len(original_snr):
        return np.arange(len(full_snr), dtype=np.int64)
    n_source = len(full_snr) // len(DEV_GRID)
    result = []
    for serial in range(n_source):
        base = serial * len(TRAIN_GRID)
        for value in DEV_GRID:
            offset = int(np.flatnonzero(np.isclose(TRAIN_GRID, value))[0])
            result.append(base + offset)
    return np.asarray(result, dtype=np.int64)

