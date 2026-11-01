"""MCS-AMC-AWGN13 v2 正式候选数据的冻结物理内核与随机调度。

本模块只定义信号、标签、随机流和 SNR 换算；正式 SNR 网格仍由命令行显式给出。
禁止把 source_uid、SNR 或其他元数据作为模型输入。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import upfirdn


GENERATOR_VERSION = "mcs_amc_awgn13_v2_formal_candidate_20261028"
FS = 2.0e9
L = 40
M = 8
COSET = np.asarray([16, 11, 1, 0, 27, 24, 37, 31], dtype=np.int64)
N_FULL = 40960
N_MCS = 1024
RS = 20.0e6
SPS = 100
ALPHA = 0.35
RRC_SPAN = 8
TIMING = 0
SLOT_BW = 50.0e6
CENTERS = np.asarray([-575.0e6 + SLOT_BW * k for k in range(24)], dtype=np.float64)
MODULATIONS = [
    "APSK16", "APSK32", "APSK64", "ASK8", "BPSK", "OQPSK", "PSK16",
    "PSK8", "QAM128", "QAM16", "QAM256", "QAM64", "QPSK",
]
DEFAULT_DEVELOPMENT_ESN0_GRID = np.asarray(
    [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    dtype=np.float64,
)

# 对当前白噪声生成方式，MCS 各支路在期望上的每分量采样点 SNR 与
# Full 域每分量采样点 SNR 相同。显式提供这一口径，避免把 Es/N0
# 误当成模型入口处的 SNR；二者相差 10log10(Fs/Rs)=20 dB。
DEFAULT_DEVELOPMENT_MCS_COMPONENT_SNR_GRID = np.arange(-10.0, 12.0, 2.0, dtype=np.float64)


def stable_seed(*parts: object) -> int:
    """由命名字段生成稳定的 64 bit seed，避免随机流互相复用。"""
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def source_uid(master_seed: int, task: str, split: str, serial: int) -> str:
    payload = f"{GENERATOR_VERSION}|{master_seed}|{task}|{split}|{serial}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def component_uid(uid: str, component_index: int) -> str:
    payload = f"{GENERATOR_VERSION}|{uid}|component|{component_index}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def view_uid(uid: str, esn0_db: float) -> str:
    payload = f"{GENERATOR_VERSION}|{uid}|esn0_db|{float(esn0_db):+.9f}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def named_seed(master_seed: int, task: str, split: str, uid: str, stream: str) -> int:
    return stable_seed(GENERATOR_VERSION, master_seed, task, split, uid, stream)


def esn0_to_full_component_snr_db(esn0_db: np.ndarray | float) -> np.ndarray:
    """Es/N0 转为每路 Full 域每复采样点 component SNR。"""
    return np.asarray(esn0_db, dtype=np.float64) - 10.0 * np.log10(FS / RS)


def full_component_to_esn0_db(full_snr_db: np.ndarray | float) -> np.ndarray:
    return np.asarray(full_snr_db, dtype=np.float64) + 10.0 * np.log10(FS / RS)


def mcs_component_snr_to_esn0_db(mcs_snr_db: np.ndarray | float) -> np.ndarray:
    """将模型入口处的目标每分量 MCS SNR 转为生成器使用的 Es/N0。"""
    return full_component_to_esn0_db(mcs_snr_db)


def esn0_to_mcs_component_snr_db(esn0_db: np.ndarray | float) -> np.ndarray:
    """将生成器 Es/N0 转为模型入口处期望的每分量 MCS SNR。"""
    return esn0_to_full_component_snr_db(esn0_db)


def full_component_to_slot50_snr_db(full_snr_db: np.ndarray | float) -> np.ndarray:
    return np.asarray(full_snr_db, dtype=np.float64) + 10.0 * np.log10(FS / SLOT_BW)


def rrc_taps() -> np.ndarray:
    n = np.arange(-RRC_SPAN * SPS // 2, RRC_SPAN * SPS // 2 + 1, dtype=np.float64)
    t = n / float(SPS)
    h = np.zeros_like(t)
    for index, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[index] = 1.0 - ALPHA + 4.0 * ALPHA / np.pi
        elif abs(abs(4.0 * ALPHA * ti) - 1.0) < 1e-10:
            h[index] = ALPHA / np.sqrt(2.0) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * ALPHA))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * ALPHA))
            )
        else:
            h[index] = (
                np.sin(np.pi * ti * (1.0 - ALPHA))
                + 4.0 * ALPHA * ti * np.cos(np.pi * ti * (1.0 + ALPHA))
            ) / (np.pi * ti * (1.0 - (4.0 * ALPHA * ti) ** 2))
    h /= np.sqrt(np.sum(h * h))
    return (h * np.sqrt(float(SPS))).astype(np.float64)


def _normalize(points: np.ndarray) -> np.ndarray:
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


def constellation(name: str) -> np.ndarray:
    if name == "BPSK":
        return np.asarray([1, -1], dtype=np.complex128)
    if name in {"QPSK", "OQPSK"}:
        return _normalize(np.asarray([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j], dtype=np.complex128))
    if name == "ASK8":
        return _normalize(np.asarray([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.complex128))
    if name.startswith("PSK"):
        order = int(name[3:])
        return np.exp(1j * 2.0 * np.pi * np.arange(order) / order).astype(np.complex128)
    if name == "QAM128":
        levels_i = np.arange(15, -16, -2, dtype=np.float64)
        levels_q = np.arange(7, -8, -2, dtype=np.float64)
        return _normalize(np.asarray([i + 1j * q for q in levels_q for i in levels_i], dtype=np.complex128))
    if name.startswith("QAM"):
        side = int(round(np.sqrt(int(name[3:]))))
        levels = np.arange(side - 1, -side, -2, dtype=np.float64)
        return _normalize(np.asarray([i + 1j * q for q in levels for i in levels], dtype=np.complex128))
    apsk = {
        "APSK16": ([4, 12], [1.0, 2.85], [np.pi / 4.0, np.pi / 12.0]),
        "APSK32": ([4, 12, 16], [1.0, 2.72, 4.87], [np.pi / 4.0, np.pi / 12.0, 0.0]),
        "APSK64": ([4, 12, 20, 28], [1.0, 2.35, 3.75, 5.25], [np.pi / 4.0, np.pi / 12.0, 0.0, np.pi / 28.0]),
    }
    if name in apsk:
        counts, radii, phases = apsk[name]
        points: list[complex] = []
        for count, radius, phase in zip(counts, radii, phases):
            angles = phase + 2.0 * np.pi * np.arange(count) / count
            points.extend(radius * np.exp(1j * angles))
        return _normalize(np.asarray(points, dtype=np.complex128))
    raise ValueError(f"未知调制: {name}")


TAPS = rrc_taps()
CONSTELLATIONS = {name: constellation(name) for name in MODULATIONS}
RAW_INDEX = np.arange(N_FULL, dtype=np.float64)


def balanced_single_schedule(split: str, count: int, master_seed: int) -> np.ndarray:
    base = np.arange(13, dtype=np.int16)
    repeats, remainder = divmod(count, 13)
    schedule = np.tile(base, repeats)
    rng = np.random.default_rng(stable_seed(GENERATOR_VERSION, master_seed, "single", split, "single_class_schedule"))
    if remainder:
        schedule = np.concatenate((schedule, base[rng.permutation(13)[:remainder]]))
    return schedule[rng.permutation(count)]


def balanced_dual_schedule(split: str, count: int, master_seed: int) -> np.ndarray:
    pairs = np.asarray([(a, b) for a in range(13) for b in range(13)], dtype=np.int16)
    repeats, remainder = divmod(count, 169)
    schedule = np.tile(pairs, (repeats, 1))
    rng = np.random.default_rng(stable_seed(GENERATOR_VERSION, master_seed, "dual", split, "pair_schedule"))
    if remainder:
        schedule = np.vstack((schedule, pairs[rng.permutation(169)[:remainder]]))
    return schedule[rng.permutation(count)]


def layout_schedule(task: str, split: str, count: int, master_seed: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(GENERATOR_VERSION, master_seed, task, split, "layout_schedule"))
    n_signal = 1 if task == "single" else 2
    return np.stack([rng.choice(24, size=n_signal, replace=False) for _ in range(count)]).astype(np.int16)


def split_schedule(task: str, split: str, count: int, master_seed: int) -> dict[str, np.ndarray]:
    if task == "single":
        mods = balanced_single_schedule(split, count, master_seed)[:, None]
    elif task == "dual":
        mods = balanced_dual_schedule(split, count, master_seed)
    else:
        raise ValueError(f"未知任务: {task}")
    slots = layout_schedule(task, split, count, master_seed)
    return {"mods_component_order": mods, "slots_component_order": slots}


def shaped_waveform(mod_id: int, symbol_seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(symbol_seed)
    name = MODULATIONS[int(mod_id)]
    points = CONSTELLATIONS[name]
    n_symbols = int(math.ceil(N_FULL / SPS)) + RRC_SPAN + 8
    symbol_indices = rng.integers(0, len(points), size=n_symbols, dtype=np.int32)
    symbols = points[symbol_indices]
    delay = (len(TAPS) - 1) // 2
    if name == "OQPSK":
        i_full = upfirdn(TAPS, symbols.real, up=SPS)
        q_full = upfirdn(TAPS, symbols.imag, up=SPS)
        q_delay = SPS // 2
        out = i_full[delay:delay + N_FULL] + 1j * q_full[delay - q_delay:delay - q_delay + N_FULL]
    else:
        full = upfirdn(TAPS, symbols, up=SPS)
        out = full[delay:delay + N_FULL]
    if len(out) != N_FULL:
        raise RuntimeError(f"{name} 有效窗口长度错误: {len(out)}")
    out = out.astype(np.complex64, copy=False)
    out = out / np.sqrt(np.mean(np.abs(out) ** 2)).astype(np.float32)
    return out, symbol_indices


def generate_clean_source(
    task: str,
    split: str,
    serial: int,
    master_seed: int,
    mods_component_order: np.ndarray,
    slots_component_order: np.ndarray,
) -> dict[str, Any]:
    uid = source_uid(master_seed, task, split, serial)
    n_signal = 1 if task == "single" else 2
    components = np.empty((n_signal, N_FULL), dtype=np.complex64)
    phases = np.empty(n_signal, dtype=np.float64)
    symbol_seeds = np.empty(n_signal, dtype=np.uint64)
    phase_seeds = np.empty(n_signal, dtype=np.uint64)
    symbol_hashes: list[str] = []
    component_hashes: list[str] = []
    component_uids: list[str] = []
    for component in range(n_signal):
        cuid = component_uid(uid, component)
        sym_seed = named_seed(master_seed, task, split, uid, f"symbols/component{component}")
        pha_seed = named_seed(master_seed, task, split, uid, f"phase/component{component}")
        baseband, symbol_indices = shaped_waveform(int(mods_component_order[component]), sym_seed)
        phase = float(np.random.default_rng(pha_seed).uniform(0.0, 2.0 * np.pi))
        slot = int(slots_component_order[component])
        carrier = np.exp(1j * (2.0 * np.pi * CENTERS[slot] * RAW_INDEX / FS + phase)).astype(np.complex64)
        components[component] = baseband * carrier
        phases[component] = phase
        symbol_seeds[component] = sym_seed
        phase_seeds[component] = pha_seed
        symbol_hashes.append(array_hash(symbol_indices))
        component_hashes.append(array_hash(components[component]))
        component_uids.append(cuid)
    order = np.argsort(slots_component_order)
    slots = np.asarray(slots_component_order, dtype=np.int16)[order]
    mods = np.asarray(mods_component_order, dtype=np.int16)[order]
    components = components[order]
    phases = phases[order]
    symbol_seeds = symbol_seeds[order]
    phase_seeds = phase_seeds[order]
    symbol_hashes = [symbol_hashes[int(i)] for i in order]
    component_hashes = [component_hashes[int(i)] for i in order]
    component_uids = [component_uids[int(i)] for i in order]
    clean = np.sum(components, axis=0, dtype=np.complex64)
    y_occ = np.zeros(24, dtype=np.uint8)
    y_mod = np.full(24, -1, dtype=np.int8)
    y_occ[slots] = 1
    y_mod[slots] = mods.astype(np.int8)
    return {
        "source_uid": uid,
        "component_uid": component_uids,
        "n_signal": n_signal,
        "occupied_slots": slots,
        "modulation_labels": mods,
        "y_occ": y_occ,
        "y_mod": y_mod,
        "phases_rad": phases,
        "symbol_seed": symbol_seeds,
        "phase_seed": phase_seeds,
        "symbol_hash": symbol_hashes,
        "component_hash": component_hashes,
        "components": components,
        "component_power_full": np.mean(np.abs(components) ** 2, axis=1).astype(np.float64),
        "clean_full": clean,
        "clean_full_hash": array_hash(clean),
    }


def generate_view(
    source: dict[str, Any],
    task: str,
    split: str,
    master_seed: int,
    esn0_db: float,
) -> dict[str, Any]:
    full_target = float(esn0_to_full_component_snr_db(esn0_db))
    component_reference_power = float(np.mean(source["component_power_full"]))
    target_noise_power = component_reference_power / (10.0 ** (full_target / 10.0))
    uid = str(source["source_uid"])
    noise_seed = named_seed(master_seed, task, split, uid, f"noise/esn0/{float(esn0_db):+.9f}")
    rng = np.random.default_rng(noise_seed)
    sigma = np.sqrt(target_noise_power / 2.0)
    noise = (rng.normal(0.0, sigma, N_FULL) + 1j * rng.normal(0.0, sigma, N_FULL)).astype(np.complex64)
    noisy = source["clean_full"] + noise
    clean_mcs = source["clean_full"].reshape(N_MCS, L)[:, COSET]
    noise_mcs = noise.reshape(N_MCS, L)[:, COSET]
    noisy_mcs = noisy.reshape(N_MCS, L)[:, COSET]
    mcs_iq = np.empty((N_MCS, 2 * M), dtype=np.float32)
    mcs_iq[:, 0::2] = noisy_mcs.real
    mcs_iq[:, 1::2] = noisy_mcs.imag
    noise_power_measured = float(np.mean(np.abs(noise) ** 2))
    measured_component = 10.0 * np.log10(source["component_power_full"] / noise_power_measured)
    measured_total = 10.0 * np.log10(float(np.mean(np.abs(source["clean_full"]) ** 2)) / noise_power_measured)
    mcs_noise_power = np.mean(np.abs(noise_mcs) ** 2, axis=0).astype(np.float64)
    mcs_clean_power = np.mean(np.abs(clean_mcs) ** 2, axis=0).astype(np.float64)
    mcs_snr = 10.0 * np.log10(mcs_clean_power / mcs_noise_power)
    return {
        "view_uid": view_uid(uid, esn0_db),
        "noise_seed": np.uint64(noise_seed),
        "target_esn0_db": np.float32(esn0_db),
        "target_full_component_snr_db": np.float32(full_target),
        "target_slot50_snr_db": np.float32(full_component_to_slot50_snr_db(full_target)),
        "target_total_snr_db": np.float32(full_target + 10.0 * np.log10(source["n_signal"])),
        "target_noise_power_full": np.float64(target_noise_power),
        "noise_power_full_measured": np.float64(noise_power_measured),
        "measured_full_component_snr_db": measured_component.astype(np.float32),
        "measured_full_total_snr_db": np.float32(measured_total),
        "mcs_noise_power_per_branch": mcs_noise_power,
        "mcs_clean_power_per_branch": mcs_clean_power,
        "mcs_snr_per_branch_db": mcs_snr.astype(np.float32),
        "noise_full": noise,
        "noisy_full": noisy,
        "mcs_iq": mcs_iq,
    }


def frozen_physical_config() -> dict[str, Any]:
    return {
        "Fs": FS,
        "L": L,
        "M": M,
        "Coset": COSET.tolist(),
        "N_full": N_FULL,
        "N_mcs": N_MCS,
        "slot_count": 24,
        "slot_bandwidth_hz": SLOT_BW,
        "slot_centers_hz": CENTERS.tolist(),
        "modulations": MODULATIONS,
        "RRC_alpha": ALPHA,
        "RRC_span_symbols": RRC_SPAN,
        "RRC_taps": len(TAPS),
        "Rs": RS,
        "SPS": SPS,
        "timing_samples": TIMING,
        "component_power": 1.0,
        "awgn_domain": "Full 域仅加入一次，随后逐元素抽取 MCS",
        "iq_order_full": "[..., 0]=I, [..., 1]=Q",
        "iq_order_mcs": "[I0,Q0,I1,Q1,...,I7,Q7]",
    }


def validate_frozen_physics() -> None:
    assert FS == 2.0e9 and RS == 20.0e6 and SPS == 100
    assert L == 40 and M == 8 and N_FULL == L * N_MCS
    assert COSET.tolist() == [16, 11, 1, 0, 27, 24, 37, 31]
    assert len(CENTERS) == 24 and len(MODULATIONS) == 13
    assert ALPHA == 0.35 and RRC_SPAN == 8 and TIMING == 0
    assert len(TAPS) == 801 and np.allclose(TAPS, TAPS[::-1])
    assert np.isclose(np.sum(TAPS * TAPS), SPS)
    assert all(np.isclose(np.mean(np.abs(points) ** 2), 1.0) for points in CONSTELLATIONS.values())
