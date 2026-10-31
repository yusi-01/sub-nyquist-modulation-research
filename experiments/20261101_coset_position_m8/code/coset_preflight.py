"""只用冻结物理常数选择位置对照，并校准 Coset/FFT 索引。"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "coset_position_20261031"
OUT.mkdir(exist_ok=True)
FS, L, M = 2_000_000_000.0, 40, 8
BASE = np.array([16, 11, 1, 0, 27, 24, 37, 31], dtype=int)
CENTERS = np.array([-575e6 + 50e6 * k for k in range(24)])
K = CENTERS / (FS / L)
assert np.array_equal(K, np.arange(-12, 12) + .5)
FFT_BINS = ((K - .5).astype(int) % L).tolist()
assert FFT_BINS == list(range(28, 40)) + list(range(12))


def matrix(coset):
    return np.exp(2j * np.pi * np.outer(coset, K) / L) / np.sqrt(M)


def measures(coset):
    a = matrix(coset)
    gram = a.conj().T @ a
    off = np.abs(gram)
    np.fill_diagonal(off, 0)
    mu = float(off.max())
    return {"max_coherence": mu, "mean_squared_offdiag": float(np.mean(off**2)),
            "max_pair_condition": float(np.sqrt((1 + mu) / (1 - mu))) if mu < 1 - 1e-12 else None,
            "matrix_rank": int(np.linalg.matrix_rank(a)), "min_nonzero_singular": float(np.linalg.svd(a, compute_uv=False)[-1])}


# 单音校准：Coset 抽样的相位与正号 A_C 一致；Full 40点块乘半bin去旋后，FFT峰索引与slot对齐。
n = np.arange(40)
calibration = []
for slot, f in enumerate(CENTERS):
    tone = np.exp(2j * np.pi * f * np.arange(2 * L) / FS)
    mcs = tone.reshape(2, L)[:, BASE]
    expected0 = np.sqrt(M) * matrix(BASE)[:, slot]
    expected1 = -expected0
    err = max(np.max(np.abs(mcs[0] - expected0)), np.max(np.abs(mcs[1] - expected1)))
    spectrum = np.fft.fft(tone[:L] * np.exp(-1j * np.pi * n / L))
    peak = int(np.argmax(np.abs(spectrum)))
    calibration.append({"slot": slot, "center_hz": float(f), "k_halfbin": float(K[slot]),
                        "fft_bin": peak, "phase_error": float(err)})
assert max(x["phase_error"] for x in calibration) < 1e-12
assert [x["fft_bin"] for x in calibration] == FFT_BINS

# 候选只由矩阵性质选择；固定搜索随机流，不访问任何标签或准确率。
rng = np.random.default_rng(271063)
seen = {tuple(sorted(BASE.tolist()))}
candidates = []
for _ in range(12000):
    c = np.sort(rng.choice(L, M, replace=False))
    key = tuple(c.tolist())
    if key in seen:
        continue
    seen.add(key)
    met = measures(c)
    candidates.append((c, met))
base_metrics = measures(BASE)
low = min(candidates, key=lambda x: (x[1]["max_coherence"], x[1]["mean_squared_offdiag"]))
# 高相关候选限制在非完全退化区，避免用可完全重合的列制造平凡失败。
high_pool = [x for x in candidates if .75 <= x[1]["max_coherence"] <= .85 and x[1]["matrix_rank"] == M]
assert high_pool and low[1]["max_coherence"] < base_metrics["max_coherence"]
high = min(high_pool, key=lambda x: abs(x[1]["max_coherence"] - .80))

# 保持基线的分支排序规则：按原 Coset 的秩次将替代集合映射到固定16个输入通道。
ranks = np.argsort(np.argsort(BASE))
def ordered(sorted_coset):
    return sorted_coset[ranks].astype(int).tolist()

record = {"status": "PASS", "Fs": FS, "L": L, "M": M,
          "slot_centers_hz": CENTERS.tolist(), "fft_bin_after_halfbin_derotation": FFT_BINS,
          "calibration": calibration,
          "matrix_definition": "A[m,k]=exp(+j*2*pi*coset[m]*(slot_center[k]/50MHz)/40)/sqrt(8)",
          "candidate_search": {"seed": 271063, "draws": 12000, "selection_uses_test_accuracy": False},
          "arms": {"baseline": {"coset_ordered": BASE.tolist(), "metrics": base_metrics},
                   "lower_coherence": {"coset_ordered": ordered(low[0]), "metrics": low[1]},
                   "higher_coherence": {"coset_ordered": ordered(high[0]), "metrics": high[1]}},
          "representation": "候选样本按基线分支秩次放入固定的16个输入通道；其余64通道始终为零"}
(OUT / "coset_preflight.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({name: arm for name, arm in record["arms"].items()}, ensure_ascii=False, indent=2))
