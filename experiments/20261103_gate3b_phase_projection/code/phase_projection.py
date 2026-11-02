"""冻结 Coset 的全频率相位匹配编码；仅使用 8 路 MCS IQ。"""
from __future__ import annotations

import numpy as np
import torch

FS = 2_000_000_000
N = 40960
L = 40
M = 8
Q = N // L
COSET = np.array([16, 11, 1, 0, 27, 24, 37, 31], dtype=np.int64)
SCALE = 8.0


def phase_numpy() -> np.ndarray:
    """B[m,q,p]，正号与 Full DFT 逆变换及已校准 A_C 一致。"""
    k = np.arange(Q)[:, None] + Q * np.arange(L)[None, :]
    return np.exp(2j * np.pi * COSET[:, None, None] * k[None, :, :] / N) / np.sqrt(L)


def encode_numpy(mcs_iq: np.ndarray) -> np.ndarray:
    """原始 H5 的 [1024,16] IQ 转为 [80,1024]，幅值固定除以 8。"""
    x = np.asarray(mcs_iq)
    if x.shape != (Q, 2 * M) or not np.isfinite(x).all():
        raise ValueError("MCS IQ 形状或有限值不符合冻结合同")
    y = (x[:, 0::2].astype(np.float64) + 1j * x[:, 1::2].astype(np.float64)).T / SCALE
    spectrum = np.fft.fft(y, axis=-1, norm="ortho")
    projected = np.einsum("mq,mqp->pq", spectrum, phase_numpy().conj(), optimize=True)
    out = np.stack((projected.real, projected.imag), axis=1).reshape(2 * L, Q).astype(np.float32)
    if not np.isfinite(out).all():
        raise FloatingPointError("编码出现非有限数")
    return out


def reconstruct_mcs_numpy(encoded: np.ndarray) -> np.ndarray:
    """从编码数值恢复原 8 路 MCS IQ，用于信息保持门禁。"""
    z = np.asarray(encoded, dtype=np.float64)
    if z.shape != (2 * L, Q):
        raise ValueError("编码形状错误")
    z_complex = z[0::2] + 1j * z[1::2]
    spectrum = np.einsum("pq,mqp->mq", z_complex, phase_numpy(), optimize=True)
    y = np.fft.ifft(spectrum, axis=-1, norm="ortho").T * SCALE
    recovered = np.empty((Q, 2 * M), dtype=np.float32)
    recovered[:, 0::2] = y.real
    recovered[:, 1::2] = y.imag
    return recovered


class PhaseProjection(torch.nn.Module):
    """GPU 可用的确定性编码层；仅注册常量缓冲区，无可训练参数。"""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("phase", torch.from_numpy(phase_numpy().astype(np.complex64)), persistent=False)

    def forward(self, mcs_iq: torch.Tensor) -> torch.Tensor:
        # 输入为原 H5 的 [batch,16,1024]；不读取 slot、SNR、标签或 Full 波形。
        if mcs_iq.ndim != 3 or mcs_iq.shape[1:] != (2 * M, Q):
            raise ValueError("MCS IQ 批量形状错误")
        y = torch.complex(mcs_iq[:, 0::2].float(), mcs_iq[:, 1::2].float()) / SCALE
        spectrum = torch.fft.fft(y, dim=-1, norm="ortho")
        z = torch.einsum("bmq,mqp->bpq", spectrum, self.phase.conj())
        out = torch.stack((z.real, z.imag), dim=-2).reshape(-1, 2 * L, Q)
        if not torch.isfinite(out).all():
            raise FloatingPointError("编码出现非有限数")
        return out
