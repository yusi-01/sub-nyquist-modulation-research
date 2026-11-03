"""GATE-3E 冻结双视图与固定融合系数；不读取真实槽、标签或真实 SNR。"""
from __future__ import annotations

import torch


class DualViewTransform(torch.nn.Module):
    """Raw 与 GATE-3D 投影视图；残差能量只用于固定 α。"""

    def __init__(self):
        super().__init__()
        from blind_input import BlindProjector
        self.projector = BlindProjector()
        assert sum(p.numel() for p in self.parameters()) == 0
        self.epsilon = torch.finfo(torch.float32).eps

    def forward(self, x: torch.Tensor, support: torch.Tensor):
        assert x.ndim == 3 and x.shape[1:] == (16, 1024)
        assert support.shape == (len(x), 2)
        raw = x.float() / 8.0
        projected = self.projector(x, support)
        z = torch.complex(raw[:, 0::2], raw[:, 1::2])
        spectrum = torch.fft.fft(z, dim=-1, norm="ortho")
        corrected = spectrum * self.projector.phase.conj()[None]
        idx = self.projector.lookup[support[:, 0], support[:, 1]]
        assert bool((idx >= 0).all())
        inside = torch.einsum("bij,bjq->biq", self.projector.matrices[idx], corrected)
        residual = corrected - inside
        sigma2 = residual.abs().square().sum(dim=(1, 2)) / (6 * 1024)
        projected_energy = inside.abs().square().sum(dim=(1, 2)) / 1024
        clean_est = torch.clamp(projected_energy - 2 * sigma2, min=0)
        gamma = clean_est / (2 * sigma2 + self.epsilon)
        alpha = 1 / (2 + gamma)
        if not (torch.isfinite(raw).all() and torch.isfinite(projected).all()
                and torch.isfinite(alpha).all() and torch.isfinite(gamma).all()
                and torch.isfinite(sigma2).all()):
            raise FloatingPointError("双视图或固定融合系数出现非有限值")
        if not bool(((alpha > 0) & (alpha <= 0.5)).all()):
            raise FloatingPointError("固定融合系数超出 (0,0.5]")
        return raw, projected, alpha, gamma, sigma2


def fused_logits(model, raw, projected, alpha):
    """一次拼接前向，BatchNorm 对两视图同时更新；只输出一组 26 维融合 logits。"""
    batch = len(raw)
    assert projected.shape == raw.shape and alpha.shape == (batch,)
    both = model(torch.cat((raw, projected), dim=0))
    logits_raw, logits_proj = both.chunk(2, dim=0)
    logits = (1 - alpha[:, None]) * logits_raw + alpha[:, None] * logits_proj
    return logits, logits_raw, logits_proj

