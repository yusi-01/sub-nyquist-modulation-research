"""只测固定编码的 GPU 显存与运行时；不构建 CNN 或执行优化步骤。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from phase_projection import PhaseProjection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert args.batch == 512 and torch.cuda.is_available()
    device = torch.device("cuda")
    torch.manual_seed(271060)
    encoder = PhaseProjection().to(device).eval()
    x = torch.randn(args.batch, 16, 1024, device=device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.no_grad():
        for _ in range(3):
            start = time.perf_counter()
            z = encoder(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            assert z.shape == (args.batch, 80, 1024) and torch.isfinite(z).all()
    result = {"gate": "PASS", "batch": args.batch, "trainable_parameters": 0,
              "device": torch.cuda.get_device_name(0),
              "encoder_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
              "encoder_seconds_per_batch": times, "output_shape": list(z.shape)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
