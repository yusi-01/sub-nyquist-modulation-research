from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from paired_common import atomic_json, install_paths, sha256


class LazyFull(Dataset):
    def __init__(self, path: Path):
        self.path = path; self.h5 = None
        with h5py.File(path, "r") as h: self.n = len(h["Y"])
    def __len__(self): return self.n
    def __getitem__(self, i):
        if self.h5 is None: self.h5 = h5py.File(self.path, "r", swmr=True)
        return torch.from_numpy(np.ascontiguousarray(self.h5["X"][i], dtype=np.float32)), torch.tensor(int(self.h5["Y"][i]), dtype=torch.long)


def loader(path, batch, workers, shuffle, seed):
    kw = dict(dataset=LazyFull(path), batch_size=batch, shuffle=shuffle, drop_last=shuffle, num_workers=workers,
              pin_memory=True, persistent_workers=workers > 0, generator=torch.Generator().manual_seed(seed))
    if workers: kw["prefetch_factor"] = 2
    return DataLoader(**kw)


def estimate_macs(model, device):
    total = [0]
    hooks = []
    def conv_hook(module, inputs, output):
        total[0] += int(output.numel() * (module.in_channels // module.groups) * module.kernel_size[0])
    def linear_hook(module, inputs, output):
        total[0] += int(output.numel() * module.in_features)
    for module in model.modules():
        if isinstance(module, nn.Conv1d): hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear): hooks.append(module.register_forward_hook(linear_hook))
    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
        model(torch.zeros(1, 80, 1024, device=device))
    for hook in hooks: hook.remove()
    return total[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True); ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--seed", type=int, default=271060)
    ap.add_argument("--batch-size", type=int, default=1024); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    if args.output.exists(): raise SystemExit(f"refuse overwrite: {args.output}")
    if not torch.cuda.is_available(): raise SystemExit("CUDA required")
    args.output.mkdir(parents=True)
    core, _ = install_paths(args.project_root)
    from utils.runtime import configure_runtime, enable_resnet_only_pnn_import, save_compatible_state_dict
    configure_runtime(args.seed, deterministic=False, tf32=True); enable_resnet_only_pnn_import()
    from models.pnn1 import Res1dNet311
    from utils.utils import init_weights, WarmUpLR
    torch.cuda.set_device(0); device = torch.device("cuda")
    train = loader(args.data_root / "train.h5", args.batch_size, args.workers, True, args.seed)
    val = loader(args.data_root / "validation.h5", args.batch_size, args.workers, False, args.seed)
    model = init_weights(Res1dNet311(in_channel=80, classes_num=13)).to(device)
    params = sum(p.numel() for p in model.parameters())
    macs = estimate_macs(model, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup)
    warmup = WarmUpLR(optimizer, args.warmup)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    resource = {"host_ram_total_gib": None, "gpu": torch.cuda.get_device_name(0),
                "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
                "batch_size": args.batch_size, "workers": args.workers, "parameters": params,
                "approx_macs_per_sample": macs, "approx_flops_per_sample": 2 * macs}
    try:
        import psutil
        resource["host_ram_total_gib"] = psutil.virtual_memory().total / 2**30
    except Exception: pass
    # One real batch forward/backward is the only OOM probe. No retry and no silent batch change.
    try:
        x, y = next(iter(train)); x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True): loss = criterion(model(x).squeeze(1), y)
        scaler.scale(loss).backward(); optimizer.zero_grad(set_to_none=True); del x, y, loss; torch.cuda.empty_cache()
        resource["batch_probe"] = "PASS"
    except torch.cuda.OutOfMemoryError:
        resource["batch_probe"] = "FAIL_OOM"; atomic_json(args.output / "resource_preflight.json", resource); raise
    atomic_json(args.output / "resource_preflight.json", resource)
    history = []; best = -1.0; best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; seen = 0; started = time.time()
        for x, y in train:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True): loss = criterion(model(x).squeeze(1), y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += float(loss) * len(y); seen += len(y)
        model.eval(); correct = 0; n = 0
        with torch.inference_mode():
            for x, y in val:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=True): pred = model(x).squeeze(1).argmax(1)
                correct += int((pred == y).sum()); n += len(y)
        val_acc = correct / n
        row = {"epoch": epoch, "train_loss": total_loss / seen, "validation_accuracy": val_acc,
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - started}
        history.append(row); print(json.dumps(row), flush=True)
        # Match the existing MCS rule exactly: latest tied Validation maximum wins.
        if val_acc >= best:
            best, best_epoch = val_acc, epoch
            save_compatible_state_dict(model, str(args.output), "best_checkpoint_basic1.pt", "best_checkpoint_basic.pt")
            atomic_json(args.output / "best_checkpoint_metadata.json", {"selected_on": "validation_accuracy", "best": best,
                         "epoch": best_epoch, "tie_rule": "latest_ge", "development_test_used": False})
        if epoch < args.warmup: warmup.step()
        else: scheduler.step()
        atomic_json(args.output / "history.json", {"history": history})
    config = {"seed": args.seed, "batch_size": args.batch_size, "epochs": args.epochs, "warmup": args.warmup,
              "optimizer": "Adam", "lr": 0.01, "weight_decay": 0.0001, "scheduler": "CosineAnnealingLR",
              "label_smoothing": 0.1, "amp": True, "tf32": True, "model": "Res1dNet311", "in_channel": 80,
              "parameters": params, "approx_macs_per_sample": macs, "approx_flops_per_sample": 2 * macs,
              "input": "full40960_complex_reversible_polyphase_80x1024_div8", "per_sample_rms": False,
              "best_validation_accuracy": best, "best_epoch": best_epoch,
              "checkpoint_sha256": sha256(args.output / "best_checkpoint_basic.pt")}
    atomic_json(args.output / "training_config.json", config); print(json.dumps(config, ensure_ascii=False))


if __name__ == "__main__": main()
