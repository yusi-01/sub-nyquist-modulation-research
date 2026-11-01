"""冻结双信号 SHELL 协议；按源分块读取，先 MCS 后 Full。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

LABELS = ["APSK16", "APSK32", "APSK64", "ASK8", "BPSK", "OQPSK", "PSK16", "PSK8", "QAM128", "QAM16", "QAM256", "QAM64", "QPSK"]
COSET = np.array([16, 11, 1, 0, 27, 24, 37, 31])


class Views(Dataset):
    def __init__(self, path: Path, arm: str, full_cache: Path | None):
        self.path, self.arm, self.full_cache = path, arm, full_cache
        self.h = self.cache = None
        with h5py.File(path) as h:
            self.sources, self.views = h["mcs_iq"].shape[:2]
            assert self.views == 11 and h["clean_full_iq"].shape == (self.sources, 40960, 2)
        if arm == "full" and full_cache is not None:
            with h5py.File(full_cache) as h:
                assert h["noisy_full_iq"].shape == (self.sources, 11, 40960, 2)

    def __len__(self):
        return self.sources * self.views

    def __getitem__(self, index):
        if self.h is None:
            self.h = h5py.File(self.path, "r")
        i, j = divmod(index, self.views)
        if self.arm == "mcs":
            x = np.ascontiguousarray(self.h["mcs_iq"][i, j].T / 8.0, dtype=np.float32)
        else:
            if self.full_cache is not None:
                if self.cache is None:
                    self.cache = h5py.File(self.full_cache, "r")
                full = self.cache["noisy_full_iq"][i, j]
            else:
                clean = self.h["clean_full_iq"][i]
                power = float(self.h["target_noise_power_full"][i, j])
                rng = np.random.default_rng(int(self.h["noise_seed"][i, j]))
                sigma = np.sqrt(power / 2.0)
                noise = (rng.normal(0, sigma, 40960) + 1j * rng.normal(0, sigma, 40960)).astype(np.complex64)
                z = clean[:, 0] + 1j * clean[:, 1] + noise
                full = np.stack((z.real, z.imag), axis=1)
            x = np.ascontiguousarray(full.reshape(1024, 40, 2).transpose(1, 2, 0).reshape(80, 1024) / 8.0, dtype=np.float32)
        y = self.h["modulation_labels"][i].astype(np.int64)
        target = np.zeros(26, np.float32)
        target[y[0]] = 1.0
        target[13 + y[1]] = 1.0
        return torch.from_numpy(x), torch.from_numpy(target), torch.from_numpy(y), index


def metrics(y, p):
    acc = float(np.mean(y == p))
    pair = float(np.mean(np.all(y == p, axis=1)))
    recall = []
    f1 = []
    for k in range(13):
        tp = int(np.sum((y == k) & (p == k)))
        fn = int(np.sum((y == k) & (p != k)))
        fp = int(np.sum((y != k) & (p == k)))
        recall.append(float(tp / (tp + fn)) if tp + fn else None)
        f1.append(float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0)
    return {"component_modacc": acc, "pair_exact": pair, "macro_f1": float(np.mean(f1)), "recall": dict(zip(LABELS, recall))}


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, ids = [], [], []
    loss_sum = 0.0; n = 0
    for x, target, y, idx in loader:
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            z = model(x.to(device, non_blocking=True))
            smoothed = (target * (1 - 0.026) + 0.026 / 26).to(device, non_blocking=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, smoothed)
        loss_sum += float(loss.item()) * len(x); n += len(x)
        z = z.float().cpu().numpy()
        ps.append(np.stack((z[:, :13].argmax(1), z[:, 13:].argmax(1)), axis=1))
        ys.append(y.numpy())
        ids.append(idx.numpy())
    y, p, ids = np.concatenate(ys), np.concatenate(ps), np.concatenate(ids)
    result = metrics(y, p); result["loss"] = loss_sum / n
    return result, y, p, ids


def write_predictions(path, dataset_path, y, p, ids):
    with h5py.File(dataset_path) as h, path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_uid", "view_uid", "noise_seed", "full_component_snr_db", "slot_1", "slot_2", "true_1", "true_2", "pred_1", "pred_2", "correct_1", "correct_2", "pair_exact"])
        for actual, pred, idx in zip(y, p, ids):
            i, j = divmod(int(idx), 11)
            uid = h["source_uid"][i].decode()
            vid = h["view_uid"][i, j].decode()
            slots = h["occupied_slots"][i]
            w.writerow([uid, vid, int(h["noise_seed"][i, j]), float(h["target_full_component_snr_db"][i, j]), int(slots[0]), int(slots[1]), LABELS[actual[0]], LABELS[actual[1]], LABELS[pred[0]], LABELS[pred[1]], int(actual[0] == pred[0]), int(actual[1] == pred[1]), int(np.all(actual == pred))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--arm", choices=["mcs", "full"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-cache", type=Path)
    ap.add_argument("--shell-root", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=271060)
    args = ap.parse_args()
    assert args.seed == 271060 and args.batch == 512 and args.epochs == 200
    assert not args.out.exists()
    sys.path.insert(0, str(args.shell_root))
    from utils.runtime import configure_runtime, enable_resnet_only_pnn_import
    configure_runtime(args.seed, deterministic=False, tf32=True)
    enable_resnet_only_pnn_import()
    from models.pnn1 import Res1dNet31
    from utils.utils import WarmUpLR, init_weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda"
    paths = {s: args.data / "dual" / s / "dataset.h5" for s in ("train", "validation", "development_test")}
    datasets = {s: Views(p, args.arm, args.full_cache / (s + ".h5") if args.full_cache else None) for s, p in paths.items()}
    assert [datasets[s].sources for s in paths] == [16900, 3380, 3380]
    args.out.mkdir(parents=True)
    (args.out / "config.json").write_text(json.dumps({"arm": args.arm, "seed": args.seed, "epochs": args.epochs, "batch": args.batch, "data": str(args.data), "full_cache": str(args.full_cache) if args.full_cache else None, "checkpoint_rule": "Validation component ModAcc >= best; tie overwrites"}, ensure_ascii=False, indent=2))
    g = torch.Generator().manual_seed(args.seed)
    tr = DataLoader(datasets["train"], batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, generator=g)
    va = DataLoader(datasets["validation"], batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    model = init_weights(Res1dNet31(in_channel=16 if args.arm == "mcs" else 80, classes_num=26)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - 5)
    warmup = WarmUpLR(opt, 5)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; n = 0
        for x, target, _, _ in tr:
            x = x.to(device, non_blocking=True)
            target = (target * (1 - 0.026) + 0.026 / 26).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                z = model(x)
                loss = criterion(z, target)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += float(loss.item()) * len(x); n += len(x)
        vm, _, _, _ = evaluate(model, va, device)
        row = {"epoch": epoch, "train_loss": total / n, "validation_loss": vm["loss"], "validation_component_modacc": vm["component_modacc"], "validation_pair_exact": vm["pair_exact"], "validation_macro_f1": vm["macro_f1"], "lr": opt.param_groups[0]["lr"]}
        history.append(row)
        if vm["component_modacc"] >= best:
            best = vm["component_modacc"]
            torch.save(model.state_dict(), args.out / "best_checkpoint_mr.pt")
            (args.out / "best_validation.json").write_text(json.dumps({"epoch": epoch, "metrics": vm}, ensure_ascii=False, indent=2))
        with (args.out / "history.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, row.keys()); w.writeheader(); w.writerows(history)
        if epoch < 5: warmup.step()
        else: cosine.step()
        print("EPOCH", args.arm, epoch, row, flush=True)
    model.load_state_dict(torch.load(args.out / "best_checkpoint_mr.pt", map_location=device))
    de = DataLoader(datasets["development_test"], batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    dm, y, p, ids = evaluate(model, de, device)
    (args.out / "development.json").write_text(json.dumps(dm, ensure_ascii=False, indent=2))
    write_predictions(args.out / "development_predictions.csv", paths["development_test"], y, p, ids)


if __name__ == "__main__":
    main()

