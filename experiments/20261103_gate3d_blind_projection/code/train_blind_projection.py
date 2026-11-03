"""GATE-3D：冻结盲投影与原生 SHELL16 的唯一训练。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

LABELS = ["APSK16", "APSK32", "APSK64", "ASK8", "BPSK", "OQPSK", "PSK16", "PSK8", "QAM128", "QAM16", "QAM256", "QAM64", "QPSK"]
COSET = np.array([16, 11, 1, 0, 27, 24, 37, 31])



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
def evaluate(model, loader, device, projector):
    model.eval()
    ys, ps, ids = [], [], []
    loss_sum = 0.0; n = 0
    for x, target, y, idx, support in loader:
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            z = model(projector(x.to(device, non_blocking=True),support.to(device, non_blocking=True)))
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
    ap.add_argument("--arm", choices=["blind"], required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--gate3c-code", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    
    ap.add_argument("--shell-root", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=271060)
    args = ap.parse_args()
    assert args.seed == 271060 and args.batch == 512 and args.epochs == 200
    assert not args.out.exists()
    sys.path.insert(0, str(args.gate3c_code))
    sys.path.insert(0, str(args.shell_root))
    from blind_input import BlindViews, BlindProjector
    from utils.runtime import configure_runtime, enable_resnet_only_pnn_import
    configure_runtime(args.seed, deterministic=False, tf32=True)
    enable_resnet_only_pnn_import()
    from models.pnn1 import Res1dNet31
    from utils.utils import WarmUpLR, init_weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda"
    paths = {s: args.data / "dual" / s / "dataset.h5" for s in ("train", "validation", "development_test")}
    datasets = {s: BlindViews(p, args.cache_root / (s + "_support.csv")) for s, p in paths.items()}
    assert [datasets[s].sources for s in paths] == [16900, 3380, 3380]
    args.out.mkdir(parents=True)
    (args.out / "config.json").write_text(json.dumps({"cache_sha256": {s: hashlib.sha256((args.cache_root/(s+"_support.csv")).read_bytes()).hexdigest() for s in paths},"arm": args.arm, "seed": args.seed, "epochs": args.epochs, "batch": args.batch, "data": str(args.data), "cache_root": str(args.cache_root), "gate3c_code": str(args.gate3c_code), "checkpoint_rule": "Validation component ModAcc >= best; tie overwrites"}, ensure_ascii=False, indent=2))
    g = torch.Generator().manual_seed(args.seed)
    tr = DataLoader(datasets["train"], batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, generator=g)
    va = DataLoader(datasets["validation"], batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    model = init_weights(Res1dNet31(in_channel=16, classes_num=26)).to(device)
    projector = BlindProjector().to(device).eval()
    assert sum(p.numel() for p in projector.parameters()) == 0
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - 5)
    warmup = WarmUpLR(opt, 5)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; n = 0
        for x, target, _, _, support in tr:
            x = projector(x.to(device, non_blocking=True), support.to(device, non_blocking=True))
            target = (target * (1 - 0.026) + 0.026 / 26).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                z = model(x)
                loss = criterion(z, target)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += float(loss.item()) * len(x); n += len(x)
        vm, _, _, _ = evaluate(model, va, device, projector)
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
    dm, y, p, ids = evaluate(model, de, device, projector)
    (args.out / "development.json").write_text(json.dumps(dm, ensure_ascii=False, indent=2))
    write_predictions(args.out / "development_predictions.csv", paths["development_test"], y, p, ids)


if __name__ == "__main__":
    main()

