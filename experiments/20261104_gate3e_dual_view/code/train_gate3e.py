"""GATE-3E 唯一训练：固定 α、共享原生 SHELL16、一次融合 BCE。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from dual_view import DualViewTransform, fused_logits

LABELS = ["APSK16", "APSK32", "APSK64", "ASK8", "BPSK", "OQPSK", "PSK16", "PSK8",
          "QAM128", "QAM16", "QAM256", "QAM64", "QPSK"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    recall, f1 = [], []
    for k in range(13):
        tp = int(np.sum((y == k) & (p == k)))
        fn = int(np.sum((y == k) & (p != k)))
        fp = int(np.sum((y != k) & (p == k)))
        recall.append(float(tp / (tp + fn)) if tp + fn else None)
        f1.append(float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0)
    return {"component_modacc": float(np.mean(y == p)),
            "pair_exact": float(np.mean(np.all(y == p, axis=1))),
            "macro_f1": float(np.mean(f1)), "recall": dict(zip(LABELS, recall))}


def decode(logits: np.ndarray) -> np.ndarray:
    return np.stack((logits[:, :13].argmax(1), logits[:, 13:].argmax(1)), axis=1)


@torch.inference_mode()
def evaluate(model, transform, loader, device, save_outputs=False):
    model.eval()
    truths, indices, fused_preds = [], [], []
    raw_logits, proj_logits, fused_logits_arr, alphas, support_arr = [], [], [], [], []
    loss_sum, count = 0.0, 0
    for x, target, y, idx, support in loader:
        x = x.to(device, non_blocking=True)
        support_gpu = support.to(device, non_blocking=True)
        raw, proj, alpha, _, _ = transform(x, support_gpu)
        with torch.cuda.amp.autocast(enabled=True):
            fused, raw_logit, proj_logit = fused_logits(model, raw, proj, alpha)
            smooth = (target * (1 - .026) + .026 / 26).to(device, non_blocking=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(fused, smooth)
        loss_sum += float(loss.item()) * len(x)
        count += len(x)
        fused_np = fused.float().cpu().numpy()
        truths.append(y.numpy()); indices.append(idx.numpy()); fused_preds.append(decode(fused_np))
        if save_outputs:
            raw_logits.append(raw_logit.float().cpu().numpy())
            proj_logits.append(proj_logit.float().cpu().numpy())
            fused_logits_arr.append(fused_np)
            alphas.append(alpha.cpu().numpy())
            support_arr.append(support.numpy())
    y, idx, p = np.concatenate(truths), np.concatenate(indices), np.concatenate(fused_preds)
    result = metrics(y, p); result["loss"] = loss_sum / count
    if not save_outputs:
        return result, None
    outputs = {"y": y, "row_index": idx, "fused_pred": p,
               "raw_logits": np.concatenate(raw_logits).astype(np.float32),
               "projection_logits": np.concatenate(proj_logits).astype(np.float32),
               "fused_logits": np.concatenate(fused_logits_arr).astype(np.float32),
               "alpha": np.concatenate(alphas).astype(np.float32),
               "estimated_support": np.concatenate(support_arr).astype(np.int16)}
    assert np.array_equal(outputs["row_index"], np.arange(len(outputs["row_index"])))
    assert np.array_equal(decode(outputs["fused_logits"]), outputs["fused_pred"])
    assert np.isfinite(outputs["raw_logits"]).all()
    assert np.isfinite(outputs["projection_logits"]).all()
    assert np.isfinite(outputs["fused_logits"]).all()
    return result, outputs


def save_development(path: Path, h5_path: Path, outputs: dict) -> None:
    np.savez_compressed(path / "development_branch_logits.npz",
                        row_index=outputs["row_index"],
                        raw_logits=outputs["raw_logits"],
                        projection_logits=outputs["projection_logits"],
                        fused_logits=outputs["fused_logits"],
                        alpha=outputs["alpha"],
                        estimated_support=outputs["estimated_support"])
    raw_pred = decode(outputs["raw_logits"])
    proj_pred = decode(outputs["projection_logits"])
    fused_pred = outputs["fused_pred"]
    with h5py.File(h5_path, "r") as h, (path / "development_predictions.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "source_uid", "view_uid", "noise_seed", "full_component_snr_db",
                         "slot_1", "slot_2", "true_1", "true_2", "est_1", "est_2", "alpha",
                         "raw_pred_1", "raw_pred_2", "projection_pred_1", "projection_pred_2",
                         "fused_pred_1", "fused_pred_2", "raw_correct_1", "raw_correct_2",
                         "projection_correct_1", "projection_correct_2", "fused_correct_1",
                         "fused_correct_2", "raw_pair_exact", "projection_pair_exact", "fused_pair_exact"])
        for k, index in enumerate(outputs["row_index"]):
            source, view = divmod(int(index), 11)
            actual = outputs["y"][k]
            slots = h["occupied_slots"][source]
            branch = [raw_pred[k], proj_pred[k], fused_pred[k]]
            correct = [(p == actual).astype(int) for p in branch]
            writer.writerow([int(index), h["source_uid"][source].decode(),
                             h["view_uid"][source, view].decode(), int(h["noise_seed"][source, view]),
                             float(h["target_full_component_snr_db"][source, view]),
                             int(slots[0]), int(slots[1]), LABELS[actual[0]], LABELS[actual[1]],
                             int(outputs["estimated_support"][k, 0]), int(outputs["estimated_support"][k, 1]),
                             float(outputs["alpha"][k]),
                             *[LABELS[p] for pred in branch for p in pred],
                             *[int(v) for c in correct for v in c],
                             *[int(np.all(c)) for c in correct]])


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("data", "cache_root", "gate3c_code", "gate3d_code", "shell_root", "out"):
        ap.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=271060)
    a = ap.parse_args()
    assert (a.seed, a.batch, a.epochs) == (271060, 512, 200)
    assert not a.out.exists()
    sys.path.insert(0, str(a.gate3c_code))
    sys.path.insert(0, str(a.gate3d_code))
    sys.path.insert(0, str(a.shell_root))
    from blind_input import BlindViews
    from utils.runtime import configure_runtime, enable_resnet_only_pnn_import
    configure_runtime(a.seed, deterministic=False, tf32=True)
    enable_resnet_only_pnn_import()
    from models.pnn1 import Res1dNet31
    from utils.utils import WarmUpLR, init_weights

    device = torch.device("cuda:0")
    assert torch.cuda.is_available()
    splits = ("train", "validation", "development_test")
    paths = {s: a.data / "dual" / s / "dataset.h5" for s in splits}
    datasets = {s: BlindViews(paths[s], a.cache_root / (s + "_support.csv")) for s in splits}
    assert [datasets[s].sources for s in splits] == [16900, 3380, 3380]
    a.out.mkdir(parents=True)
    config = {"model": "Res1dNet31(16,26) shared dual-view", "seed": a.seed,
              "epochs": a.epochs, "batch_original_views": a.batch,
              "batch_cnn_inputs": 2 * a.batch,
              "data": str(a.data), "cache_root": str(a.cache_root),
              "gate3c_code": str(a.gate3c_code), "gate3d_code": str(a.gate3d_code),
              "alpha_formula": "1/(2+max(||P v||^2/Q-2*sigma2,0)/(2*sigma2+float32_eps))",
              "epsilon": float(torch.finfo(torch.float32).eps),
              "checkpoint_rule": "Validation component ModAcc >= best; tie overwrites",
              "h5_sha256": {s: digest(paths[s]) for s in splits},
              "support_cache_sha256": {s: digest(a.cache_root / (s + "_support.csv")) for s in splits}}
    (a.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    g = torch.Generator().manual_seed(a.seed)
    train = DataLoader(datasets["train"], batch_size=a.batch, shuffle=True, num_workers=4,
                       pin_memory=True, persistent_workers=True, generator=g)
    val = DataLoader(datasets["validation"], batch_size=a.batch, shuffle=False, num_workers=4,
                     pin_memory=True, persistent_workers=True)
    model = init_weights(Res1dNet31(in_channel=16, classes_num=26)).to(device)
    assert sum(p.numel() for p in model.parameters()) == 6320026
    transform = DualViewTransform().to(device).eval()
    assert sum(p.numel() for p in transform.parameters()) == 0
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs - 5)
    warmup = WarmUpLR(optimizer, 5)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    best, history = -1.0, []
    for epoch in range(1, a.epochs + 1):
        model.train(); total, count = 0.0, 0
        for x, target, _, _, support in train:
            raw, proj, alpha, _, _ = transform(x.to(device, non_blocking=True),
                                                support.to(device, non_blocking=True))
            smooth = (target * (1 - .026) + .026 / 26).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                logits, _, _ = fused_logits(model, raw, proj, alpha)
                loss = criterion(logits, smooth)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += float(loss.item()) * len(x); count += len(x)
        vm, _ = evaluate(model, transform, val, device)
        row = {"epoch": epoch, "train_loss": total / count, "validation_loss": vm["loss"],
               "validation_component_modacc": vm["component_modacc"],
               "validation_pair_exact": vm["pair_exact"], "validation_macro_f1": vm["macro_f1"],
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        if vm["component_modacc"] >= best:
            best = vm["component_modacc"]
            torch.save(model.state_dict(), a.out / "best_checkpoint_mr.pt")
            (a.out / "best_validation.json").write_text(
                json.dumps({"epoch": epoch, "metrics": vm}, ensure_ascii=False, indent=2), encoding="utf-8")
        with (a.out / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, row.keys()); writer.writeheader(); writer.writerows(history)
        if epoch < 5: warmup.step()
        else: cosine.step()
        print("EPOCH gate3e", epoch, row, flush=True)
    model.load_state_dict(torch.load(a.out / "best_checkpoint_mr.pt", map_location=device))
    dev_loader = DataLoader(datasets["development_test"], batch_size=a.batch, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    dev_metrics, outputs = evaluate(model, transform, dev_loader, device, save_outputs=True)
    (a.out / "development.json").write_text(json.dumps(dev_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_development(a.out, paths["development_test"], outputs)
    print("GATE3E_DONE", json.dumps({"best_validation_modacc": best,
                                     "development": dev_metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

