from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from mask_common import LABELS, atomic_json, apply_mask, decode, install_paths, normalize_original_mcs, sha256


class DevDataset(Dataset):
    def __init__(self, path: Path, masked: bool):
        self.path, self.masked, self.h5 = path, masked, None
        with h5py.File(path, "r") as handle:
            self.length = len(handle["Y"])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.h5 is None:
            self.h5 = h5py.File(self.path, "r", swmr=True)
        x = self.h5["X"][index].astype(np.float32)
        return torch.from_numpy(apply_mask(x) if self.masked else np.ascontiguousarray(x)), index


class MCSDataset(Dataset):
    def __init__(self, path: Path):
        self.path, self.h5 = path, None
        with h5py.File(path, "r") as handle:
            self.length = len(handle["Y"])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.h5 is None:
            self.h5 = h5py.File(self.path, "r", swmr=True)
        return torch.from_numpy(normalize_original_mcs(self.h5["X"][index])), index


def infer(model, dataset, batch, workers, device):
    kwargs = dict(dataset=dataset, batch_size=batch, shuffle=False, num_workers=workers,
                  pin_memory=True, persistent_workers=workers > 0)
    if workers:
        kwargs["prefetch_factor"] = 2
    outputs = []
    with torch.inference_mode():
        for x, _ in DataLoader(**kwargs):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(outputs)


def statistics(y, logits):
    shifted = logits - logits.max(1, keepdims=True)
    prob = np.exp(shifted)
    prob /= prob.sum(1, keepdims=True)
    pred = prob.argmax(1)
    confusion = np.zeros((13, 13), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    recall = np.divide(np.diag(confusion), confusion.sum(1), out=np.zeros(13), where=confusion.sum(1) > 0)
    precision = np.divide(np.diag(confusion), confusion.sum(0), out=np.zeros(13), where=confusion.sum(0) > 0)
    f1 = np.divide(2 * recall * precision, recall + precision, out=np.zeros(13), where=(recall + precision) > 0)
    return {
        "modacc": float(np.mean(y == pred)), "macro_f1": float(f1.mean()), "confusion": confusion.tolist(),
        "recall": {LABELS[i]: float(recall[i]) for i in range(13)},
        "precision": {LABELS[i]: float(precision[i]) for i in range(13)},
        "predicted_count": {LABELS[i]: int(np.sum(pred == i)) for i in range(13)},
        "mean_predicted_confidence": {LABELS[i]: (float(prob[pred == i, i].mean()) if np.any(pred == i) else None) for i in range(13)},
    }, pred


def paired_ci(a, b, seed, nboot=10000):
    delta = a.astype(np.int8) - b.astype(np.int8)
    rng, values = np.random.default_rng(seed), np.empty(nboot)
    for i in range(nboot):
        values[i] = delta[rng.integers(0, len(delta), len(delta))].mean()
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def identity(path: Path):
    with h5py.File(path, "r") as handle:
        return {"y": handle["Y"][:].astype(int), "snr": handle["snr_db"][:].astype(np.float32),
                "uid": [decode(v) for v in handle["source_uid"][:]], "view": [decode(v) for v in handle["view_uid"][:]],
                "noise": handle["noise_seed"][:], "slot": handle["occupied_slot"][:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--paired-run", type=Path, required=True)
    parser.add_argument("--mask-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-h5", type=Path, help="可选：新建的11档Full development H5")
    parser.add_argument("--mcs-h5", type=Path, help="与--dev-h5配套的原16通道MCS development H5")
    parser.add_argument("--mcs-checkpoint", type=Path, help="与--mcs-h5配套的原16通道MCS checkpoint")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=271060)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"拒绝覆盖: {args.output}")
    if any(v is not None for v in (args.mcs_h5, args.mcs_checkpoint)) and not all(v is not None for v in (args.mcs_h5, args.mcs_checkpoint)):
        raise SystemExit("--mcs-h5 与 --mcs-checkpoint 必须同时提供")
    args.output.mkdir(parents=True)
    (args.output / "figures").mkdir()

    install_paths(args.project_root)
    from utils.runtime import configure_runtime, enable_resnet_only_pnn_import
    configure_runtime(args.seed, deterministic=False, tf32=True)
    enable_resnet_only_pnn_import()
    from models.pnn1 import Res1dNet311
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full, mask = Res1dNet311(in_channel=80, classes_num=13), Res1dNet311(in_channel=80, classes_num=13)
    full_ckpt = args.paired_run / "full_model/best_checkpoint_basic.pt"
    mask_ckpt = args.mask_run / "best_checkpoint_mask.pt"
    full.load_state_dict(torch.load(full_ckpt, map_location="cpu"))
    mask.load_state_dict(torch.load(mask_ckpt, map_location="cpu"))
    full.to(device).eval()
    mask.to(device).eval()
    h5path = args.dev_h5 or (args.paired_run / "full_data/development_test.h5")
    full_logits = infer(full, DevDataset(h5path, False), args.batch_size, args.workers, device)
    mask_logits = infer(mask, DevDataset(h5path, True), args.batch_size, args.workers, device)
    ids = identity(h5path)
    y, snr, uid, view, noise, slot = (ids[k] for k in ("y", "snr", "uid", "view", "noise", "slot"))

    if args.mcs_h5:
        mcs_ids = identity(args.mcs_h5)
        for key in ("y", "snr", "noise", "slot"):
            if not np.array_equal(ids[key], mcs_ids[key]):
                raise RuntimeError(f"11档Full/MCS身份字段不一致: {key}")
        for key in ("uid", "view"):
            if ids[key] != mcs_ids[key]:
                raise RuntimeError(f"11档Full/MCS身份字段不一致: {key}")
        mcs = Res1dNet311(in_channel=16, classes_num=13)
        mcs.load_state_dict(torch.load(args.mcs_checkpoint, map_location="cpu"))
        mcs.to(device).eval()
        mcs_logits = infer(mcs, MCSDataset(args.mcs_h5), args.batch_size, args.workers, device)
        mcs_hash, mcs_source = sha256(args.mcs_checkpoint), "checkpoint_recalculation"
    else:
        old = pd.read_csv(args.paired_run / "paired_development_results/paired_predictions.csv")
        mcs_logits = old[[f"mcs_logit_{name}" for name in LABELS]].to_numpy(np.float32)
        if uid != old.source_uid.tolist() or not np.array_equal(noise, old.noise_seed.to_numpy(dtype=np.uint64)):
            raise RuntimeError("旧MCS预测与Full开发集身份不匹配")
        full_old_mismatch = int(np.sum(full_logits.argmax(1) != old.full_pred_id.to_numpy()))
        full_old_mismatch_rate = full_old_mismatch / len(old)
        if full_old_mismatch_rate > 1e-3:
            raise RuntimeError(f"Full复算预测与旧CSV差异率超限: {full_old_mismatch}/{len(old)}={full_old_mismatch_rate:.6%}")
        mcs_hash = json.loads((args.paired_run / "paired_development_results/paired_metrics.json").read_text())["checkpoints"]["mcs"]
        mcs_source = "existing_paired_prediction_csv"

    grid = np.asarray(sorted(np.unique(snr)), dtype=np.float32)
    result = {"scope": "development_test_only", "seed": args.seed, "snr_grid": [float(v) for v in grid],
              "mcs_source": mcs_source, "datasets": {"full": str(h5path), "mcs": str(args.mcs_h5) if args.mcs_h5 else None},
              "checkpoints": {"full": sha256(full_ckpt), "mcs_mask": sha256(mask_ckpt), "mcs_original": mcs_hash}, "per_snr": {}}
    if not args.mcs_h5:
        result["old_full_prediction_recalculation"] = {"mismatch": full_old_mismatch, "rows": len(old),
                                                       "mismatch_rate": full_old_mismatch_rate, "threshold": 1e-3}
    for value in grid:
        selected, arms, predictions = np.isclose(snr, value), {}, {}
        for name, logits in (("full", full_logits), ("mcs_mask", mask_logits), ("mcs_original", mcs_logits)):
            arms[name], predictions[name] = statistics(y[selected], logits[selected])
        fc, mc = predictions["full"] == y[selected], predictions["mcs_mask"] == y[selected]
        result["per_snr"][f"{value:+g}"] = {**arms,
            "full_minus_mask": arms["full"]["modacc"] - arms["mcs_mask"]["modacc"],
            "paired_95ci": paired_ci(fc, mc, args.seed + int(value * 10)),
            "four_cells": {"both_correct": int(np.sum(fc & mc)), "full_only": int(np.sum(fc & ~mc)),
                            "mask_only": int(np.sum(~fc & mc)), "both_wrong": int(np.sum(~fc & ~mc))},
            "mask_minus_original": arms["mcs_mask"]["modacc"] - arms["mcs_original"]["modacc"]}

    low_stable = all(result["per_snr"][f"{s:+g}"]["paired_95ci"][0] > 0 for s in (-10.0, -6.0, -2.0))
    high = result["per_snr"]["+10"]
    original_high_delta = high["full"]["modacc"] - high["mcs_original"]["modacc"]
    mask_cross_significant = high["paired_95ci"][1] < 0
    if not mask_cross_significant and original_high_delta < 0:
        high_text = "原16通道MCS的高SNR反超在同架构mask后消失或不显著，支持输入表示/优化偏向解释"
    elif mask_cross_significant:
        high_text = "高SNR反超在同架构mask后仍显著，不能归因于16/80通道架构差异"
    else:
        high_text = "高SNR结果不确定"
    result["research_judgment"] = {"A_low_snr_full_advantage_stable": low_stable, "B_high_snr_interpretation": high_text,
        "C_sampling_attribution_evidence": "PARTIAL" if low_stable else "INCONCLUSIVE",
        "D_next_priority": "多seed复验" if low_stable else "继续输入表示/有效fan-in控制，不进入模型优化"}
    history = json.loads((args.mask_run / "history.json").read_text(encoding="utf-8"))["history"]
    best_epoch = json.loads((args.mask_run / "best_checkpoint_metadata.json").read_text())["epoch"]
    result["training_diagnostics"] = {"epochs": len(history),
        "collapsed_validation_classes_by_epoch": {str(x["epoch"]): [LABELS[i] for i, n in enumerate(x["validation_predicted_counts"]) if n < 10] for x in history if "validation_predicted_counts" in x},
        "bn_first": history[0].get("normalization_probe"), "bn_best": history[best_epoch - 1].get("normalization_probe"),
        "bn_last": history[-1].get("normalization_probe")}
    atomic_json(args.output / "control_metrics.json", result)

    fields = ["source_uid", "view_uid", "noise_seed", "snr_db", "occupied_slot", "true_id", "true_label", "full_pred", "mask_pred", "original_mcs_pred"] + [f"full_logit_{x}" for x in LABELS] + [f"mask_logit_{x}" for x in LABELS] + [f"original_mcs_logit_{x}" for x in LABELS]
    with (args.output / "control_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(y)):
            row = {"source_uid": uid[i], "view_uid": view[i], "noise_seed": int(noise[i]), "snr_db": float(snr[i]),
                   "occupied_slot": int(slot[i]), "true_id": int(y[i]), "true_label": LABELS[y[i]],
                   "full_pred": LABELS[int(full_logits[i].argmax())], "mask_pred": LABELS[int(mask_logits[i].argmax())],
                   "original_mcs_pred": LABELS[int(mcs_logits[i].argmax())]}
            for prefix, logits in (("full", full_logits), ("mask", mask_logits), ("original_mcs", mcs_logits)):
                row.update({f"{prefix}_logit_{LABELS[j]}": float(logits[i, j]) for j in range(13)})
            writer.writerow(row)

    xs = grid.astype(float)
    curves = {name: np.array([result["per_snr"][f"{s:+g}"][name]["modacc"] for s in xs]) for name in ("full", "mcs_mask", "mcs_original")}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, marker in (("full", "o"), ("mcs_mask", "s"), ("mcs_original", "^")):
        ax.plot(xs, curves[name], marker + "-", label=name)
    ax.set(xlabel="Full component SNR (dB)", ylabel="ModAcc", xticks=xs)
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "figures/control_curve.png", dpi=200)
    plt.close(fig)

    lines = ["# 单信号 Full/MCS-mask 同架构控制结果", "", "> Development Test，single seed；未读取 Final Test。", "",
             "|SNR|Full|MCS-mask|原16ch MCS|Full-mask|95%CI|mask-原MCS|", "|---:|---:|---:|---:|---:|---:|---:|"]
    for value in xs:
        row = result["per_snr"][f"{value:+g}"]
        lines.append(f"|{value:+g}|{row['full']['modacc']:.4f}|{row['mcs_mask']['modacc']:.4f}|{row['mcs_original']['modacc']:.4f}|{row['full_minus_mask']:+.4f}|[{row['paired_95ci'][0]:+.4f},{row['paired_95ci'][1]:+.4f}]|{row['mask_minus_original']:+.4f}|")
    judgment = result["research_judgment"]
    lines += ["", "## 研究判断", "", f"- A 低SNR Full优势稳定：{judgment['A_low_snr_full_advantage_stable']}。",
              f"- B：{judgment['B_high_snr_interpretation']}。", f"- C 采样结构归因证据：{judgment['C_sampling_attribution_evidence']}。",
              f"- D 下一优先级：{judgment['D_next_priority']}。", "",
              "完整四格统计、13类混淆矩阵、Recall、Precision、预测频数、平均预测置信度及训练归一化诊断见 control_metrics.json；逐样本三模型 logits 见 control_predictions.csv。", "",
              "本结果只能隔离输入通道架构差异的一部分；零填充还会改变第一层有效 fan-in，需结合 conv0/BN0 诊断解释。"]
    (args.output / "单信号_Full_MCS-mask_同架构控制报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "snr_grid": result["snr_grid"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
