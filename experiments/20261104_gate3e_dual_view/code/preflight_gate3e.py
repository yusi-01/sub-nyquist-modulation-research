"""正式 v2 H5 的 GATE-3E 单批前后向和 Train/Validation α 门禁。"""
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe(v: np.ndarray) -> dict:
    return {"mean": float(np.mean(v)), "median": float(np.median(v)),
            "p05": float(np.quantile(v, .05)), "p95": float(np.quantile(v, .95)),
            "min": float(np.min(v)), "max": float(np.max(v))}


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("data", "cache_root", "gate3c_code", "gate3d_code", "shell_root", "out"):
        ap.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    sys.path.insert(0, str(a.gate3c_code))
    sys.path.insert(0, str(a.gate3d_code))
    sys.path.insert(0, str(a.shell_root))
    from gate3c_physical import blind_somp
    from blind_input import BlindViews, BlindProjector
    from dual_view import DualViewTransform, fused_logits
    from utils.runtime import configure_runtime, enable_resnet_only_pnn_import

    configure_runtime(271060, deterministic=False, tf32=True)
    enable_resnet_only_pnn_import()
    from models.pnn1 import Res1dNet31
    from utils.utils import init_weights

    device = torch.device("cuda:0")
    assert torch.cuda.is_available()
    paths = {s: a.data / "dual" / s / "dataset.h5" for s in ("train", "validation")}
    datasets = {s: BlindViews(p, a.cache_root / (s + "_support.csv")) for s, p in paths.items()}
    assert [datasets[s].sources for s in paths] == [16900, 3380]
    transform = DualViewTransform().to(device).eval()
    assert sum(p.numel() for p in transform.parameters()) == 0
    model = init_weights(Res1dNet31(in_channel=16, classes_num=26)).to(device).train()
    assert sum(p.numel() for p in model.parameters()) == 6320026
    bn = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    ln = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
    dropout = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert bn and not ln

    batch = next(iter(DataLoader(datasets["train"], batch_size=512, shuffle=False, num_workers=0)))
    x, target, y, idx, support = batch
    x, target, support = x.to(device), target.to(device), support.to(device)
    # 当前 x 尚未缩放；Raw 的唯一缩放发生在变换内部。
    raw, proj, alpha, gamma, sigma2 = transform(x, support)
    assert raw.shape == proj.shape == (512, 16, 1024)
    assert torch.equal(raw, x / 8)
    assert torch.equal(proj, BlindProjector().to(device)(x, support))
    with h5py.File(paths["train"], "r") as h:
        source_indices = idx.numpy() // 11
        slots = h["occupied_slots"][:][source_indices]
        labels = h["modulation_labels"][:][source_indices]
        assert np.all(slots[:, 0] < slots[:, 1])
        assert np.array_equal(y.numpy(), labels)
        assert (target.sum(dim=1) == 2).all()
        assert (target[:, :13].sum(dim=1) == 1).all()
        assert (target[:, 13:].sum(dim=1) == 1).all()
        assert np.array_equal(target[:, :13].argmax(1).cpu().numpy(), labels[:, 0])
        assert np.array_equal(target[:, 13:].argmax(1).cpu().numpy(), labels[:, 1])
        from gate3c_physical import complex_mcs
        for row_index in range(3):
            source, view = divmod(row_index, 11)
            estimate, _ = blind_somp(np.fft.fft(complex_mcs(h["mcs_iq"][source, view]), axis=1, norm="ortho"))
            assert set(estimate.tolist()) == set(support[row_index].cpu().tolist())
    assert bool((alpha > 0).all() and (alpha <= .5).all())
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = [int(m.num_batches_tracked.item()) for m in bn]
    with torch.cuda.amp.autocast(enabled=True):
        logits, logits_raw, logits_proj = fused_logits(model, raw, proj, alpha)
        smoothed = target * (1 - .026) + .026 / 26
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, smoothed)
    assert logits.shape == logits_raw.shape == logits_proj.shape == (512, 26)
    assert bool(torch.isfinite(logits).all() and torch.isfinite(loss))
    loss_value = float(loss.item())
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    after = [int(m.num_batches_tracked.item()) for m in bn]
    assert all(y == x + 1 for x, y in zip(before, after))
    torch.cuda.synchronize(device)
    memory = {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
    # 本门禁绝不执行 optimizer.step。
    del logits, logits_raw, logits_proj, loss, model, raw, proj, alpha, gamma, sigma2
    torch.cuda.empty_cache()

    rows = []
    wrong_rows = []
    for split, dataset in datasets.items():
        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)
        with h5py.File(paths[split], "r") as h:
            true_slots = h["occupied_slots"][:]
            snrs = h["target_full_component_snr_db"][:]
            for x, _, _, index, support in loader:
                # 特征/α 计算仅接收 MCS IQ 与冻结盲支持；真值和 SNR 在计算后才用于 QA 分组。
                with torch.inference_mode():
                    _, _, alpha, gamma, sigma2 = transform(
                        x.to(device, non_blocking=True), support.to(device, non_blocking=True))
                aa, gg, nn = alpha.cpu().numpy(), gamma.cpu().numpy(), sigma2.cpu().numpy()
                ids = index.numpy()
                ss = support.numpy()
                for k, row_index in enumerate(ids):
                    source, view = divmod(int(row_index), 11)
                    actual = true_slots[source]
                    exact = int(set(map(int, ss[k])) == set(map(int, actual)))
                    snr = int(snrs[source, view])
                    rows.append((split, snr, exact, float(aa[k]), float(gg[k]), float(nn[k])))
                    if not exact:
                        wrong_rows.append((split, int(row_index), snr, int(actual[0]), int(actual[1]),
                                           int(ss[k, 0]), int(ss[k, 1]), float(aa[k]), float(gg[k]), float(nn[k])))
    values = np.array(rows, dtype=object)
    summaries = []
    support_summaries = []
    for split in paths:
        for snr in range(-10, 11, 2):
            selected = values[(values[:, 0] == split) & (values[:, 1] == snr)]
            assert len(selected) == datasets[split].sources
            finite = np.isfinite(selected[:, 3:].astype(float)).all(axis=1)
            assert finite.all()
            alpha_values = selected[:, 3].astype(float)
            assert np.all((alpha_values > 0) & (alpha_values <= .5))
            summaries.append({"split": split, "snr_db_for_qa_only": snr, "views": len(selected),
                              "support_errors": int(np.sum(selected[:, 2].astype(int) == 0)),
                              "nonfinite": int(np.sum(~finite)),
                              **{"alpha_" + k: v for k, v in describe(alpha_values).items()},
                              **{"gamma_" + k: v for k, v in describe(selected[:, 4].astype(float)).items()},
                              **{"sigma2_" + k: v for k, v in describe(selected[:, 5].astype(float)).items()}})
            for exact in (0, 1):
                group = selected[selected[:, 2].astype(int) == exact]
                if len(group):
                    support_summaries.append({"split": split, "snr_db_for_qa_only": snr,
                                              "support_exact_for_qa_only": exact, "views": len(group),
                                              **{"alpha_" + k: v for k, v in describe(group[:, 3].astype(float)).items()},
                                              **{"gamma_" + k: v for k, v in describe(group[:, 4].astype(float)).items()},
                                              **{"sigma2_" + k: v for k, v in describe(group[:, 5].astype(float)).items()}})
    for split in paths:
        means = [r["alpha_mean"] for r in summaries if r["split"] == split]
        assert means[0] > means[-1]
        # 总体下降趋势；不把有限样本的每一步严格单调当作物理定律。
        assert np.corrcoef(np.arange(11), means)[0, 1] < -.8
    a.out.mkdir(parents=True)
    with (a.out / "alpha_Train_Validation逐SNR审计.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, summaries[0].keys()); writer.writeheader(); writer.writerows(summaries)
    with (a.out / "alpha_支持正确错误分组.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, support_summaries[0].keys()); writer.writeheader(); writer.writerows(support_summaries)
    with (a.out / "支持错误alpha审计.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "row_index", "snr_db_for_qa_only", "true_slot_1", "true_slot_2",
                         "estimated_slot_1", "estimated_slot_2", "alpha", "gamma", "sigma2"])
        writer.writerows(wrong_rows)
    report = {"status": "PASS", "python": sys.version, "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
              "model_parameters": 6320026, "batch_original_views": 512,
              "batch_cnn_inputs": 1024, "batchnorm_modules": len(bn),
              "layernorm_modules": len(ln), "dropout_modules": len(dropout),
              "functional_dropout_in_res1dnet31": True,
              "batchnorm_update_count_each": 1,
              "loss": loss_value, "memory": memory,
              "rows": {s: datasets[s].sources * 11 for s in paths},
              "support_errors": {s: sum(r[0] == s for r in wrong_rows) for s in paths},
              "input_sha256": {s: sha256(paths[s]) for s in paths},
              "support_cache_sha256": {s: sha256(a.cache_root / (s + "_support.csv")) for s in paths}}
    (a.out / "P0门禁摘要.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("GATE3E_P0_PASS", json.dumps({"memory": memory, "support_errors": report["support_errors"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

