"""MCS-mask80 的数据、模型、初始化及单 batch 资源门禁。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_mask80 import COSET, Views, state_hash


def check_split(path: Path, cache_path: Path) -> dict:
    dataset = Views(path, "mask80", None)
    assert dataset.views == 11
    with h5py.File(path) as h, h5py.File(cache_path) as cache:
        slots = h["occupied_slots"][:]
        labels = h["modulation_labels"][:]
        y_mod = h["y_mod"][:]
        assert np.all(slots[:, 0] < slots[:, 1])
        assert np.array_equal(np.take_along_axis(y_mod, slots, axis=1), labels)
        indices = [0, dataset.sources // 2, dataset.sources - 1]
        count = 0
        active = np.asarray([x for c in COSET for x in (2 * c, 2 * c + 1)], dtype=np.int64)
        inactive = np.setdiff1d(np.arange(80), active)
        for i in indices:
            for j in (0, 5, 10):
                x, _, y, idx = dataset[i * 11 + j]
                x = x.numpy()
                mcs = h["mcs_iq"][i, j].T / 8.0
                full = cache["noisy_full_iq"][i, j].reshape(1024, 40, 2).transpose(1, 2, 0).reshape(80, 1024) / 8.0
                assert x.shape == (80, 1024) and np.array_equal(x[active], mcs)
                assert np.array_equal(x[active], full[active])
                assert np.count_nonzero(x[inactive]) == 0
                assert np.array_equal(y.numpy(), labels[i]) and idx == i * 11 + j
                count += 1
        return {"sources": dataset.sources, "views": dataset.sources * 11,
                "sampled_views_checked_against_mcs_and_full": count,
                "all_sources_label_order_checked": True, "active_channels": active.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--full-cache", type=Path, required=True)
    p.add_argument("--full-checkpoint", type=Path, required=True)
    p.add_argument("--shell-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    assert not args.out.exists()
    splits = ("train", "validation", "development_test")
    checked = {s: check_split(args.data / "dual" / s / "dataset.h5", args.full_cache / (s + ".h5"))
               for s in splits}
    assert [checked[s]["sources"] for s in splits] == [16900, 3380, 3380]

    sys.path.insert(0, str(args.shell_root))
    from utils.runtime import configure_runtime, enable_resnet_only_pnn_import
    from utils.utils import init_weights
    configure_runtime(271060, deterministic=False, tf32=True)
    enable_resnet_only_pnn_import()
    from models.pnn1 import Res1dNet31
    full_state = torch.load(args.full_checkpoint, map_location="cpu")
    model = init_weights(Res1dNet31(in_channel=80, classes_num=26))
    assert set(model.state_dict()) == set(full_state)
    assert all(model.state_dict()[name].shape == full_state[name].shape for name in full_state)
    params = sum(t.numel() for t in model.parameters())
    full_params = sum(full_state[name].numel() for name, _ in model.named_parameters())
    assert params == full_params
    expected_hash = state_hash(model.state_dict())
    # 重新播种只核对初始化协议的确定性；历史 Full 实际初态未保存，不虚称实物相同。
    configure_runtime(271060, deterministic=False, tf32=True)
    reference = init_weights(Res1dNet31(in_channel=80, classes_num=26))
    tensor_equal = all(torch.equal(model.state_dict()[name], reference.state_dict()[name])
                       for name in model.state_dict())
    assert tensor_equal and expected_hash == state_hash(reference.state_dict())
    del reference, full_state

    device = torch.device("cuda")
    model = model.to(device)
    loader = DataLoader(Views(args.data / "dual" / "train" / "dataset.h5", "mask80", None),
                        batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    x, target, _, _ = next(iter(loader))
    assert x.shape == (512, 80, 1024) and target.shape == (512, 26)
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.amp.autocast(enabled=True):
        logits = model(x.to(device, non_blocking=True))
        smooth = (target * (1 - 0.026) + 0.026 / 26).to(device, non_blocking=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, smooth)
    assert logits.shape == (512, 26) and torch.isfinite(loss)
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    result = {"gate": "PASS", "split_checks": checked, "parameter_count": params,
              "full80_parameter_count": full_params, "tensor_shapes_match_full_checkpoint": True,
              "two_reconstructed_initial_states_equal_tensorwise": True,
              "historical_full_actual_initial_state_saved": False,
              "expected_initial_state_sha256": expected_hash,
              "batch512_forward_backward_pass": True, "peak_allocated_gib": peak,
              "input_scale": "physical_iq_div_8", "coset": COSET.tolist()}
    args.out.mkdir(parents=True)
    (args.out / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

