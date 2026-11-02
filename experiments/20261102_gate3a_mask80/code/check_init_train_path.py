"""按正式训练脚本的导入和构造顺序重建 mask80 初始状态。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_mask80 import Views, state_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--shell-root", type=Path, required=True)
    p.add_argument("--full-checkpoint", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.shell_root))
    from utils.runtime import configure_runtime, enable_resnet_only_pnn_import
    configure_runtime(271060, deterministic=False, tf32=True)
    enable_resnet_only_pnn_import()
    from models.pnn1 import Res1dNet31
    from utils.utils import WarmUpLR, init_weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = {s: args.data / "dual" / s / "dataset.h5" for s in ("train", "validation", "development_test")}
    datasets = {s: Views(path, "mask80", None) for s, path in paths.items()}
    assert [datasets[s].sources for s in paths] == [16900, 3380, 3380]
    g = torch.Generator().manual_seed(271060)
    tr = DataLoader(datasets["train"], batch_size=512, shuffle=True, num_workers=4,
                    pin_memory=True, persistent_workers=True, generator=g)
    va = DataLoader(datasets["validation"], batch_size=512, shuffle=False, num_workers=4,
                    pin_memory=True, persistent_workers=True)
    model = init_weights(Res1dNet31(in_channel=80, classes_num=26)).to(device)
    full_state = torch.load(args.full_checkpoint, map_location="cpu")
    assert set(model.state_dict()) == set(full_state)
    assert all(model.state_dict()[name].shape == full_state[name].shape for name in full_state)
    print(json.dumps({"initial_state_sha256": state_hash(model.state_dict()),
                      "parameter_count": sum(x.numel() for x in model.parameters()),
                      "full_checkpoint_tensor_shapes_match": True,
                      "historical_full_initial_state_saved": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()

