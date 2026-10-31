from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mask_common import ACTIVE_CHANNELS, COSET, SOURCE_CHANNELS, atomic_json, apply_mask, install_paths, sha256


class LazyMaskedFull(Dataset):
    def __init__(self, path: Path):
        self.path = path; self.h5 = None
        with h5py.File(path, "r") as h5: self.length = len(h5["Y"])
    def __len__(self): return self.length
    def __getitem__(self, index):
        if self.h5 is None: self.h5 = h5py.File(self.path, "r", swmr=True)
        x = apply_mask(self.h5["X"][index].astype(np.float32))
        return torch.from_numpy(x), torch.tensor(int(self.h5["Y"][index]), dtype=torch.long)


def make_loader(path, batch_size, workers, train, seed):
    kwargs = dict(dataset=LazyMaskedFull(path), batch_size=batch_size, shuffle=train, drop_last=train,
                  num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                  generator=torch.Generator().manual_seed(seed))
    if workers: kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def estimate_macs(model, device):
    total=[0];hooks=[]
    def conv(module,inputs,output):total[0]+=int(output.numel()*(module.in_channels//module.groups)*module.kernel_size[0])
    def linear(module,inputs,output):total[0]+=int(output.numel()*module.in_features)
    for module in model.modules():
        if isinstance(module,nn.Conv1d):hooks.append(module.register_forward_hook(conv))
        elif isinstance(module,nn.Linear):hooks.append(module.register_forward_hook(linear))
    model.eval()
    with torch.inference_mode(),torch.cuda.amp.autocast(enabled=True):model(torch.zeros(1,80,1024,device=device))
    for hook in hooks:hook.remove()
    return total[0]


def bn_probe(model, batch, device):
    captured = {}
    def capture_conv(module, inputs, output): captured["conv"] = output.detach().float().cpu()
    def capture_bn(module, inputs, output): captured["bn"] = output.detach().float().cpu()
    h1 = model.conv0.register_forward_hook(capture_conv)
    h2 = model.bn0.register_forward_hook(capture_bn)
    was_training = model.training; model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        model(batch.to(device, non_blocking=True))
    h1.remove(); h2.remove(); model.train(was_training)
    active = model.conv0.weight[:, ACTIVE_CHANNELS]
    inactive_index = np.setdiff1d(np.arange(80), ACTIVE_CHANNELS)
    inactive = model.conv0.weight[:, inactive_index]
    return {"conv_std": float(captured["conv"].std()), "bn_std": float(captured["bn"].std()),
            "bn_running_mean_abs_max": float(model.bn0.running_mean.abs().max()),
            "bn_running_var_min": float(model.bn0.running_var.min()), "bn_running_var_max": float(model.bn0.running_var.max()),
            "conv_active_weight_norm": float(active.norm()), "conv_inactive_weight_norm": float(inactive.norm())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True); parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mapping-gate", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=271060); parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    mapping = json.loads(args.mapping_gate.read_text(encoding="utf-8"))
    if mapping.get("status") != "PASS" or mapping.get("coset_ordered") != COSET.tolist():
        raise SystemExit("映射门禁未通过或 Coset 不一致")
    if args.output.exists(): raise SystemExit(f"拒绝覆盖: {args.output}")
    if not torch.cuda.is_available(): raise SystemExit("训练要求 CUDA")
    args.output.mkdir(parents=True)
    install_paths(args.project_root)
    from utils.runtime import configure_runtime, enable_resnet_only_pnn_import, save_compatible_state_dict
    from utils.utils import init_weights, WarmUpLR
    configure_runtime(args.seed, deterministic=False, tf32=True); enable_resnet_only_pnn_import()
    from models.pnn1 import Res1dNet311
    torch.cuda.set_device(0);device = torch.device("cuda")
    train_loader = make_loader(args.data_root / "train.h5", args.batch_size, args.workers, True, args.seed)
    val_loader = make_loader(args.data_root / "validation.h5", args.batch_size, args.workers, False, args.seed)
    model = init_weights(Res1dNet311(in_channel=80, classes_num=13)).to(device)
    params=sum(p.numel() for p in model.parameters());macs=estimate_macs(model,device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup)
    warmup = WarmUpLR(optimizer, args.warmup)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    resource={"host_ram_total_gib":None,"gpu":torch.cuda.get_device_name(0),"gpu_total_gib":torch.cuda.get_device_properties(0).total_memory/2**30,
              "batch_size":args.batch_size,"workers":args.workers,"parameters":params,"approx_macs_per_sample":macs,"approx_flops_per_sample":2*macs}
    try:
        import psutil
        resource["host_ram_total_gib"]=psutil.virtual_memory().total/2**30
    except Exception:pass
    try:
        x,y=next(iter(train_loader));x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):loss=criterion(model(x),y)
        scaler.scale(loss).backward();optimizer.zero_grad(set_to_none=True);del x,y,loss;torch.cuda.empty_cache();resource["batch_probe"]="PASS"
    except torch.cuda.OutOfMemoryError:
        resource["batch_probe"]="FAIL_OOM";atomic_json(args.output/"resource_preflight.json",resource);raise
    atomic_json(args.output/"resource_preflight.json",resource)
    probe_batch = next(iter(val_loader))[0][:min(64, args.batch_size)]
    best = -1.0; best_epoch = 0; history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time(); model.train(); loss_sum = 0.0; seen = 0
        for x,y in train_loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True): loss=criterion(model(x),y)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            loss_sum += float(loss) * len(y); seen += len(y)
        model.eval(); correct=0;n=0;predicted=np.zeros(13,dtype=np.int64)
        with torch.inference_mode():
            for x,y in val_loader:
                x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
                with torch.cuda.amp.autocast(enabled=True): pred=model(x).argmax(1)
                correct += int((pred==y).sum());n += len(y);predicted += np.bincount(pred.cpu().numpy(),minlength=13)
        val_acc=correct/n
        row={"epoch":epoch,"train_loss":loss_sum/seen,"validation_accuracy":val_acc,"validation_predicted_counts":predicted.tolist(),
             "lr":optimizer.param_groups[0]["lr"],"seconds":time.time()-started,"normalization_probe":bn_probe(model,probe_batch,device)}
        history.append(row);print(json.dumps(row),flush=True)
        if val_acc >= best:
            best=val_acc;best_epoch=epoch
            save_compatible_state_dict(model,str(args.output),"best_checkpoint_mask1.pt","best_checkpoint_mask.pt")
            atomic_json(args.output/"best_checkpoint_metadata.json",{"selected_on":"validation_accuracy","best":best,"epoch":best_epoch,"tie_rule":"latest_ge","development_test_used":False})
        if epoch < args.warmup: warmup.step()
        else: scheduler.step()
        atomic_json(args.output/"history.json",{"history":history})
    config={"arm":"mcs_mask_80ch_position_control","seed":args.seed,"batch_size":args.batch_size,"epochs":args.epochs,"warmup":args.warmup,
            "optimizer":"Adam","lr":0.01,"weight_decay":0.0001,"scheduler":"CosineAnnealingLR","label_smoothing":0.1,
            "amp":True,"tf32":True,"model":"Res1dNet311","in_channel":80,"parameters":params,"approx_macs_per_sample":macs,"approx_flops_per_sample":2*macs,
            "coset":COSET.tolist(),"active_channels":ACTIVE_CHANNELS.tolist(),"source_channels":SOURCE_CHANNELS.tolist(),"zero_channels":64,
            "representation":"candidate source channels mapped to fixed baseline active channels in branch-rank order",
            "per_sample_rms":False,"best_validation_accuracy":best,"best_epoch":best_epoch,
            "checkpoint_sha256":sha256(args.output/"best_checkpoint_mask.pt")}
    atomic_json(args.output/"training_config.json",config);print(json.dumps(config,ensure_ascii=False))


if __name__ == "__main__": main()
