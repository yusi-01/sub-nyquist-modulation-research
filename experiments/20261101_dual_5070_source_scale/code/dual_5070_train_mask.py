"""旧1690源加新增3380源的唯一MCS-mask80训练；仅Validation选模。"""
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from dual_full_mcs_train_dev import Views, evaluate
from v2_shell_models_portable import DualV6


def init_model(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return DualV6(in_channels=80, num_classes=13)


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', type=Path, required=True)
    ap.add_argument('--append', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    seed = 271070
    old_h5 = args.old/'dual/train/dataset.h5'
    add_h5 = args.append/'dual/train_append5070/dataset.h5'
    val_h5 = args.old/'dual/validation/dataset.h5'
    train = ConcatDataset([Views(old_h5,'mask'), Views(add_h5,'mask')])
    validation = Views(val_h5,'mask')
    assert len(train) == 5070*11 and len(validation) == 338*11
    args.out.mkdir(parents=True,exist_ok=False)
    # 用原代码相同的种子与初始化路径重放一次，逐张量核对新模型初值。
    replay = init_model(seed)
    initial_hash = state_hash(replay)
    model = init_model(seed)
    assert all(torch.equal(replay.state_dict()[k],v) for k,v in model.state_dict().items())
    assert state_hash(model) == initial_hash
    torch.save({'state_dict':model.state_dict(),'seed':seed,'sha256':initial_hash},args.out/'initial_state.pt')
    device = torch.device('cuda')
    assert torch.cuda.is_available()
    model.to(device)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train,batch_size=256,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,generator=g)
    val_loader = DataLoader(validation,batch_size=256,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
    opt = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=200)
    old_steps = math.ceil(1690*11/256)
    new_steps = len(train_loader)
    diag_epoch = round(200*old_steps/new_steps)
    config = {'seed':seed,'arm':'mask','model':'DualV6(80,13)','sources':5070,'old_sources':1690,
        'new_sources':3380,'batch':256,'optimizer':'Adam','lr':.001,'weight_decay':1e-4,
        'label_smoothing':.1,'epochs':200,'checkpoint_selection':'Validation PairExact; Macro-F1 tie-break',
        'old_steps_per_epoch':old_steps,'new_steps_per_epoch':new_steps,'old_200_epoch_steps':200*old_steps,
        'equal_update_diagnostic_epoch':diag_epoch,'initial_state_sha256':initial_hash,
        'old_data':str(args.old),'append_data':str(args.append)}
    (args.out/'training_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    history=[]; best=(-1,-1); best_epoch=None; steps=0
    for epoch in range(1,201):
        model.train(); losses=[]
        for x,y,_ in train_loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            z=model(x); loss=F.cross_entropy(z.reshape(-1,13),y.reshape(-1),label_smoothing=.1)
            loss.backward();opt.step();steps+=1
            losses.append((float(loss.item()),len(x)))
        scheduler.step(); vm,_,_,_=evaluate(model,val_loader,device)
        row={'epoch':epoch,'optimizer_steps':steps,'train_loss':sum(a*n for a,n in losses)/sum(n for _,n in losses),
            'validation_pair_exact':vm['pair_exact'],'validation_component_modacc':vm['component_modacc'],
            'validation_macro_f1':vm['macro_f1'],'validation_loss':vm['loss'],'lr':scheduler.get_last_lr()[0]}
        history.append(row)
        if (vm['pair_exact'],vm['macro_f1'])>best:
            best=(vm['pair_exact'],vm['macro_f1']);best_epoch=epoch
            torch.save({'model':model.state_dict(),'epoch':epoch,'validation':vm,'config':config},args.out/'best_validation.pt')
        if epoch==diag_epoch:
            torch.save({'model':model.state_dict(),'epoch':epoch,'optimizer_steps':steps,'role':'equal_update_diagnostic_only'},args.out/'equal_update_diagnostic.pt')
        with (args.out/'history.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
        print('epoch',epoch,'steps',steps,'train_loss',row['train_loss'],'val_pair',vm['pair_exact'],flush=True)
    (args.out/'completion.json').write_text(json.dumps({'best_epoch':best_epoch,'best_validation_pair_exact':best[0],
        'total_optimizer_steps':steps,'diagnostic_epoch':diag_epoch},ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__': main()
