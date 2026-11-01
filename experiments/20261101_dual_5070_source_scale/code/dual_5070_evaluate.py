"""冻结Validation最佳模型后，在同一验证和开发源上作旧/新模型比较。"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from dual_full_mcs_train_dev import Views, evaluate, metrics, LABELS
from v2_shell_models_portable import DualV6


def run(model, path, device):
    ds=Views(path,'mask')
    loader=DataLoader(ds,batch_size=256,shuffle=False,num_workers=2,pin_memory=True)
    met,idx,y,p=evaluate(model,loader,device)
    assert len(idx)==len(ds)
    rows=[]; classes=[]
    with h5py.File(path,'r') as h:
        assert len(idx)==len(h['source_uid'])*11
        for j,snr in enumerate(range(-10,11,2)):
            use=idx%11==j
            sm=metrics(y[use],p[use]); rows.append({'snr_db':snr,**sm})
        for i,label in enumerate(LABELS):
            truth=y.reshape(-1)==i; pred=p.reshape(-1)==i
            classes.append({'label':label,'support':int(truth.sum()),
                'recall':float((truth&pred).sum()/truth.sum()),
                'predicted_count':int(pred.sum())})
    return met,rows,classes,idx,y,p


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    assert not args.out.exists()
    old=args.run/'train_mask_seed271070/best_validation.pt'
    new=args.run/'train_mask_5070_seed271070_20261101/best_validation.pt'
    data=args.run/'data_dual_dev1690_seed271070/dual'
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.out.mkdir(parents=True,exist_ok=False)
    summary={}; snr_rows=[]; class_rows=[]; predictions={}
    for arm,ckpath in (('old1690',old),('new5070',new)):
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        model=DualV6(in_channels=80,num_classes=13)
        model.load_state_dict(ck['model']);model.to(device).eval()
        summary[arm]={'best_validation_epoch':int(ck['epoch']),'splits':{}}
        for split in ('train','validation','development_test'):
            m,rows,classes,idx,y,p=run(model,data/split/'dataset.h5',device)
            summary[arm]['splits'][split]=m
            for row in rows:snr_rows.append({'arm':arm,'split':split,**row})
            for row in classes:class_rows.append({'arm':arm,'split':split,**row})
            if split in ('validation','development_test'):
                predictions[(arm,split)]=(idx,y,p)
    for split in ('validation','development_test'):
        ia,ya,pa=predictions[('old1690',split)]
        ib,yb,pb=predictions[('new5070',split)]
        assert np.array_equal(ia,ib) and np.array_equal(ya,yb)
        d=np.all(pb==yb,axis=1).astype(int)-np.all(pa==ya,axis=1).astype(int)
        summary['paired_'+split]={'pair_exact_difference':float(d.mean()),
            'new_only_correct':int((d==1).sum()),'old_only_correct':int((d==-1).sum())}
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,rows in (('snr_metrics.csv',snr_rows),('class_metrics.csv',class_rows)):
        with (args.out/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
