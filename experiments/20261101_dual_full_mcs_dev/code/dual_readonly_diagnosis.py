"""既有最佳 checkpoint 的训练/验证泛化及源标签顺序核查；不训练、不读 Final Test。"""
from pathlib import Path
import argparse,json,sys
import h5py,numpy as np,torch
from torch.utils.data import DataLoader
from dual_full_mcs_train_dev import Views,evaluate,LABELS
from v2_shell_models_portable import DualV6
import v2_formal_core as core

ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
data=args.run/'data_dual_dev1690_seed271070'; out=args.run/'readonly_diagnosis.json'
assert data.is_dir() and not out.exists()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
record={'status':'PASS','arms':{},'source_regeneration':{}}
cfg=json.loads((data/'config_snapshot.json').read_text(encoding='utf-8'))
for split in ('train','validation','development_test'):
    schedule=core.split_schedule('dual',split,cfg['counts_by_task']['dual'][split],cfg['master_seed'])
    path=data/'dual'/split/'dataset.h5'
    with h5py.File(path) as h:
        n=len(h['source_uid']); checks=[]
        for row in (0,n//2,n-1):
            src=core.generate_clean_source('dual',split,row,cfg['master_seed'],schedule['mods_component_order'][row],schedule['slots_component_order'][row])
            good=(h['source_uid'][row].decode()==src['source_uid'] and np.array_equal(h['modulation_labels'][row],src['modulation_labels']) and np.array_equal(h['occupied_slots'][row],src['occupied_slots']) and h['clean_full_hash'][row].decode()==src['clean_full_hash'])
            checks.append(bool(good))
        assert all(checks),split
        record['source_regeneration'][split]={'checked_rows':[0,n//2,n-1],'pass':all(checks)}
for arm in ('full','mask'):
    ck=torch.load(args.run/f'train_{arm}_seed271070'/'best_validation.pt',map_location='cpu',weights_only=False)
    model=DualV6(in_channels=80,num_classes=13);model.load_state_dict(ck['model']);model.to(device)
    record['arms'][arm]={'checkpoint_epoch':int(ck['epoch']),'splits':{}}
    for split in ('train','validation'):
        ds=Views(data/'dual'/split/'dataset.h5',arm)
        loader=DataLoader(ds,batch_size=256,shuffle=False,num_workers=2,pin_memory=device.type=='cuda')
        met,idx,y,p=evaluate(model,loader,device)
        class_rows=[]
        yt=y.reshape(-1);pt=p.reshape(-1)
        for i,label in enumerate(LABELS):
            true=yt==i; pred=pt==i
            class_rows.append({'label':label,'support':int(true.sum()),'recall':float((true&pred).sum()/true.sum()),'predicted_count':int(pred.sum())})
        record['arms'][arm]['splits'][split]={'n_views':len(idx),'metrics':met,'class_metrics':class_rows}
out.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS',[(a,{s:round(v['metrics']['pair_exact'],4) for s,v in d['splits'].items()}) for a,d in record['arms'].items()])
