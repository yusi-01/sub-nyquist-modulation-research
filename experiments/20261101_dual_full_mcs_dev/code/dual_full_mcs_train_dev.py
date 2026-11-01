"""双信号 Full80/MCS-mask80 单种子开发控制；只读 Validation 选点与 Development 评价。"""
from pathlib import Path
import argparse, csv, json, random, sys
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from v2_shell_models_portable import DualV6

COSET=np.asarray([16,11,1,0,27,24,37,31])
ACTIVE=np.asarray([x for c in COSET for x in (2*c,2*c+1)])
LABELS=['APSK16','APSK32','APSK64','ASK8','BPSK','OQPSK','PSK16','PSK8','QAM128','QAM16','QAM256','QAM64','QPSK']

class Views(Dataset):
    def __init__(self,path,arm):
        self.path=Path(path); self.arm=arm; self.h=None
        with h5py.File(self.path) as f:
            self.ns,self.nsnr=f['noisy_full_iq'].shape[:2]
            assert self.nsnr==11 and f['modulation_labels'].shape==(self.ns,2)
            assert f.attrs['task']=='dual' and list(f.attrs['Coset'])==COSET.tolist()
    def __len__(self): return self.ns*self.nsnr
    def __getitem__(self,idx):
        if self.h is None: self.h=h5py.File(self.path,'r')
        row,j=divmod(idx,self.nsnr)
        iq=self.h['noisy_full_iq'][row,j].reshape(1024,40,2).transpose(1,2,0).reshape(80,1024)
        if self.arm=='mask':
            x=np.zeros((80,1024),dtype=np.float32)
            x[ACTIVE]=iq[ACTIVE]
        else: x=np.ascontiguousarray(iq,dtype=np.float32)
        y=self.h['modulation_labels'][row].astype(np.int64)
        return torch.from_numpy(x),torch.from_numpy(y),idx

def metrics(y,p):
    pair=float(np.mean(np.all(y==p,axis=1)))
    component=float(np.mean(y==p))
    f1=[]
    for i in range(13):
        tp=np.sum((y==i)&(p==i)); fp=np.sum((y!=i)&(p==i)); fn=np.sum((y==i)&(p!=i))
        f1.append(0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
    return {'pair_exact':pair,'component_modacc':component,'macro_f1':float(np.mean(f1))}

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); ys=[]; ps=[]; ids=[]; losses=[]
    for x,y,idx in loader:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        logits=model(x); losses.append((float(F.cross_entropy(logits.reshape(-1,13),y.reshape(-1)).item()),len(x)))
        ys.append(y.cpu().numpy()); ps.append(logits.argmax(-1).cpu().numpy()); ids.append(idx.numpy())
    y=np.concatenate(ys); p=np.concatenate(ps); idx=np.concatenate(ids)
    m=metrics(y,p); m['loss']=sum(a*n for a,n in losses)/sum(n for _,n in losses)
    return m,idx,y,p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--arm',choices=('full','mask'),required=True); ap.add_argument('--seed',type=int,default=271070)
    ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--batch',type=int,default=256)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type=='cuda','正式开发训练需要 CUDA'
    roots={s:args.data/'dual'/s/'dataset.h5' for s in ('train','validation','development_test')}
    for p in roots.values(): assert p.is_file(),p
    tr=Views(roots['train'],args.arm); va=Views(roots['validation'],args.arm)
    g=torch.Generator().manual_seed(args.seed)
    train_loader=DataLoader(tr,batch_size=args.batch,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,generator=g)
    val_loader=DataLoader(va,batch_size=args.batch,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
    model=DualV6(in_channels=80,num_classes=13).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    config={'arm':args.arm,'seed':args.seed,'epochs':args.epochs,'batch':args.batch,'optimizer':'Adam','lr':.001,'weight_decay':1e-4,'label_smoothing':.1,'checkpoint_selection':'Validation PairExact; Macro-F1 tie-break','model':'DualV6 80 input 2x13','source_data':str(args.data.resolve())}
    (args.out/'training_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    history=[]; best=(-1,-1); best_epoch=None
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for x,y,_ in train_loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits=model(x); loss=F.cross_entropy(logits.reshape(-1,13),y.reshape(-1),label_smoothing=.1)
            loss.backward(); opt.step(); losses.append((float(loss.item()),len(x)))
        scheduler.step(); vm,_,_,_=evaluate(model,val_loader,device)
        row={'epoch':epoch,'train_loss':sum(a*n for a,n in losses)/sum(n for _,n in losses),'validation_pair_exact':vm['pair_exact'],'validation_component_modacc':vm['component_modacc'],'validation_macro_f1':vm['macro_f1'],'validation_loss':vm['loss'],'lr':scheduler.get_last_lr()[0]}
        history.append(row)
        if (vm['pair_exact'],vm['macro_f1'])>best:
            best=(vm['pair_exact'],vm['macro_f1']); best_epoch=epoch
            torch.save({'model':model.state_dict(),'epoch':epoch,'validation':vm,'config':config},args.out/'best_validation.pt')
        with (args.out/'history.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerows(history)
        print(args.arm,epoch,row['train_loss'],vm['pair_exact'],vm['macro_f1'],flush=True)
    ck=torch.load(args.out/'best_validation.pt',map_location='cpu'); model.load_state_dict(ck['model']); model.to(device)
    de=Views(roots['development_test'],args.arm)
    dev_loader=DataLoader(de,batch_size=args.batch,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
    overall,idx,y,p=evaluate(model,dev_loader,device)
    rows=[]
    with h5py.File(roots['development_test']) as f:
        for n,(k,yy,pp) in enumerate(zip(idx,y,p)):
            source,j=divmod(int(k),11)
            assert np.array_equal(yy,f['modulation_labels'][source])
            slots=f['occupied_slots'][source]
            rows.append({'source_uid':f['source_uid'][source].decode(),'view_uid':f['view_uid'][source,j].decode(),
                         'noise_seed':int(f['noise_seed'][source,j]),'snr_db':float(f['target_full_component_snr_db'][source,j]),
                         'measured_full_component_snr_1_db':float(f['measured_full_component_snr_db'][source,j,0]),
                         'measured_full_component_snr_2_db':float(f['measured_full_component_snr_db'][source,j,1]),
                         'slot_1':int(slots[0]),'slot_2':int(slots[1]),'true_1':LABELS[int(yy[0])],'true_2':LABELS[int(yy[1])],
                         'pred_1':LABELS[int(pp[0])],'pred_2':LABELS[int(pp[1])],
                         'correct_1':int(yy[0]==pp[0]),'correct_2':int(yy[1]==pp[1]),'pair_exact':int(np.all(yy==pp))})
    with (args.out/'development_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.out/'evaluation_summary.json').write_text(json.dumps({'best_validation_epoch':best_epoch,'best_validation_pair_exact':best[0],'development':overall,'rows':len(rows)},ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
