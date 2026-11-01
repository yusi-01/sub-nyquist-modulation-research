"""单种子双信号 Development 同源配对分析；真实槽与 η 仅用于评价后诊断。"""
from pathlib import Path
import argparse
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser(); ap.add_argument('--full',type=Path,required=True);ap.add_argument('--mask',type=Path,required=True);ap.add_argument('--geometry',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
keys=['source_uid','view_uid','noise_seed','snr_db','slot_1','slot_2','true_1','true_2','measured_full_component_snr_1_db','measured_full_component_snr_2_db']
f=pd.read_csv(args.full,dtype={'source_uid':str,'view_uid':str,'noise_seed':str}).sort_values('view_uid').reset_index(drop=True)
m=pd.read_csv(args.mask,dtype={'source_uid':str,'view_uid':str,'noise_seed':str}).sort_values('view_uid').reset_index(drop=True)
assert len(f)==len(m)==338*11 and f[keys].equals(m[keys]),'Full/MCS 逐源身份不一致'
assert not f.duplicated(['source_uid','view_uid']).any()
g=pd.read_csv(args.geometry);g=g[g.arm=='baseline'][['slot_i','slot_j','rho_abs','gram_condition','eta','spacing_slots']].rename(columns={'slot_i':'slot_1','slot_j':'slot_2'})
x=f[keys+['pred_1','pred_2','correct_1','correct_2','pair_exact']].rename(columns={k:'full_'+k for k in ['pred_1','pred_2','correct_1','correct_2','pair_exact']}).merge(m[['view_uid','pred_1','pred_2','correct_1','correct_2','pair_exact']].rename(columns={k:'mask_'+k for k in ['pred_1','pred_2','correct_1','correct_2','pair_exact']}),on='view_uid',validate='one_to_one').merge(g,on=['slot_1','slot_2'],validate='many_to_one')
assert len(x)==len(f) and x.source_uid.nunique()==338
x['full_only']=(x.full_pair_exact.eq(1)&x.mask_pair_exact.eq(0)).astype(int)
x['mask_only']=(x.full_pair_exact.eq(0)&x.mask_pair_exact.eq(1)).astype(int)
x['delta_pair']=x.full_pair_exact-x.mask_pair_exact
x['eta_bin']=pd.cut(x.eta,[1,1.05,1.15,1.30,np.inf],labels=['1.00–1.05','1.05–1.15','1.15–1.30','>1.30'],include_lowest=True)
x.to_csv(args.out/'逐样本Full_MCS_物理关联.csv',index=False,encoding='utf-8-sig')
labels=sorted(set(x.true_1)|set(x.true_2))
def macro(y,p):
    vals=[]
    for z in labels:
        tp=((y==z)&(p==z)).sum();fp=((y!=z)&(p==z)).sum();fn=((y==z)&(p!=z)).sum()
        vals.append(0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
    return np.mean(vals)
rng=np.random.default_rng(20261101); rows=[]
for snr,d in x.groupby('snr_db'):
    source=d.groupby('source_uid').delta_pair.mean().to_numpy();boot=source[rng.integers(0,len(source),(2000,len(source)))].mean(1)
    row={'snr_db':snr,'sources':d.source_uid.nunique(),'views':len(d),'full_component_modacc':d[['full_correct_1','full_correct_2']].to_numpy().mean(),'mask_component_modacc':d[['mask_correct_1','mask_correct_2']].to_numpy().mean(),'full_pair_exact':d.full_pair_exact.mean(),'mask_pair_exact':d.mask_pair_exact.mean(),'delta_pair_exact':d.delta_pair.mean(),'paired_ci_low':np.quantile(boot,.025),'paired_ci_high':np.quantile(boot,.975),'full_only':d.full_only.sum(),'mask_only':d.mask_only.sum()}
    for arm in ('full','mask'):
        row[arm+'_macro_f1']=macro(d[['true_1','true_2']].to_numpy().reshape(-1),d[[arm+'_pred_1',arm+'_pred_2']].to_numpy().reshape(-1))
        row[arm+'_slot1_modacc']=d[arm+'_correct_1'].mean();row[arm+'_slot2_modacc']=d[arm+'_correct_2'].mean()
    rows.append(row)
pd.DataFrame(rows).to_csv(args.out/'逐SNR配对统计.csv',index=False,encoding='utf-8-sig')
strata=x.groupby(['snr_db','eta_bin'],observed=True).agg(n=('source_uid','size'),unique_sources=('source_uid','nunique'),eta_min=('eta','min'),eta_max=('eta','max'),full_pair_exact=('full_pair_exact','mean'),mask_pair_exact=('mask_pair_exact','mean'),full_only_rate=('full_only','mean'),mask_only_rate=('mask_only','mean'),delta_pair=('delta_pair','mean')).reset_index()
strata['small_cell']=strata.n<30
strata.to_csv(args.out/'逐SNR_η分层.csv',index=False,encoding='utf-8-sig')
class_rows=[]
for snr,d in x.groupby('snr_db'):
    for arm in ('full','mask'):
        truth=d[['true_1','true_2']].to_numpy().reshape(-1)
        pred=d[[arm+'_pred_1',arm+'_pred_2']].to_numpy().reshape(-1)
        for label in labels:
            tp=int(((truth==label)&(pred==label)).sum());support=int((truth==label).sum());count=int((pred==label).sum())
            class_rows.append({'snr_db':snr,'arm':arm,'label':label,'support':support,'recall':tp/support if support else np.nan,'precision':tp/count if count else 0,'predicted_count':count})
pd.DataFrame(class_rows).to_csv(args.out/'逐SNR逐类指标.csv',index=False,encoding='utf-8-sig')
x[x.full_only==1].to_csv(args.out/'Full正确_MCS错误样本.csv',index=False,encoding='utf-8-sig')
for col in ('spacing_slots','true_1','true_2'):
    tab=x.groupby(col).agg(n=('source_uid','size'),full_only=('full_only','mean'),mask_only=('mask_only','mean'),delta_pair=('delta_pair','mean'),eta_mean=('eta','mean')).reset_index()
    tab.to_csv(args.out/f'额外错误按{col}.csv',index=False,encoding='utf-8-sig')
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
z=pd.DataFrame(rows)
axs[0].plot(z.snr_db,z.full_pair_exact,'o-',label='Full');axs[0].plot(z.snr_db,z.mask_pair_exact,'o-',label='MCS mask');axs[0].set(xlabel='Full component SNR (dB)',ylabel='PairExact');axs[0].legend()
axs[1].errorbar(z.snr_db,z.delta_pair_exact,yerr=[z.delta_pair_exact-z.paired_ci_low,z.paired_ci_high-z.delta_pair_exact],fmt='o-');axs[1].axhline(0,color='gray',lw=.8);axs[1].set(xlabel='Full component SNR (dB)',ylabel='Full − MCS PairExact')
fig.savefig(args.out/'逐SNR配对表现.png',dpi=180)
print('PASS',len(x),x.source_uid.nunique(),args.out)
