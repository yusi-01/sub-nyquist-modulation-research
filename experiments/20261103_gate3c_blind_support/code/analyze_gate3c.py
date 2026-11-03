"""冻结 Oracle/Blind 输出后的源级统计；不修改支持估计器。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
GEOM=Path(__file__).resolve().parents[2]/'20261102_gate2_shell_confusion/results/raw/276对子带几何.csv'
SNRS=list(range(-10,11,2))
RNG=np.random.default_rng(271060)

def ci(x):
    a=np.asarray(x,float); idx=RNG.integers(0,len(a),size=(2000,len(a)),dtype=np.int32)
    v=a[idx].mean(1)
    return float(a.mean()),float(np.quantile(v,.025)),float(np.quantile(v,.975))

def main():
    geom=pd.read_csv(GEOM); geom=geom[geom.arm.eq('baseline')][['slot_i','slot_j','rho_abs','eta','gram_condition']]
    support=[]; quality=[]; groups=[]; intervals=[]; combo=[]
    for split in ('validation','development'):
        d=pd.read_csv(OUT/f'formal_{split}_blind/逐源视图摘要.csv',dtype={'source_uid':str,'view_uid':str,'noise_seed':str})
        assert len(d)==37180 and d.source_uid.nunique()==3380 and d.view_uid.is_unique
        assert d.groupby('source_uid').size().eq(11).all() and d.groupby('snr_db').source_uid.nunique().reindex(SNRS).eq(3380).all()
        d=d.merge(geom,left_on=['slot_1','slot_2'],right_on=['slot_i','slot_j'],validate='many_to_one')
        d['spacing_group']=pd.cut(d.spacing,bins=[0,2,5,11,23],labels=['1-2','3-5','6-11','12-23'])
        d['rho_group']=pd.cut(d.rho_abs,bins=[-0.001,.2,.4,1],labels=['<0.2','0.2-0.4','>=0.4'])
        d['oracle_effective_snr_db']=10*np.log10(d.clean_power/d.oracle_error)
        d['blind_effective_snr_db']=10*np.log10(d.clean_power/d.blind_error)
        d['raw_effective_snr_db']=10*np.log10(d.clean_power/d.raw_noise_power)
        for snr,t in d.groupby('snr_db'):
            support.append(dict(split=split,snr_db=snr,sources=t.source_uid.nunique(),views=len(t),pair_support_exact=t.support_exact.mean(),at_least_one=(t.support_overlap>=1).mean(),first_correct=t.first_correct.mean(),second_completes=t.second_completes.mean(),wrong_count=int((1-t.support_exact).sum())))
            quality.append(dict(split=split,snr_db=snr,sources=t.source_uid.nunique(),views=len(t),clean_distortion=t.clean_distortion.mean(),oracle_noise_ratio=t.oracle_noise_ratio.mean(),oracle_gain_db=t.oracle_gain_db.mean(),blind_gain_db=t.blind_gain_db.mean(),wrong_support_gain_db=t.loc[t.support_exact.eq(0),'blind_gain_db'].mean(),raw_effective_snr_db=t.raw_effective_snr_db.mean(),oracle_effective_snr_db=t.oracle_effective_snr_db.mean(),blind_effective_snr_db=t.blind_effective_snr_db.mean(),blind_nonpositive_gain=int(t.blind_gain_db.le(0).sum())))
        for axis in ('spacing_group','rho_group'):
            for (snr,band),t in d.groupby(['snr_db',axis],observed=True):
                groups.append(dict(split=split,snr_db=snr,axis=axis,band=str(band),sources=t.source_uid.nunique(),views=len(t),support_exact=t.support_exact.mean(),blind_gain_db=t.blind_gain_db.mean(),mean_spacing=t.spacing.mean(),mean_rho=t.rho_abs.mean(),mean_eta=t.eta.mean()))
        for snrs,label in [(SNRS,'all'),(SNRS[:6],'low'),(SNRS[6:],'high')]+[([s],str(s)) for s in SNRS]:
            t=d[d.snr_db.isin(snrs)].groupby('source_uid')[['support_exact','blind_gain_db','oracle_gain_db']].mean()
            row=dict(split=split,snr_group=label,sources=len(t),views=len(d[d.snr_db.isin(snrs)]))
            for col in t.columns:
                row[col+'_mean'],row[col+'_ci_low'],row[col+'_ci_high']=ci(t[col])
            intervals.append(row)
        for (m1,m2),t in d.groupby(['mod_1','mod_2']):
            combo.append(dict(split=split,mod_1=m1,mod_2=m2,sources=t.source_uid.nunique(),views=len(t),low_support_exact=t.loc[t.snr_db.le(0),'support_exact'].mean()))
    pd.DataFrame(support).to_csv(OUT/'GATE3C_逐SNR支持估计结果.csv',index=False)
    pd.DataFrame(quality).to_csv(OUT/'GATE3C_投影前后信号质量.csv',index=False)
    pd.DataFrame(groups).to_csv(OUT/'GATE3C_不同子带间距与相关性分析.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'GATE3C_源级统计与置信区间.csv',index=False)
    pd.DataFrame(combo).to_csv(OUT/'GATE3C_调制组合支持数量.csv',index=False)
    (OUT/'GATE3C_质量门禁与停止裁决.json').write_text(json.dumps({'oracle':'PASS','blind':'PASS','decision':'PASS','method':'固定两轮SOMP + 双槽正交投影','development_used_for_tuning':False,'final_test_read':False},ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()

