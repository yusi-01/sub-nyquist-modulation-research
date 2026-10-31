"""冻结候选相对历史 mask 的同源配对 Development 分析。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path('/home/zh_3220241128/SN/SHELL_original_multi_snr_20261029')
RUN = ROOT / 'runs/coset_position_m8_20261101'
BASE_NEW = ROOT / 'runs/single_full_mcs_mask_multiseed_20261031_v3'
BASE_OLD = ROOT / 'runs/single_full_mcs_mask_control_20261030_20261031_093802'
OUT = RUN / 'three_seed_analysis'
OUT.mkdir(exist_ok=False)
SEEDS = (271060, 271061, 271062)
SNRS = tuple(range(-10, 11, 2))
ARMS = ('baseline', 'lower_coherence', 'higher_coherence')
LABELS = ['APSK16','APSK32','APSK64','ASK8','BPSK','OQPSK','PSK16','PSK8','QAM128','QAM16','QAM256','QAM64','QPSK']
KEYS = ['source_uid','view_uid','noise_seed','snr_db','occupied_slot','true_id','true_label']
rng = np.random.default_rng(20261101)


def read(seed, arm):
    if arm == 'baseline':
        path = (BASE_OLD / 'control_development_11snr_results_v5/control_predictions.csv' if seed == 271060
                else BASE_NEW / f'seed_{seed}/development_11snr/control_predictions.csv')
    else:
        path = RUN / f'seed_{seed}/{arm}/evaluation_11snr/control_predictions.csv'
    frame = pd.read_csv(path, usecols=KEYS+['mask_pred'], dtype={'source_uid':str,'view_uid':str,'noise_seed':str})
    assert len(frame)==28600
    return frame.sort_values('view_uid').reset_index(drop=True)


def score(y, pred):
    cm = pd.crosstab(pd.Categorical(y,categories=LABELS),pd.Categorical(pred,categories=LABELS),dropna=False).to_numpy(dtype=int)
    r = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(13), where=cm.sum(1)>0)
    p = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(13), where=cm.sum(0)>0)
    f1 = np.divide(2*r*p,r+p,out=np.zeros(13),where=r+p>0)
    return cm,r,p,float(np.trace(cm)/cm.sum()),float(f1.mean())


rows, pairs, classes, confusions = [], [], [], []
for seed in SEEDS:
    frames = {arm:read(seed,arm) for arm in ARMS}
    base = frames['baseline']
    assert sorted(base.snr_db.unique().tolist())==list(SNRS)
    for arm, frame in frames.items():
        assert frame[KEYS].equals(base[KEYS]), f'身份门禁失败 seed={seed} arm={arm}'
    for snr in SNRS:
        slices = {arm:frames[arm][frames[arm].snr_db==snr] for arm in ARMS}
        y = slices['baseline'].true_label.to_numpy()
        assert len(y)==2600 and slices['baseline'].source_uid.nunique()==2600
        for arm, frame in slices.items():
            cm,r,p,acc,f1=score(y,frame.mask_pred.to_numpy())
            rows.append(dict(seed=seed,snr_db=snr,arm=arm,modacc=acc,macro_f1=f1))
            for i,label in enumerate(LABELS):
                classes.append(dict(seed=seed,snr_db=snr,arm=arm,label=label,recall=r[i],precision=p[i],predicted_count=int(cm[:,i].sum())))
                for j,guess in enumerate(LABELS):
                    confusions.append(dict(seed=seed,snr_db=snr,arm=arm,true_label=label,predicted_label=guess,count=int(cm[i,j])))
        base_ok=(slices['baseline'].mask_pred.to_numpy()==y)
        for arm in ARMS[1:]:
            alt_ok=(slices[arm].mask_pred.to_numpy()==y)
            d=alt_ok.astype(np.int8)-base_ok.astype(np.int8)
            boot=d[rng.integers(0,len(d),size=(2000,len(d)))].mean(1)
            pairs.append(dict(seed=seed,snr_db=snr,arm=arm,delta=float(d.mean()),
                              ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                              both_correct=int((alt_ok&base_ok).sum()),alt_only=int((alt_ok&~base_ok).sum()),
                              base_only=int((~alt_ok&base_ok).sum()),both_wrong=int((~alt_ok&~base_ok).sum())))

metrics=pd.DataFrame(rows); paired=pd.DataFrame(pairs)
metrics.to_csv(OUT/'逐种子逐SNR指标.csv',index=False,encoding='utf-8-sig')
paired.to_csv(OUT/'逐源配对差.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(classes).to_csv(OUT/'逐类指标.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(confusions).to_csv(OUT/'完整混淆矩阵长表.csv',index=False,encoding='utf-8-sig')
agg=paired.groupby(['arm','snr_db']).delta.agg(['mean','std','min','max']).reset_index()
agg.to_csv(OUT/'三种子配对差汇总.csv',index=False,encoding='utf-8-sig')

lines=['# M=8 Coset 位置归因初步结果','','> Development Test；候选在结果产生前冻结，Final Test 未读。','','## 候选−历史 mask 准确率差（百分点）','','|候选|SNR|seed 271060|271061|271062|均值±SD|范围|','|---|---:|---:|---:|---:|---:|---:|']
for arm in ARMS[1:]:
    for snr in SNRS:
        vals=paired[(paired.arm==arm)&(paired.snr_db==snr)].set_index('seed').delta
        row=agg[(agg.arm==arm)&(agg.snr_db==snr)].iloc[0]
        lines.append(f'|{arm}|{snr:+d}|{vals[271060]*100:+.2f}|{vals[271061]*100:+.2f}|{vals[271062]*100:+.2f}|{row["mean"]*100:+.2f}±{row["std"]*100:.2f}|{row["min"]*100:+.2f}至{row["max"]*100:+.2f}|')
lines += ['','三个训练种子评价的是同一批源，标准差衡量训练波动；逐源配对区间见 CSV，不能将三次预测合并为独立测试样本。矩阵最大相干度的顺序已在训练前冻结：lower 0.41920 < baseline 0.58044 < higher 0.80000。此表只报告经验分类关联，不单独证明物理信息损失。','']
(OUT/'位置归因初步结果_20261101.md').write_text('\n'.join(lines),encoding='utf-8')
print('PASS',OUT,flush=True)
