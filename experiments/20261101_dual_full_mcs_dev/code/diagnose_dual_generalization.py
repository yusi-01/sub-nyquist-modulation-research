"""只读汇总双信号最佳 checkpoint 的训练源记忆化与独立源泛化差距。"""
from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'双信号单种子_20261101'
OUT=ROOT/'泛化诊断_20261101';OUT.mkdir(exist_ok=True)
d=json.loads((ROOT/'readonly_diagnosis.json').read_text(encoding='utf-8'))
rows=[];classes=[]
for arm,v in d['arms'].items():
    for split,z in v['splits'].items():
        rows.append({'arm':arm,'split':split,'checkpoint_epoch':v['checkpoint_epoch'],'n_views':z['n_views'],**z['metrics']})
        for c in z['class_metrics']: classes.append({'arm':arm,'split':split,**c})
pd.DataFrame(rows).to_csv(OUT/'训练与验证泛化差距.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(classes).to_csv(OUT/'训练与验证逐类.csv',index=False,encoding='utf-8-sig')
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for arm,color in [('full','tab:blue'),('mask','tab:orange')]:
    h=pd.read_csv(ROOT/f'train_{arm}_seed271070'/'history.csv')
    axs[0].plot(h.epoch,h.train_loss,label=f'{arm} train',color=color)
    axs[0].plot(h.epoch,h.validation_loss,label=f'{arm} validation',color=color,linestyle='--')
    axs[1].plot(h.epoch,h.validation_pair_exact,label=arm,color=color)
    best=d['arms'][arm]['checkpoint_epoch'];axs[1].axvline(best,color=color,alpha=.5,linestyle=':')
axs[0].set(xlabel='Epoch',ylabel='Cross entropy',title='Train/Validation gap');axs[0].legend(fontsize=8)
axs[1].set(xlabel='Epoch',ylabel='Validation PairExact',title='Validation selection');axs[1].legend(fontsize=8)
fig.savefig(OUT/'训练验证泛化差距.png',dpi=180)
print('PASS',OUT)
