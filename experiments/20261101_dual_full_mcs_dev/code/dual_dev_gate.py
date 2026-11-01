"""双信号开发数据训练前 P0 门禁；不访问 Final Test。"""
from pathlib import Path
import argparse,json
import h5py,numpy as np
import v2_formal_core as core

p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,required=True); args=p.parse_args()
cfg=json.loads((args.data/'config_snapshot.json').read_text(encoding='utf-8'))
assert cfg['tasks']==['dual'] and cfg['splits']==['train','validation','development_test']
assert cfg['physical']['Coset']==[16,11,1,0,27,24,37,31]
assert cfg['store_noisy_full'] is True
assert cfg['full_component_snr_grid_db']==list(range(-10,11,2))
expect={'train':1690,'validation':338,'development_test':338}
ids={}; noise={}; results={}
for split,n in expect.items():
    with h5py.File(args.data/'dual'/split/'dataset.h5') as f:
        assert f['source_uid'].shape==(n,) and f['view_uid'].shape==(n,11)
        assert f['noisy_full_iq'].shape==(n,11,40960,2)
        assert f['mcs_iq'].shape==(n,11,1024,16)
        ids[split]=set(map(bytes,f['source_uid'][:]))
        noise[split]=set(map(int,f['noise_seed'][:].reshape(-1)))
        assert len(ids[split])==n and len(noise[split])==n*11
        y=f['modulation_labels'][:].astype(int); slots=f['occupied_slots'][:]
        count=np.bincount(y[:,0]*13+y[:,1],minlength=169)
        original=core.split_schedule('dual',split,n,cfg['master_seed'])['mods_component_order'].astype(int)
        original_count=np.bincount(original[:,0]*13+original[:,1],minlength=169)
        assert len(count)==169 and original_count.min()==original_count.max()==n//169
        assert np.all(slots[:,0]<slots[:,1]) and np.all((slots>=0)&(slots<24))
        power=f['component_power_full'][:]
        assert np.max(np.abs(power-1))<1e-4
        assert np.allclose(f['target_full_component_snr_db'][:],np.arange(-10,11,2)[None,:])
        worst=0.0
        for i in (0,n//2,n-1):
            for j in (0,5,10):
                full=f['noisy_full_iq'][i,j].reshape(1024,40,2)[:,[16,11,1,0,27,24,37,31],:].reshape(1024,16)
                worst=max(worst,float(np.max(np.abs(full-f['mcs_iq'][i,j]))))
        assert worst==0,worst
        results[split]={'sources':n,'views':n*11,'generation_order_combo_repeats':int(original_count.min()),'frequency_order_combo_min':int(count.min()),'frequency_order_combo_max':int(count.max()),'frequency_order_missing_combos':int(np.sum(count==0)),'max_full_mcs_abs_error':worst}
for a,b in [('train','validation'),('train','development_test'),('validation','development_test')]:
    assert not ids[a]&ids[b] and not noise[a]&noise[b]
args.out.write_text(json.dumps({'status':'PASS','results':results,'source_and_noise_split_isolation':True},ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS',results)
