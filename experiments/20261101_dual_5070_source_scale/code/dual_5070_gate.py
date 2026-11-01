"""追加训练源的P0门禁：源、噪声、标签、物理映射和旧文件保护。"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

import v2_formal_core as core


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--old',type=Path,required=True)
    ap.add_argument('--append',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    assert not args.out.exists()
    files={
        'old_train':args.old/'dual/train/dataset.h5',
        'validation':args.old/'dual/validation/dataset.h5',
        'development':args.old/'dual/development_test/dataset.h5',
        'append':args.append/'dual/train_append5070/dataset.h5'}
    ids={}; hashes={}; noises={}; file_hashes={}
    coset=np.asarray([16,11,1,0,27,24,37,31])
    for name,path in files.items():
        assert path.is_file(),path
        file_hashes[name]=sha(path)
        with h5py.File(path,'r') as h:
            n=len(h['source_uid']); assert n==({'old_train':1690,'validation':338,'development':338,'append':3380}[name])
            assert h['noisy_full_iq'].shape[:2]==(n,11)
            assert list(h.attrs['Coset'])==coset.tolist()
            assert np.array_equal(h.attrs['esn0_grid_db'],np.arange(10,31,2))
            assert np.array_equal(h['target_full_component_snr_db'][0],np.arange(-10,11,2))
            ids[name]=set(x.decode() for x in h['source_uid'][:])
            hashes[name]=set(x.decode() for x in h['clean_full_hash'][:])
            noises[name]=set(int(x) for x in h['noise_seed'][:].ravel())
            assert len(ids[name])==n and len(hashes[name])==n and len(noises[name])==n*11
            assert h['modulation_labels'].shape==(n,2)
            assert np.all((h['modulation_labels'][:]>=0)&(h['modulation_labels'][:]<13))
            for i in range(n):
                iq=h['clean_full_iq'][i]
                clean=iq[:,0]+1j*iq[:,1]
                assert core.array_hash(clean.astype(np.complex64))==h['clean_full_hash'][i].decode(),(name,i)
            for i in (0,n//2,n-1):
                for j in (0,4,10):
                    full=h['noisy_full_iq'][i,j]
                    sampled=full.reshape(1024,40,2)[:,coset,:].reshape(1024,16)
                    assert np.array_equal(sampled,h['mcs_iq'][i,j]),(name,i,j)
    names=list(files)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            assert not(ids[a]&ids[b]),('UID',a,b)
            assert not(hashes[a]&hashes[b]),('clean hash',a,b)
            assert not(noises[a]&noises[b]),('noise seed',a,b)
    assert all(x.startswith('dual:train_append5070:') for x in ids['append']) if any(':' in x for x in ids['append']) else True
    result={'status':'PASS','source_counts':{k:len(v) for k,v in ids.items()},'h5_sha256':file_hashes,
        'source_uid_sha256':{k:hashlib.sha256('\n'.join(sorted(v)).encode()).hexdigest() for k,v in ids.items()},
        'clean_hash_set_sha256':{k:hashlib.sha256('\n'.join(sorted(v)).encode()).hexdigest() for k,v in hashes.items()},
        'noise_seed_set_sha256':{k:hashlib.sha256('\n'.join(map(str,sorted(v))).encode()).hexdigest() for k,v in noises.items()},
        'mapping_checked_rows':'first,middle,last','mapping_checked_snr_indices':[0,4,10]}
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS',result['source_counts'],flush=True)


if __name__=='__main__':main()
