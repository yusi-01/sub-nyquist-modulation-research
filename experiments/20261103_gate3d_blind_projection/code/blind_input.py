"""冻结盲支持缓存与 GPU 双槽投影；不使用真实槽作分类输入。"""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from gate3c_physical import A, C, D, Q, projection


class BlindViews(Dataset):
    def __init__(self, h5_path: Path, cache_path: Path):
        self.path = h5_path
        self.h = None
        cache = pd.read_csv(cache_path, dtype={'source_uid':str,'view_uid':str})
        with h5py.File(h5_path, 'r') as h:
            self.sources, self.views = h['mcs_iq'].shape[:2]
            assert self.views == 11 and len(cache) == self.sources*11
            uids = [v.decode() for v in h['source_uid'][:]]
            assert cache.row_index.to_numpy().tolist() == list(range(len(cache)))
            assert np.array_equal(cache.source_uid.to_numpy().reshape(self.sources,11)[:,0],uids)
            assert np.array_equal(cache.view_uid.to_numpy().reshape(self.sources,11),np.vectorize(bytes.decode)(h['view_uid'][:]))
        self.support = cache[['est_1','est_2']].to_numpy(np.int64)
        assert np.all((self.support>=0)&(self.support<24)) and np.all(self.support[:,0]!=self.support[:,1])

    def __len__(self): return self.sources*self.views

    def __getitem__(self,index):
        if self.h is None: self.h=h5py.File(self.path,'r')
        i,j=divmod(index,11)
        x=np.ascontiguousarray(self.h['mcs_iq'][i,j].T,dtype=np.float32)
        y=self.h['modulation_labels'][i].astype(np.int64)
        target=np.zeros(26,np.float32); target[y[0]]=1; target[13+y[1]]=1
        return torch.from_numpy(x),torch.from_numpy(target),torch.from_numpy(y),index,torch.from_numpy(self.support[index].copy())


class BlindProjector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        matrices=[]; lookup=np.full((24,24),-1,np.int64)
        for i in range(24):
            for j in range(i+1,24):
                u,_=np.linalg.qr(A[:,[i,j]])
                lookup[i,j]=lookup[j,i]=len(matrices)
                matrices.append(u@u.conj().T)
        self.register_buffer('matrices',torch.from_numpy(np.stack(matrices).astype(np.complex64)),persistent=False)
        self.register_buffer('lookup',torch.from_numpy(lookup),persistent=False)
        self.register_buffer('phase',torch.from_numpy(D.astype(np.complex64)),persistent=False)

    def forward(self,raw_iq:torch.Tensor,support:torch.Tensor)->torch.Tensor:
        assert raw_iq.ndim==3 and raw_iq.shape[1:]==(16,Q) and support.shape==(len(raw_iq),2)
        idx=self.lookup[support[:,0],support[:,1]]
        assert bool((idx>=0).all())
        z=torch.complex(raw_iq[:,0::2].float(),raw_iq[:,1::2].float())/8
        spectrum=torch.fft.fft(z,dim=-1,norm='ortho')
        corrected=spectrum*self.phase.conj()[None]
        projected=torch.einsum('bij,bjq->biq',self.matrices[idx],corrected)*self.phase[None]
        y=torch.fft.ifft(projected,dim=-1,norm='ortho')
        out=torch.stack((y.real,y.imag),dim=2).reshape(-1,16,Q)
        if not bool(torch.isfinite(out).all()): raise FloatingPointError('投影出现非有限值')
        return out

