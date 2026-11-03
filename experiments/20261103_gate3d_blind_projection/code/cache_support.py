"""仅缓存冻结 SOMP 的盲支持；不读取真实 slot、标签、SNR 或 Full。"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--h5', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--gate3c-code', type=Path, required=True)
args = ap.parse_args()
sys.path.insert(0, str(args.gate3c_code))
from gate3c_physical import blind_somp, complex_mcs

assert not args.out.exists()
with h5py.File(args.h5, 'r') as h, args.out.open('w', newline='', encoding='utf-8') as f:
    n, views = h['mcs_iq'].shape[:2]
    assert views == 11
    w = csv.writer(f)
    w.writerow(['row_index','source_uid','view_uid','est_1','est_2'])
    for i in range(n):
        uid = h['source_uid'][i].decode()
        for j in range(views):
            x = complex_mcs(h['mcs_iq'][i,j])
            y = np.fft.fft(x, axis=1, norm='ortho')
            est, _ = blind_somp(y)
            assert len(set(est)) == 2
            w.writerow([i*views+j,uid,h['view_uid'][i,j].decode(),int(est[0]),int(est[1])])
print('CACHE_PASS',n,n*views,hashlib.sha256(args.out.read_bytes()).hexdigest(),flush=True)

