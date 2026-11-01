"""冻结双信号 checkpoint 的同训练源新噪声只读诊断；仅评估 -2 dB。"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from v2_shell_models_portable import DualV6

COSET = np.array([16, 11, 1, 0, 27, 24, 37, 31])
ACTIVE = np.array([v for c in COSET for v in (2*c, 2*c+1)])


def input_tensor(iq, arm):
    x = iq.reshape(1024, 40, 2).transpose(1, 2, 0).reshape(80, 1024)
    if arm == 'mask':
        y = np.zeros_like(x)
        y[ACTIVE] = x[ACTIVE]
        x = y
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def evaluate_batch(model, device, data):
    with torch.no_grad():
        return model(torch.stack(data).to(device)).argmax(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--include-new', action='store_true')
    args = ap.parse_args()
    assert not args.out.exists()
    data = args.run/'data_dual_dev1690_seed271070/dual/train/dataset.h5'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result = {'snr_db': -2, 'source_count': 1690, 'noise_seed_rule': '271071 + source_row, independent of stored noise seed', 'arms': {}}
    with h5py.File(data, 'r') as h:
        assert len(h['source_uid']) == 1690
        assert np.all(h['target_full_component_snr_db'][:, 4] == -2)
        arms = [('full','train_full_seed271070/best_validation.pt'),
                ('mask','train_mask_seed271070/best_validation.pt')]
        if args.include_new:
            arms.append(('mask5070','train_mask_5070_seed271070_20261101/best_validation.pt'))
        for arm, ckname in arms:
            ck = torch.load(args.run/ckname, map_location='cpu', weights_only=False)
            model = DualV6(in_channels=80, num_classes=13)
            model.load_state_dict(ck['model']); model.to(device).eval()
            counts = {'original_pair': 0, 'fresh_pair': 0, 'original_component': 0, 'fresh_component': 0}
            batch_old, batch_new, labels = [], [], []
            for row in range(1690):
                clean_iq = h['clean_full_iq'][row]
                clean = clean_iq[:, 0] + 1j*clean_iq[:, 1]
                power = float(h['target_noise_power_full'][row, 4])
                rng = np.random.default_rng(271071 + row)
                sigma = np.sqrt(power/2)
                noise = (rng.normal(0, sigma, len(clean)) + 1j*rng.normal(0, sigma, len(clean))).astype(np.complex64)
                fresh = clean + noise
                fresh_iq = np.stack((fresh.real, fresh.imag), axis=-1)
                representation = 'full' if arm == 'full' else 'mask'
                batch_old.append(input_tensor(h['noisy_full_iq'][row, 4], representation))
                batch_new.append(input_tensor(fresh_iq, representation))
                labels.append(h['modulation_labels'][row])
                if len(labels) == 64 or row == 1689:
                    y = np.stack(labels)
                    old = evaluate_batch(model, device, batch_old)
                    new = evaluate_batch(model, device, batch_new)
                    counts['original_pair'] += int(np.all(old == y, axis=1).sum())
                    counts['fresh_pair'] += int(np.all(new == y, axis=1).sum())
                    counts['original_component'] += int((old == y).sum())
                    counts['fresh_component'] += int((new == y).sum())
                    batch_old, batch_new, labels = [], [], []
            result['arms'][arm] = {**counts, 'original_pair_exact': counts['original_pair']/1690,
                'fresh_pair_exact': counts['fresh_pair']/1690,
                'original_modacc': counts['original_component']/3380,
                'fresh_modacc': counts['fresh_component']/3380}
            if args.include_new:
                with h5py.File(args.run/'data_dual_dev1690_seed271070/dual/validation/dataset.h5','r') as hv:
                    val_x=[];val_y=[]; correct_pair=0;correct_comp=0
                    for row in range(338):
                        val_x.append(input_tensor(hv['noisy_full_iq'][row,4],representation))
                        val_y.append(hv['modulation_labels'][row])
                        if len(val_x)==64 or row==337:
                            y=np.stack(val_y); pred=evaluate_batch(model,device,val_x)
                            correct_pair+=int(np.all(pred==y,axis=1).sum())
                            correct_comp+=int((pred==y).sum())
                            val_x=[];val_y=[]
                    result['arms'][arm]['independent_validation_minus2_pair_exact']=correct_pair/338
                    result['arms'][arm]['independent_validation_minus2_modacc']=correct_comp/676
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
