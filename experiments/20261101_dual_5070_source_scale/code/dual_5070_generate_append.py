"""仅追加3380个双信号训练源；旧1690源及验证集保持原样。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import v2_formal_core as core
from v2_generate_formal_dataset import write_task_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert args.old.is_dir() and not args.out.exists()
    cfg = json.loads((args.old/'config_snapshot.json').read_text(encoding='utf-8'))
    assert cfg['counts_by_task']['dual'] == {'train':1690,'validation':338,'development_test':338}
    assert cfg['full_component_snr_grid_db'] == list(range(-10,11,2))
    assert cfg['code_sha256']['v2_formal_core.py'] == core.sha256_file(Path(core.__file__))
    esn0 = np.asarray(cfg['esn0_grid_db'], dtype=float)
    assert esn0.tolist() == list(range(10,31,2))
    args.out.mkdir(parents=True, exist_ok=False)
    result = write_task_split(args.out, 'dual', 'train_append5070', 3380,
        cfg['master_seed'], esn0, cfg['config_sha256'],
        cfg['code_sha256']['v2_formal_core.py'], True, False, 8)
    (args.out/'append_manifest.json').write_text(json.dumps({
        'role':'additional_training_sources_only', 'old_data':str(args.old),
        'namespace':'dual/train_append5070', 'source_count':3380,
        'old_config_sha256':cfg['config_sha256'], 'new_data':result},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
