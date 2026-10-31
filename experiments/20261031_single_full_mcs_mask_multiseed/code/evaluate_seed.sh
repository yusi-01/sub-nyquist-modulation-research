#!/usr/bin/env bash
set -euo pipefail
SEED=${1:?训练种子}
GPU=${2:?GPU 编号}
case "$SEED" in 271061|271062) ;; *) echo '不支持的种子' >&2; exit 2;; esac
case "$GPU" in 0|1|2|3) ;; *) echo '不支持的 GPU' >&2; exit 2;; esac
ROOT=/home/zh_3220241128/SN/SHELL_original_multi_snr_20261029
OUT=$ROOT/runs/single_full_mcs_mask_multiseed_20261031_v3
SEED_DIR=$OUT/seed_$SEED
CONTROL=$ROOT/runs/single_full_mcs_mask_control_20261030_20261031_093802
MCS_DATA=$ROOT/datasets/winner/development_single/component_sample/single/development_test.h5
MCS_CKPT=$ROOT/runs/full_component_diagnostic_200e/seed_271060/single/best_checkpoint_basic.pt
PYTHON=/home/user/anaconda3/bin/python
cd "$ROOT"
"$PYTHON" - "$SEED_DIR" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
for arm in ('full','mask'):
    d=p/f'{arm}_model'
    cfg=json.loads((d/'training_config.json').read_text())
    best=json.loads((d/'best_checkpoint_metadata.json').read_text())
    history=json.loads((d/'history.json').read_text())['history']
    assert cfg['epochs']==200 and len(history)==200
    assert cfg['seed']==int(p.name.split('_')[1])
    assert cfg['batch_size']==1024 and cfg['model']=='Res1dNet311' and cfg['parameters']==3885965
    assert best['selected_on']=='validation_accuracy' and not best['development_test_used']
    assert (d/('best_checkpoint_basic.pt' if arm=='full' else 'best_checkpoint_mask.pt')).is_file()
print('训练完成与 Validation 选点门禁 PASS',flush=True)
PY
[[ ! -e "$SEED_DIR/development_11snr" ]] || { echo '拒绝覆盖已有评价'; exit 2; }
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$ROOT/single_full_mcs_mask_control_20261030/04_evaluate_control.py" \
  --project-root "$ROOT" --paired-run "$SEED_DIR" --mask-run "$SEED_DIR/mask_model" \
  --dev-h5 "$CONTROL/development_11snr.h5" --mcs-h5 "$MCS_DATA" --mcs-checkpoint "$MCS_CKPT" \
  --output "$SEED_DIR/development_11snr" --batch-size 256 --workers 4 --seed "$SEED"
