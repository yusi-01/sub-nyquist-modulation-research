#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zh_3220241128/SN/SHELL_original_multi_snr_20261029
PAIR=$ROOT/runs/single_full_mcs_paired_20261030_20261030_170342
CONTROL=$ROOT/runs/single_full_mcs_mask_control_20261030_20261031_093802
OUT=$ROOT/runs/single_full_mcs_mask_multiseed_20261031_v3
DATA=$PAIR/full_data
GPU=0
PYTHON=/home/user/anaconda3/bin/python
FULL_CODE=$ROOT/single_full_mcs_paired_20261030
MASK_CODE=$ROOT/single_full_mcs_mask_control_20261030
cd "$ROOT"
[[ ! -e "$OUT" ]] || { echo "拒绝覆盖: $OUT"; exit 2; }
mkdir "$OUT"
exec > >(tee "$OUT/pipeline.log") 2>&1
cp "$FULL_CODE/02_train_full.py" "$FULL_CODE/paired_common.py" "$OUT/"
cp "$MASK_CODE/03_train_mcs_mask.py" "$MASK_CODE/mask_common.py" "$OUT/"
sha256sum "$OUT"/*.py > "$OUT/code_sha256.txt"
cp "$ROOT/runs/single_full_mcs_mask_multiseed_20261031_v2/input_sha256.txt" "$OUT/input_sha256.txt"
sha256sum "$PAIR/full_model/best_checkpoint_basic.pt" "$CONTROL/mcs_mask_model/best_checkpoint_mask.pt" > "$OUT/historical_checkpoint_sha256.txt"
"$PYTHON" - "$PAIR/full_data/manifest.json" "$CONTROL/mapping_gate.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));g=json.load(open(sys.argv[2]))
assert m['status']=='PASS' and g['status']=='PASS' and g['same_initial_state']
assert all(m['files'][s]['snr_grid_db']==[float(x) for x in range(-10,11,2)] for s in ('train','validation'))
assert all(m['exhaustive_pair_check'][s]['bitwise'] for s in ('train','validation'))
print('数据、映射及初始化历史门禁 PASS',flush=True)
PY
"$PYTHON" - "$PAIR/full_data/manifest.json" "$OUT/input_sha256.txt" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))['files']
lines=open(sys.argv[2]).read().splitlines()
assert len(lines)==2
for line,split in zip(lines,('train','validation')):
    digest,path=line.split(maxsplit=1)
    assert digest==m[split]['sha256'] and path.endswith('/'+split+'.h5')
print('输入 H5 重算哈希与历史 manifest 一致',flush=True)
PY

preflight() {
  echo "资源检查 $(date -Is)"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
  free -h
  df -h "$ROOT"
  "$PYTHON" - <<'PY'
import psutil,subprocess
assert psutil.virtual_memory().available >= 50*2**30, '主机可用内存不足50GiB'
out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
assert int(out.splitlines()[0].strip()) >= 40000, 'GPU0空闲显存不足40GiB'
for p in psutil.process_iter(['pid','cmdline']):
    cmd=' '.join(p.info['cmdline'] or [])
    if p.pid != psutil.Process().pid and ('02_train_full.py' in cmd or '03_train_mcs_mask.py' in cmd):
        raise RuntimeError(f'检测到训练进程 {p.pid}: {cmd[:150]}')
PY
}

for SEED in 271061 271062; do
  for ARM in full mask; do
    preflight
    TARGET="$OUT/seed_${SEED}/${ARM}_model"
    [[ ! -e "$TARGET" ]] || { echo "拒绝覆盖: $TARGET"; exit 2; }
    mkdir -p "$OUT/seed_${SEED}"
    echo "开始 seed=$SEED arm=$ARM $(date -Is)"
    if [[ "$ARM" == full ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$FULL_CODE/02_train_full.py" --project-root "$ROOT" --data-root "$DATA" --output "$TARGET" --seed "$SEED" --batch-size 1024 --workers 4 --epochs 200 2>&1 | tee "$OUT/seed_${SEED}/full_train.log"
    else
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$MASK_CODE/03_train_mcs_mask.py" --project-root "$ROOT" --data-root "$DATA" --mapping-gate "$CONTROL/mapping_gate.json" --output "$TARGET" --seed "$SEED" --batch-size 1024 --workers 4 --epochs 200 2>&1 | tee "$OUT/seed_${SEED}/mask_train.log"
    fi
    echo "完成 seed=$SEED arm=$ARM $(date -Is)"
  done
done
echo "四次训练完成 $(date -Is)"
