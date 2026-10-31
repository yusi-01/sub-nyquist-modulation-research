#!/usr/bin/env bash
set -euo pipefail
SEED=${1:?seed}
GPU=${2:?GPU}
case "$SEED" in 271060|271061|271062) ;; *) echo '未注册的 seed' >&2; exit 2;; esac
case "$GPU" in 0|1|2) ;; *) echo '未注册的 GPU' >&2; exit 2;; esac
ROOT=/home/zh_3220241128/SN/SHELL_original_multi_snr_20261029
PKG=$ROOT/coset_position_20261101
OUT=$ROOT/runs/coset_position_m8_20261101
DATA=$ROOT/runs/single_full_mcs_paired_20261030_20261030_170342/full_data
PYTHON=/home/user/anaconda3/bin/python
cd "$ROOT"
mkdir -p "$OUT/seed_$SEED"
for ARM in lower_coherence higher_coherence; do
  "$PYTHON" - "$PKG/coset_preflight.json" "$PKG/mapping_gates/mapping_gate_$ARM.json" "$GPU" <<'PY'
import json,subprocess,sys,psutil
p=json.load(open(sys.argv[1])); g=json.load(open(sys.argv[2])); gpu=int(sys.argv[3])
assert p['status']=='PASS' and g['status']=='PASS' and p['arms'][g['name']]['coset_ordered']==g['coset_ordered']
assert p['Fs']==2e9 and p['L']==40 and p['M']==8
free=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
assert free[gpu]>=40000, f'GPU{gpu} 空闲显存不足40GiB'
assert psutil.virtual_memory().available>=55*2**30, '主机可用内存不足55GiB'
print(f"启动门禁PASS {g['name']} GPU{gpu} 空闲{free[gpu]}MiB RAM可用{psutil.virtual_memory().available/2**30:.1f}GiB",flush=True)
PY
  TARGET=$OUT/seed_$SEED/$ARM
  [[ ! -e "$TARGET" ]] || { echo "拒绝覆盖 $TARGET"; exit 2; }
  COSET_CSV=$("$PYTHON" - "$PKG/coset_preflight.json" "$ARM" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(','.join(map(str,p['arms'][sys.argv[2]]['coset_ordered'])))
PY
)
  echo "开始 seed=$SEED arm=$ARM gpu=$GPU coset=$COSET_CSV $(date -Is)" | tee -a "$OUT/seed_$SEED/runner.log"
  POSITION_COSET_CSV="$COSET_CSV" CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$PKG/code/train_position_mask.py" \
    --project-root "$ROOT" --data-root "$DATA" --mapping-gate "$PKG/mapping_gates/mapping_gate_$ARM.json" \
    --output "$TARGET" --seed "$SEED" --batch-size 1024 --workers 4 --epochs 200 \
    2>&1 | tee "$OUT/seed_$SEED/${ARM}_train.log"
  echo "完成 seed=$SEED arm=$ARM $(date -Is)" | tee -a "$OUT/seed_$SEED/runner.log"
done
