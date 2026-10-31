#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zh_3220241128/SN/SHELL_original_multi_snr_20261029
PAIR=$ROOT/runs/single_full_mcs_paired_20261030_20261030_170342
CONTROL=$ROOT/runs/single_full_mcs_mask_control_20261030_20261031_093802
OUT=$ROOT/runs/single_full_mcs_mask_multiseed_20261031_v3
PYTHON=/home/user/anaconda3/bin/python
ARM=${1:?full 或 mask}
case "$ARM" in
  full) GPU=1; CODE=$ROOT/single_full_mcs_paired_20261030/02_train_full.py;;
  mask) GPU=2; CODE=$ROOT/single_full_mcs_mask_control_20261030/03_train_mcs_mask.py;;
  *) echo '仅允许 full 或 mask' >&2; exit 2;;
esac
cd "$ROOT"
TARGET=$OUT/seed_271062/${ARM}_model
[[ ! -e "$TARGET" ]] || { echo "拒绝覆盖: $TARGET"; exit 2; }
"$PYTHON" - "$GPU" <<'PY'
import subprocess,sys,psutil
gpu=int(sys.argv[1])
free=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
assert free[gpu]>=40000, f'GPU{gpu} 空闲显存不足40GiB'
assert psutil.virtual_memory().available>=60*2**30, '主机可用内存不足60GiB'
print(f'GPU{gpu} 空闲 {free[gpu]} MiB，主机可用 {psutil.virtual_memory().available/2**30:.1f} GiB',flush=True)
PY
mkdir -p "$OUT/seed_271062"
echo "启动 seed=271062 arm=$ARM GPU=$GPU $(date -Is)" | tee -a "$OUT/parallel_launch.log"
if [[ "$ARM" == full ]]; then
  exec env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$CODE" --project-root "$ROOT" --data-root "$PAIR/full_data" --output "$TARGET" --seed 271062 --batch-size 1024 --workers 4 --epochs 200
else
  exec env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$CODE" --project-root "$ROOT" --data-root "$PAIR/full_data" --mapping-gate "$CONTROL/mapping_gate.json" --output "$TARGET" --seed 271062 --batch-size 1024 --workers 4 --epochs 200
fi
