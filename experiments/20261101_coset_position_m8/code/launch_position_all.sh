#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zh_3220241128/SN/SHELL_original_multi_snr_20261029
PKG=$ROOT/coset_position_20261101
OUT=$ROOT/runs/coset_position_m8_20261101
cd "$ROOT"
[[ ! -e "$OUT" ]] || { echo "拒绝覆盖: $OUT"; exit 2; }
mkdir "$OUT"
cp -a "$PKG" "$OUT/code_snapshot"
sha256sum "$PKG"/code/*.py "$PKG"/code/*.sh "$PKG"/coset_preflight.json "$PKG"/mapping_gates/*.json > "$OUT/code_and_gate_sha256.txt"
sha256sum -c "$ROOT/runs/single_full_mcs_mask_multiseed_20261031_v3/input_sha256.txt" | tee "$OUT/input_hash_check.log"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv | tee "$OUT/gpu_before.csv"
free -h | tee "$OUT/ram_before.txt"
df -h "$OUT" | tee "$OUT/disk_before.txt"
for spec in 271060:0 271061:1 271062:2; do
  SEED=${spec%:*}; GPU=${spec#*:}
  nohup bash "$PKG/code/run_position_seed.sh" "$SEED" "$GPU" > "$OUT/seed_${SEED}_launcher.log" 2>&1 < /dev/null &
  sleep 45
  tail -n 2 "$OUT/seed_${SEED}_launcher.log"
done
echo "三种子各自串行两臂已启动 $(date -Is)"
