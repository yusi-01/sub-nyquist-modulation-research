#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zh_3220241128/SN/SHELL_original_multi_snr_20261029
PKG=$ROOT/coset_position_20261101
OUT=$ROOT/runs/coset_position_m8_20261101
BASE_NEW=$ROOT/runs/single_full_mcs_mask_multiseed_20261031_v3
BASE_OLD=$ROOT/runs/single_full_mcs_paired_20261030_20261030_170342
CONTROL=$ROOT/runs/single_full_mcs_mask_control_20261030_20261031_093802
MCS_DATA=$ROOT/datasets/winner/development_single/component_sample/single/development_test.h5
MCS_CKPT=$ROOT/runs/full_component_diagnostic_200e/seed_271060/single/best_checkpoint_basic.pt
PYTHON=/home/user/anaconda3/bin/python
cd "$ROOT"

for attempt in $(seq 1 720); do
  ready=1
  for seed in 271060 271061 271062; do
    for arm in lower_coherence higher_coherence; do
      [[ -f "$OUT/seed_$seed/$arm/training_config.json" ]] || ready=0
    done
  done
  if [[ "$ready" == 1 ]]; then break; fi
  if (( attempt % 30 == 0 )); then echo "等待训练完成 $(date -Is)"; fi
  sleep 60
done
[[ "$ready" == 1 ]] || { echo '12小时内六个模型未全部完成，停止评价'; exit 3; }

for seed in 271060 271061 271062; do
  if [[ "$seed" == 271060 ]]; then PAIRED=$BASE_OLD; else PAIRED=$BASE_NEW/seed_$seed; fi
  for arm in lower_coherence higher_coherence; do
    RUN=$OUT/seed_$seed/$arm
    "$PYTHON" - "$PKG/coset_preflight.json" "$RUN/training_config.json" "$RUN/history.json" "$RUN/best_checkpoint_metadata.json" "$seed" "$arm" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2])); h=json.load(open(sys.argv[3]))['history']; b=json.load(open(sys.argv[4]))
assert c['seed']==int(sys.argv[5]) and c['coset']==p['arms'][sys.argv[6]]['coset_ordered']
assert c['epochs']==200 and len(h)==200 and c['batch_size']==1024 and c['parameters']==3885965
assert b['selected_on']=='validation_accuracy' and not b['development_test_used']
print('评价前训练门禁PASS',sys.argv[5],sys.argv[6],flush=True)
PY
    [[ ! -e "$RUN/evaluation_11snr" ]] || { echo "拒绝覆盖 $RUN/evaluation_11snr"; exit 2; }
    COSET_CSV=$("$PYTHON" - "$PKG/coset_preflight.json" "$arm" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(','.join(map(str,p['arms'][sys.argv[2]]['coset_ordered'])))
PY
)
    echo "开始评价 seed=$seed arm=$arm $(date -Is)"
    POSITION_COSET_CSV="$COSET_CSV" CUDA_VISIBLE_DEVICES=3 "$PYTHON" "$PKG/code/evaluate_position_mask.py" \
      --project-root "$ROOT" --paired-run "$PAIRED" --mask-run "$RUN" \
      --dev-h5 "$CONTROL/development_11snr.h5" --mcs-h5 "$MCS_DATA" --mcs-checkpoint "$MCS_CKPT" \
      --output "$RUN/evaluation_11snr" --batch-size 256 --workers 4 --seed "$seed" \
      2>&1 | tee "$RUN/evaluate.log"
  done
done
echo "六个位置模型11档评价完成 $(date -Is)"
"$PYTHON" "$PKG/code/analyze_position_results.py" | tee "$OUT/analysis.log"
