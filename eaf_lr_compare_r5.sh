#!/usr/bin/env bash
# Created: 2026-08-04
# Last modified: 2026-08-04
#
# Freeze the r=5 learning-rate schedule before committing to the ladder. Run on EAF.
#
# The original schedule multiplies by 0.65 every epoch past 30, so the rate falls from
# 3.7e-5 at epoch 30 to 6.8e-9 at epoch 50 -- 12 of 50 epochs run below 1e-6 and change
# nothing. The 'slow' alternative keeps the identical epoch<10 warm-up, then decays 0.93
# per epoch with a 1e-5 floor.
#
# Only the alternative is run here: the original at 10M already exists for all three seeds
# in out_r5_teacher, so those serve as the control. Seeds 0 and 1 are the informative pair
# -- 0 was the poor basin (val p_L 0.09136), 1 was the parity-level run (0.08423). The
# question is whether 'slow' lifts 0 without hurting 1.
#
# Validation only. The sealed test is never read.
set -euo pipefail
RT="${RT:-$HOME/rcnn_threshold}"
POOL="$RT/pools_r5/data_d5_p0.010_r5_FORMAL.npz"
OUT="${OUT:-$RT/out_r5_lrcompare}"
SEEDS="${SEEDS:-0 1}"
PY="${PY:-python}"
[ -f "$POOL" ] || { echo "MISSING $POOL"; exit 1; }

for s in $SEEDS; do
  tag="r5_lrslow_ntr10000000_seed${s}"
  d="$OUT/slow/seed${s}"; mkdir -p "$d/ckpt"
  echo "=============== $tag ==============="
  $PY train_one.py --d 5 --p 0.010 --rounds 5 --seed "$s" \
    --pool "$POOL" --n-train 10000000 --n-test 200000 \
    --val-start 20000000 --val-n 200000 --seal-test \
    --lr-schedule slow \
    --epochs 50 --batch-size 10000 --patience 5 --save-weights \
    --out-dir "$d" --ckpt-dir "$d/ckpt" --run-tag "$tag" 2>&1 \
    | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique"

  $PY eval_on_tail.py --weights "$d/ckpt/${tag}.best.weights.h5" \
    --d 5 --p 0.010 --rounds 5 --n-test 200000 --eval-start 20000000 \
    --pool "$POOL" --out-csv "$OUT/val_scores_best_ckpt.csv" 2>&1 \
    | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique"
  echo
done

echo "=============================================================="
echo "slow-schedule results (.best re-scored on validation):"
cat "$OUT/val_scores_best_ckpt.csv" 2>/dev/null
echo
echo "control -- original schedule, same rung, from out_r5_teacher:"
echo "  seed 0  0.091355     seed 1  0.084230     seed 2  0.085710"
echo "  MWPM on validation = 0.084215"
