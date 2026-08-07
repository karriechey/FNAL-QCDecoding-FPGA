#!/usr/bin/env bash
# Created: 2026-08-07
# Last updated: 2026-08-07
#
# Exp 10 RCNN arm: d=9, r=9, fixed 10M training shots, 3 seeds. Run on EAF.
#
# This is a distance-scaling study, not a data-volume ladder. There is exactly one rung
# (10M) and the only things that differ from the established d=5, r=5 teacher runs are
# --d and --rounds. Every other flag below is copied from eaf_run_rcnn_ladder_r5.sh
# unchanged, because a hyperparameter that moves with distance would confound the very
# comparison the experiment exists to make.
#
# TIMING MODE. The d=9 per-epoch cost is unmeasured on GPU. The RCNN has 49 kernels at
# d=9 against 9 at d=5 and 9 rounds against 5, so it is expected to be near 10x the
# d=5 teacher's ~4.7 h/seed -- roughly 47 h/seed, 140 h for three seeds. Do not accept
# that estimate. Run one epoch first and read the real number:
#
#   EPOCHS=1 SEEDS=0 OUTROOT=$HOME/rcnn_threshold/out_d9_timing bash eaf_run_rcnn_d9.sh
#
# then multiply by 50 epochs x 3 seeds and decide. The timing job writes to its own
# OUTROOT so it can never be mistaken for a real result.
#
# Full sweep, once the cost is known and accepted:
#
#   bash eaf_run_rcnn_d9.sh                      # seeds 0 1 2
#   SEEDS="0" bash eaf_run_rcnn_d9.sh            # one seed per server, split by hand
#
# Reporting partition is the VALIDATION block [10.0M, 10.2M). --seal-test keeps the test
# block [10.2M, 10.4M) unread, matching the d=5 protocol: every d=5,r=5 number in hand is
# a validation number, so the distance comparison is like-for-like and both sealed sets
# survive for the paper's one-shot final report.
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
POOL="${POOL:-$RT/pools_d9/data_d9_p0.010_r9_FORMAL.npz}"
VAL_START="${VAL_START:-10000000}"
VAL_N="${VAL_N:-200000}"
NTRAIN="${NTRAIN:-10000000}"
OUTROOT="${OUTROOT:-$RT/out_d9_rcnn}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-5}"
# Batch 10,000 is what every d=5 run used, so it stays the default. Whether it FITS at
# d=9 is an open question -- activation memory grows ~10x with the kernel and round
# counts while the weights do not, and a local batch-10,000 probe was OOM-killed on a
# 16 GB host. Run probe_batch_d9.py on this GPU before trusting the default. If a smaller
# batch is forced here, the d=5 reference has to be re-run at the same batch or the
# distance comparison is confounded by a batch change.
BATCH="${BATCH:-10000}"
PY="${PY:-python}"

[ -f "$POOL" ] || { echo "MISSING $POOL -- build it first:"; \
  echo "  python gen_pool_r5.py --d 9 --rounds 9 --p 0.010 --n-train 10000000 \\"; \
  echo "      --n-val 200000 --n-test 200000 --out-dir $RT/pools_d9 --tag FORMAL"; exit 1; }

n=0; for s in $SEEDS; do n=$((n+1)); done
echo "Exp 10 RCNN d=9 r=9: $n runs"
echo "  pool=$POOL"
echo "  n_train=$NTRAIN  epochs=$EPOCHS  batch=$BATCH  seeds=$SEEDS"
echo "  validation [$VAL_START, $((VAL_START+VAL_N)))   sealed test not read"
echo "  out=$OUTROOT"
[ "$EPOCHS" = "1" ] && echo "  *** TIMING MODE: 1 epoch, this is not a result ***"
echo

i=0
for s in $SEEDS; do
  i=$((i+1))
  tag="d9_rcnn_ntr${NTRAIN}_seed${s}"
  out="$OUTROOT/seed${s}"
  echo "=============== [$i/$n] $tag ==============="
  mkdir -p "$out/ckpt"
  $PY train_one.py --d 9 --p 0.010 --rounds 9 --seed "$s" \
    --pool "$POOL" \
    --n-train "$NTRAIN" --n-test "$VAL_N" \
    --val-start "$VAL_START" --val-n "$VAL_N" --seal-test \
    --epochs "$EPOCHS" --batch-size "$BATCH" --patience "$PATIENCE" --save-weights \
    --out-dir "$out" --ckpt-dir "$out/ckpt" --run-tag "$tag" 2>&1 \
    | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique"

  # Re-score the .best checkpoint explicitly. A run's own p_L comes from whatever weights
  # were in memory when fit() returned, which is the restored best only if early stopping
  # fired -- so it is not comparable across seeds. This is.
  $PY eval_on_tail.py --weights "$out/ckpt/${tag}.best.weights.h5" \
    --d 9 --p 0.010 --rounds 9 --n-test "$VAL_N" --eval-start "$VAL_START" \
    --pool "$POOL" --out-csv "$OUTROOT/val_scores_best_ckpt.csv" 2>&1 \
    | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique"
  echo
done

echo "=============================================================="
echo "done. Collate with:  python collate_d9.py"
