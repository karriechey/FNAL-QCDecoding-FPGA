#!/usr/bin/env bash
# Created: 2026-08-07
# Last updated: 2026-08-07
#
# Exp 10 student arm: MLP + GRU at d=9, r=9, fixed 10M shots, 3 seeds each. Run on EAF.
#
# HARD LABEL, alpha=1.0. This matters and is not a default worth changing quietly: the
# d=5, r=5, 10M reference these runs are compared against is out_student_r5_hard (GRU
# 0.08145, MLP 0.09756 on the validation block). Mixing a hard-label d=5 point with a
# distilled d=9 point would make the distance contrast meaningless. train_student.py
# writes alpha and temperature into every result row, so the CSV records the regime.
#
# Hard label also means these nine runs need no teacher, so they are independent of the
# RCNN arm and can start as soon as the pool exists.
#
# Sizes are the d=5 values, deliberately NOT retuned for d=9:
#   MLP hidden=(209,209)   69,389 params at d=5  ->  194,789 at d=9  (2.81x)
#   GRU units=140          69,441 params at d=5  ->   92,961 at d=9  (1.34x)
# The parameter growth is a result of the experiment, not a thing to correct for. Exp 10
# asks how the existing architecture families scale; forcing equal parameter counts is a
# different study and is explicitly out of scope here.
#
#   bash eaf_run_student_d9.sh                   # mlp + gru, seeds 0 1 2
#   ARCHS=gru bash eaf_run_student_d9.sh         # one architecture per server
#
# --eval-start pins scoring to the VALIDATION block [10.0M, 10.2M). Without it the trainer
# scores the last --n-test shots, which on this pool is the sealed test partition.
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
POOL="${POOL:-$RT/pools_d9/data_d9_p0.010_r9_FORMAL.npz}"
VAL_START="${VAL_START:-10000000}"
VAL_N="${VAL_N:-200000}"
NTRAIN="${NTRAIN:-10000000}"
OUTDIR="${OUTDIR:-$RT/out_d9_student_hard}"
ARCHS="${ARCHS:-mlp gru}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-50}"
MLP_HIDDEN="${MLP_HIDDEN:-209 209}"
GRU_UNITS="${GRU_UNITS:-140}"
LR="${LR:-0.003}"
PY="${PY:-python}"

[ -f "$POOL" ] || { echo "MISSING $POOL -- build it first:"; \
  echo "  python gen_pool_r5.py --d 9 --rounds 9 --p 0.010 --n-train 10000000 \\"; \
  echo "      --n-val 200000 --n-test 200000 --out-dir $RT/pools_d9 --tag FORMAL"; exit 1; }
mkdir -p "$OUTDIR"

n=0; for a in $ARCHS; do for s in $SEEDS; do n=$((n+1)); done; done
echo "Exp 10 student d=9 r=9 hard-label: $n runs -> $OUTDIR"
echo "  pool=$POOL"
echo "  n_train=$NTRAIN  epochs=$EPOCHS  archs=$ARCHS  seeds=$SEEDS  lr=$LR"
echo "  mlp hidden=($MLP_HIDDEN)   gru units=$GRU_UNITS   (d=5 sizes, not retuned)"
echo "  validation [$VAL_START, $((VAL_START+VAL_N)))   sealed test not read"
echo

i=0
for arch in $ARCHS; do
  if [ "$arch" = "gru" ]; then SIZE="--hidden --units $GRU_UNITS"; else SIZE="--hidden $MLP_HIDDEN"; fi
  for s in $SEEDS; do
    i=$((i+1))
    # The learning rate is in the tag so a diagnostic at another LR cannot overwrite the
    # runs it is being compared against.
    tag="d9hard_${arch}_ntr${NTRAIN}_seed${s}_lr${LR}${TAGSUF:-}"
    echo "=============== [$i/$n] $tag ==============="
    $PY train_student.py --student "$arch" --inputs evts $SIZE \
      --d 9 --p 0.010 --rounds 9 \
      --alpha 1.0 --temperature 1.0 --lr "$LR" \
      --seed "$s" --n-train "$NTRAIN" --n-test "$VAL_N" --eval-start "$VAL_START" \
      --val-start "$VAL_START" --val-n "$VAL_N" \
      --epochs "$EPOCHS" --batch-size 10000 --no-early-stopping \
      --pool "$POOL" --out-dir "$OUTDIR" --tag "$tag" 2>&1 \
      | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique|^Epoch |- loss:"
    echo
  done
done

echo "=============================================================="
echo "done. Collate with:  python collate_d9.py"
