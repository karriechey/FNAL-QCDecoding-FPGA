#!/usr/bin/env bash
# Created: 2026-08-18
# Last updated: 2026-08-18
#
# GH200 A/B: d=9, r=9, p=0.004, 10M training shots, batch 5,000, seed 0.
# One thing differs between the two arms: --lr-schedule.
#
#   arm A   --lr-schedule original    the schedule every earlier run used
#   arm B   --lr-schedule slow        same warm-up, 0.93/epoch decay, 1e-5 floor
#
# Everything else -- pool, seed, partition indices, batch size, epoch budget,
# architecture, determinism flags -- is held fixed, so any difference in the final
# evaluation-block p_L is attributable to the schedule alone.
#
# ONE ARM PER INVOCATION. Both arms must share a RUNROOT so their outputs sit together
# and the geometry file written once applies to both:
#
#   STAMP=$(date -u +%Y%m%dT%H%M%SZ)
#   RUNROOT=$HOME/rcnn_threshold/results/gh200_d9_p004_lr_ab_$STAMP
#   ARM=original RUNROOT=$RUNROOT bash gh200_run_lr_ab_d9.sh
#   ARM=slow     RUNROOT=$RUNROOT bash gh200_run_lr_ab_d9.sh
#
# Output-collision safety. `lr_schedule` appears in the per-arm output directory, in the
# checkpoint --run-tag, and in the evaluation CSV filename. train_one.py's own result
# filename is built from (d, p, rounds, seed, n_train) only and carries no schedule, so
# the two arms would write the same basename; the per-arm directory is what keeps them
# apart, and this script refuses to start if that directory already holds a result.
# train_one.py's resume identity does include lr_schedule, so a resume can never load one
# arm's checkpoint into the other even if the directories were confused.
#
# Fixed epoch budget, no early stopping. The best epoch is selected afterwards from the
# ModelCheckpoint file, which monitors val_loss with save_best_only. With early stopping
# off there is no restore_best_weights, so the weights in memory when fit() returns are
# the FINAL epoch, not the best -- which is why the scoring step below loads
# .best.weights.h5 explicitly rather than trusting the training job's own p_L.
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
SCRATCH="${SCRATCH:-$RT}"

D=9
ROUNDS=9
P="${P:-0.004}"
SEED="${SEED:-0}"
NTRAIN="${NTRAIN:-10000000}"
BATCH="${BATCH:-5000}"
EPOCHS="${EPOCHS:-50}"

ARM="${ARM:-}"
case "$ARM" in
  original|slow) ;;
  *) echo "[ab] set ARM=original or ARM=slow. STOP."; exit 1 ;;
esac

POOL="${POOL:-$SCRATCH/pools_d9_p004_19M/data_d9_p0.004_r9_FORMAL.npz}"

# Partition boundaries. Identical to the Experiment 13 layout, so this A/B is comparable
# to the p=0.004 ladder rungs. Changing any one of these makes it incomparable.
VAL_START="${VAL_START:-15000000}"      # monitoring validation block start
VAL_N="${VAL_N:-200000}"                # 200k, ends at 15.2M
EVAL_START="${EVAL_START:-15200000}"    # final shared evaluation block start
EVAL_N="${EVAL_N:-1800000}"             # 1.8M, ends at 17.0M
SEALED_N="${SEALED_N:-2000000}"         # sealed test [17.0M, 19.0M)
POOL_MIN="${POOL_MIN:-19000000}"

RUNROOT="${RUNROOT:-$RT/results/gh200_d9_p004_lr_ab}"
ARMROOT="$RUNROOT/arm_${ARM}"

# Pinned interpreter: TensorFlow 2.15.x / Keras 2. CNNModel.py does not build under
# Keras 3 -- the state correlator's initializer compares a tuple against a list and
# rejects identical shapes.
PY="${PY:-$HOME/FNAL-QCDecoding-FPGA/.venv/bin/python}"
[ -x "$PY" ] || { echo "[ab] pinned interpreter missing: $PY. STOP."; exit 1; }
"$PY" -c "import tensorflow as tf; assert tf.__version__.startswith('2.15.'), \
  f'TensorFlow {tf.__version__}, this study is pinned to 2.15.x'; \
  print('[ab] TensorFlow', tf.__version__)" 2>/dev/null \
  || { echo "[ab] interpreter is not the pinned TensorFlow 2.15.x stack. STOP."; exit 1; }

# Determinism. Both variables are read by TensorFlow during its own import, so a script
# that exports them after the process starts gets no determinism and no warning.
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
DETERMINISM="${DETERMINISM:---require-determinism}"

# Resume by default, so a killed process costs the epoch in flight rather than the run.
# A resumed run reshuffles from a fresh RNG stream and is therefore not bit-identical to
# an uninterrupted one; train_one.py records resumed_from_epoch in the result row.
RESUME="${RESUME:---resume}"

[ -f "$POOL" ] || { echo "[ab] missing pool: $POOL. STOP."; exit 1; }

echo "[ab] checking pool layout: $POOL"
"$PY" - "$POOL" "$POOL_MIN" "$EVAL_START" "$EVAL_N" "$VAL_START" "$VAL_N" "$SEALED_N" \
      "$NTRAIN" <<'PYCHECK'
import sys
import numpy as np

(pool, pool_min, eval_start, eval_n, val_start, val_n, sealed_n, ntrain) = sys.argv[1:9]
pool_min, eval_start, eval_n = int(pool_min), int(eval_start), int(eval_n)
val_start, val_n, sealed_n, ntrain = int(val_start), int(val_n), int(sealed_n), int(ntrain)

with np.load(pool) as z:
    n = z['det_evts'].shape[0]
    shapes = {k: z[k].shape for k in ('measurements', 'det_evts', 'flips') if k in z}
print(f"[ab]   shots {n:,}   " + "  ".join(f"{k}{v}" for k, v in shapes.items()))

if n < pool_min:
    raise SystemExit(f"[ab] pool holds {n:,} shots, layout needs {pool_min:,}. STOP.")
if ntrain > val_start:
    raise SystemExit(f"[ab] training prefix [0, {ntrain:,}) reaches into the validation "
                     f"block at {val_start:,}. STOP.")
if val_start + val_n > eval_start:
    raise SystemExit("[ab] validation overlaps evaluation. STOP.")
if eval_start + eval_n > n - sealed_n:
    raise SystemExit("[ab] evaluation overlaps the sealed test block. STOP.")
print(f"[ab]   train      [{0:,}, {ntrain:,})")
print(f"[ab]   validation [{val_start:,}, {val_start + val_n:,})")
print(f"[ab]   evaluation [{eval_start:,}, {eval_start + eval_n:,})")
print(f"[ab]   sealed     [{n - sealed_n:,}, {n:,})   all disjoint")
PYCHECK

tag="d9_p004_rcnn_ntr${NTRAIN}_seed${SEED}_lr${ARM}"
mkdir -p "$ARMROOT/ckpt"

# Collision guard. A finished result in this directory means this arm already ran; a
# rerun would overwrite it. Results are append-only until the paper is written.
if [ -f "$ARMROOT/rcnn_d${D}_p${P}_r${ROUNDS}_seed${SEED}_ntr${NTRAIN}.csv" ] \
   && [ -z "${ALLOW_OVERWRITE:-}" ]; then
  echo "[ab] $ARMROOT already holds a result row for this configuration."
  echo "[ab] Move it aside, choose a different RUNROOT, or set ALLOW_OVERWRITE=1. STOP."
  exit 1
fi

echo
echo "GH200 LR A/B: d=$D r=$ROUNDS p=$P  arm=$ARM"
echo "  pool        $POOL"
echo "  n_train     $NTRAIN   batch $BATCH   epochs $EPOCHS (fixed, no early stopping)"
echo "  seed        $SEED"
echo "  lr schedule $ARM"
echo "  determin.   TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1 ${DETERMINISM:-off}"
echo "  resume      ${RESUME:-off}"
echo "  run tag     $tag"
echo "  out         $ARMROOT"
echo

"$PY" train_one.py --d "$D" --p "$P" --rounds "$ROUNDS" --seed "$SEED" \
  --pool "$POOL" \
  --n-train "$NTRAIN" --n-test "$SEALED_N" \
  --val-start "$VAL_START" --val-n "$VAL_N" --seal-test \
  --epochs "$EPOCHS" --batch-size "$BATCH" --no-early-stopping \
  --lr-schedule "$ARM" --save-weights $DETERMINISM $RESUME \
  --out-dir "$ARMROOT" --ckpt-dir "$ARMROOT/ckpt" --run-tag "$tag" 2>&1 \
  | tee -a "$ARMROOT/train.log" \
  | grep --line-buffered -Ev "cuda_|Unable to register|^Total number|^Number of unique"

# Score the best-validation checkpoint on the shared 1.8M evaluation block, with MWPM
# decoded on those exact shots and paired shot by shot. Both arms append to the same CSV,
# which is what makes the comparison paired and the McNemar counts meaningful.
"$PY" eval_on_tail.py --weights "$ARMROOT/ckpt/${tag}.best.weights.h5" \
  --d "$D" --p "$P" --rounds "$ROUNDS" \
  --n-test "$EVAL_N" --eval-start "$EVAL_START" \
  --batch-size "$BATCH" --mcnemar \
  --dump-per-shot "$ARMROOT/per_shot_${tag}.npz" \
  --pool "$POOL" --out-csv "$RUNROOT/eval_1p8M_best_ckpt.csv" 2>&1 \
  | tee -a "$ARMROOT/eval.log" \
  | grep --line-buffered -Ev "cuda_|Unable to register|^Total number|^Number of unique"

echo
echo "[ab] arm $ARM done -> $RUNROOT/eval_1p8M_best_ckpt.csv"
echo "[ab] per-shot dump  -> $ARMROOT/per_shot_${tag}.npz"
echo "[ab] once BOTH arms have finished, pair them directly:"
echo "[ab]   \$PY pair_two_decoders.py --a $RUNROOT/arm_original/per_shot_d9_p004_rcnn_ntr${NTRAIN}_seed${SEED}_lroriginal.npz \\"
echo "[ab]                             --b $RUNROOT/arm_slow/per_shot_d9_p004_rcnn_ntr${NTRAIN}_seed${SEED}_lrslow.npz"
