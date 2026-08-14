#!/usr/bin/env bash
# Created: 2026-08-13
# Last updated: 2026-08-13
#
# Below-threshold distance scaling at p=0.004, for d=5,r=5 and d=9,r=9.
#
# The noise model is unchanged. Only p changes.
#
# Rungs. 10M and 15M, nested. The 15M prefix contains every shot the 10M prefix used, so
# the difference between the two rungs reflects added training data rather than a
# differently sampled dataset. They are correlated estimates and cannot be treated as
# independent measurements when judging significance.
#
# Batch size. 5,000 at both distances. d=9, r=9 at batch 10,000 exhausted a 40 GB MIG
# slice during the backward pass, failing on a 17 MB allocation at epoch 6
# (shape[49,10000,9]). Activation memory scales with batch. Neither distance has a trained
# reference at p=0.004, so fixing 5,000 for both defines a matched protocol.
#
# Partitions, identical at both distances. The pool holds 19M shots.
#
#   training      [0, N)                nested prefixes: 10M, 15M
#   validation    [15.0M, 15.2M)        200k, early stopping and checkpoint selection
#   evaluation    [15.2M, 17.0M)        1.8M, scored once from the .best checkpoint
#   test          [17.0M, 19.0M)        sealed, untouched in this phase
#
# Validation and evaluation are disjoint by construction: validation ends at 15.2M and
# evaluation begins there. The largest training prefix stops at 15.0M, so no training shot
# appears in either. --seal-test keeps train_one.py away from the final 2M.
#
# Resume. Every rung runs with --resume, and train_one.py writes a resume checkpoint after
# every epoch holding model weights, Adam slot variables, the learning rate, and the
# EarlyStopping and ModelCheckpoint state. Re-running the identical command after a restart picks
# up where it stopped; the resume state is keyed to the exact configuration and refuses to
# load into a different one. A resumed run reshuffles from a fresh RNG stream, so it is not
# bit-identical to one that ran straight through even under --require-determinism, and the
# result row records the epoch it resumed from.
#
# Usage, one distance per pod:
#
#   D=5 ROUNDS=5 POOL=<d5 pool> bash eaf_run_ladder_p004.sh
#   D=9 ROUNDS=9 POOL=<d9 pool> bash eaf_run_ladder_p004.sh
#
#   NTRAINS=10000000 ...        one rung only
#   SEEDS="1 2" ...             replicate a configuration after seed 0 has been read
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
SCRATCH="${SCRATCH:-/scratch/fast/7DayLifetime/kchey}"

D="${D:-5}"
ROUNDS="${ROUNDS:-$D}"
P="${P:-0.004}"
POOL="${POOL:-$SCRATCH/pools_d${D}_p004/data_d${D}_p${P}_r${ROUNDS}_FORMAL.npz}"

# Rungs, nested, smallest first so the cheaper result arrives first.
NTRAINS="${NTRAINS:-10000000 15000000}"
# Seed 0 only to start. Seeds 1 and 2 are queued later, on the configurations worth
# replicating, once the seed-0 trend has been read.
SEEDS="${SEEDS:-0}"

BATCH="${BATCH:-5000}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-5}"
LR_SCHEDULE="${LR_SCHEDULE:-original}"

# Partition boundaries. These are the protocol. Changing one makes the run incomparable
# to every other rung.
VAL_START="${VAL_START:-15000000}"     # validation block start
VAL_N="${VAL_N:-200000}"               # validation size, ends at 15.2M
EVAL_START="${EVAL_START:-15200000}"   # evaluation start, where validation ends
EVAL_N="${EVAL_N:-1800000}"            # evaluation size, ends at 17.0M
SEALED_N="${SEALED_N:-2000000}"        # sealed test tail [17.0M, 19.0M)
POOL_MIN="${POOL_MIN:-19000000}"       # a smaller pool cannot hold this layout

OUTROOT="${OUTROOT:-$RT/out_d${D}_p004_ladder}"

# Pinned interpreter. The pod's `python` is /opt/conda/bin/python, carrying TensorFlow
# 2.16 / Keras 3, which fails inside CNNModel.py on a tuple-vs-list shape comparison.
# This study is pinned to the repo .venv at TensorFlow 2.15.1.
PY="${PY:-$HOME/QuantumDecoderQKeras/.venv/bin/python}"
[ -x "$PY" ] || { echo "[ladder] pinned interpreter missing or not executable: $PY"; \
  echo "[ladder] expected the repo .venv (TensorFlow 2.15.1). STOP."; exit 1; }
echo "[ladder] interpreter $PY"
"$PY" -c "import tensorflow as tf; assert tf.__version__.startswith('2.15.'), \
  f'TensorFlow {tf.__version__}, this study is pinned to 2.15.x'; \
  print('[ladder] TensorFlow', tf.__version__)" 2>/dev/null \
  || { echo "[ladder] interpreter is not the pinned TensorFlow 2.15.x stack. STOP."; exit 1; }

# Determinism. slice_guard_r5.assert_deterministic_env() requires both variables present
# before the Python process starts, since TensorFlow reads them during its own import and
# a script that sets them afterwards gets no determinism and no warning. Exporting here
# means the run does not depend on what the pod happens to have in its environment.
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
DETERMINISM="${DETERMINISM:---require-determinism}"

# Resume by default. Override with RESUME= to force a run to start from epoch 0.
RESUME="${RESUME:---resume}"

[ -f "$POOL" ] || { echo "missing pool: $POOL"; exit 1; }

# Refuse to run against a pool that cannot hold the partition layout. Checked by reading
# the array header rather than the filename, so a mislabelled pool is caught here instead
# of producing results that silently overlap the sealed test block.
echo "[ladder] checking pool layout: $POOL"
$PY - "$POOL" "$POOL_MIN" "$EVAL_START" "$EVAL_N" "$VAL_START" "$VAL_N" "$SEALED_N" <<'PYCHECK'
import sys
import numpy as np

(pool, pool_min, eval_start, eval_n, val_start, val_n, sealed_n) = sys.argv[1:8]
pool_min, eval_start, eval_n = int(pool_min), int(eval_start), int(eval_n)
val_start, val_n, sealed_n = int(val_start), int(val_n), int(sealed_n)

with np.load(pool) as z:
    n = z['det_evts'].shape[0]
    shapes = {k: z[k].shape for k in ('measurements', 'det_evts', 'flips') if k in z}
print(f"[ladder]   shots {n:,}   " + "  ".join(f"{k}{v}" for k, v in shapes.items()))

if n < pool_min:
    raise SystemExit(f"[ladder] pool holds {n:,} shots, layout needs {pool_min:,}. STOP.")
if val_start + val_n > eval_start:
    raise SystemExit("[ladder] validation overlaps evaluation. STOP.")
if eval_start + eval_n > n - sealed_n:
    raise SystemExit("[ladder] evaluation overlaps the sealed test block. STOP.")
print(f"[ladder]   validation [{val_start:,}, {val_start + val_n:,})  "
      f"evaluation [{eval_start:,}, {eval_start + eval_n:,})  "
      f"sealed [{n - sealed_n:,}, {n:,})  all disjoint")
PYCHECK

mkdir -p "$OUTROOT"

n_runs=0; for ntr in $NTRAINS; do for s in $SEEDS; do n_runs=$((n_runs+1)); done; done
echo
echo "Below-threshold ladder: d=$D r=$ROUNDS p=$P  ($n_runs runs)"
echo "  pool       $POOL"
echo "  rungs      $NTRAINS   (nested prefixes, correlated estimates)"
echo "  seeds      $SEEDS"
echo "  batch      $BATCH    epochs $EPOCHS   patience $PATIENCE   lr $LR_SCHEDULE"
echo "  determin.  TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1 ${DETERMINISM:-off}"
echo "  resume     ${RESUME:-off}"
echo "  validation [$VAL_START, $((VAL_START+VAL_N)))"
echo "  evaluation [$EVAL_START, $((EVAL_START+EVAL_N)))   scored from .best"
echo "  test       sealed, last $SEALED_N shots"
echo "  out        $OUTROOT"
echo

i=0
for ntr in $NTRAINS; do
  # A training prefix must stop before the validation block, or the model would be scored
  # on shots it trained on.
  if [ "$ntr" -gt "$VAL_START" ]; then
    echo "[ladder] rung $ntr exceeds the validation start $VAL_START -- skipping"
    continue
  fi
  for s in $SEEDS; do
    i=$((i+1))
    tag="d${D}_p004_rcnn_ntr${ntr}_seed${s}"
    out="$OUTROOT/ntr${ntr}_seed${s}"
    echo "=============== [$i/$n_runs] $tag ==============="
    mkdir -p "$out/ckpt"

    $PY train_one.py --d "$D" --p "$P" --rounds "$ROUNDS" --seed "$s" \
      --pool "$POOL" \
      --n-train "$ntr" --n-test "$SEALED_N" \
      --val-start "$VAL_START" --val-n "$VAL_N" --seal-test \
      --epochs "$EPOCHS" --batch-size "$BATCH" --patience "$PATIENCE" \
      --lr-schedule "$LR_SCHEDULE" --save-weights $DETERMINISM $RESUME \
      --out-dir "$out" --ckpt-dir "$out/ckpt" --run-tag "$tag" 2>&1 \
      | grep --line-buffered -Ev "cuda_|Unable to register|^Total number|^Number of unique"

    # Score the .best checkpoint on the evaluation block, immediately, so each rung's
    # result is in hand before the next one starts. Done as a separate step because a
    # run's own p_L comes from whichever weights were in memory when fit() returned, which
    # is the restored best only when early stopping fired.
    #
    # --mcnemar pairs the model against MWPM shot by shot on these exact shots, which is
    # what any ratio or significance claim has to be built from.
    $PY eval_on_tail.py --weights "$out/ckpt/${tag}.best.weights.h5" \
      --d "$D" --p "$P" --rounds "$ROUNDS" \
      --n-test "$EVAL_N" --eval-start "$EVAL_START" \
      --batch-size "$BATCH" --mcnemar \
      --pool "$POOL" --out-csv "$OUTROOT/eval_1p8M_best_ckpt.csv" 2>&1 \
      | grep --line-buffered -Ev "cuda_|Unable to register|^Total number|^Number of unique"
    echo "[ladder] rung $ntr seed $s scored -> $OUTROOT/eval_1p8M_best_ckpt.csv"
    echo
  done
done

echo "=============================================================="
echo "done. Results: $OUTROOT/eval_1p8M_best_ckpt.csv"
echo
echo "Reading the ladder: 10M and 15M share training shots, so their difference is not an"
echo "independent measurement, and 15M is only a 1.5x increase over 10M. Binomial error on"
echo "the 1.8M evaluation block is the floor on the uncertainty, not the whole of it --"
echo "seed-to-seed variation can be larger. If the decision to extend past 15M turns on a"
echo "small 10M->15M difference, add seeds at that rung before deciding."
