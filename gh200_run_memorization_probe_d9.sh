#!/usr/bin/env bash
# Created: 2026-08-18
# Last updated: 2026-08-18
#
# d=9 memorization probe. Can the RCNN fit a small training set at all?
#
# Purpose. This exists only to interpret the main A/B. If the model memorizes a small set,
# no hard capacity limit is established and a large-scale optimization or representation
# difficulty stays plausible. If it cannot memorize a small set after a sufficient update
# budget, that is much stronger evidence for a model or architecture capacity limitation.
# It is not itself a decoder result and must never be scored against MWPM.
#
# Budget in optimizer steps, not epochs. A 50k set at batch 5,000 gives 10 steps per epoch,
# which is step-starved regardless of how many epochs are run. This uses batch 500 over a
# 50k set: 100 steps per epoch, 600 epochs, 60,000 optimizer updates -- the same order as
# the main run consumes before its curve flattens (2,000 steps/epoch x ~30 epochs).
#
# Constant learning rate, no decay, no early stopping. --lr-schedule flat holds 0.003 for
# every epoch and skips the warm-up ramp. 0.003 is the rate the hard-label MLP and GRU
# students used on this same d=9 pool, where the GRU did learn, so it is the justified
# constant here rather than an arbitrary one.
#
# Stopping rule. Train loss below ~0.02, or train accuracy above ~99%, means the small set
# is memorized and the probe can be stopped. train_one.py has no callback for a train-metric
# threshold, so this is done by watching the log and killing the process. That is safe: the
# resume checkpoint and its JSON, holding the full merged history, are written after every
# epoch, so a killed probe keeps its readout. Otherwise it runs the full 600 epochs.
#
#   RUNROOT=... bash gh200_run_memorization_probe_d9.sh
#
#   NTRAIN=10000 EPOCHS=3000 ...   the 10k variant, same 60,000 updates
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
SCRATCH="${SCRATCH:-$RT}"

D=9
ROUNDS=9
P="${P:-0.004}"
SEED="${SEED:-0}"

NTRAIN="${NTRAIN:-50000}"     # small training subset, the thing being memorized
BATCH="${BATCH:-500}"
EPOCHS="${EPOCHS:-600}"       # 50,000/500 = 100 steps/epoch -> 60,000 optimizer updates
LR_SCHEDULE="${LR_SCHEDULE:-flat}"

POOL="${POOL:-$SCRATCH/pools_d9_p004_19M/data_d9_p0.004_r9_FORMAL.npz}"

# A small validation block immediately after the training subset. train_one.py needs a
# validation source to record val_loss, and the checkpoint callback monitors it. It is a
# bookkeeping requirement here, not the readout: the readout is TRAIN loss and accuracy.
# Kept small so it costs almost nothing per epoch.
VAL_START="${VAL_START:-$((NTRAIN))}"
VAL_N="${VAL_N:-20000}"
SEALED_N="${SEALED_N:-2000000}"   # --seal-test keeps the probe away from the tail

RUNROOT="${RUNROOT:-$RT/results/gh200_d9_p004_memorization_probe}"
OUT="$RUNROOT/ntr${NTRAIN}_b${BATCH}_lr${LR_SCHEDULE}_seed${SEED}"

PY="${PY:-$HOME/FNAL-QCDecoding-FPGA/.venv/bin/python}"
[ -x "$PY" ] || { echo "[probe] pinned interpreter missing: $PY. STOP."; exit 1; }

export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1

[ -f "$POOL" ] || { echo "[probe] missing pool: $POOL. STOP."; exit 1; }
mkdir -p "$OUT/ckpt"

steps=$(( (NTRAIN + BATCH - 1) / BATCH ))
echo
echo "d=9 memorization probe"
echo "  pool        $POOL"
echo "  train       [0, $NTRAIN)      validation [$VAL_START, $((VAL_START+VAL_N)))"
echo "  batch       $BATCH   -> $steps steps/epoch"
echo "  epochs      $EPOCHS  -> $((steps * EPOCHS)) optimizer updates"
echo "  lr          $LR_SCHEDULE (constant 0.003, no warm-up, no decay)"
echo "  readout     TRAIN loss / accuracy. Stop at loss < 0.02 or accuracy > 0.99."
echo "  out         $OUT"
echo

"$PY" train_one.py --d "$D" --p "$P" --rounds "$ROUNDS" --seed "$SEED" \
  --pool "$POOL" \
  --n-train "$NTRAIN" --n-test "$SEALED_N" \
  --val-start "$VAL_START" --val-n "$VAL_N" --seal-test \
  --epochs "$EPOCHS" --batch-size "$BATCH" --no-early-stopping \
  --lr-schedule "$LR_SCHEDULE" --save-weights --require-determinism --resume \
  --out-dir "$OUT" --ckpt-dir "$OUT/ckpt" --run-tag "probe_ntr${NTRAIN}_b${BATCH}" 2>&1 \
  | tee -a "$OUT/probe.log" \
  | grep --line-buffered -Ev "cuda_|Unable to register|^Total number|^Number of unique"

echo
echo "[probe] done -> $OUT"
echo "[probe] read the TRAIN loss/accuracy curve from $OUT/ckpt/probe_ntr${NTRAIN}_b${BATCH}.resume.json"
