#!/usr/bin/env bash
# Created: 2026-07-30
# Last modified: 2026-07-30
#
# Select the paper's teacher and build the caches the distillation grid needs.
# RUN THIS ON EAF -- the grid has to run there to stay comparable to the teacher sweep,
# so the caches must be built against EAF's pools, not copied from the Mac.
#
# What it does, in order:
#   0. Print the pool fingerprints, so a cache built elsewhere can be checked against
#      these pools before being trusted.
#   1. Score all three FP32 teacher checkpoints on the VALIDATION partition -- the last
#      --val-split fraction of each teacher's training prefix, which Keras held out during
#      training and which is disjoint from both the training shots and the sealed tail.
#   2. Report the three validation p_L values so the median teacher can be designated.
#      Median, not best: the best of three is a favourable draw rather than a typical one.
#
#      Selection never touches the 200k tail. The tail is sealed until the designated
#      teacher is frozen, and is then read once for reporting (step 4).
#   3. Build the teacher caches for the designated teacher -- one training-prefix cache
#      covering every ladder rung, and the 200k tail cache that enables the
#      ambiguous-band agreement columns.
#
# Prints the three numbers and stops; set TEACHER_SEED and re-run with BUILD_CACHES=1
# once the median is known.
#
# Usage on EAF:
#   bash eaf_teacher_selection.sh                      # steps 0-2, scoring only
#   TEACHER_SEED=1 BUILD_CACHES=1 bash eaf_teacher_selection.sh   # step 3
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
POOLS="$RT/pools"
POOL_DIR="${POOL_DIR:-pools}"
TRAIN_POOL="$RT/$POOL_DIR/data_d5_p0.010_r3.npz"
TAIL_POOL="$POOLS/data_d5_p0.010_r3_TAIL200k.npz"     # gen-seed 43, disjoint from 42
WDIR="$RT/out_t200k_w"
OUTDIR="$RT/teacher"
NTE=200000
NTR="${NTR:-10000000}"     # one prefix cache covers every ladder rung, since rungs nest
VAL_SPLIT="${VAL_SPLIT:-0.2}"   # must match what train_one.py used
PY="${PY:-python}"

# Keras validation_split takes the LAST fraction of the training prefix, before shuffling.
# For a 10M prefix at 0.2 that is shots [8,000,000, 10,000,000): held out during training,
# disjoint from the sealed tail.
VAL_START=$($PY -c "print(int($NTR * (1 - $VAL_SPLIT)))")
VAL_N=$(( NTR - VAL_START ))

# Where the evaluation tail lives. pools_t200k carries its own tail as the last 200k of
# the training file (n_total 10.2M = 10M train + 200k tail); pools/ uses the separate
# gen-seed-43 TAIL200k file starting at 0.
if [ -f "$RT/$POOL_DIR/data_d5_p0.010_r3_TAIL200k.npz" ]; then
  TAIL_SRC="$RT/$POOL_DIR/data_d5_p0.010_r3_TAIL200k.npz"
  TAIL_START=0
else
  TAIL_SRC="$TRAIN_POOL"
  TAIL_START=$(( $($PY -c "import numpy as np,sys;print(np.load(sys.argv[1])['measurements'].shape[0])" "$TRAIN_POOL") - NTE ))
fi

mkdir -p "$OUTDIR"

for f in "$TRAIN_POOL" "$TAIL_SRC"; do
  [ -f "$f" ] || { echo "MISSING $f -- build it with make_fresh_tail.py before running."; exit 1; }
done

echo "=============================================================="
echo "0. Pool fingerprints (compare against any cache built off-EAF)"
echo "=============================================================="
$PY - "$TRAIN_POOL" "$TAIL_SRC" <<'PYEOF'
import numpy as np, os, sys
sys.path.insert(0, os.getcwd())
from dump_teacher_probs import pool_fingerprint
for name, path in [('train', sys.argv[1]), ('tail ', sys.argv[2])]:
    z = np.load(path)
    sha, shape = pool_fingerprint(z)
    print(f"  {name}: {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}")
    print(f"         flips_sha256={sha[:24]}...  measurements{shape}")
PYEOF
echo
echo "  train_student.py refuses a cache whose fingerprint does not match the pool, so"
echo "  this print is a convenience check rather than the safety net."
echo

if [ "${BUILD_CACHES:-0}" != "1" ]; then
  echo "=============================================================="
  echo "1-2. Score all three teachers on the VALIDATION partition"
  echo "     shots [$VAL_START, $NTR) of $(basename "$TRAIN_POOL")  --  the tail stays sealed"
  echo "=============================================================="
  # UTC-stamped: an earlier scoring pass is never removed.
  SCORES="$OUTDIR/teacher_val_scores_$(date -u +%Y%m%dT%H%M%SZ).csv"
  for s in 0 1 2; do
    echo "--- seed $s ---"
    # no --mcnemar: MWPM on the validation slice is not a quantity we report, and
    # decoding it would only invite comparing a selection metric to a baseline
    $PY eval_on_tail.py \
      --weights "$WDIR/rcnn_d5_p0.010_r3_seed${s}_ntr10000000.weights.h5" \
      --d 5 --p 0.010 --rounds 3 --n-test "$VAL_N" --eval-start "$VAL_START" \
      --pool "$TRAIN_POOL" --data-dir "$RT/$POOL_DIR" --out-csv "$SCORES"
  done
  echo
  echo "=============================================================="
  echo "Three teachers on the validation partition:"
  echo "=============================================================="
  $PY - "$SCORES" <<'PYEOF'
import csv, sys, statistics
rows = list(csv.DictReader(open(sys.argv[1])))
vals = [(r['weights'], float(r['p_L'])) for r in rows]
for w, pl in vals:
    print(f"  {w:52s} validation p_L={pl:.6f}")
pls = sorted(v[1] for v in vals)
med = statistics.median(pls)
winner = [w for w, pl in vals if pl == med]
print(f"\n  median validation p_L = {med:.6f}  ->  {winner[0] if winner else '?'}")
print("  Selection metric: p_L on the held-out validation partition. The 200k tail was")
print("  not read, and stays sealed until the designated teacher is frozen.")
print("\n  Then re-run:")
print("    TEACHER_SEED=<n> BUILD_CACHES=1 bash eaf_teacher_selection.sh")
PYEOF
  exit 0
fi

# ---------------------------------------------------------------------------
S="${TEACHER_SEED:?set TEACHER_SEED to the median seed from the scoring step}"
W="$WDIR/rcnn_d5_p0.010_r3_seed${S}_ntr10000000.weights.h5"
echo "=============================================================="
echo "3. Building caches for the designated teacher: seed $S"
echo "=============================================================="

echo "--- training prefix, shots [0, $NTR) of $(basename "$TRAIN_POOL") ---"
$PY dump_teacher_probs.py --weights "$W" \
  --d 5 --p 0.010 --rounds 3 --n-start 0 --n-shots "$NTR" \
  --pool "$TRAIN_POOL" --out "$OUTDIR/teacher_seed${S}_prefix${NTR}_${POOL_DIR}.npz"

# train_student.py checks the cache length equals --n-test before using it.
echo "--- 200k evaluation tail (shots [$TAIL_START, $((TAIL_START+NTE))) of $(basename "$TAIL_SRC")) ---"
$PY dump_teacher_probs.py --weights "$W" \
  --d 5 --p 0.010 --rounds 3 --n-start "$TAIL_START" --n-shots "$NTE" \
  --pool "$TAIL_SRC" --out "$OUTDIR/teacher_seed${S}_tail${NTE}_${POOL_DIR}.npz"

# ---------------------------------------------------------------------------
# Step 4: the sealed tail, read once, after the teacher is designated and frozen.
# This is reporting, not selection -- nothing downstream branches on it.
echo
echo "=============================================================="
echo "4. Frozen teacher on the sealed 200k tail (reporting only)"
echo "=============================================================="
FINAL="$OUTDIR/teacher_seed${S}_final_tail_$(date -u +%Y%m%dT%H%M%SZ).csv"
$PY eval_on_tail.py --weights "$W" \
  --d 5 --p 0.010 --rounds 3 --n-test "$NTE" \
  --pool "$TAIL_SRC" --data-dir "$RT/$POOL_DIR" --mcnemar --out-csv "$FINAL"
echo "  reported -> $FINAL"

echo
echo "Done. The grid can now run with:"
echo "  --teacher-cache      $OUTDIR/teacher_seed${S}_prefix${NTR}_${POOL_DIR}.npz"
echo "  --teacher-tail-cache $OUTDIR/teacher_seed${S}_tail${NTE}_${POOL_DIR}.npz"
echo "  --pool               $TRAIN_POOL"
echo "  --lr 0.003 --temperature 1.0"
