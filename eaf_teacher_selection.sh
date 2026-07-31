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
#   1. Score all three FP32 teacher checkpoints (seeds 0/1/2) on the FRESH 200k tail,
#      with --mcnemar so MWPM is re-decoded on that same tail rather than read from a
#      stored column. Baselines are tail-specific; the pools/mwpm_baseline.csv value
#      (0.0518, a 10k tail) does not describe this tail.
#   2. Report the three tail p_L values so the MEDIAN teacher can be designated as the
#      paper's teacher. Median, not best: picking the best of three on the same tail the
#      students are later scored on would select a favourable draw and leak it into every
#      downstream number.
#   3. Build the teacher caches for the designated teacher -- the 1M training prefix and
#      the 200k tail. The tail cache is what enables the ambiguous-band agreement columns.
#
# This script does NOT pick the teacher for you. It prints the three numbers and stops;
# set TEACHER_SEED and re-run with BUILD_CACHES=1 once the median is known.
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
NTR=1000000
PY="${PY:-python}"

mkdir -p "$OUTDIR"

for f in "$TRAIN_POOL" "$TAIL_POOL"; do
  [ -f "$f" ] || { echo "MISSING $f -- build it with make_fresh_tail.py before running."; exit 1; }
done

echo "=============================================================="
echo "0. Pool fingerprints (compare against any cache built off-EAF)"
echo "=============================================================="
$PY - <<'PYEOF'
import numpy as np, os, sys
sys.path.insert(0, os.getcwd())
from dump_teacher_probs import pool_fingerprint
for name, path in [('train', os.path.expanduser('~/rcnn_threshold/pools/data_d5_p0.010_r3.npz')),
                   ('tail ', os.path.expanduser('~/rcnn_threshold/pools/data_d5_p0.010_r3_TAIL200k.npz'))]:
    z = np.load(path)
    sha, shape = pool_fingerprint(z)
    print(f"  {name} pool: flips_sha256={sha[:24]}...  measurements{shape}")
PYEOF
echo
echo "  NOTE: the Mac-built 1M prefix caches carry pool flips_sha256 67995c29ee99...."
echo "  If the train pool above matches that prefix, those caches are valid here and"
echo "  step 3's prefix build is redundant. If it does NOT match, the pools differ and"
echo "  the caches MUST be rebuilt here -- train_student.py will refuse them either way,"
echo "  so this is a convenience check, not the safety net."
echo

if [ "${BUILD_CACHES:-0}" != "1" ]; then
  echo "=============================================================="
  echo "1-2. Score all three teachers on the FRESH 200k tail"
  echo "=============================================================="
  # UTC-stamped: an earlier scoring pass is never removed.
  SCORES="$OUTDIR/teacher_tail_scores_$(date -u +%Y%m%dT%H%M%SZ).csv"
  for s in 0 1 2; do
    echo "--- seed $s ---"
    $PY eval_on_tail.py \
      --weights "$WDIR/rcnn_d5_p0.010_r3_seed${s}_ntr10000000.weights.h5" \
      --d 5 --p 0.010 --rounds 3 --n-test "$NTE" \
      --pool "$TAIL_POOL" --mcnemar --out-csv "$SCORES"
  done
  echo
  echo "=============================================================="
  echo "Three teachers on the fresh 200k tail:"
  echo "=============================================================="
  $PY - "$SCORES" <<'PYEOF'
import csv, sys, statistics
rows = list(csv.DictReader(open(sys.argv[1])))
vals = [(r['weights'], float(r['p_L']), float(r['mwpm_p_L'])) for r in rows]
for w, pl, mw in vals:
    print(f"  {w:52s} p_L={pl:.6f}")
pls = sorted(v[1] for v in vals)
med = statistics.median(pls)
winner = [w for w, pl, _ in vals if pl == med]
print(f"\n  MWPM on this tail (re-decoded): {vals[0][2]:.6f}")
print(f"  median p_L = {med:.6f}  ->  {winner[0] if winner else '?'}")
print("\n  Designate that seed as the paper's teacher. Then re-run:")
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

echo "--- 1M training prefix (from the TRAIN pool) ---"
$PY dump_teacher_probs.py --weights "$W" \
  --d 5 --p 0.010 --rounds 3 --n-start 0 --n-shots "$NTR" \
  --pool "$TRAIN_POOL" --out "$OUTDIR/teacher_seed${S}_prefix${NTR}_${POOL_DIR}.npz"

# The tail cache covers the WHOLE fresh tail pool: that file is 200k shots of pure
# evaluation data, so the range is [0, 200000) of THAT file, not a slice of the training
# pool. train_student.py checks the cache length equals --n-test before using it.
echo "--- 200k evaluation tail (from the TAIL pool) ---"
$PY dump_teacher_probs.py --weights "$W" \
  --d 5 --p 0.010 --rounds 3 --n-start 0 --n-shots "$NTE" \
  --pool "$TAIL_POOL" --out "$OUTDIR/teacher_seed${S}_tail${NTE}_${POOL_DIR}.npz"

echo
echo "Done. The grid can now run with:"
echo "  --teacher-cache      $OUTDIR/teacher_seed${S}_prefix${NTR}_${POOL_DIR}.npz"
echo "  --teacher-tail-cache $OUTDIR/teacher_seed${S}_tail${NTE}_${POOL_DIR}.npz"
echo "  --test-pool          $TAIL_POOL"
echo "  --lr 0.003 --temperature 1.0"
