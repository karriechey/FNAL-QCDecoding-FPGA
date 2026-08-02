#!/usr/bin/env bash
# Created: 2026-08-01
# Last modified: 2026-08-01
#
# Student data-volume ladder: p_L vs training-set size for the MLP and GRU students,
# on the same rungs, the same pool and the same tail as the RCNN ladder.
# Run of EAF.
#
# The grid:
#   rung   in {100k, 300k, 800k, 2M, 5M, 10M}   matching the RCNN ladder
#   arch   in {mlp, gru}
#   alpha  in {1.0, 0.0}
#   seed   in {0, 1, 2}
#   = 72 runs
#
# Why both alpha arms:
#   alpha=1.0 (hard labels) is the like-for-like architecture comparison: the RCNN ladder
#   was trained on hard labels, so only this arm can be overlaid on it as an architecture
#   result. alpha=0.0 (distilled) is the deployable method. The gap between the two curves
#   is itself a result -- whether distillation buys more at small N, where the hard label
#   carries least information.
#
# Why this pool
#   pools_t200k is the pool the RCNN ladder ran on: 10.2M shots = 10M training capacity +
#   a fixed 200k tail, nested prefixes growing from the front. Every rung and every
#   architecture therefore trains on the same shots and is scored on the same tail, with
#   one baseline (MWPM 0.04875, decoded on that tail). The 100k-5M RCNN rungs have no
#   saved weights, so they can never be re-scored onto a different tail -- this is the
#   only tail every point can share.
#
#   No --test-pool: the tail is the last 200k of the training file, and train_student.py's
#   `ntr <= N - nte` assert enforces disjointness (10,000,000 <= 10,000,000 at the top
#   rung, exact by construction).
#
# Note on the distilled curve
#   The teacher was trained on 10M shots. A distilled student at the 100k rung therefore
#   has indirect access to far more data than a 100k RCNN did, so the distilled curve is
#   not a like-for-like point on a data-volume axis. State this in any figure caption.
#
# Usage:  bash eaf_run_ladder.sh
#         ARCHS=mlp bash eaf_run_ladder.sh              # one architecture
#         RUNGS="100000 300000" bash eaf_run_ladder.sh  # quick check
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
POOL_DIR="${POOL_DIR:-pools_t200k}"
TEACHER_SEED="${TEACHER_SEED:-1}"
TRAIN_POOL="$RT/$POOL_DIR/data_d5_p0.010_r3.npz"
PREFIX_CACHE="$RT/teacher/teacher_seed${TEACHER_SEED}_prefix10000000_${POOL_DIR}.npz"
TAIL_CACHE="$RT/teacher/teacher_seed${TEACHER_SEED}_tail200000_${POOL_DIR}.npz"
OUTDIR="${OUTDIR:-$RT/out_student_ladder_$POOL_DIR}"
NTE=200000
RUNGS="${RUNGS:-100000 300000 800000 2000000 5000000 10000000}"
ARCHS="${ARCHS:-mlp gru}"
ALPHAS="${ALPHAS:-1.0 0.0}"
SEEDS="${SEEDS:-0 1 2}"
PY="${PY:-python}"

for f in "$TRAIN_POOL" "$PREFIX_CACHE" "$TAIL_CACHE"; do
  [ -f "$f" ] || { echo "MISSING $f"; echo "Build the caches first: POOL_DIR=$POOL_DIR TEACHER_SEED=$TEACHER_SEED BUILD_CACHES=1 bash eaf_teacher_selection.sh"; exit 1; }
done
mkdir -p "$OUTDIR"

n_runs=0
for a in $ALPHAS; do for arch in $ARCHS; do for n in $RUNGS; do for s in $SEEDS; do
  n_runs=$((n_runs+1)); done; done; done; done
echo "student ladder: $n_runs runs -> $OUTDIR"
echo "  rungs=$RUNGS"
echo "  archs=$ARCHS  alphas=$ALPHAS  seeds=$SEEDS"
echo "  pool=$POOL_DIR  teacher=seed$TEACHER_SEED"
echo

i=0
for a in $ALPHAS; do
  for arch in $ARCHS; do
    # GRU takes no Dense stack after the recurrence; MLP uses the default two layers.
    [ "$arch" = "gru" ] && HID="--hidden" || HID=""
    for n in $RUNGS; do
      for s in $SEEDS; do
        i=$((i+1))
        tag="ladder_${arch}_alpha${a}_ntr${n}_seed${s}_${POOL_DIR}"
        echo "=============== [$i/$n_runs] $tag ==============="
        $PY train_student.py \
          --student "$arch" --inputs evts $HID \
          --alpha "$a" --temperature 1.0 --lr 0.003 \
          --seed "$s" --n-train "$n" --n-test "$NTE" \
          --epochs 50 --batch-size 10000 --no-early-stopping \
          --teacher-cache "$PREFIX_CACHE" \
          --teacher-tail-cache "$TAIL_CACHE" \
          --pool "$TRAIN_POOL" --data-dir "$RT/$POOL_DIR" \
          --out-dir "$OUTDIR" --tag "$tag" 2>&1 \
          | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique|^Epoch |- loss:"
        echo
      done
    done
  done
done

echo "=============================================================="
echo "Ladder complete."
echo "=============================================================="
$PY - "$OUTDIR" <<'PYEOF'
import csv, glob, os, sys, statistics as st, collections
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], 'ladder_*.csv'))):
    rows += list(csv.DictReader(open(f)))
key = lambda r: (r['student'], r['alpha'], int(r['n_train']))
groups = collections.defaultdict(list)
for r in rows:
    groups[key(r)].append(float(r['p_L']))
print(f"  {'arch':>5} {'alpha':>6} {'n_train':>10} {'seeds':>6} {'mean p_L':>10} {'std':>9}")
for k in sorted(groups):
    v = groups[k]
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f"  {k[0]:>5} {k[1]:>6} {k[2]:>10,} {len(v):>6} {st.mean(v):>10.5f} {sd:>9.5f}")
print()
print("  Same tail as the RCNN ladder: MWPM = 0.04875 on the pools_t200k 200k tail.")
print("  alpha=1.0 is the like-for-like architecture comparison against the RCNN curve;")
print("  alpha=0.0 is the deployable distilled student and is NOT a like-for-like point")
print("  on the data-volume axis (its teacher saw 10M shots).")
PYEOF
