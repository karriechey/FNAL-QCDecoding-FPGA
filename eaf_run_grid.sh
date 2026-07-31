#!/usr/bin/env bash
# Created: 2026-07-31
# Last modified: 2026-07-31
#
# Phase 2 de-risk grid: does distillation beat hard-label training for the MLP student?
# RUN ON EAF. Requires the seed-1 teacher caches from eaf_teacher_selection.sh.
#
# THE GRID
#   alpha in {0.0, 1.0} x seed in {0,1,2}  = 6 runs
#     alpha=0.0 -> pure distillation (learns only from the teacher's soft output)
#     alpha=1.0 -> pure hard labels   (control arm: what the architecture does alone)
#   Three student seeds per arm, because a one-seed difference between the arms is not
#   evidence of anything -- the whole question is whether the arms differ by more than
#   seed scatter. Same code path for both arms (only --alpha changes), so the comparison
#   cannot be confounded by two different trainers.
#
# FIXED, NOT SWEPT (settled earlier, do not vary here)
#   --lr 0.003          pinned by the LR de-risk; the inherited train_one.py scheduler is
#                       deliberately NOT used (it opens at 1e-2 to kick the teacher's
#                       zero-init state correlator, a layer the student does not have)
#   --temperature 1.0
#   --student mlp       GRU is parked pending the QGRU reset_after and input-layout checks
#   --inputs evts       detector events only; 'evts+bits' is a separate question
#   fixed 50 epochs, no early stopping -- both arms get an identical budget, so neither
#   can win by training longer
#
# SCORING
#   --test-pool is the fresh 200k tail (gen-seed 43), disjoint from the training pool by
#   construction. train_student.py suppresses the stored-baseline MWPM ratio for this
#   pool; the valid comparison is MWPM = 0.049405, re-decoded on this same tail, and the
#   seed-1 teacher's 0.046395 on it.
#
# Usage:  bash eaf_run_grid.sh            # all 6 runs
#         SEEDS="0" bash eaf_run_grid.sh  # single seed, e.g. a quick check first
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
TEACHER_SEED="${TEACHER_SEED:-1}"
PREFIX_CACHE="$RT/teacher/teacher_seed${TEACHER_SEED}_prefix1000000.npz"
TAIL_CACHE="$RT/teacher/teacher_seed${TEACHER_SEED}_tail200000.npz"
TAIL_POOL="$RT/pools/data_d5_p0.010_r3_TAIL200k.npz"
POOL_DIR="${POOL_DIR:-pools}"          # which training pool; part of every output name
TRAIN_POOL="$RT/$POOL_DIR/data_d5_p0.010_r3.npz"
OUTDIR="${OUTDIR:-$RT/out_student_grid_$POOL_DIR}"
NTR="${NTR:-1000000}"
NTE=200000
SEEDS="${SEEDS:-0 1 2}"
ALPHAS="${ALPHAS:-0.0 1.0}"
PY="${PY:-python}"

for f in "$PREFIX_CACHE" "$TAIL_CACHE" "$TAIL_POOL" "$TRAIN_POOL"; do
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
done
mkdir -p "$OUTDIR"

echo "grid: alphas={$ALPHAS} x seeds={$SEEDS}  ntr=$NTR  teacher=seed$TEACHER_SEED"
echo "out:  $OUTDIR"
echo

for a in $ALPHAS; do
  for s in $SEEDS; do
    tag="grid_mlp_alpha${a}_seed${s}_ntr${NTR}_${POOL_DIR}"
    echo "=============== $tag ==============="
    $PY train_student.py \
      --student mlp --inputs evts \
      --alpha "$a" --temperature 1.0 --lr 0.003 \
      --seed "$s" --n-train "$NTR" --n-test "$NTE" \
      --epochs 50 --batch-size 10000 --no-early-stopping \
      --tail-diagnostics \
      --teacher-cache "$PREFIX_CACHE" \
      --teacher-tail-cache "$TAIL_CACHE" \
      --pool "$TRAIN_POOL" --test-pool "$TAIL_POOL" \
      --out-dir "$OUTDIR" --tag "$tag" 2>&1 | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique"
    echo
  done
done

echo "=============================================================="
echo "Grid complete. Summary:"
echo "=============================================================="
$PY - "$OUTDIR" <<'PYEOF'
import csv, glob, os, sys, statistics
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], 'grid_*.csv'))):
    rows += list(csv.DictReader(open(f)))
cols = ['alpha', 'seed', 'p_L', 'agree_all', 'agree_ambiguous', 'student_p_L_ambiguous',
        'teacher_p_L_ambiguous', 'best_val_loss']
print('  ' + ' '.join(f'{c:>22s}' for c in cols))
for r in sorted(rows, key=lambda r: (r['alpha'], r['seed'])):
    print('  ' + ' '.join(f"{r.get(c,''):>22s}" for c in cols))
print()
for a in sorted({r['alpha'] for r in rows}):
    pls = [float(r['p_L']) for r in rows if r['alpha'] == a]
    arm = 'distillation' if float(a) == 0.0 else 'hard labels'
    sd = statistics.stdev(pls) if len(pls) > 1 else 0.0
    print(f"  alpha={a} ({arm:12s}) p_L = {statistics.mean(pls):.5f} +/- {sd:.5f}  (n={len(pls)})")
print()
print("  Reference on this SAME tail: teacher(seed1) p_L = 0.046395, MWPM = 0.049405.")
print("  The bar for the student is staying under MWPM, not merely tracking the teacher.")
print("  Read agree_ambiguous, NOT agree_all -- the teacher is confident on ~80% of shots,")
print("  so the raw rate is dominated by the easy majority and reads misleadingly high.")
PYEOF
