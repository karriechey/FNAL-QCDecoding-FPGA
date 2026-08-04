#!/usr/bin/env bash
# Created: 2026-08-03
# Last modified: 2026-08-03
#
# Is the students' floor architectural, or just capacity?
# Run on EAF.
#
# The ladder ran students well below the teacher's size:
#   RCNN teacher            51,547 params
#   MLP hidden=(128,128)    25,985  (0.50x)
#   GRU units=64            17,153  (0.33x)
# so "the RCNN's structure matters" is confounded with "the RCNN is 2-3x bigger". This
# re-runs both students at matched size:
#   MLP hidden=(192,192)    51,265  (0.994x)
#   GRU units=119           51,528  (0.9996x)
#
# Two rungs, not one: 10M and 20M. One point cannot distinguish a student that has moved
# to a lower floor from one that has merely slid along the same curve.
#
# Grid: {mlp, gru} x {alpha 1.0, 0.0} x {10M, 20M} x 3 seeds = 24 runs.
#
# 10M trains on pools_t200k (the ladder's pool); 20M on pools_20M. Both score on the
# pools_t200k tail, MWPM 0.04875, matching every other point in the figure.
#
# Usage:  bash eaf_run_matched.sh
#         ARCHS=mlp bash eaf_run_matched.sh
#         RUNGS=10000000 bash eaf_run_matched.sh
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
TEACHER_SEED="${TEACHER_SEED:-1}"
NTE=200000
MLP_HIDDEN="${MLP_HIDDEN:-192 192}"
GRU_UNITS="${GRU_UNITS:-119}"

POOL_10M="$RT/pools_t200k/data_d5_p0.010_r3.npz"
POOL_20M="$RT/pools_20M/data_d5_p0.010_r3.npz"
CACHE_10M="$RT/teacher/teacher_seed${TEACHER_SEED}_prefix10000000_pools_t200k.npz"
CACHE_20M="$RT/teacher/teacher_seed${TEACHER_SEED}_prefix20000000_pools_20M.npz"
TAIL_POOL="$POOL_10M"        # score on its last 200k
TAIL_CACHE="$RT/teacher/teacher_seed${TEACHER_SEED}_tail200000_pools_t200k.npz"
OUTDIR="${OUTDIR:-$RT/out_student_matched}"
ARCHS="${ARCHS:-mlp gru}"
ALPHAS="${ALPHAS:-1.0 0.0}"
RUNGS="${RUNGS:-10000000 20000000}"
SEEDS="${SEEDS:-0 1 2}"
PY="${PY:-python}"

for f in "$POOL_10M" "$TAIL_CACHE"; do
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
done
mkdir -p "$OUTDIR"

n_runs=0
for a in $ALPHAS; do for arch in $ARCHS; do for n in $RUNGS; do for s in $SEEDS; do
  n_runs=$((n_runs+1)); done; done; done; done
echo "matched-capacity runs: $n_runs -> $OUTDIR"
echo "  mlp hidden=($MLP_HIDDEN)   gru units=$GRU_UNITS   target 51,547 (RCNN)"
echo "  rungs=$RUNGS  alphas=$ALPHAS  seeds=$SEEDS"
echo

i=0
for a in $ALPHAS; do
  for arch in $ARCHS; do
    if [ "$arch" = "gru" ]; then
      SIZE="--hidden --units $GRU_UNITS"; szlab="u${GRU_UNITS}"
    else
      SIZE="--hidden $MLP_HIDDEN";        szlab="h$(echo "$MLP_HIDDEN" | tr ' ' '-')"
    fi
    for n in $RUNGS; do
      # each rung trains on the pool that holds that many shots
      if [ "$n" -gt 10000000 ]; then
        POOL="$POOL_20M"; CACHE="$CACHE_20M"; pooltag="pools20M"
      else
        POOL="$POOL_10M"; CACHE="$CACHE_10M"; pooltag="poolst200k"
      fi
      for f in "$POOL" "$CACHE"; do
        [ -f "$f" ] || { echo "MISSING $f (needed for the ${n} rung)"; exit 1; }
      done
      for s in $SEEDS; do
        i=$((i+1))
        tag="matched_${arch}_${szlab}_alpha${a}_ntr${n}_seed${s}_${pooltag}"
        echo "=============== [$i/$n_runs] $tag ==============="
        $PY train_student.py \
          --student "$arch" --inputs evts $SIZE \
          --alpha "$a" --temperature 1.0 --lr 0.003 \
          --seed "$s" --n-train "$n" --n-test "$NTE" \
          --epochs 50 --batch-size 10000 --no-early-stopping \
          --teacher-cache "$CACHE" --teacher-tail-cache "$TAIL_CACHE" \
          --pool "$POOL" --test-pool "$TAIL_POOL" \
          --out-dir "$OUTDIR" --tag "$tag" 2>&1 \
          | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique|^Epoch |- loss:"
        echo
      done
    done
  done
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SUMMARY="$RT/summary_matched_${STAMP}.csv"
ARCHIVE="$HOME/results_matched_${STAMP}.tgz"

echo "=============================================================="
echo "Matched-capacity runs complete."
echo "=============================================================="
$PY - "$OUTDIR" "$SUMMARY" <<'PYEOF'
import csv, glob, os, sys, statistics as st, collections
outdir, summary_path = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(outdir, 'matched_*.csv'))):
    if '.superseded_' in os.path.basename(f):
        continue
    rows += list(csv.DictReader(open(f)))
if rows:
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

MW = 0.04875
# what the same arms did at the sizes the ladder used
SMALL = {('mlp', '1.0', 10000000): 0.05431, ('mlp', '0.0', 10000000): 0.05350,
         ('gru', '1.0', 10000000): 0.05625, ('gru', '0.0', 10000000): 0.05567,
         ('mlp', '1.0', 20000000): 0.05203, ('mlp', '0.0', 20000000): 0.05215,
         ('gru', '1.0', 20000000): 0.05292, ('gru', '0.0', 20000000): 0.05261}
g = collections.defaultdict(list)
params = {}
for r in rows:
    k = (r['student'], r['alpha'], int(r['n_train']))
    g[k].append(float(r['p_L']))
    params[(r['student'],)] = r.get('n_params', '?')
print(f"  {'arch':>4} {'alpha':>5} {'n_train':>11} {'n':>2} {'p_L':>9} {'std':>8} {'xMWPM':>6} "
      f"{'small-model':>12} {'change':>9}")
for k in sorted(g):
    v = g[k]; m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    prev = SMALL.get(k)
    ch = f"{m - prev:+9.5f}" if prev else " " * 9
    pv = f"{prev:12.5f}" if prev else " " * 12
    print(f"  {k[0]:>4} {k[1]:>5} {k[2]:>11,} {len(v):>2} {m:>9.5f} {sd:>8.5f} {m/MW:>6.3f} {pv} {ch}")
print(f"\n  params: " + "  ".join(f"{a[0]}={params[a]}" for a in sorted(params)) + "   RCNN=51,547")
print(f"  MWPM {MW} · RCNN 10M 0.04628 (0.949x) · RCNN 20M 0.04801 (0.985x)")
print("  A lower floor at matched size => the earlier gap was capacity.")
print("  Unchanged => the gap is architectural, and capacity was not the constraint.")
print(f"\n  {len(rows)} runs collated -> {summary_path}")
PYEOF

tar czf "$ARCHIVE" -C "$RT" "$(basename "$OUTDIR")"
echo "  archive -> $ARCHIVE  ($(du -h "$ARCHIVE" | cut -f1))"
echo
echo "  Download: $SUMMARY"
