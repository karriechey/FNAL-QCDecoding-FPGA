#!/usr/bin/env bash
# Created: 2026-08-03
# Last modified: 2026-08-03
#
# Does either student reach MWPM parity at 20M shots?
# Run on EAF.
#
# Why this run
#   Both student curves are still descending at 10M where the RCNN had flattened. A
#   power-law fit to the last three rungs (2M/5M/10M) puts all four arms within 2% of
#   MWPM at 20M and crossing parity near 22M, so 20M is exactly where the answer is
#   uncertain: either the descent continues and a synthesizable student can match MWPM
#   given data, or the students hit their own floor as the RCNN did.
#
# The pool
#   Reproduces the original Experiment 5 pool: 20.2M shots at gen-seed 42, verified
#   bit-identical by its flips sha256. EAF's pools_20M was cleared, but the draw is
#   deterministic in (n, seed), so regenerating is cheaper than moving 3.4 GB.
#
#   The sample stream depends on n as well as the seed, so this pool is NOT a superset of
#   pools_t200k (10.2M, also seed 42) -- they are independent draws. That is what makes it
#   safe to score these runs on the pools_t200k tail: those shots appear in no training
#   prefix here, and the rung lands on the same axis and the same MWPM (0.04875) as every
#   other point in the figure.
#
# The grid: {mlp, gru} x {alpha 1.0, 0.0} x 3 seeds = 12 runs at n_train = 20M.
#
# Usage:  bash eaf_run_20M.sh
#         SKIP_POOL=1 bash eaf_run_20M.sh    # pool and teacher cache already built
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
TEACHER_SEED="${TEACHER_SEED:-1}"
GEN_SEED="${GEN_SEED:-42}"      # the original Experiment 5 pool
NPOOL="${NPOOL:-20200000}"      # 20M training capacity + its own 200k tail
EXPECT_SHA="90f0a00cfbf9e356"
NTR="${NTR:-20000000}"
NTE=200000

POOL20="$RT/pools_20M/data_d5_p0.010_r3.npz"
TAIL_POOL="$RT/pools_t200k/data_d5_p0.010_r3.npz"   # score on its last 200k, MWPM 0.04875
PREFIX_CACHE="$RT/teacher/teacher_seed${TEACHER_SEED}_prefix${NTR}_pools_20M.npz"
TAIL_CACHE="$RT/teacher/teacher_seed${TEACHER_SEED}_tail200000_pools_t200k.npz"
WDIR="$RT/out_t200k_w"
OUTDIR="${OUTDIR:-$RT/out_student_20M}"
ARCHS="${ARCHS:-mlp gru}"
ALPHAS="${ALPHAS:-1.0 0.0}"
SEEDS="${SEEDS:-0 1 2}"
PY="${PY:-python}"

[ -f "$TAIL_POOL" ]  || { echo "MISSING $TAIL_POOL"; exit 1; }
[ -f "$TAIL_CACHE" ] || { echo "MISSING $TAIL_CACHE -- build it with eaf_teacher_selection.sh"; exit 1; }
mkdir -p "$OUTDIR" "$RT/pools_20M"

if [ "${SKIP_POOL:-0}" != "1" ]; then
  if [ -f "$POOL20" ]; then
    echo "pool already present: $POOL20"
  else
    echo "=============================================================="
    echo "1. Reproducing the Experiment 5 pool: $NPOOL shots, gen-seed $GEN_SEED"
    echo "   (fails unless the draw is bit-identical to the original)"
    echo "=============================================================="
    $PY make_fresh_tail.py --d 5 --p 0.010 --rounds 3 \
      --n "$NPOOL" --gen-seed "$GEN_SEED" --allow-training-seed \
      --expect-flips-sha "$EXPECT_SHA" --out "$POOL20"
  fi

  if [ -f "$PREFIX_CACHE" ]; then
    echo "teacher cache already present: $PREFIX_CACHE"
  else
    echo "=============================================================="
    echo "2. Teacher outputs over the 20M pool (seed $TEACHER_SEED)"
    echo "=============================================================="
    $PY dump_teacher_probs.py \
      --weights "$WDIR/rcnn_d5_p0.010_r3_seed${TEACHER_SEED}_ntr10000000.weights.h5" \
      --d 5 --p 0.010 --rounds 3 --n-start 0 --n-shots "$NTR" \
      --pool "$POOL20" --out "$PREFIX_CACHE"
  fi
fi

n_runs=0
for a in $ALPHAS; do for arch in $ARCHS; do for s in $SEEDS; do n_runs=$((n_runs+1)); done; done; done
echo
echo "=============================================================="
echo "3. $n_runs runs at n_train=$NTR -> $OUTDIR"
echo "=============================================================="

i=0
for a in $ALPHAS; do
  for arch in $ARCHS; do
    [ "$arch" = "gru" ] && HID="--hidden" || HID=""
    for s in $SEEDS; do
      i=$((i+1))
      tag="n20M_${arch}_alpha${a}_ntr${NTR}_seed${s}_pools20M"
      echo "=============== [$i/$n_runs] $tag ==============="
      $PY train_student.py \
        --student "$arch" --inputs evts $HID \
        --alpha "$a" --temperature 1.0 --lr 0.003 \
        --seed "$s" --n-train "$NTR" --n-test "$NTE" \
        --epochs 50 --batch-size 10000 --no-early-stopping \
        --teacher-cache "$PREFIX_CACHE" \
        --teacher-tail-cache "$TAIL_CACHE" \
        --pool "$POOL20" --test-pool "$TAIL_POOL" \
        --out-dir "$OUTDIR" --tag "$tag" 2>&1 \
        | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique|^Epoch |- loss:"
      echo
    done
  done
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SUMMARY="$RT/summary_20M_${STAMP}.csv"
ARCHIVE="$HOME/results_20M_${STAMP}.tgz"

echo "=============================================================="
echo "20M run complete."
echo "=============================================================="
$PY - "$OUTDIR" "$SUMMARY" <<'PYEOF'
import csv, glob, os, sys, statistics as st, collections
outdir, summary_path = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(outdir, 'n20M_*.csv'))):
    if '.superseded_' in os.path.basename(f):
        continue
    rows += list(csv.DictReader(open(f)))
if rows:
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

MW = 0.04875   # re-decoded on the pools_t200k tail, the tail these runs score on
g = collections.defaultdict(list)
for r in rows:
    g[(r['student'], r['alpha'])].append(float(r['p_L']))
print(f"  {'arch':>5} {'alpha':>6} {'seeds':>6} {'mean p_L':>10} {'std':>9} {'xMWPM':>7}")
for k in sorted(g):
    v = g[k]
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f"  {k[0]:>5} {k[1]:>6} {len(v):>6} {st.mean(v):>10.5f} {sd:>9.5f} {st.mean(v)/MW:>7.3f}")
print(f"\n  MWPM on this tail = {MW}.  10M reference (same tail):")
print("    mlp distilled 0.05350 (1.098x) · mlp hard 0.05431 · gru distilled 0.05567 · gru hard 0.05625")
print("  Power-law fit predicted ~0.0495 (1.015x) for all four arms at 20M.")
print("  Below prediction => still descending. At or above => a floor, as the RCNN hit.")
print(f"\n  {len(rows)} runs collated -> {summary_path}")
PYEOF

tar czf "$ARCHIVE" -C "$RT" "$(basename "$OUTDIR")"
echo "  archive -> $ARCHIVE  ($(du -h "$ARCHIVE" | cut -f1))"
echo
echo "  Download: $SUMMARY"
