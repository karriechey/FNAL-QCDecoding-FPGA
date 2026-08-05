#!/usr/bin/env bash
# Created: 2026-08-04
# Last modified: 2026-08-04
#
# Hard-label student ladder at d=5, r=5. Run on EAF.
#
# Independent of the RCNN: no teacher, no distillation, so it does not wait on the
# learning-rate decision or on any teacher checkpoint. alpha=1.0 throughout.
#
# Matched to the r=5 teacher's 69,485 parameters:
#   MLP hidden=(209,209)  69,389
#   GRU units=140         69,441
# Both take det_evts only, so all three architectures see the same input channel.
#
# Scores the validation partition [20.0M, 20.2M) via --eval-start. Without that the
# trainer would score the last --n-test shots, which on this pool is the sealed test.
set -euo pipefail
RT="${RT:-$HOME/rcnn_threshold}"
POOL="$RT/pools_r5/data_d5_p0.010_r5_FORMAL.npz"
VAL_START=20000000
VAL_N=200000
OUTDIR="${OUTDIR:-$RT/out_student_r5_hard}"
RUNGS="${RUNGS:-100000 300000 800000 2000000 5000000 10000000 20000000}"
ARCHS="${ARCHS:-mlp gru}"
SEEDS="${SEEDS:-0 1 2}"
MLP_HIDDEN="${MLP_HIDDEN:-209 209}"
GRU_UNITS="${GRU_UNITS:-140}"
PY="${PY:-python}"
[ -f "$POOL" ] || { echo "MISSING $POOL"; exit 1; }
mkdir -p "$OUTDIR"

n=0; for a in $ARCHS; do for r in $RUNGS; do for s in $SEEDS; do n=$((n+1)); done; done; done
echo "student r=5 hard-label ladder: $n runs -> $OUTDIR"
echo "  mlp hidden=($MLP_HIDDEN)  gru units=$GRU_UNITS   target 69,485 (RCNN r=5)"
echo "  validation [$VAL_START, $((VAL_START+VAL_N)))   sealed test not read"
echo

i=0
for arch in $ARCHS; do
  if [ "$arch" = "gru" ]; then SIZE="--hidden --units $GRU_UNITS"; else SIZE="--hidden $MLP_HIDDEN"; fi
  for ntr in $RUNGS; do
    for s in $SEEDS; do
      i=$((i+1))
      tag="r5hard_${arch}_ntr${ntr}_seed${s}"
      echo "=============== [$i/$n] $tag ==============="
      $PY train_student.py --student "$arch" --inputs evts $SIZE \
        --d 5 --p 0.010 --rounds 5 \
        --alpha 1.0 --temperature 1.0 --lr 0.003 \
        --seed "$s" --n-train "$ntr" --n-test "$VAL_N" --eval-start "$VAL_START" \
        --epochs 50 --batch-size 10000 --no-early-stopping \
        --pool "$POOL" --out-dir "$OUTDIR" --tag "$tag" 2>&1 \
        | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique|^Epoch |- loss:"
      echo
    done
  done
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SUMMARY="$RT/student_r5_hard_${STAMP}.csv"
echo "=============================================================="
$PY - "$OUTDIR" "$SUMMARY" <<'PYEOF'
import csv, glob, os, sys, statistics as st, collections
outdir, out = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(outdir, 'r5hard_*.csv'))):
    if '.superseded_' in os.path.basename(f):
        continue
    rows += list(csv.DictReader(open(f)))
if rows:
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
MW = 0.084215
g = collections.defaultdict(list)
for r in rows:
    g[(r['student'], int(r['n_train']))].append(float(r['p_L']))
print(f"  {'arch':>5} {'n_train':>11} {'seeds':>6} {'val p_L':>9} {'std':>8} {'xMWPM':>7}")
for k in sorted(g, key=lambda k: (k[0], k[1])):
    v = g[k]
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f"  {k[0]:>5} {k[1]:>11,} {len(v):>6} {st.mean(v):>9.5f} {sd:>8.5f} {st.mean(v)/MW:>7.3f}")
print(f"\n  MWPM on validation = {MW}.  Hard labels only; the distilled arms need a teacher")
print(f"  and wait on the r=5 schedule decision.  {len(rows)} runs -> {out}")
PYEOF
