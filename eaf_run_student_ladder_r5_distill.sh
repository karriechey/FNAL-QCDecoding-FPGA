#!/usr/bin/env bash
# Created: 2026-08-05
# Last modified: 2026-08-05
#
# Distilled student ladder at d=5, r=5 (alpha=0.0). Run on EAF.
#
# Companion to eaf_run_student_ladder_r5.sh, which ran the same architectures on hard
# labels (alpha=1.0). Same rungs, same seeds, same architectures, same validation
# partition, so the two ladders differ in exactly one variable: where the target comes
# from. That is the method claim being tested.
#
# What the hard-label ladder already showed, and what this is aimed at:
#   at 20M the GRU student (0.933x MWPM) BEATS its teacher (0.966x), so distillation has
#   no headroom there. At 800k the ordering reverses -- GRU 1.850x vs teacher 1.609x --
#   which is the data-starved regime where soft targets classically help. The low rungs
#   are the informative part of this plot, not the deep ones.
#
# Stage 1 dumps the frozen teacher's output once. Stage 2 runs the ladder against it.
# The cache is absolute-indexed by shot_idx, so ONE dump over [0, 20M) serves every rung.
#
# All output lands in ONE timestamped folder (driver.log, summary.csv, runs/, MANIFEST).
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
POOL="$RT/pools_r5/data_d5_p0.010_r5_FORMAL.npz"
PY="${PY:-$HOME/QuantumDecoderQKeras/.venv/bin/python}"

# The teacher: the MEDIAN of the three 20M seeds on the validation partition, not the best.
# seed0 = 0.081585, seed1 = 0.08264, seed2 = 0.07980. Picking the best seed would make the
# teacher an outlier draw and inflate whatever the students inherit from it.
TEACHER_SEED="${TEACHER_SEED:-0}"
TEACHER="$RT/out_r5_ladder/ntr20000000/seed${TEACHER_SEED}/rcnn_d5_p0.010_r5_seed${TEACHER_SEED}_ntr20000000.weights.h5"

VAL_START=20000000
VAL_N=200000
MAX_NTR=20000000

RUNGS="${RUNGS:-100000 300000 800000 2000000 5000000 10000000 20000000}"
ARCHS="${ARCHS:-gru mlp}"
SEEDS="${SEEDS:-0 1 2}"
MLP_HIDDEN="${MLP_HIDDEN:-209 209}"
GRU_UNITS="${GRU_UNITS:-140}"
ALPHA="${ALPHA:-0.0}"
TEMP="${TEMP:-1.0}"
LR="${LR:-0.003}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$RT/results/student_r5_distill_${STAMP}"
RUNS="$OUT/runs"
CACHE_DIR="$RT/teacher_r5"
CACHE_TRAIN="$CACHE_DIR/teacher_r5_seed${TEACHER_SEED}_first20M.npz"
CACHE_TAIL="$CACHE_DIR/teacher_r5_seed${TEACHER_SEED}_val200k.npz"

mkdir -p "$RUNS" "$CACHE_DIR" "$RT/transfer"

[ -f "$POOL" ]    || { echo "MISSING pool $POOL"; exit 1; }
[ -f "$TEACHER" ] || { echo "MISSING teacher $TEACHER"; exit 1; }

# ---------------------------------------------------------------------------------------
# MANIFEST: what was launched, against which code and which inputs. Written before any
# compute, so a run that dies still leaves a record of what it was trying to do.
# ---------------------------------------------------------------------------------------
{
  echo "experiment    : r=5 distilled student ladder (alpha=$ALPHA, T=$TEMP)"
  echo "launched      : $(date -u '+%Y-%m-%dT%H:%M:%SZ')  on $(hostname)"
  echo "pool          : $POOL"
  echo "teacher       : $TEACHER   (median 20M seed)"
  echo "teacher sha   : $(sha256sum "$TEACHER" | cut -c1-16)"
  echo "cache (train) : $CACHE_TRAIN   shots [0, $MAX_NTR)"
  echo "cache (tail)  : $CACHE_TAIL    shots [$VAL_START, $((VAL_START+VAL_N)))"
  echo "rungs         : $RUNGS"
  echo "archs         : $ARCHS   mlp hidden=($MLP_HIDDEN)  gru units=$GRU_UNITS"
  echo "seeds         : $SEEDS"
  echo "lr            : $LR"
  echo "validation    : [$VAL_START, $((VAL_START+VAL_N)))   sealed test not read"
  echo "code sha      : $(sha256sum train_student.py | cut -c1-16)  train_student.py"
  echo "              : $(sha256sum StudentModels.py | cut -c1-16)  StudentModels.py"
} | tee "$OUT/MANIFEST.txt"
echo

# ---------------------------------------------------------------------------------------
# Stage 1: teacher cache. Skipped if it already exists -- the teacher is frozen, so the
# cache is a pure function of (checkpoint, pool, shot range) and never needs redoing.
# train_student.py re-verifies the fingerprints anyway, so a stale cache fails loudly
# rather than being silently trusted here.
# ---------------------------------------------------------------------------------------
for spec in "train:$CACHE_TRAIN:0:$MAX_NTR" "tail:$CACHE_TAIL:$VAL_START:$VAL_N"; do
  name="${spec%%:*}"; rest="${spec#*:}"
  path="${rest%%:*}"; rest="${rest#*:}"
  start="${rest%%:*}"; nshots="${rest#*:}"
  if [ -f "$path" ]; then
    echo "[cache] $name already exists, skipping: $path"
  else
    echo "[cache] dumping $name: shots [$start, $((start+nshots))) -> $path"
    $PY dump_teacher_probs.py \
      --weights "$TEACHER" --d 5 --p 0.010 --rounds 5 \
      --pool "$POOL" --n-start "$start" --n-shots "$nshots" \
      --batch-size 10000 --out "$path"
  fi
done
echo

# ---------------------------------------------------------------------------------------
# Stage 2: the ladder.
# ---------------------------------------------------------------------------------------
n=0; for a in $ARCHS; do for r in $RUNGS; do for s in $SEEDS; do n=$((n+1)); done; done; done
echo "distilled ladder: $n runs -> $RUNS"
echo

i=0
for arch in $ARCHS; do
  if [ "$arch" = "gru" ]; then SIZE="--hidden --units $GRU_UNITS"; else SIZE="--hidden $MLP_HIDDEN"; fi
  for ntr in $RUNGS; do
    for s in $SEEDS; do
      i=$((i+1))
      # alpha in the tag, so the distilled runs can never overwrite the hard-label ones
      # and both ladders can share a collation glob.
      tag="r5dist_${arch}_ntr${ntr}_seed${s}_a${ALPHA}_T${TEMP}_lr${LR}${TAGSUF:-}"
      echo "=============== [$i/$n] $tag ==============="
      $PY train_student.py --student "$arch" --inputs evts $SIZE \
        --d 5 --p 0.010 --rounds 5 \
        --alpha "$ALPHA" --temperature "$TEMP" --lr "$LR" \
        --teacher-cache "$CACHE_TRAIN" --teacher-tail-cache "$CACHE_TAIL" \
        --seed "$s" --n-train "$ntr" --n-test "$VAL_N" --eval-start "$VAL_START" \
        --val-start "$VAL_START" --val-n "$VAL_N" \
        --epochs 50 --batch-size 10000 --no-early-stopping \
        --pool "$POOL" --out-dir "$RUNS" --tag "$tag" 2>&1 \
        | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique|^Epoch |- loss:"
      echo
    done
  done
done

# ---------------------------------------------------------------------------------------
# Collate. Reads only this run's folder, so a concurrent ladder cannot leak into it.
# ---------------------------------------------------------------------------------------
echo "=============================================================="
$PY - "$RUNS" "$OUT/summary.csv" <<'PYEOF'
import csv, glob, os, sys, statistics as st, collections
runs, out = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(runs, 'r5dist_*.csv'))):
    if '.superseded_' in os.path.basename(f):
        continue
    rows += list(csv.DictReader(open(f)))
if rows:
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

# MWPM on the FORMAL pool's validation partition [20.0M, 20.2M). Not from any log and not
# from mwpm_baseline.csv: that file describes a different tail and is not portable.
MW = 0.084215
# Hard-label means from the completed alpha=1.0 ladder, for the side-by-side that is the
# whole point of running this. Distillation is a claim about the DIFFERENCE.
HARD = {('gru', 100000): 0.25080, ('gru', 300000): 0.20930, ('gru', 800000): 0.15583,
        ('gru', 2000000): 0.11000, ('gru', 5000000): 0.08846, ('gru', 10000000): 0.08145,
        ('gru', 20000000): 0.07854,
        ('mlp', 100000): 0.32611, ('mlp', 300000): 0.24391, ('mlp', 800000): 0.17218,
        ('mlp', 2000000): 0.12939, ('mlp', 5000000): 0.10623, ('mlp', 10000000): 0.09756,
        ('mlp', 20000000): 0.09295}

g = collections.defaultdict(list)
for r in rows:
    g[(r['student'], int(r['n_train']))].append(float(r['p_L']))
print(f"  {'arch':>5} {'n_train':>11} {'seeds':>6} {'dist p_L':>9} {'std':>8} "
      f"{'xMWPM':>7} {'hard p_L':>9} {'delta':>9}")
for k in sorted(g, key=lambda k: (k[0], k[1])):
    v = g[k]
    sd = st.stdev(v) if len(v) > 1 else 0.0
    m = st.mean(v)
    h = HARD.get(k)
    # delta < 0 means distillation helped at this rung.
    d = f"{m - h:+.5f}" if h else "".rjust(9)
    hs = f"{h:.5f}" if h else "".rjust(9)
    print(f"  {k[0]:>5} {k[1]:>11,} {len(v):>6} {m:>9.5f} {sd:>8.5f} "
          f"{m / MW:>7.3f} {hs:>9} {d:>9}")
print(f"\n  MWPM on validation = {MW}.  delta = distilled - hard; negative = distillation")
print(f"  helped. Expect the effect, if any, at the LOW rungs.  {len(rows)} runs -> {out}")
PYEOF

tar czf "$RT/transfer/student_r5_distill_${STAMP}.tgz" -C "$RT/results" "student_r5_distill_${STAMP}"
echo
echo "transfer -> $RT/transfer/student_r5_distill_${STAMP}.tgz"
