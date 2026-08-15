#!/usr/bin/env bash
# Created: 2026-08-06
# Last modified: 2026-08-06
#
# Rung-matched-teacher distillation ladder at d=5, r=5. Run on EAF.
#
# Why this exists
# eaf_run_student_ladder_r5_distill.sh supervised every rung with one teacher trained on
# 20M shots. So the distilled student at 800k beat RCNN@800k while being taught by a model
# that had seen 19.2M shots it never saw. Two effects are mixed there and that design
# cannot separate them:
#
#   denoising           soft targets express how ambiguous a syndrome was; the hard bit
#                       cannot. A property of the target form.
#   data amplification  the targets carry information from shots outside the student's
#                       training prefix. A property of the teacher's budget.
#
# Here each rung's teacher is the RCNN trained on that same rung's n_train. Teacher and
# student then see the same shots and the same information, so any surviving gain is denoising
# alone -- the claim "soft targets are intrinsically better than hard ones".
#
# Expected outcome if the 20M-teacher gains were mostly amplification: these deltas are
# much smaller than that run's -0.023 at 800k, possibly ~0. That would be a real result,
# not a failure -- it would say the low-rung win came from the teacher's extra data.
#
# GRU only. The MLP never crosses MWPM and does not inform the deployment decision; adding
# it would triple the cost to sharpen a curve nobody will quantize.
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
POOL="$RT/pools_r5/data_d5_p0.010_r5_FORMAL.npz"
PY="${PY:-$HOME/QuantumDecoderQKeras/.venv/bin/python}"
LADDER="$RT/out_r5_ladder"

VAL_START=20000000
VAL_N=200000
RUNGS="${RUNGS:-100000 300000 800000 2000000 5000000 10000000 20000000}"
SEEDS="${SEEDS:-0 1 2}"
GRU_UNITS="${GRU_UNITS:-140}"
ALPHA="${ALPHA:-0.0}"
TEMP="${TEMP:-1.0}"
LR="${LR:-0.003}"

# Each rung needs its own cache over [0, n_train), and they sum to ~650 MB across the
# ladder. Disk on the pod runs at ~91%, so by default a rung's training cache is removed
# once its three seeds are done. Caches are a pure function of (checkpoint, pool, range)
# and regenerate in minutes; results are never touched by this.
KEEP_CACHES="${KEEP_CACHES:-0}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$RT/results/student_r5_distill_matched_${STAMP}"
RUNS="$OUT/runs"
CACHE_DIR="$RT/teacher_r5_matched"
mkdir -p "$RUNS" "$CACHE_DIR" "$RT/transfer"

[ -f "$POOL" ] || { echo "missing pool $POOL"; exit 1; }

# ---------------------------------------------------------------------------------------
# Pick each rung's teacher: the median of that rung's three seeds on the validation
# partition, matching how the 20M supervisor was chosen. Median rather than best, so no
# rung inherits an outlier draw -- and so the per-rung choice rule is identical to the one
# the 20M run used, leaving the teacher's data budget as the only thing that changed
# between the two experiments.
# ---------------------------------------------------------------------------------------
TEACHER_MAP="$OUT/teacher_map.txt"
$PY - "$RT" "$RUNGS" > "$TEACHER_MAP" <<'PYEOF'
import csv, glob, os, sys, statistics as st, collections
rt, rungs = sys.argv[1], [int(x) for x in sys.argv[2].split()]

# Read the collated ladder csv rather than out_r5_ladder/val_scores_best_ckpt.csv: the
# collated file carries n_train and seed as real columns and covers the 10M rung, whose
# runs live in out_r5_teacher/ and whose checkpoint names carry no ntr to parse.
# Later files win on a repeated (n_train, seed), same rule as the notebook loader.
by_run = {}
for f in sorted(glob.glob(os.path.join(rt, 'rcnn_r5_ladder_*.csv'))):
    for r in csv.DictReader(open(f)):
        by_run[(int(r['n_train']), int(r['seed']))] = float(r['p_L'])
by = collections.defaultdict(dict)
for (n, seed), pl in by_run.items():
    by[n][seed] = pl

# Two directory layouts hold these checkpoints: the ladder rungs under
# out_r5_ladder/ntr<N>/seed<S>/, and the 10M rung under out_r5_teacher/seed<S>/. The
# filename convention is the same in both, so try each and emit the one that exists.
def find_ckpt(n, seed):
    for cand in (os.path.join(rt, 'out_r5_ladder', f'ntr{n}', f'seed{seed}'),
                 os.path.join(rt, 'out_r5_teacher', f'seed{seed}')):
        path = os.path.join(cand, f'rcnn_d5_p0.010_r5_seed{seed}_ntr{n}.weights.h5')
        if os.path.exists(path):
            return path
    return ''

for n in rungs:
    seeds = by.get(n)
    if not seeds or len(seeds) < 3:
        print(f"skipping rung {n}: have {len(seeds) if seeds else 0} seeds, need 3",
              file=sys.stderr)
        continue
    med = st.median(seeds.values())
    seed = min(seeds, key=lambda s: abs(seeds[s] - med))
    ckpt = find_ckpt(n, seed)
    if not ckpt:
        print(f"skipping rung {n}: no checkpoint for seed {seed}", file=sys.stderr)
        continue
    print(f"{n} {seed} {seeds[seed]:.6f} {ckpt}")
PYEOF
echo "rung-matched teachers (rung / seed / p_L / checkpoint):"
sed 's/^/  /' "$TEACHER_MAP"
echo

{
  echo "experiment    : r=5 rung-matched-teacher distillation ladder (alpha=$ALPHA, T=$TEMP)"
  echo "purpose       : separate denoising from data amplification in Experiment 9"
  echo "launched      : $(date -u '+%Y-%m-%dT%H:%M:%SZ')  on $(hostname)"
  echo "pool          : $POOL"
  echo "arch          : gru units=$GRU_UNITS   seeds: $SEEDS   lr: $LR"
  echo "validation    : [$VAL_START, $((VAL_START+VAL_N)))   sealed test not read"
  echo "teachers      : median seed of each rung, from $RT/rcnn_r5_ladder_*.csv"
  sed 's/^/                rung /' "$TEACHER_MAP"
  echo "code sha      : $(sha256sum train_student.py | cut -c1-16)  train_student.py"
} | tee "$OUT/MANIFEST.txt"
echo

n=0; for r in $RUNGS; do for s in $SEEDS; do n=$((n+1)); done; done
echo "rung-matched ladder: $n runs -> $RUNS"
echo

i=0
while read -r ntr tseed tpl TEACHER; do
  [ -n "${ntr:-}" ] || continue
  if [ ! -f "$TEACHER" ]; then
    echo "!! missing teacher for rung $ntr: $TEACHER -- skipping rung"
    continue
  fi
  CACHE_TRAIN="$CACHE_DIR/teacher_r5_ntr${ntr}_seed${tseed}_train.npz"
  CACHE_VAL="$CACHE_DIR/teacher_r5_ntr${ntr}_seed${tseed}_val200k.npz"

  # Both caches are per-rung: the supervisor differs by rung, so its output on the
  # validation block differs too and cannot be shared with another rung.
  for spec in "train:$CACHE_TRAIN:0:$ntr" "val:$CACHE_VAL:$VAL_START:$VAL_N"; do
    name="${spec%%:*}"; rest="${spec#*:}"
    path="${rest%%:*}"; rest="${rest#*:}"
    start="${rest%%:*}"; nshots="${rest#*:}"
    if [ -f "$path" ]; then
      echo "[cache] rung $ntr $name exists, skipping"
    else
      echo "[cache] rung $ntr $name: shots [$start, $((start+nshots))) (teacher seed $tseed, p_L=$tpl)"
      $PY dump_teacher_probs.py \
        --weights "$TEACHER" --d 5 --p 0.010 --rounds 5 \
        --pool "$POOL" --n-start "$start" --n-shots "$nshots" \
        --batch-size 10000 --out "$path"
    fi
  done

  for s in $SEEDS; do
    i=$((i+1))
    tag="r5distm_gru_ntr${ntr}_seed${s}_a${ALPHA}_T${TEMP}_lr${LR}${TAGSUF:-}"
    echo "=============== [$i/$n] $tag   (teacher: rung $ntr seed $tseed) ==============="
    $PY train_student.py --student gru --inputs evts --hidden --units "$GRU_UNITS" \
      --d 5 --p 0.010 --rounds 5 \
      --alpha "$ALPHA" --temperature "$TEMP" --lr "$LR" \
      --teacher-cache "$CACHE_TRAIN" --teacher-tail-cache "$CACHE_VAL" \
      --val-teacher-cache "$CACHE_VAL" \
      --seed "$s" --n-train "$ntr" --n-test "$VAL_N" --eval-start "$VAL_START" \
      --val-start "$VAL_START" --val-n "$VAL_N" \
      --epochs 50 --batch-size 10000 --no-early-stopping \
      --pool "$POOL" --out-dir "$RUNS" --tag "$tag" 2>&1 \
      | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique|^Epoch |- loss:"
    echo
  done

  if [ "$KEEP_CACHES" = "0" ]; then
    rm -f "$CACHE_TRAIN"
    echo "[cache] removed rung $ntr training cache (KEEP_CACHES=1 to retain)"
  fi
done < "$TEACHER_MAP"

echo "=============================================================="
$PY - "$RUNS" "$OUT/summary.csv" "$TEACHER_MAP" <<'PYEOF'
import csv, glob, os, sys, statistics as st, collections
runs, out, tmap = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for f in sorted(glob.glob(os.path.join(runs, 'r5distm_*.csv'))):
    rows += list(csv.DictReader(open(f)))
if rows:
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

MW = 0.084215
# The two arms this run exists to sit between: hard labels, and distillation from the
# single 20M teacher. If matched-teacher lands near HARD, the Experiment 9 low-rung gain
# was mostly the teacher's extra data. If it lands near BIG, it was the target form.
HARD = {100000: 0.25080, 300000: 0.20930, 800000: 0.15583, 2000000: 0.11000,
        5000000: 0.08846, 10000000: 0.08145, 20000000: 0.07854}
BIG20M = {100000: 0.23822, 300000: 0.18817, 800000: 0.13304, 2000000: 0.10199,
          5000000: 0.08798, 10000000: 0.08382, 20000000: 0.08176}
teach = {}
for line in open(tmap):
    parts = line.split()
    if len(parts) == 3:
        teach[int(parts[0])] = float(parts[2])

g = collections.defaultdict(list)
for r in rows:
    g[int(r['n_train'])].append(float(r['p_L']))
print(f"  {'n_train':>11} {'seeds':>6} {'matched':>9} {'std':>8} {'hard':>9} "
      f"{'20M-teach':>10} {'vs hard':>9} {'teacher':>9} {'share':>7}")
for n in sorted(g):
    v = g[n]
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    h, b = HARD.get(n), BIG20M.get(n)
    dh = f"{m - h:+.5f}" if h else ""
    # What fraction of the 20M-teacher gain survives when the teacher's data budget is
    # taken away. Near 0 = that gain was data amplification; near 1 = it was denoising.
    share = ""
    if h and b and abs(b - h) > 1e-9:
        share = f"{(m - h) / (b - h):.2f}"
    print(f"  {n:>11,} {len(v):>6} {m:>9.5f} {sd:>8.5f} {h if h else 0:>9.5f} "
          f"{b if b else 0:>10.5f} {dh:>9} {teach.get(n, float('nan')):>9.5f} {share:>7}")
print(f"\n  MWPM = {MW}.  'share' = (matched - hard) / (20M-teacher - hard): the fraction of")
print(f"  Experiment 9's gain that survives when teacher and student see the same shots.")
print(f"  {len(rows)} runs -> {out}")
PYEOF

tar czf "$RT/transfer/student_r5_distill_matched_${STAMP}.tgz" -C "$RT/results" "student_r5_distill_matched_${STAMP}"
echo
echo "transfer -> $RT/transfer/student_r5_distill_matched_${STAMP}.tgz"
