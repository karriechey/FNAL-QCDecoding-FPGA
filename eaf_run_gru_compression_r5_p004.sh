#!/usr/bin/env bash
# Created: 2026-08-17
# Last modified: 2026-08-17
#
# GRU parameter-reduction and distillation study at d=5, r=5, p=0.004. Run on EAF.
#
# The question: how far can the GRU decoder be compressed before it stops matching MWPM,
# and does supervision from a full-size GRU teacher hold performance up as the parameter
# budget falls?
#
# Four GRU widths, all measured with count_params() on a built model rather than estimated
# (the count is quadratic in width, so ratios of widths are not ratios of parameters):
#
#     units=140   69,441 params   full size (matched to the r=5 RCNN's 69,485)
#     units=70    20,021 params   ~20k
#     units=46     9,845 params   ~10k
#     units=34     6,053 params   ~6k
#
# At each reduced width, two arms differing in exactly one variable -- where the training
# target comes from:
#     alpha=1.0   hard Stim logical-flip labels
#     alpha=0.0   pure distillation from the frozen full-size GRU teacher, T=1
#
# Partitions, batch size and schedule are inherited unchanged from the Experiment 13
# below-threshold protocol, so these runs sit on the same footing as the p=0.004 RCNN
# ladder. The sealed test block [17M, 19M) is never read by anything in this script.
#
# Stages:
#   1  train three full-size GRU teachers (seeds 0/1/2, hard labels)
#   2  select the teacher seed with the lowest best_val_loss on the validation block
#   3  dump that teacher's frozen output over the training prefix, the validation block,
#      and the evaluation block
#   4  train the 18 reduced students (3 widths x 2 arms x 3 seeds)
#   5  collate
#
# All output lands in ONE timestamped folder: driver.log, MANIFEST.txt, runs/, teacher/.
set -euo pipefail

# ---------------------------------------------------------------------------------------
# Determinism. Exported BEFORE any python starts, because TensorFlow selects deterministic
# kernels at import time; setting them later would be silently ineffective. Every training
# process below runs with --require-determinism, which hard-fails if these are missing, so
# a launcher that forgot them cannot quietly produce nondeterministic runs.
# ---------------------------------------------------------------------------------------
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1

RT="${RT:-$HOME/rcnn_threshold}"
REPO="${REPO:-$HOME/QuantumDecoderQKeras}"
PY="${PY:-$REPO/.venv/bin/python}"
POOL="${POOL:-$RT/pools_d5_p004_19M/data_d5_p0.004_r5_FORMAL.npz}"

# --- Experiment 13 partitions, d=5 r=5 p=0.004, 19M pool -------------------------------
# training   [0, 10M)
# validation [15.0M, 15.2M)   early stopping, checkpoint selection, teacher selection
# evaluation [15.2M, 17.0M)   scored once, from the best checkpoint
# test       [17.0M, 19.0M)   SEALED -- not read here
N_TRAIN="${N_TRAIN:-10000000}"
VAL_START="${VAL_START:-15000000}"
VAL_N="${VAL_N:-200000}"
EVAL_START="${EVAL_START:-15200000}"
EVAL_N="${EVAL_N:-1800000}"
# Overridable only so a local smoke test can scale the whole layout down onto a small
# pool. On the formal 19M pool it is 17,000,000 and is not passed.
SEALED_START="${SEALED_START:-17000000}"

# MWPM on the exact evaluation block, measured 2026-08-14 (exp13 protocol, table of
# record). Carried here for the console summary only; the collator recomputes ratios from
# this same constant so the two cannot disagree.
MWPM="${MWPM:-0.007572}"

# --- study grid -------------------------------------------------------------------------
TEACHER_UNITS="${TEACHER_UNITS:-140}"
STUDENT_UNITS="${STUDENT_UNITS:-70 46 34}"
SEEDS="${SEEDS:-0 1 2}"
TEACHER_SEEDS="${TEACHER_SEEDS:-0 1 2}"
MODES="${MODES:-hard distill}"

# --- recipe, identical across every run here --------------------------------------------
BATCH="${BATCH:-5000}"
LR="${LR:-0.003}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-5}"
TEMP="${TEMP:-1.0}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="${OUT:-$RT/results/gru_compression_r5_p004_${STAMP}}"
RUNS="$OUT/runs"
TEACHER_DIR="$OUT/teacher"
CKPT_DIR="$OUT/ckpt"
LOG="$OUT/driver.log"

mkdir -p "$RUNS" "$TEACHER_DIR" "$CKPT_DIR" "$RT/transfer"

# Everything from here on goes to the console AND to driver.log, so a run that is watched
# live and a run that is read from a tarball later see the same record. Done with exec
# rather than piping the whole script, so `set -e` still aborts on the failing command
# rather than on the pipeline's last one.
exec > >(tee -a "$LOG") 2>&1

# ---------------------------------------------------------------------------------------
# Preflight. Everything that would waste GPU hours if wrong is checked before the first
# run, not discovered at hour three.
# ---------------------------------------------------------------------------------------
[ -f "$POOL" ] || { echo "MISSING pool $POOL"; exit 1; }
[ -x "$PY" ]   || { echo "MISSING python $PY"; exit 1; }

"$PY" - "$POOL" "$N_TRAIN" "$VAL_START" "$VAL_N" "$EVAL_START" "$EVAL_N" "$SEALED_START" <<'PYEOF'
"""Verify the pool holds the partition layout, and that no block overlaps another."""
import sys, numpy as np
pool, ntr, v0, vn, e0, en, sealed = sys.argv[1], *map(int, sys.argv[2:])
z = np.load(pool, mmap_mode='r')
N = z['det_evts'].shape[0]
v1, e1 = v0 + vn, e0 + en
print(f"[preflight] pool {pool}")
print(f"[preflight] shots={N:,}  det_evts{z['det_evts'].shape}")
print(f"[preflight] train [0, {ntr:,})  val [{v0:,}, {v1:,})  eval [{e0:,}, {e1:,})  "
      f"sealed [{sealed:,}, {N:,})")
assert N >= sealed, f"pool holds {N:,} shots, sealed block starts at {sealed:,}"
assert ntr <= v0, f"training prefix [0, {ntr:,}) overlaps validation at {v0:,}"
assert v1 <= e0, f"validation [{v0:,}, {v1:,}) overlaps evaluation at {e0:,}"
assert e1 <= sealed, f"evaluation [{e0:,}, {e1:,}) runs into the sealed test at {sealed:,}"
print("[preflight] partitions disjoint; sealed test not touched by this layout")
PYEOF

# Confirm the four widths land where the study says they do. Printed, not assumed: if
# StudentModels or the detector layout ever changes, this catches it before 21 runs.
"$PY" - "$TEACHER_UNITS" $STUDENT_UNITS <<'PYEOF'
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from StudentModels import build_gru_student
print("[preflight] measured GRU capacities (d=5, r=5, evts):")
for u in map(int, sys.argv[1:]):
    m = build_gru_student(d=5, rounds=5, inputs='evts', units=u, hidden=(), p=0.004)
    print(f"[preflight]   units={u:4d}  params={m.count_params():,}")
PYEOF

# ---------------------------------------------------------------------------------------
# MANIFEST: written before any compute, so a run that dies still leaves a record of what
# it was trying to do and against which code.
# ---------------------------------------------------------------------------------------
{
  echo "experiment     : GRU parameter reduction + distillation, d=5 r=5 p=0.004"
  echo "launched       : $(date -u '+%Y-%m-%dT%H:%M:%SZ')  on $(hostname)"
  echo "pool           : $POOL"
  echo "training       : [0, $N_TRAIN)"
  echo "validation     : [$VAL_START, $((VAL_START + VAL_N)))"
  echo "evaluation     : [$EVAL_START, $((EVAL_START + EVAL_N)))"
  echo "sealed test    : [$SEALED_START, end of pool)   NOT READ"
  echo "mwpm (eval blk): $MWPM"
  echo "teacher units  : $TEACHER_UNITS   seeds $TEACHER_SEEDS"
  echo "student units  : $STUDENT_UNITS   seeds $SEEDS   modes $MODES"
  echo "recipe         : batch=$BATCH lr=$LR epochs=$EPOCHS patience=$PATIENCE T=$TEMP"
  echo "determinism    : TF_DETERMINISTIC_OPS=$TF_DETERMINISTIC_OPS "\
       "TF_CUDNN_DETERMINISTIC=$TF_CUDNN_DETERMINISTIC"
  echo "eval-block use : scored ONCE after training, from the val_select-best checkpoint."
  echo "                 The evaluation teacher cache is diagnostics-only: it is read"
  echo "                 after model.fit returns and feeds the student-teacher agreement"
  echo "                 columns alone. No loss, callback, checkpoint monitor, early-stop"
  echo "                 criterion or teacher selection reads the evaluation block."
  echo "                 --tail-diagnostics is not passed (guarded at launch)."
  echo "code sha       : $(sha256sum train_student.py | cut -c1-16)  train_student.py"
  echo "               : $(sha256sum StudentModels.py | cut -c1-16)  StudentModels.py"
  echo "               : $(sha256sum dump_teacher_probs.py | cut -c1-16)  dump_teacher_probs.py"
  echo "               : $(sha256sum "$0" | cut -c1-16)  $(basename "$0")"
} | tee "$OUT/MANIFEST.txt"
echo

# Common arguments. Every run in this study shares them, so a difference between two runs
# can only come from the grid variables spelled out at the call site.
#
# --tail-diagnostics is deliberately absent and must stay absent. It scores the evaluation
# block once per epoch and writes the result into the epoch stream. That would not change
# what any callback selects -- EarlyStopping and ModelCheckpoint both monitor val_loss on
# [15.0M, 15.2M) -- but it would put evaluation-block numbers in front of a human while the
# run is still in flight, which is the same leak by a slower route. The guard below fails
# the launch rather than trusting the list to stay clean through future edits.
COMMON=(--student gru --inputs evts --hidden
        --d 5 --p 0.004 --rounds 5
        --pool "$POOL"
        --n-train "$N_TRAIN"
        --val-start "$VAL_START" --val-n "$VAL_N"
        --eval-start "$EVAL_START" --n-test "$EVAL_N"
        --batch-size "$BATCH" --lr "$LR"
        --epochs "$EPOCHS" --patience "$PATIENCE"
        --require-determinism
        --ckpt-dir "$CKPT_DIR")

case " ${COMMON[*]} " in
  *" --tail-diagnostics "*)
    echo "ABORT: --tail-diagnostics is in the shared arguments. The evaluation block is"
    echo "       scored once, after training, from the val_select-best checkpoint. Nothing"
    echo "       in this study may see it per-epoch."
    exit 1 ;;
esac

# Filter of TensorFlow's per-epoch chatter. The driver log stays readable; the per-run CSV
# and history.json keep everything that matters.
QUIET='cuda_|Unable to register|^Total number|^Number of unique|^Epoch |- loss:'

# ---------------------------------------------------------------------------------------
# Stage 1 -- full-size GRU teachers, hard labels, three seeds.
#
# These are not only the teacher candidates: all three are the full-size reference point on
# the final plot, reported as a 3-seed mean +/- std exactly like every student point. Only
# the distillation source is a single seed.
# ---------------------------------------------------------------------------------------
echo "=== stage 1: full-size GRU teachers (units=$TEACHER_UNITS, hard labels) ==="
for s in $TEACHER_SEEDS; do
  tag="gruc_p004_teacher_u${TEACHER_UNITS}_hard_seed${s}_ntr${N_TRAIN}"
  if [ -f "$RUNS/$tag.csv" ]; then
    echo "--- $tag: result exists, skipping"
    continue
  fi
  echo "--- $tag"
  "$PY" train_student.py "${COMMON[@]}" \
    --units "$TEACHER_UNITS" --alpha 1.0 --temperature "$TEMP" \
    --seed "$s" --out-dir "$RUNS" --tag "$tag" --run-tag "$tag" 2>&1 | grep -Ev "$QUIET"
  echo
done

# ---------------------------------------------------------------------------------------
# Stage 2 -- pick the distillation teacher.
#
# Selected on best_val_loss over the validation block [15.0M, 15.2M), which is the same
# quantity the checkpoint callback minimises and is disjoint from the evaluation block, so
# nothing about this choice is informed by the numbers that get reported. The chosen seed
# is an upper draw of three by construction; that is recorded here so the distilled arm's
# inheritance from it stays visible.
# ---------------------------------------------------------------------------------------
echo "=== stage 2: teacher selection on best_val_loss ==="
BEST_SEED=$("$PY" - "$RUNS" "$TEACHER_UNITS" "$OUT/teacher_selection.csv" <<'PYEOF'
import csv, glob, os, sys
runs, units, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for f in sorted(glob.glob(os.path.join(runs, f'gruc_p004_teacher_u{units}_hard_seed*.csv'))):
    for r in csv.DictReader(open(f)):
        rows.append(r)
if not rows:
    sys.exit(f"[select] no teacher runs found in {runs}")
rows.sort(key=lambda r: float(r['best_val_loss']))
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['seed', 'best_val_loss', 'eval_p_L', 'selected'])
    w.writeheader()
    for i, r in enumerate(rows):
        w.writerow({'seed': r['seed'], 'best_val_loss': r['best_val_loss'],
                    'eval_p_L': r['p_L'], 'selected': int(i == 0)})
for i, r in enumerate(rows):
    mark = '  <-- teacher' if i == 0 else ''
    print(f"[select] seed {r['seed']}  best_val_loss={float(r['best_val_loss']):.6f}  "
          f"eval p_L={float(r['p_L']):.6f}{mark}", file=sys.stderr)
print(rows[0]['seed'])
PYEOF
)
echo "[select] distillation teacher = seed $BEST_SEED"

TEACHER_CKPT="$CKPT_DIR/gruc_p004_teacher_u${TEACHER_UNITS}_hard_seed${BEST_SEED}_ntr${N_TRAIN}.best.weights.h5"
[ -f "$TEACHER_CKPT" ] || { echo "MISSING teacher checkpoint $TEACHER_CKPT"; exit 1; }
echo "[select] checkpoint $TEACHER_CKPT"
echo "[select] sha $(sha256sum "$TEACHER_CKPT" | cut -c1-16)"
echo

# ---------------------------------------------------------------------------------------
# Stage 3 -- freeze the teacher's output over every block a distilled run touches.
#
# THREE dumps, not two. The distillation loss packs its target as [hard_label,
# teacher_logit], and the validation loss is computed with that same loss -- so with
# alpha<1 the validation block needs teacher logits too, or early stopping and checkpoint
# selection would be run against a teacher that says p=0.5 on every validation shot.
# train_student.py refuses to start without --val-teacher-cache for exactly this reason.
#
#   [0, N_TRAIN)              --teacher-cache        the training target
#   [VAL_START, +VAL_N)       --val-teacher-cache    the validation target
#   [EVAL_START, +EVAL_N)     --teacher-tail-cache   diagnostics only: student-vs-teacher
#                                                    agreement in the ambiguous band
#
# The caches are absolute-indexed by shot_idx and the teacher is frozen, so one set of
# dumps serves all 9 distilled runs.
# ---------------------------------------------------------------------------------------
echo "=== stage 3: dump frozen teacher output (seed $BEST_SEED) ==="
CACHE_TRAIN="$TEACHER_DIR/teacher_gru_u${TEACHER_UNITS}_seed${BEST_SEED}_train0_${N_TRAIN}.npz"
CACHE_VAL="$TEACHER_DIR/teacher_gru_u${TEACHER_UNITS}_seed${BEST_SEED}_val${VAL_START}_${VAL_N}.npz"
CACHE_EVAL="$TEACHER_DIR/teacher_gru_u${TEACHER_UNITS}_seed${BEST_SEED}_eval${EVAL_START}_${EVAL_N}.npz"

dump_block () {  # $1 = n-start, $2 = n-shots, $3 = out path, $4 = human label
  if [ -f "$3" ]; then echo "--- $4 cache exists, skipping"; return; fi
  echo "--- dumping $4: shots [$1, $(($1 + $2)))"
  "$PY" dump_teacher_probs.py \
    --teacher-arch gru --units "$TEACHER_UNITS" \
    --weights "$TEACHER_CKPT" \
    --d 5 --p 0.004 --rounds 5 \
    --pool "$POOL" --n-start "$1" --n-shots "$2" \
    --batch-size "$BATCH" --out "$3" 2>&1 | grep -Ev "$QUIET"
}
dump_block 0 "$N_TRAIN" "$CACHE_TRAIN" "training prefix"
dump_block "$VAL_START" "$VAL_N" "$CACHE_VAL" "validation block"
dump_block "$EVAL_START" "$EVAL_N" "$CACHE_EVAL" "evaluation block"
echo

# ---------------------------------------------------------------------------------------
# Stage 4 -- the reduced students. 3 widths x 2 arms x 3 seeds = 18 runs.
#
# The hard and distilled arms share every argument except --alpha and the three cache
# paths, so the comparison at each width isolates the supervision signal.
# ---------------------------------------------------------------------------------------
echo "=== stage 4: reduced GRU students ==="
total=0; for u in $STUDENT_UNITS; do for m in $MODES; do for s in $SEEDS; do
  total=$((total + 1)); done; done; done
i=0
for u in $STUDENT_UNITS; do
  for mode in $MODES; do
    for s in $SEEDS; do
      i=$((i + 1))
      tag="gruc_p004_student_u${u}_${mode}_seed${s}_ntr${N_TRAIN}"
      if [ -f "$RUNS/$tag.csv" ]; then
        echo "--- [$i/$total] $tag: result exists, skipping"
        continue
      fi
      echo "=============== [$i/$total] $tag ==============="
      if [ "$mode" = "hard" ]; then
        ARM=(--alpha 1.0)
      else
        ARM=(--alpha 0.0
             --teacher-cache "$CACHE_TRAIN"
             --val-teacher-cache "$CACHE_VAL"
             --teacher-tail-cache "$CACHE_EVAL")
      fi
      "$PY" train_student.py "${COMMON[@]}" \
        --units "$u" --temperature "$TEMP" "${ARM[@]}" \
        --seed "$s" --out-dir "$RUNS" --tag "$tag" --run-tag "$tag" 2>&1 | grep -Ev "$QUIET"
      echo
    done
  done
done

# ---------------------------------------------------------------------------------------
# Stage 5 -- collate.
# ---------------------------------------------------------------------------------------
echo "=== stage 5: collate ==="
"$PY" collate_gru_compression_p004.py --runs "$RUNS" --out "$OUT" --mwpm "$MWPM"

# --exclude before the paths: BSD tar (a local smoke run on macOS) treats a trailing
# --exclude as a filename, and errors out. GNU tar on the pod accepts either order.
tar -czf "$RT/transfer/gru_compression_r5_p004_${STAMP}.tgz" \
    --exclude='*.weights.h5' --exclude='*.npz' \
    -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
echo "done. results -> $OUT"
echo "      tarball -> $RT/transfer/gru_compression_r5_p004_${STAMP}.tgz"
echo "      (checkpoints and teacher caches excluded from the tarball: too large, and"
echo "       reproducible from the pool plus the checkpoints that stay on the pod)"
