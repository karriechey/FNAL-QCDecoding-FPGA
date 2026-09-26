#!/usr/bin/env bash
# Created: 2026-09-22
# Last modified: 2026-09-22
#
# Distil the 20M-shot u140 GRU (Run 11, 0.947x MWPM at d=5 r=5 p=0.004) into smaller GRUs
# for hls4ml / FPGA. FP32 only; quantization is a separate, later experiment. Run on EAF.
#
# Two stages, run separately so the teacher gate is read by a person before any student
# trains:
#
#   STAGE=prep   (default)
#       1. preflight: resolve and hash both pools and the teacher checkpoint, check the
#          partitions, print measured parameter counts
#       2. dump the frozen teacher over four blocks:
#            primary  [0, 15M)             training, primary part
#            extension [0, 5M)             training, extension part
#            primary  [15.0M, 15.2M)       validation
#            primary  [15.2M, 17.0M)       evaluation (diagnostics + teacher gate only)
#       3. verify all four caches and run the teacher gate: teacher p_L vs MWPM p_L on the
#          evaluation block. Writes PREP_OK only if the teacher beats MWPM there.
#       Then stops.
#
#   STAGE=train OUT=<the prep folder>
#       Requires PREP_OK. Trains the grid (4 widths x {hard, distill} x 3 seeds = 24 runs),
#       skipping any run whose folder already has COMPLETE, then collates and tars.
#       Safe to re-run after an interruption: finished runs are skipped and a half-finished
#       run's folder is renamed to <tag>.attempt_<UTC> (kept, never deleted) before retry.
#
# Recipe (Run 11, the teacher's own): Adam lr 3e-3 constant, batch 10,000, 200 epochs,
# early stopping off, min-val_loss checkpoint scored once on the evaluation block.
# Determinism variables exported below, before any python starts.
#
# Everything lands in ONE folder: $RT/results/gru_distill_20m_<UTC>/
#   MANIFEST.txt  driver.log  preflight_inputs.json  preflight_caches.json  PREP_OK
#   teacher/      four teacher caches
#   runs/<tag>/   one folder per run: train.log, <tag>.csv, history, checkpoints,
#                 per-shot eval npz, teacher_alignment.json, COMPLETE
#   summary_<UTC>.csv  collated rows (a new file per collate, never overwritten)
set -euo pipefail

export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1

STAGE="${STAGE:-prep}"

RT="${RT:-$HOME/rcnn_threshold}"
REPO="${REPO:-$HOME/QuantumDecoderQKeras}"
PY="${PY:-$REPO/.venv/bin/python}"
POOLDIR="${POOLDIR:-/scratch/fast/7DayLifetime/kchey/pools_d5_p004_19M}"
POOL="${POOL:-$POOLDIR/data_d5_p0.004_r5_FORMAL.npz}"
EXTRA_POOL="${EXTRA_POOL:-$POOLDIR/data_d5_p0.004_r5_TRAINONLY_seed43.npz}"

# Teacher: Run 11 seed 1, selected by lowest best_val_loss (0.019599) among the three
# seeds. Copied from grace1 to EAF by hand; TEACHER_SHA256 is the sha256sum taken on
# grace1 before the copy, so a damaged transfer stops the preflight.
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$RT/teacher_gru20m/gru_d5_p004_u140_ext5000000_hard_seed1_ntr15000000_b10000.best.weights.h5}"
# sha256sum taken on grace1, 2026-09-22.
TEACHER_SHA256="${TEACHER_SHA256:-4451f0573a52ffdb71e8e2152e4e4f6648f71fa670bcf9b6e7ff094f5f18d2a5}"
TEACHER_UNITS=140

# Pool of record on EAF (Experiment 13/14): gen_seed 42. Regenerating with the same seed on
# the same architecture must reproduce this SHA; anything else is a different pool.
EXPECT_MAIN_FLIPS_SHA="${EXPECT_MAIN_FLIPS_SHA:-8b45765edef2007426341bd0620033f6217ba4bd84d6ebb4d30bfcfddd559c66}"
EXPECT_EXTRA_FLIPS_SHA="${EXPECT_EXTRA_FLIPS_SHA:-}"

# Partitions (Experiment 13 protocol; extension adds 5M training-only shots).
MAIN_N_TRAIN="${MAIN_N_TRAIN:-15000000}"
EXTRA_N="${EXTRA_N:-5000000}"
VAL_START="${VAL_START:-15000000}"
VAL_N="${VAL_N:-200000}"
EVAL_START="${EVAL_START:-15200000}"
EVAL_N="${EVAL_N:-1800000}"
SEALED_START="${SEALED_START:-17000000}"

# Grid.
STUDENT_UNITS="${STUDENT_UNITS:-100 70 46 34}"
MODES="${MODES:-hard distill}"
SEEDS="${SEEDS:-0 1 2}"
TEMP="${TEMP:-1.0}"

# Recipe.
BATCH="${BATCH:-10000}"
LR="${LR:-0.003}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-0}"      # 0 = early stopping off

# Concurrency. 1 = one run at a time. Each 20M run holds ~12 GB of float32 inputs plus
# TensorFlow's copy, so check the pod's memory limit (printed below) before raising this.
MAX_JOBS="${MAX_JOBS:-1}"
GPU_MEM_MIB="${GPU_MEM_MIB:-}"

# Smoke tests only: record a teacher-gate failure instead of stopping on it.
ALLOW_GATE_FAIL="${ALLOW_GATE_FAIL:-0}"

sha () { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi; }
utc () { date -u +%Y%m%dT%H%M%SZ; }

if [ "$STAGE" = "prep" ]; then
  OUT="${OUT:-$RT/results/gru_distill_20m_$(utc)}"
elif [ "$STAGE" = "train" ]; then
  [ -n "${OUT:-}" ] || { echo "STAGE=train needs OUT=<the prep folder>. STOP."; exit 1; }
  [ -f "$OUT/PREP_OK" ] || { echo "no $OUT/PREP_OK: run STAGE=prep first and read the teacher gate. STOP."; exit 1; }
else
  echo "STAGE=$STAGE; expected prep or train. STOP."; exit 1
fi
TEACHER_DIR="$OUT/teacher"
RUNS="$OUT/runs"
mkdir -p "$OUT" "$TEACHER_DIR" "$RUNS" "$RT/transfer"
exec > >(tee -a "$OUT/driver.log") 2>&1

cd "$REPO"
[ -x "$PY" ] || { echo "MISSING python $PY. STOP."; exit 1; }

CACHE_MAIN="$TEACHER_DIR/teacher_u${TEACHER_UNITS}_main_0_${MAIN_N_TRAIN}.npz"
CACHE_EXTRA="$TEACHER_DIR/teacher_u${TEACHER_UNITS}_extra_0_${EXTRA_N}.npz"
CACHE_VAL="$TEACHER_DIR/teacher_u${TEACHER_UNITS}_val_${VAL_START}_${VAL_N}.npz"
CACHE_EVAL="$TEACHER_DIR/teacher_u${TEACHER_UNITS}_eval_${EVAL_START}_${EVAL_N}.npz"

COMMON_PART=(--main-pool "$POOL" --extra-pool "$EXTRA_POOL"
             --main-n-train "$MAIN_N_TRAIN" --extra-n "$EXTRA_N"
             --val-start "$VAL_START" --val-n "$VAL_N"
             --eval-start "$EVAL_START" --eval-n "$EVAL_N" --sealed-start "$SEALED_START")

echo "=============== gru_distill_20m  STAGE=$STAGE  $(utc) on $(hostname) ==============="
echo "[mem] pod memory limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo unknown) bytes"
free -g 2>/dev/null | head -2 || true
nvidia-smi -L 2>/dev/null || echo "[gpu] nvidia-smi unavailable"

# ======================================================================================
if [ "$STAGE" = "prep" ]; then
  {
    echo "experiment     : GRU distillation from the 20M u140 teacher, d=5 r=5 p=0.004, FP32"
    echo "launched       : $(date -u '+%Y-%m-%dT%H:%M:%SZ')  on $(hostname)"
    echo "primary pool   : $POOL"
    echo "extension pool : $EXTRA_POOL"
    echo "teacher        : $TEACHER_WEIGHTS"
    echo "teacher sha    : $(sha "$TEACHER_WEIGHTS" | cut -c1-64)"
    echo "training       : primary [0, $MAIN_N_TRAIN) + extension [0, $EXTRA_N)"
    echo "validation     : [$VAL_START, $((VAL_START + VAL_N)))  checkpoint selection"
    echo "evaluation     : [$EVAL_START, $((EVAL_START + EVAL_N)))  scored once per run"
    echo "sealed test    : [$SEALED_START, end)  NOT READ"
    echo "grid           : units {$STUDENT_UNITS} x modes {$MODES} x seeds {$SEEDS}"
    echo "distill arm    : alpha=0.0 T=$TEMP   hard arm: alpha=1.0"
    echo "recipe         : Adam lr=$LR constant, batch=$BATCH, epochs=$EPOCHS, patience=$PATIENCE (0=off)"
    echo "selection      : min val_loss checkpoint; evaluation block read only after training"
    echo "determinism    : TF_DETERMINISTIC_OPS=$TF_DETERMINISTIC_OPS TF_CUDNN_DETERMINISTIC=$TF_CUDNN_DETERMINISTIC"
    echo "allow gate fail: $ALLOW_GATE_FAIL  (must be 0 for the real study)"
    for f in train_gru_distillation_20m.py gru_distill_20m_preflight.py dump_teacher_probs.py \
             train_student.py StudentModels.py eval_on_tail.py "$0"; do
      echo "code sha       : $(sha "$f" | cut -c1-16)  $(basename "$f")"
    done
  } | tee "$OUT/MANIFEST.txt"
  echo

  echo "=== prep 1/3: inputs, hashes, partitions ==="
  "$PY" gru_distill_20m_preflight.py --stage inputs "${COMMON_PART[@]}" \
    --teacher-weights "$TEACHER_WEIGHTS" --teacher-units "$TEACHER_UNITS" \
    --teacher-sha256 "$TEACHER_SHA256" --student-units $STUDENT_UNITS \
    --expect-main-flips-sha "$EXPECT_MAIN_FLIPS_SHA" \
    --expect-extra-flips-sha "$EXPECT_EXTRA_FLIPS_SHA" \
    --out-json "$OUT/preflight_inputs.json"
  echo

  echo "=== prep 2/3: dump the frozen teacher over four blocks ==="
  dump () {  # $1 pool  $2 n-start  $3 n-shots  $4 out  $5 label
    if [ -f "$4" ]; then echo "--- $5: cache exists, keeping it"; return; fi
    echo "--- $5: shots [$2, $(($2 + $3))) of $(basename "$1")"
    "$PY" dump_teacher_probs.py --teacher-arch gru --units "$TEACHER_UNITS" \
      --weights "$TEACHER_WEIGHTS" --d 5 --p 0.004 --rounds 5 \
      --pool "$1" --n-start "$2" --n-shots "$3" --batch-size "$BATCH" --out "$4"
  }
  dump "$POOL"       0              "$MAIN_N_TRAIN" "$CACHE_MAIN"  "primary training"
  dump "$EXTRA_POOL" 0              "$EXTRA_N"      "$CACHE_EXTRA" "extension training"
  dump "$POOL"       "$VAL_START"   "$VAL_N"        "$CACHE_VAL"   "validation"
  dump "$POOL"       "$EVAL_START"  "$EVAL_N"       "$CACHE_EVAL"  "evaluation"
  echo

  echo "=== prep 3/3: verify caches, teacher gate ==="
  set +e
  "$PY" gru_distill_20m_preflight.py --stage caches "${COMMON_PART[@]}" \
    --teacher-weights "$TEACHER_WEIGHTS" --teacher-units "$TEACHER_UNITS" \
    --cache-main "$CACHE_MAIN" --cache-extra "$CACHE_EXTRA" \
    --cache-val "$CACHE_VAL" --cache-eval "$CACHE_EVAL" \
    --out-json "$OUT/preflight_caches.json"
  rc=$?
  set -e
  if [ $rc -eq 0 ] || { [ $rc -eq 3 ] && [ "$ALLOW_GATE_FAIL" = "1" ]; }; then
    [ $rc -eq 3 ] && echo "[gate] teacher did not beat MWPM; continuing ONLY because ALLOW_GATE_FAIL=1 (smoke)."
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$OUT/PREP_OK"
    echo
    echo "prep done. Read the TEACHER GATE block above, then train with:"
    echo "  STAGE=train OUT=$OUT bash $(basename "$0")"
    exit 0
  fi
  echo "prep stopped (exit $rc). No PREP_OK written; students will not train."
  exit $rc
fi

# ======================================================================================
# STAGE=train
TEACHER_ARGS=(--teacher-main-cache "$CACHE_MAIN" --teacher-extra-cache "$CACHE_EXTRA"
              --teacher-val-cache "$CACHE_VAL" --teacher-tail-cache "$CACHE_EVAL"
              --teacher-units "$TEACHER_UNITS" --teacher-weights "$TEACHER_WEIGHTS")
[ -n "$TEACHER_SHA256" ] && TEACHER_ARGS+=(--teacher-sha256 "$TEACHER_SHA256")
EXTRA_ARGS=()
[ -n "$GPU_MEM_MIB" ] && EXTRA_ARGS+=(--gpu-mem-mib "$GPU_MEM_MIB")

run_one () {  # $1 units  $2 mode  $3 seed
  local u="$1" mode="$2" s="$3" alpha
  local tag="gkd20m_u${u}_${mode}_seed${s}"
  local dir="$RUNS/$tag"
  if [ -f "$dir/COMPLETE" ]; then echo "--- $tag: COMPLETE, skipping"; return 0; fi
  if [ -d "$dir" ]; then
    local moved="$dir.attempt_$(utc)"
    mv "$dir" "$moved"
    echo "--- $tag: incomplete earlier attempt kept at $(basename "$moved")"
  fi
  mkdir -p "$dir"
  if [ "$mode" = "hard" ]; then alpha=1.0; else alpha=0.0; fi
  echo "--- $tag: start $(utc)  log $dir/train.log"
  # Both arms receive the same teacher caches; at alpha=1 the teacher column is
  # multiplied by zero in the loss, so the two arms differ only in alpha.
  if "$PY" train_gru_distillation_20m.py \
       --main-pool "$POOL" --main-n-train "$MAIN_N_TRAIN" \
       --extra-train-pool "$EXTRA_POOL" --extra-n "$EXTRA_N" \
       --val-start "$VAL_START" --val-n "$VAL_N" \
       --eval-start "$EVAL_START" --eval-n "$EVAL_N" --sealed-start "$SEALED_START" \
       "${TEACHER_ARGS[@]}" \
       --units "$u" --alpha "$alpha" --temperature "$TEMP" --seed "$s" \
       --batch-size "$BATCH" --lr "$LR" --epochs "$EPOCHS" --patience "$PATIENCE" \
       ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
       --out-dir "$dir" --tag "$tag" > "$dir/train.log" 2>&1; then
    echo "--- $tag: done $(utc)  $(grep -h 'EVAL \[' "$dir/train.log" | tail -1)"
  else
    echo "--- $tag: FAILED $(utc), see $dir/train.log"
    return 1
  fi
}

# Seed-major order: every width and both arms get seed 0 before any seed 1, so a partial
# grid is already a full (width x arm) picture.
echo "=== train: grid, MAX_JOBS=$MAX_JOBS ==="
fail=0
for s in $SEEDS; do
  for u in $STUDENT_UNITS; do
    for mode in $MODES; do
      if [ "$MAX_JOBS" -le 1 ]; then
        run_one "$u" "$mode" "$s" || fail=1
      else
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 20; done
        run_one "$u" "$mode" "$s" &
      fi
    done
  done
done
wait || fail=1

echo "=== collate ==="
SUMMARY="$OUT/summary_$(utc).csv"
"$PY" - "$RUNS" "$SUMMARY" <<'PYEOF'
import csv, glob, os, sys
from collections import defaultdict
runs, out = sys.argv[1], sys.argv[2]
rows = []
for done in sorted(glob.glob(os.path.join(runs, '*', 'COMPLETE'))):
    d = os.path.dirname(done)
    rows += list(csv.DictReader(open(os.path.join(d, os.path.basename(d) + '.csv'))))
if not rows:
    sys.exit('[collate] no completed runs')
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f'[collate] {len(rows)} completed runs -> {out}')
g = defaultdict(list)
for r in rows:
    g[(int(r['units']), r['mode'])].append(r)
print(f"{'units':>5} {'params':>7} {'mode':>7} {'n':>2} {'p_L mean':>9} {'x MWPM':>7} "
      f"{'agree_amb':>9}")
for (u, m), rs in sorted(g.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
    pl = [float(r['p_L']) for r in rs]
    rt = [float(r['ratio_vs_mwpm']) for r in rs]
    am = [float(r['agree_ambiguous']) for r in rs if r['agree_ambiguous'] != '']
    print(f"{u:5d} {rs[0]['n_params']:>7} {m:>7} {len(rs):2d} {sum(pl)/len(pl):9.6f} "
          f"{sum(rt)/len(rt):7.4f} {(sum(am)/len(am) if am else float('nan')):9.4f}")
PYEOF

STAMP=$(basename "$OUT")
tar -czf "$RT/transfer/${STAMP}.tgz" \
    --exclude='teacher' --exclude='*.per_shot_eval.npz' --exclude='*.attempt_*' \
    -C "$(dirname "$OUT")" "$STAMP"
echo "results -> $OUT"
echo "tarball -> $RT/transfer/${STAMP}.tgz  (student checkpoints included; teacher caches"
echo "           and per-shot npz stay on the pod)"
exit $fail
