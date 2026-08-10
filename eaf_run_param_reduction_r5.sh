#!/usr/bin/env bash
# Created: 2026-08-06
# Last updated: 2026-08-06
#
# Parameter-reduction sweep at d=5, r=5, p=0.010. Run on EAF.
#
# Independent variable: the trainable-parameter budget. Everything else is held fixed --
# 10,000,000 training shots for every run, the architecture's own existing recipe, hard
# Stim logical-flip labels only (alpha = 1.0), no distillation.
#
# Two disjoint validation slices, per specification section 2.1:
#   val_select [20,000,000, 20,100,000)  early stopping, checkpoint selection, any tuning
#   val_report [20,100,000, 20,200,000)  the reported number, never selected on
#   sealed     [20,200,000, 20,400,000)  never read anywhere in this study
#
# Configurations come from the FROZEN manifest written by parameter_search_r5.py. This
# script recomputes the manifest's SHA-256 and refuses to launch if it does not match the
# hash recorded at freeze time, so a manifest edited after the fact cannot silently change
# what gets trained.
#
# Filters (all optional):
#   ARCHS="rcnn gru mlp"          which architectures
#   FRACTIONS="1.0 0.5 0.25 0.10" which budget rungs
#   SEEDS="0 1 2"                 which seeds
#   DRY_RUN=1                     print the commands and exit without training
#
# Example, splitting the sweep across two machines:
#   ARCHS="gru" SEEDS="0 1 2" bash eaf_run_param_reduction_r5.sh
#   ARCHS="mlp" SEEDS="0 1 2" bash eaf_run_param_reduction_r5.sh
set -euo pipefail

# --- determinism, exported BEFORE any Python process starts -----------------------------
# TensorFlow reads both during its own import; setting them later has no effect. Every
# entry point below is passed --require-determinism, so a launcher that lost these lines
# produces a hard failure rather than a quietly nondeterministic run.
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
# The smoke-test partition scaling must never be active in a formal run. It is cleared
# unless PARAM_REDUCTION_R5_SMOKE=1 is set, which only the smoke test does, and which
# prints a banner into the driver log so a smoke run can never be read as a formal one.
if [ "${PARAM_REDUCTION_R5_SMOKE:-0}" != "1" ]; then
  unset SLICE_GUARD_SMOKE_DIVISOR
else
  echo "##############################################################"
  echo "# PARAM_REDUCTION_R5_SMOKE=1 -- SMOKE TEST, NOT A STUDY RUN  #"
  echo "# SLICE_GUARD_SMOKE_DIVISOR=${SLICE_GUARD_SMOKE_DIVISOR:-unset}"
  echo "##############################################################"
fi

RT="${RT:-$HOME/rcnn_threshold}"
POOL="${POOL:-$RT/pools_r5/data_d5_p0.010_r5_FORMAL.npz}"
MANIFEST_DIR="${MANIFEST_DIR:-$RT/param_reduction_r5}"
MANIFEST="$MANIFEST_DIR/param_manifest_r5.json"
MANIFEST_HASH_FILE="$MANIFEST.sha256"
OUTROOT="${OUTROOT:-$RT/out_param_reduction_r5}"

PY="${PY:-python}"
# The slice boundaries come from slice_guard_r5.py rather than from literals here, so
# there is exactly one definition of the partitions in the whole study and the smoke test
# exercises this script rather than a scaled copy of it.
read -r N_TRAIN VAL_SELECT_START VAL_SELECT_N VAL_REPORT_START VAL_REPORT_N SEALED_START SEALED_N <<EOF
$($PY -c "
import slice_guard_r5 as g
print(g.TRAIN_STOP, g.VAL_SELECT_START, g.VAL_SELECT_STOP - g.VAL_SELECT_START,
      g.VAL_REPORT_START, g.VAL_REPORT_STOP - g.VAL_REPORT_START,
      g.SEALED_START, g.SEALED_STOP - g.SEALED_START)" 2>/dev/null | tail -1)
EOF
[ -n "${SEALED_N:-}" ] || { echo "could not read the partition boundaries from slice_guard_r5.py"; exit 1; }
FORMAL_N_TRAIN="$N_TRAIN"

# --- GPU smoke mode ---------------------------------------------------------------------
# GPU_SMOKE=1 shortens the TRAINING PREFIX and nothing else. The pool, the frozen manifest,
# the model code, val_select, val_report and the sealed-test guard are all the formal ones,
# so this exercises the study's real path on the real hardware. That is the point: the
# partition-scaling smoke mode in slice_guard_r5.py cannot be used here, because it would
# stop the run from touching the real pool and the real manifest.
#
# Four things stop a GPU-smoke artifact from being read as a study result:
#   1. it must write to an output root whose name contains 'gpusmoke', and the script
#      refuses to start otherwise;
#   2. every run tag carries the literal 'pr5gpusmoke';
#   3. score_val_report_r5.py stamps gpu_smoke=true into the summary and the per-shot file;
#   4. collate_param_reduction_r5.py rejects any run whose n_train is not the full training
#      partition, so a smoke artifact cannot enter a results table even by accident.
TAG_PREFIX="pr5"
if [ "${GPU_SMOKE:-0}" = "1" ]; then
  N_TRAIN="${N_TRAIN_OVERRIDE:?GPU_SMOKE=1 requires N_TRAIN_OVERRIDE (e.g. 50000)}"
  if [ "$N_TRAIN" -ge "$FORMAL_N_TRAIN" ]; then
    echo "GPU_SMOKE=1 with N_TRAIN_OVERRIDE=$N_TRAIN is not a smoke test"
    echo "(the formal training prefix is $FORMAL_N_TRAIN). STOP."
    exit 1
  fi
  TAG_PREFIX="pr5gpusmoke"
  case "$OUTROOT" in
    *gpusmoke*) : ;;
    *) echo "GPU_SMOKE=1 requires an OUTROOT containing 'gpusmoke', so smoke artifacts"
       echo "cannot land in the formal output tree. Got: $OUTROOT. STOP."; exit 1 ;;
  esac
  mkdir -p "$OUTROOT"
  cat > "$OUTROOT/GPU_SMOKE_DO_NOT_USE.txt" <<MARK
These artifacts came from a GPU smoke test, not from the study.
The training prefix was $N_TRAIN shots, not $FORMAL_N_TRAIN.
No number in this directory may be quoted, plotted, or collated as a study result.
MARK
  echo "##############################################################"
  echo "# GPU_SMOKE=1 -- training prefix cut to $N_TRAIN shots"
  echo "# Formal partitions, formal manifest, formal pool. NOT a run."
  echo "##############################################################"
fi

ARCHS="${ARCHS:-rcnn gru mlp}"
FRACTIONS="${FRACTIONS:-1.0 0.5 0.25 0.10}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-10000}"
STUDENT_LR="${STUDENT_LR:-0.003}"
PY="${PY:-python}"
DRY_RUN="${DRY_RUN:-0}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUTROOT"
DRIVER_LOG="$OUTROOT/driver_${STAMP}.log"
# Every guarded index range from every child process lands in one append-only file, so
# the pilot can confirm afterwards that nothing bypassed the guard.
export SLICE_GUARD_LOG="$OUTROOT/slice_guard_${STAMP}.jsonl"

# --- preflight --------------------------------------------------------------------------
[ -f "$POOL" ] || { echo "MISSING pool: $POOL"; exit 1; }
[ -f "$MANIFEST" ] || { echo "MISSING manifest: $MANIFEST -- run parameter_search_r5.py"; exit 1; }
[ -f "$MANIFEST_HASH_FILE" ] || { echo "MISSING $MANIFEST_HASH_FILE -- the manifest was never frozen"; exit 1; }

RECORDED_HASH=$(awk '{print $1}' "$MANIFEST_HASH_FILE")
ACTUAL_HASH=$(sha256sum "$MANIFEST" | awk '{print $1}')
if [ "$RECORDED_HASH" != "$ACTUAL_HASH" ]; then
  echo "MANIFEST HASH MISMATCH"
  echo "  recorded at freeze time: $RECORDED_HASH"
  echo "  computed now:            $ACTUAL_HASH"
  echo "The manifest changed after it was frozen. Every run already tagged with the old"
  echo "hash describes different models. STOP."
  exit 1
fi

GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then GIT_SHA="${GIT_SHA}-dirty"; fi
CNNMODEL_HASH=$(sha256sum CNNModel.py | awk '{print $1}')

{
  echo "=============================================================="
  echo "parameter-reduction sweep  d=5 r=5 p=0.010"
  echo "  started        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "  host           : $(hostname)"
  echo "  git SHA        : $GIT_SHA"
  echo "  manifest       : $MANIFEST"
  echo "  manifest sha256: $ACTUAL_HASH"
  echo "  CNNModel.py sha256: $CNNMODEL_HASH"
  echo "  pool           : $POOL"
  echo "  train          : [0, $N_TRAIN)${GPU_SMOKE:+   <-- GPU SMOKE, the formal prefix is $FORMAL_N_TRAIN}"
  echo "  val_select     : [$VAL_SELECT_START, $((VAL_SELECT_START+VAL_SELECT_N)))"
  echo "  val_report     : [$VAL_REPORT_START, $((VAL_REPORT_START+VAL_REPORT_N)))"
  echo "  sealed test    : [$SEALED_START, $((SEALED_START+SEALED_N)))  NEVER READ"
  echo "  archs          : $ARCHS"
  echo "  fractions      : $FRACTIONS"
  echo "  seeds          : $SEEDS"
  echo "  recipe         : epochs=$EPOCHS batch=$BATCH, no early stopping,"
  echo "                   RCNN = train_one.py 'original' LR schedule (adam),"
  echo "                   students = constant lr=$STUDENT_LR (adam), alpha=1.0 T=1.0"
  echo "  slice guard log: $SLICE_GUARD_LOG"
  echo "  out            : $OUTROOT"
  echo "=============================================================="
} | tee -a "$DRIVER_LOG"

# manifest_lookup <arch> <fraction> <field>
# Reads one field out of the frozen manifest. Returns the literal string UNREACHABLE for
# a rung the parameter search marked as unreachable, so the loop can skip it explicitly
# rather than inventing a configuration for it.
manifest_lookup() {
  $PY - "$MANIFEST" "$1" "$2" "$3" <<'PYEOF'
import json, sys
manifest, arch, frac, field = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
m = json.load(open(manifest))
hits = [c for c in m['configurations']
        if c['architecture'] == arch and abs(c['target_fraction'] - frac) < 1e-9]
if len(hits) != 1:
    sys.exit(f"expected one manifest row for {arch} @ {frac}, found {len(hits)}")
row = hits[0]
if row['reachable'] != 'YES' and field != 'reason':
    # Only the reason is readable for an unreachable rung; every other field would be a
    # configuration that does not exist.
    print('UNREACHABLE'); raise SystemExit(0)
if field == 'args':
    import json as j
    print(j.dumps(j.loads(row['constructor_args'])))
else:
    print(row[field])
PYEOF
}

# Count the runs first so the log has a denominator.
total=0
for arch in $ARCHS; do for frac in $FRACTIONS; do
  ap=$(manifest_lookup "$arch" "$frac" actual_params)
  [ "$ap" = "UNREACHABLE" ] && continue
  for s in $SEEDS; do total=$((total+1)); done
done; done
echo "planned runs: $total" | tee -a "$DRIVER_LOG"
echo | tee -a "$DRIVER_LOG"

i=0
for arch in $ARCHS; do
  for frac in $FRACTIONS; do
    ACTUAL_PARAMS=$(manifest_lookup "$arch" "$frac" actual_params)
    if [ "$ACTUAL_PARAMS" = "UNREACHABLE" ]; then
      REASON=$(manifest_lookup "$arch" "$frac" reason || true)
      echo "SKIP $arch @ $frac -- unreachable in the frozen manifest: $REASON" | tee -a "$DRIVER_LOG"
      continue
    fi
    CARGS=$(manifest_lookup "$arch" "$frac" args)

    for seed in $SEEDS; do
      i=$((i+1))
      # The run tag carries architecture, fraction, MEASURED parameter count, seed and
      # n_train, so two rungs that happen to share a fraction label but not a parameter
      # count can never collide in the filesystem or in collation.
      TAG="${TAG_PREFIX}_${arch}_frac${frac}_p${ACTUAL_PARAMS}_seed${seed}_ntr${N_TRAIN}"
      RUN_DIR="$OUTROOT/$arch/frac${frac}/seed${seed}"
      CKPT_DIR="$RUN_DIR/ckpt"
      RUN_LOG="$RUN_DIR/${TAG}_${STAMP}.log"
      # The scorer is given this same tag, so the filenames it writes are exactly the
      # ones checked here. Deriving them separately on each side would let a formatting
      # difference (0.10 against 0.1) make every completed run look missing.
      DONE_MARK="$RUN_DIR/valreport_${TAG}.json"
      PERSHOT="$RUN_DIR/pershot_valreport_${TAG}.npz"

      echo "=============== [$i/$total] $TAG ===============" | tee -a "$DRIVER_LOG"

      # Skip only a run that is complete AND validated: the summary exists, the per-shot
      # file exists, the recorded parameter count matches this rung, and the recorded
      # manifest hash matches the frozen one. File existence alone is not enough.
      if [ -f "$DONE_MARK" ] && [ -f "$PERSHOT" ]; then
        if $PY - "$DONE_MARK" "$ACTUAL_PARAMS" "$ACTUAL_HASH" "$VAL_REPORT_START" "$((VAL_REPORT_START+VAL_REPORT_N))" <<'PYEOF'
import json, sys
p, want_params, want_hash = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lo, hi = int(sys.argv[4]), int(sys.argv[5])
d = json.load(open(p))
ok = (int(d.get('actual_params', -1)) == want_params
      and d.get('manifest_sha256') == want_hash
      and d.get('val_report_slice') == [lo, hi]
      and isinstance(d.get('p_L'), float))
sys.exit(0 if ok else 1)
PYEOF
        then
          echo "  already complete and validated -- skipping" | tee -a "$DRIVER_LOG"
          continue
        else
          echo "  existing result failed validation; it is NOT overwritten." | tee -a "$DRIVER_LOG"
          echo "  Move it aside and rerun. STOP." | tee -a "$DRIVER_LOG"
          exit 1
        fi
      fi

      mkdir -p "$CKPT_DIR"
      {
        echo "run tag        : $TAG"
        echo "started        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "manifest sha256: $ACTUAL_HASH"
        echo "git SHA        : $GIT_SHA"
        echo "CNNModel.py sha: $CNNMODEL_HASH"
        echo "architecture   : $arch"
        echo "target fraction: $frac"
        echo "actual params  : $ACTUAL_PARAMS"
        echo "constructor    : $CARGS"
        echo "seed           : $seed"
        echo "train slice    : [0, $N_TRAIN)"
        echo "val_select     : [$VAL_SELECT_START, $((VAL_SELECT_START+VAL_SELECT_N)))"
        echo "val_report     : [$VAL_REPORT_START, $((VAL_REPORT_START+VAL_REPORT_N)))"
        echo "recipe         : epochs=$EPOCHS batch=$BATCH no-early-stopping"
      } | tee -a "$RUN_LOG" "$DRIVER_LOG" > /dev/null

      if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY_RUN=1: not launching" | tee -a "$DRIVER_LOG"
        continue
      fi

      # --- train -----------------------------------------------------------------------
      # validation_data is passed explicitly from val_select in both trainers.
      # validation_split is never used: --val-start/--val-n disable it.
      if [ "$arch" = "rcnn" ]; then
        HIDDEN=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['hidden'])" "$CARGS")
        HL=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['hidden_layers'])" "$CARGS")
        NPOL=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['npol'])" "$CARGS")
        KERN=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['kernel'])" "$CARGS")
        # --n-test 200000 names the sealed block, and --seal-test is what stops it from
        # ever being read: under --seal-test the trainer scores val_select instead.
        $PY train_one.py --require-determinism \
          --d 5 --p 0.010 --rounds 5 --kernel "$KERN" --npol "$NPOL" \
          --hidden "$HIDDEN" --hidden-layers "$HL" \
          --seed "$seed" --pool "$POOL" \
          --n-train "$N_TRAIN" --n-test "$SEALED_N" \
          --val-start "$VAL_SELECT_START" --val-n "$VAL_SELECT_N" --seal-test \
          --epochs "$EPOCHS" --batch-size "$BATCH" --no-early-stopping --save-weights \
          --out-dir "$RUN_DIR" --ckpt-dir "$CKPT_DIR" --run-tag "$TAG" 2>&1 \
          | tee -a "$RUN_LOG"
      else
        if [ "$arch" = "gru" ]; then
          UNITS=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['units'])" "$CARGS")
          # A bare --hidden gives argparse an empty list: no dense head between the GRU
          # and the logit, which is the configuration the 69,441 count refers to.
          SIZE="--hidden --units $UNITS"
        else
          H=$($PY -c "import json,sys;print(' '.join(str(x) for x in json.loads(sys.argv[1])['hidden']))" "$CARGS")
          SIZE="--hidden $H"
        fi
        # --eval-start points the trainer's own scoring at val_select, not at the pool
        # tail, which on this pool is the sealed test block.
        $PY train_student.py --require-determinism \
          --student "$arch" --inputs evts $SIZE \
          --d 5 --p 0.010 --rounds 5 \
          --alpha 1.0 --temperature 1.0 --lr "$STUDENT_LR" \
          --seed "$seed" --n-train "$N_TRAIN" \
          --n-test "$VAL_SELECT_N" --eval-start "$VAL_SELECT_START" \
          --val-start "$VAL_SELECT_START" --val-n "$VAL_SELECT_N" \
          --epochs "$EPOCHS" --batch-size "$BATCH" --no-early-stopping \
          --pool "$POOL" --out-dir "$RUN_DIR" --tag "$TAG" \
          --ckpt-dir "$CKPT_DIR" --run-tag "$TAG" 2>&1 \
          | tee -a "$RUN_LOG"
      fi

      BEST="$CKPT_DIR/${TAG}.best.weights.h5"
      [ -f "$BEST" ] || { echo "MISSING best checkpoint $BEST -- training did not produce one. STOP." | tee -a "$RUN_LOG" "$DRIVER_LOG"; exit 1; }

      # --- score val_report once, from the val_select-best checkpoint --------------------
      $PY score_val_report_r5.py \
        --arch "$arch" --fraction "$frac" --seed "$seed" --n-train "$N_TRAIN" \
        --weights "$BEST" --manifest "$MANIFEST" --pool "$POOL" --run-tag "$TAG" \
        ${GPU_SMOKE:+--gpu-smoke --verify-reload --also-score-val-select} \
        --out-dir "$RUN_DIR" --batch-size "$BATCH" 2>&1 | tee -a "$RUN_LOG"

      echo "RUN COMPLETE $TAG  $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_LOG" "$DRIVER_LOG"
      echo | tee -a "$DRIVER_LOG"
    done
  done
done

echo "==============================================================" | tee -a "$DRIVER_LOG"
echo "SWEEP COMPLETE  $i/$total runs  $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$DRIVER_LOG"
echo "driver log: $DRIVER_LOG" | tee -a "$DRIVER_LOG"
echo "next: python collate_param_reduction_r5.py --out-root $OUTROOT --manifest $MANIFEST" | tee -a "$DRIVER_LOG"
