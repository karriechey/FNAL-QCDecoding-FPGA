#!/usr/bin/env bash
# Created: 2026-08-07
# Last updated: 2026-08-07
#
# GPU smoke test for the d=5, r=5 parameter-reduction study. Run on EAF, on the A100.
#
# What this is for
# ----------------
# The earlier smoke test ran on a Mac CPU against a 20,400-shot scaled pool. That proves
# the scripts work; it proves nothing about this hardware. Deterministic cuDNN kernels,
# GPU reduction order, and whether a checkpoint reload reproduces its predictions are all
# properties of the machine, and this project has already been bitten once by GPU
# reduction order at scale (the 10M QAT val_loss spike, 2026-07-19).
#
# So this test uses the REAL pool, the REAL frozen manifest, the REAL model and scoring
# code, the REAL val_select / val_report / sealed-test partitions -- and shortens only the
# training prefix, so it finishes in minutes instead of hours.
#
# It is not a study result and cannot become one:
#   * output goes to a directory whose name contains 'gpusmoke', which the launcher
#     enforces;
#   * every run tag carries 'pr5gpusmoke';
#   * gpu_smoke=true is written into every summary and every per-shot file;
#   * the collation refuses any run carrying that flag, and separately refuses any run
#     whose training prefix is not the full 10,000,000 shots.
# The last check is run here, at the end, and is expected to REJECT everything this script
# produced. A collation that accepted these runs would itself be the failure.
#
# Usage:
#   bash eaf_gpu_smoke_r5.sh
#
# Knobs (all optional):
#   SMOKE_NTRAIN=50000   training prefix, in shots
#   SMOKE_EPOCHS=2       epochs
#   SMOKE_ARCHS="gru mlp rcnn"
#   SMOKE_FRACTION=0.10  which manifest rung (0.10 is the smallest, so the fastest)
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
POOL="${POOL:-$RT/pools_r5/data_d5_p0.010_r5_FORMAL.npz}"
MANIFEST_DIR="${MANIFEST_DIR:-$RT/param_reduction_r5}"
MANIFEST="$MANIFEST_DIR/param_manifest_r5.json"
OUTROOT="${OUTROOT:-$RT/out_param_reduction_r5_gpusmoke}"

SMOKE_NTRAIN="${SMOKE_NTRAIN:-50000}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-2}"
SMOKE_BATCH="${SMOKE_BATCH:-10000}"
# The RCNN is included deliberately. It is the only architecture whose 100% rung will be
# trained, it is by far the most expensive, and it is the one whose weights live in
# CNNModel.py rather than in a Keras functional model -- so its save/reload path is the
# one least like the students'.
SMOKE_ARCHS="${SMOKE_ARCHS:-gru mlp rcnn}"
SMOKE_FRACTION="${SMOKE_FRACTION:-0.10}"
PY="${PY:-python}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/gpu_smoke_summary_${STAMP}.txt"

pass_fail=0
note() { echo "$*" | tee -a "$SUMMARY"; }
check() {  # check <label> <0|1 ok> <detail>
  if [ "$2" = "0" ]; then note "PASS  $1: $3"; else note "FAIL  $1: $3"; pass_fail=1; fi
}
# want_grep <label> <present|absent> <pattern> <file> <detail>
# Wrapped in an if, because a bare `grep -q` that finds nothing returns 1 and would abort
# the whole script under `set -e`.
want_grep() {
  local label="$1" mode="$2" pat="$3" file="$4" detail="$5" found=1
  if grep -qE "$pat" "$file" 2>/dev/null; then found=0; fi
  if [ "$mode" = "present" ]; then
    check "$label" "$found" "$detail"
  else
    check "$label" $([ "$found" -ne 0 ] && echo 0 || echo 1) "$detail"
  fi
}

note "=============================================================="
note "GPU smoke test  d=5 r=5  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
note "  host           : $(hostname)"
note "  training prefix: $SMOKE_NTRAIN shots (the study uses 10,000,000)"
note "  epochs         : $SMOKE_EPOCHS"
note "  architectures  : $SMOKE_ARCHS   at manifest fraction $SMOKE_FRACTION"
note "  out            : $OUTROOT"
note "=============================================================="

# --- 1. environment ---------------------------------------------------------------------
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
unset SLICE_GUARD_SMOKE_DIVISOR      # the PARTITION scaling must never be on here

note ""
note "--- environment ---"
note "python: $(command -v $PY)"
ENV_OUT="$OUTROOT/gpu_smoke_env_${STAMP}.txt"
set +e
$PY - > "$ENV_OUT" 2>&1 <<'PYEOF' 
# Runs before TensorFlow is imported, exactly as every study entry point does.
import os, sys
from slice_guard_r5 import assert_deterministic_env, assert_tf_version, is_smoke
assert_deterministic_env()
import tensorflow as tf
assert_tf_version(tf)
gpus = tf.config.list_physical_devices('GPU')
print(f"[gpu-smoke] visible GPUs: {gpus}")
if not gpus:
    sys.exit("[gpu-smoke] NO GPU VISIBLE -- this test is meaningless on CPU. STOP.")
print(f"[gpu-smoke] partition scaling active: {is_smoke()} (must be False)")
if is_smoke():
    sys.exit("[gpu-smoke] SLICE_GUARD_SMOKE_DIVISOR is set. STOP.")
for d in gpus:
    print(f"[gpu-smoke] {d.name}  {tf.config.experimental.get_device_details(d)}")
PYEOF
env_rc=$?
set -e
cat "$ENV_OUT" | tee -a "$SUMMARY"
check "environment" $env_rc "TF 2.15.x, determinism variables present before import, GPU visible"
if [ "$env_rc" -ne 0 ]; then
  note ""
  note "Environment check failed. Nothing further is meaningful -- stopping here."
  note "GPU SMOKE TEST: FAIL"
  exit 1
fi

# --- 2. inputs --------------------------------------------------------------------------
note ""
note "--- inputs ---"
[ -f "$POOL" ] || { note "FAIL  pool: MISSING $POOL"; exit 1; }
[ -f "$MANIFEST" ] || { note "FAIL  manifest: MISSING $MANIFEST"; exit 1; }
MHASH=$(sha256sum "$MANIFEST" | awk '{print $1}')
note "pool           : $POOL"
note "manifest       : $MANIFEST"
note "manifest sha256: $MHASH"
note "CNNModel.py    : $(sha256sum CNNModel.py | awk '{print $1}')"

# --- 3. the runs ------------------------------------------------------------------------
# Straight through eaf_run_param_reduction_r5.sh, so this is the study's launcher, its
# trainers, its checkpointing and its scorer -- not a parallel implementation.
note ""
note "--- training and scoring (via the study launcher, GPU_SMOKE=1) ---"
RUN_LOG="$OUTROOT/gpu_smoke_runs_${STAMP}.log"
set +e
GPU_SMOKE=1 N_TRAIN_OVERRIDE="$SMOKE_NTRAIN" \
  RT="$RT" POOL="$POOL" MANIFEST_DIR="$MANIFEST_DIR" OUTROOT="$OUTROOT" \
  ARCHS="$SMOKE_ARCHS" FRACTIONS="$SMOKE_FRACTION" SEEDS="0" \
  EPOCHS="$SMOKE_EPOCHS" BATCH="$SMOKE_BATCH" PY="$PY" \
  bash eaf_run_param_reduction_r5.sh > "$RUN_LOG" 2>&1
rc=$?
set -e
note "launcher exit code: $rc   (log: $RUN_LOG)"
check "launcher completed" $rc "eaf_run_param_reduction_r5.sh returned $rc"

# --- 4. per-check verification ----------------------------------------------------------
note ""
note "--- checks ---"

want_grep "partition scaling off" absent "SMOKE MODE" "$RUN_LOG" \
  "no slice_guard SMOKE MODE banner, so the partitions were the formal ones"
want_grep "frozen manifest used" present "manifest sha256: $MHASH" "$RUN_LOG" \
  "the launcher recorded $MHASH"
want_grep "training prefix shortened" absent "train \[0, 10,000,000\)" "$RUN_LOG" \
  "the trainer did NOT use the full 10M prefix, which is what makes this a smoke test"
want_grep "val_select resolved" present "validation \[20,000,000, 20,100,000\)" "$RUN_LOG" \
  "validation_data came from [20,000,000, 20,100,000)"
want_grep "validation_split disabled" present "validation_split disabled" "$RUN_LOG" \
  "explicit validation_data, never a split"
want_grep "val_report resolved" present "ALLOW \[20,100,000, 20,200,000\)" "$RUN_LOG" \
  "scoring ran on [20,100,000, 20,200,000)"
want_grep "parameter count" present "matches the manifest" "$RUN_LOG" \
  "the rebuilt model matched the frozen manifest count"
want_grep "sealed test untouched" absent "SealedTestAccess|REJECT" "$RUN_LOG" \
  "no guarded range was rejected, so nothing tried to read the sealed block"
want_grep "best checkpoint saved" present "best.weights.h5" "$RUN_LOG" \
  "a val_select-best checkpoint was written"
want_grep "reload verified on GPU" present "reload verification" "$RUN_LOG" \
  "the checkpoint was reloaded into a fresh model and re-scored"

note ""
note "reload verification, verbatim:"
grep -A2 "reload verification" "$RUN_LOG" | tee -a "$SUMMARY" || true

want_grep "no traceback" absent "Traceback" "$RUN_LOG" "no Python traceback in the log"

N_SUM=$(ls "$OUTROOT"/*/*/*/valreport_*.json 2>/dev/null | wc -l)
N_SHOT=$(ls "$OUTROOT"/*/*/*/pershot_valreport_*.npz 2>/dev/null | wc -l)
N_EXP=$(echo $SMOKE_ARCHS | wc -w)
note "summaries: $N_SUM   per-shot files: $N_SHOT   expected (at most): $N_EXP"

# --- 5. per-shot artifact contents ------------------------------------------------------
note ""
note "--- per-shot artifact ---"
SHOT_OUT="$OUTROOT/gpu_smoke_pershot_${STAMP}.txt"
set +e
$PY - "$OUTROOT" > "$SHOT_OUT" 2>&1 <<'PYEOF' 
import glob, sys
import numpy as np
root = sys.argv[1]
files = sorted(glob.glob(f'{root}/*/*/*/pershot_valreport_*.npz'))
if not files:
    sys.exit('[gpu-smoke] no per-shot files written')
need = {'shot_idx', 'truth', 'predicted_class', 'predicted_probability', 'correctness',
        'architecture', 'actual_params', 'seed', 'n_train', 'manifest_sha256', 'git_sha',
        'checkpoint_path', 'checkpoint_sha256', 'slice_start', 'slice_stop', 'gpu_smoke'}
bad = 0
for f in files:
    z = np.load(f, allow_pickle=True)
    missing = need - set(z.files)
    idx = z['shot_idx']
    ok_idx = (idx[0] == 20_100_000 and idx[-1] == 20_199_999 and len(idx) == 100_000)
    print(f"[gpu-smoke] {f.split('/')[-1]}")
    print(f"            keys missing: {sorted(missing) if missing else 'none'}")
    print(f"            shot_idx: {idx[0]}..{idx[-1]}  n={len(idx)}  global-indexed={ok_idx}")
    print(f"            gpu_smoke={z['gpu_smoke']}  n_train={z['n_train']}  "
          f"params={z['actual_params']}")
    print(f"            predicted-positive rate={float(z['predicted_class'].mean()):.4f}  "
          f"probability std={float(z['predicted_probability'].std()):.4f}")
    if missing or not ok_idx or not bool(z['gpu_smoke']):
        bad += 1
sys.exit(1 if bad else 0)
PYEOF
shot_rc=$?
set -e
cat "$SHOT_OUT" | tee -a "$SUMMARY"
check "per-shot artifact" $shot_rc "global shot indices, required keys, gpu_smoke flag present"

# --- 6. the collation must REJECT all of it ---------------------------------------------
note ""
note "--- collation refusal (this is supposed to reject every run above) ---"
COLLATE_LOG="$OUTROOT/gpu_smoke_collate_${STAMP}.log"
set +e
$PY collate_param_reduction_r5.py \
  --out-root "$OUTROOT" \
  --manifest "$MANIFEST" \
  --mwpm "$MANIFEST_DIR/mwpm_val_report_r5.json" \
  --mwpm-per-shot "$MANIFEST_DIR/mwpm_val_report_per_shot.npz" \
  --collate-dir "$OUTROOT" > "$COLLATE_LOG" 2>&1
set -e
grep -E "gpu_smoke_artifact_excluded|not_a_formal_run" "$COLLATE_LOG" | tee -a "$SUMMARY" || true
N_EXCL=$(grep -c "gpu_smoke_artifact_excluded" "$COLLATE_LOG" || true)
if [ "$N_EXCL" -ge 1 ]; then
  check "collation refuses smoke artifacts" 0 "$N_EXCL run(s) excluded, as required"
else
  check "collation refuses smoke artifacts" 1 \
    "the collation did NOT exclude these runs -- a smoke artifact could reach a results table"
fi

# --- 7. verdict -------------------------------------------------------------------------
note ""
note "=============================================================="
if [ "$pass_fail" = "0" ]; then
  note "GPU SMOKE TEST: PASS"
else
  note "GPU SMOKE TEST: FAIL -- see the FAIL lines above"
fi
note "summary : $SUMMARY"
note "run log : $RUN_LOG"
note "collate : $COLLATE_LOG"
note "=============================================================="
note ""
note "Reminder: nothing in $OUTROOT is a study result. Delete the directory once the"
note "numbers above have been read, or leave it -- the collation will keep refusing it."
exit $pass_fail
