#!/usr/bin/env bash
# Created: 2026-09-21
# Last updated: 2026-09-22
#
# d=7, r=7, p=0.004: parameter-matched Mamba against the Experiment 16 GRU. Runs on the
# GPU host (EAF pod, or grace1 inside the qdec:tf215 container -- see NEEDS_EAF_CHECK
# notes in experiment_log/docs/exp18_mamba_vs_gru_d7_DRAFT.md).
#
# One script, four modes, so every stage shares the same preflight, partitions and
# output layout:
#
#   MODE=smoke    bash eaf_run_mamba_d7.sh     3 epochs on a 500k prefix, seed 0, one LR.
#                                              Proves the pipeline on the real pool and
#                                              prints s/epoch, which is the number that
#                                              sizes everything below.
#   MODE=lrsweep  bash eaf_run_mamba_d7.sh     3 learning rates x seed 0, full 10M prefix,
#                                              LR_EPOCHS epochs. Selection is by best
#                                              val_loss on the validation block ONLY.
#                                              Nothing is scored.
#   MODE=full     bash eaf_run_mamba_d7.sh     SEEDS x 200 epochs at LR, 10M prefix.
#                                              Scored once on the tuning evaluation block
#                                              [15.2M, 17.0M) from the .best checkpoint,
#                                              matching how the GRU rows were scored.
#   MODE=eval     bash eaf_run_mamba_d7.sh     Final, run once: Mamba (every seed) and the
#                                              existing GRU checkpoints on the SEALED test
#                                              block [17M, 19M), plus MWPM on that block.
#                                              Requires GRU_CKPT_DIR (see below).
#
# Partitions are the Experiment 13 / 16 layout and are not tunable here:
#   training     [0, 10M)
#   validation   [15.0M, 15.2M)     checkpoint selection, LR selection
#   evaluation   [15.2M, 17.0M)     scored once per seed in MODE=full (tuning-side block)
#   sealed test  [17.0M, 19.0M)     read only in MODE=eval
set -euo pipefail

# Determinism, exported before any Python starts (TensorFlow reads these at import).
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

MODE="${MODE:-smoke}"

# --- paths: NEEDS_EAF_CHECK ---------------------------------------------------------------
# The d=7 pool of record (gen_seed 42, flips SHA b49fbe40786da9e6...) was generated on
# grace1 at ~/pools_d7_p004_19M/. No local evidence shows a d=7 p=0.004 pool on EAF.
# Either copy that pool + its .fingerprint.json to the EAF scratch area, or run this
# script on grace1. Pass POOL explicitly; there is no safe default.
POOL="${POOL:?set POOL=<path>/data_d7_p0.004_r7_FORMAL.npz}"
REPO="${REPO:-$HOME/QuantumDecoderQKeras}"      # NEEDS_EAF_CHECK: pod workdir name
PY="${PY:-$REPO/.venv/bin/python}"              # grace1 container: PY=python3
RT="${RT:-$HOME/rcnn_threshold}"

D=7; ROUNDS=7; P=0.004
N_TRAIN="${N_TRAIN:-10000000}"
VAL_START=15000000; VAL_N=200000
EVAL_START=15200000; EVAL_N=1800000
SEALED_START=17000000; SEALED_N=2000000

BATCH="${BATCH:-10000}"
LR="${LR:-0.003}"                # committed LR; overwrite after the sweep
EPOCHS="${EPOCHS:-200}"
SEEDS="${SEEDS:-0 1 2}"
LR_GRID="${LR_GRID:-0.001 0.003 0.01}"
LR_EPOCHS="${LR_EPOCHS:-40}"     # sweep budget; NEEDS_EAF_CHECK after the smoke s/epoch
GPU_MEM_MIB="${GPU_MEM_MIB:-}"   # per-process cap when seeds run concurrently
PARALLEL="${PARALLEL:-0}"
JIT="${JIT:-0}"                  # 1 = --jit (XLA) on every training run; decide before the sweep

# Existing GRU checkpoints for MODE=eval. On grace1 they are under
#   ~/rcnn_threshold/results/gh200_gru_d7_e200_20260819T192524Z/seed<N>/ckpt/
#   gru_d7_p004_u140_hard_seed<N>_ntr10000000_b10000.best.weights.h5
# NEEDS_EAF_CHECK: those files are on grace1, not EAF, and not on the Mac.
GRU_CKPT_DIR="${GRU_CKPT_DIR:-}"
GRU_TAG_FMT="${GRU_TAG_FMT:-gru_d7_p004_u140_hard_seed%s_ntr10000000_b10000}"

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export OUT="${OUT:-$RT/results/mamba_d7_p004_${MODE}_${STAMP}}"
mkdir -p "$OUT/runs" "$OUT/ckpt" "$RT/transfer"
LOG="$OUT/driver.log"
exec > >(tee -a "$LOG") 2>&1

QUIET='cuda_|Unable to register|^Total number|^Number of unique|oneDNN|cpu_feature'

echo "[mamba] mode=$MODE  out=$OUT"
[ -f "$POOL" ] || { echo "[mamba] missing pool $POOL. STOP."; exit 1; }
[ -f "${POOL%.npz}.fingerprint.json" ] || { echo "[mamba] no fingerprint beside $POOL. STOP."; exit 1; }
command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ] || { echo "[mamba] no python at $PY. STOP."; exit 1; }
cd "$REPO"
for f in MambaModel.py train_mamba.py eval_mamba_on_tail.py StudentModels.py train_one.py \
         eval_on_tail.py slice_guard_r5.py circuit_generators.py mwpm_on_block.py \
         eval_student_on_tail.py; do
  [ -f "$f" ] || { echo "[mamba] missing $f in $REPO -- upload it. STOP."; exit 1; }
done

# --- preflight: environment, GPU, pool geometry, partitions --------------------------------
"$PY" - "$POOL" <<'PYENV'
import os, sys, json
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
assert tf.__version__.startswith('2.15'), f'need TF 2.15.x; got {tf.__version__}'
gpus = tf.config.list_physical_devices('GPU')
print(f"[mamba] TensorFlow {tf.__version__}  GPUs: {gpus}")
if not gpus:
    raise SystemExit("[mamba] no GPU visible. STOP.")
import numpy as np, pymatching, stim
print(f"[mamba] stim {stim.__version__}  numpy {np.__version__}  pymatching ok")
pool = sys.argv[1]
z = np.load(pool)
N, w = z['det_evts'].shape
m = json.load(open(pool.replace('.npz', '.fingerprint.json')))
assert (m['d'], m['rounds'], float(m['p'])) == (7, 7, 0.004), m
assert w == 336, f"det_evts width {w}, expected 336 for d=7 r=7"
assert N >= 19_000_000, f"pool has {N:,} shots, layout needs 19M"
print(f"[mamba] pool ok: {N:,} shots, det_evts width {w}, gen_seed={m['gen_seed']}, "
      f"flips_sha={m['flips_sha256'][:16]}")
json.dump(m, open(os.path.join(os.environ['OUT'], 'pool_provenance.json'), 'w'), indent=2)
PYENV
# OUT exported at definition above, so the preflight heredoc can read it

# Parameter counts, printed into the log every run.
CUDA_VISIBLE_DEVICES= "$PY" - <<'PYPARAM' 2>&1 | grep --line-buffered -Ev "$QUIET"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from StudentModels import build_student
from MambaModel import build_mamba_decoder
g = build_student('gru', d=7, rounds=7, inputs='evts', units=140, hidden=()).count_params()
m = build_mamba_decoder(8, 48, d_model=100, n_layers=1, d_state=16, expand=2, d_conv=4).count_params()
print(f"[mamba] params: mamba={m:,}  gru={g:,}  diff={100*(m-g)/g:+.2f}%")
PYPARAM

{
  echo "experiment   : Mamba vs GRU, d=7 r=7 p=0.004  mode=$MODE"
  echo "launched     : $(date -u '+%Y-%m-%dT%H:%M:%SZ') on $(hostname)"
  echo "pool         : $POOL"
  echo "train        : [0, $N_TRAIN)"
  echo "validation   : [$VAL_START, $((VAL_START + VAL_N)))"
  echo "evaluation   : [$EVAL_START, $((EVAL_START + EVAL_N)))"
  echo "sealed test  : [$SEALED_START, $((SEALED_START + SEALED_N)))  read only in MODE=eval"
  echo "model        : mamba d_model=100 n_layers=1 d_state=16 expand=2 d_conv=4 (79,001 params)"
  echo "recipe       : Adam constant lr=$LR batch=$BATCH epochs=$EPOCHS no early stopping jit=$JIT"
  echo "lr grid      : $LR_GRID x $LR_EPOCHS epochs (MODE=lrsweep)"
  echo "seeds        : $SEEDS"
  echo "code sha     : $(sha256sum MambaModel.py | cut -c1-16)  MambaModel.py"
  echo "             : $(sha256sum train_mamba.py | cut -c1-16)  train_mamba.py"
  echo "             : $(sha256sum eval_mamba_on_tail.py | cut -c1-16)  eval_mamba_on_tail.py"
  echo "             : $(sha256sum "$0" | cut -c1-16)  $(basename "$0")"
} | tee "$OUT/MANIFEST.txt"

CAP=(); [ -n "$GPU_MEM_MIB" ] && CAP=(--gpu-mem-mib "$GPU_MEM_MIB")
JITF=(); [ "$JIT" = "1" ] && JITF=(--jit)

train_one () {   # $1 seed  $2 lr  $3 epochs  $4 n_train  $5 tag
  "$PY" train_mamba.py --d $D --p $P --rounds $ROUNDS --pool "$POOL" \
    --n-train "$4" --val-start $VAL_START --val-n $VAL_N --sealed-start $SEALED_START \
    --seed "$1" --lr "$2" --batch-size "$BATCH" --epochs "$3" --require-determinism \
    --ckpt-dir "$OUT/ckpt" --out-dir "$OUT/runs" --tag "$5" "${CAP[@]}" "${JITF[@]}" 2>&1 | grep --line-buffered -Ev "$QUIET"
}

eval_block () {  # $1 tag  $2 start  $3 n  $4 label
  "$PY" eval_mamba_on_tail.py --weights "$OUT/ckpt/$1.best.weights.h5" \
    --config "$OUT/runs/$1.config.json" --pool "$POOL" \
    --eval-start "$2" --n-test "$3" --batch-size "$BATCH" --label "$4" \
    --dump-per-shot "$OUT/per_shot_${1}_${4}.npz" --out-csv "$OUT/eval_${4}.csv" \
    "${CAP[@]}" 2>&1 | grep --line-buffered -Ev "$QUIET"
}

case "$MODE" in
  smoke)
    TAG="mamba_d7_smoke_seed0_lr${LR}"
    train_one 0 "$LR" 3 500000 "$TAG"
    eval_block "$TAG" $VAL_START 50000 smoke_val   # 50k of the validation block, not test
    echo "[mamba] smoke done. Read sec_per_epoch from $OUT/runs/$TAG.csv; scale by 20x for"
    echo "[mamba] the 10M prefix (1,000 steps/epoch vs 50) to size LR_EPOCHS and EPOCHS."
    ;;
  lrsweep)
    for lr in $LR_GRID; do
      TAG="mamba_d7_lrsweep_seed0_lr${lr}"
      [ -f "$OUT/runs/$TAG.csv" ] && { echo "[mamba] $TAG exists, skipping"; continue; }
      train_one 0 "$lr" "$LR_EPOCHS" "$N_TRAIN" "$TAG"
    done
    echo; echo "[mamba] sweep summary (validation block only; nothing scored):"
    "$PY" - "$OUT/runs" <<'PYSUM'
import csv, glob, os, sys
rows = [next(csv.DictReader(open(f))) for f in sorted(glob.glob(os.path.join(sys.argv[1], 'mamba_d7_lrsweep_*.csv')))]
rows.sort(key=lambda r: float(r['best_val_loss']))
for r in rows:
    print(f"  lr={float(r['lr']):<7g} best_val_loss={float(r['best_val_loss']):.6f} "
          f"@epoch {r['best_epoch']}/{r['epochs_ran']}  final={float(r['final_val_loss']):.6f}  "
          f"{float(r['sec_per_epoch']):.1f} s/epoch")
print(f"  -> lowest best_val_loss: lr={rows[0]['lr']}   (set LR=... for MODE=full)")
PYSUM
    ;;
  full)
    run_seed () {
      local s="$1"; local TAG="mamba_d7_p004_dm100_hard_seed${s}_ntr${N_TRAIN}_b${BATCH}_lr${LR}"
      local SLOG="$OUT/seed${s}.log"
      if [ -f "$OUT/COMPLETE_seed${s}" ]; then echo "[mamba] seed $s complete, skipping"; return 0; fi
      {
        date -u '+%Y-%m-%dT%H:%M:%SZ start'
        train_one "$s" "$LR" "$EPOCHS" "$N_TRAIN" "$TAG"
        eval_block "$TAG" $EVAL_START $EVAL_N eval
        "$PY" - "$OUT" "$TAG" "$EPOCHS" <<'PYDONE'
import json, os, sys
out, tag, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
h = json.load(open(os.path.join(out, 'runs', tag + '.history.json')))
ok = (len(h['val_loss']) == epochs
      and os.path.exists(os.path.join(out, 'ckpt', tag + '.best.weights.h5'))
      and os.path.exists(os.path.join(out, f'per_shot_{tag}_eval.npz')))
print('[complete] all artifacts present' if ok else '[complete] NOT complete')
sys.exit(0 if ok else 1)
PYDONE
        touch "$OUT/COMPLETE_seed${s}"
        date -u '+%Y-%m-%dT%H:%M:%SZ end'
      } > "$SLOG" 2>&1
    }
    if [ "$PARALLEL" = "1" ]; then
      for s in $SEEDS; do run_seed "$s" & done; wait
    else
      for s in $SEEDS; do run_seed "$s"; done
    fi
    echo "[mamba] eval-block results:"; cat "$OUT/eval_eval.csv" 2>/dev/null || true
    ;;
  eval)
    # Sealed test block, once. MWPM first (CPU), then Mamba, then the GRU checkpoints.
    "$PY" mwpm_on_block.py --pool "$POOL" --d $D --p $P --rounds $ROUNDS \
      --start $SEALED_START --n $SEALED_N \
      --out-json "$OUT/mwpm_sealed_test_d7.json" \
      --dump-per-shot "$OUT/mwpm_sealed_test_d7_per_shot.npz" 2>&1 | grep --line-buffered -Ev "$QUIET"
    for s in $SEEDS; do
      TAG="mamba_d7_p004_dm100_hard_seed${s}_ntr${N_TRAIN}_b${BATCH}_lr${LR}"
      # MODE=full wrote into its own folder; point FULL_OUT at it.
      FULL_OUT="${FULL_OUT:?set FULL_OUT=<results folder written by MODE=full>}"
      "$PY" eval_mamba_on_tail.py --weights "$FULL_OUT/ckpt/$TAG.best.weights.h5" \
        --config "$FULL_OUT/runs/$TAG.config.json" --pool "$POOL" \
        --eval-start $SEALED_START --n-test $SEALED_N --batch-size "$BATCH" \
        --label sealed_test --dump-per-shot "$OUT/per_shot_${TAG}_sealed_test.npz" \
        --out-csv "$OUT/eval_sealed_test.csv" "${CAP[@]}" 2>&1 | grep --line-buffered -Ev "$QUIET"
    done
    [ -n "$GRU_CKPT_DIR" ] || { echo "[mamba] GRU_CKPT_DIR unset; GRU not scored on the sealed block."; exit 1; }
    for s in $SEEDS; do
      GTAG=$(printf "$GRU_TAG_FMT" "$s")
      W="$GRU_CKPT_DIR/seed${s}/ckpt/$GTAG.best.weights.h5"
      [ -f "$W" ] || W="$GRU_CKPT_DIR/$GTAG.best.weights.h5"
      [ -f "$W" ] || { echo "[mamba] missing GRU checkpoint for seed $s ($GTAG). STOP."; exit 1; }
      "$PY" eval_student_on_tail.py --student gru --inputs evts --hidden --units 140 \
        --weights "$W" --d $D --p $P --rounds $ROUNDS --pool "$POOL" \
        --eval-start $SEALED_START --n-test $SEALED_N --batch-size "$BATCH" \
        --dump-per-shot "$OUT/per_shot_${GTAG}_sealed_test.npz" \
        --out-csv "$OUT/eval_sealed_test_gru.csv" "${CAP[@]}" 2>&1 | grep --line-buffered -Ev "$QUIET"
    done
    ;;
  *) echo "[mamba] MODE=$MODE; expected smoke|lrsweep|full|eval"; exit 1 ;;
esac

tar -czf "$RT/transfer/mamba_d7_p004_${MODE}_${STAMP}.tgz" --exclude='*.weights.h5' \
    -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "[mamba] done. results -> $OUT"
echo "[mamba] tarball -> $RT/transfer/mamba_d7_p004_${MODE}_${STAMP}.tgz  (per-shot npz included, checkpoints excluded)"
