#!/usr/bin/env bash
# Created: 2026-08-19
# Last updated: 2026-08-19
#
# Student distance scaling at p=0.004 on grace1 (GH200). Hard labels, batch 10,000,
# seeds 0/1/2, fixed epoch budget with validation-based checkpoint selection.
# Runs inside the qdec:tf215 podman container.
#
#     D=5 ROUNDS=5 STUDENT=gru UNITS=140       bash gh200_run_student_scaling.sh
#     D=7 ROUNDS=7 STUDENT=gru UNITS=140       bash gh200_run_student_scaling.sh
#     D=7 ROUNDS=7 STUDENT=mlp HIDDEN="160 160" bash gh200_run_student_scaling.sh
#     PARALLEL=1 runs all seeds at once on the one GPU.
#
# Widths are matched within a distance, so architecture is the only difference at that
# distance. Across distances the layer is held and the count grows with input width:
#   d=5 r=5   GRU units=140 -> 69,441    MLP hidden=(209,209) -> 69,389
#   d=7 r=7   GRU units=140 -> 79,521    MLP hidden=(160,160) -> 79,841
set -euo pipefail

D="${D:-7}"
ROUNDS="${ROUNDS:-$D}"
P="${P:-0.004}"
SEEDS="${SEEDS:-0 1 2}"
NTRAIN="${NTRAIN:-10000000}"
BATCH="${BATCH:-10000}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-0.003}"          # constant; passing --lr detaches the teacher LR schedule

STUDENT="${STUDENT:-gru}"  # gru or mlp
UNITS="${UNITS:-140}"      # GRU state width
HIDDEN="${HIDDEN:-}"       # MLP layer widths, space separated, e.g. "160 160"

# Quantization-aware training. Empty = float32 layers, the anchor every quantized run is
# measured against. W_BITS swaps in QDense/QGRU with quantized_bits(B, 1, alpha=1);
# A_BITS additionally quantizes activations. Weights-only (A_BITS empty) matches the
# teacher sweep's first stage.
W_BITS="${W_BITS:-}"
A_BITS="${A_BITS:-}"

# Early stopping off by default. The full epoch budget runs and the reported model is the
# minimum-val_loss checkpoint, so patience is not a tunable that moves the result.
# EARLY_STOP=1 restores it with PATIENCE.
EARLY_STOP="${EARLY_STOP:-0}"
PATIENCE="${PATIENCE:-5}"

# Pools sit in $HOME on grace1; pass POOL explicitly from the podman mount.
POOL="${POOL:-$HOME/pools_d${D}_p004_19M/data_d${D}_p0.004_r${ROUNDS}_FORMAL.npz}"

# Experiment 13 partitions. Changing these breaks comparability with the p=0.004 corpus.
VAL_START="${VAL_START:-15000000}"        # early stopping, checkpoint selection
VAL_N="${VAL_N:-200000}"
EVAL_START="${EVAL_START:-15200000}"      # scored once from the best checkpoint
EVAL_N="${EVAL_N:-1800000}"
SEALED_START="${SEALED_START:-17000000}"  # [17M, 19M) sealed

# PARALLEL=1 runs every seed at once on the one GPU. Each process holds ~45 GB host RAM at
# 10M shots, so three concurrent seeds need ~135 GB of the box's 572 GB.
PARALLEL="${PARALLEL:-0}"

# Stop below this much free GPU memory. sglang holds ~93.6 of 97.9 GB when up.
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"

# Per-process GPU memory cap, MiB. Unset means TensorFlow grows into whatever is free and
# keeps it: on the 97,871 MiB card a d=7 process settles at ~17,000 MiB although it needs
# 1-2 GB, and the sixth concurrent process dies on its first allocation. Set this when
# running more than four trainings at once.
GPU_MEM_MIB="${GPU_MEM_MIB:-}"

# Set before TensorFlow imports. podman does not inherit host exports.
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
# Allocate GPU memory on demand; the default reserves nearly the whole card per process.
export TF_FORCE_GPU_ALLOW_GROWTH=true

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$HOME/rcnn_threshold/results/gh200_${STUDENT}_d${D}_p004_b${BATCH}_${STAMP}}"
LOG="$OUT/driver.log"        # preflight and scheduling; per-seed logs sit in $OUT/seed<N>/
mkdir -p "$OUT"

# Inside a container, $HOME is /root and the default OUT lands on the container's own
# filesystem, which podman --rm deletes on exit. Require OUT on a bind mount, which has a
# different device number than /.
if [ -f /run/.containerenv ] || [ -f /.dockerenv ]; then
  if [ "$(stat -c %d "$OUT")" = "$(stat -c %d /)" ] && [ -z "${ALLOW_EPHEMERAL_OUT:-}" ]; then
    echo "[gru] OUT=$OUT is on the container filesystem and dies with the container."
    echo "[gru] Pass OUT=<path under a -v bind mount>, e.g. /rt/results/<name>. STOP."
    exit 1
  fi
fi

PY="${PY:-python3}"

exec > >(tee -a "$LOG") 2>&1

case "$STUDENT" in
  gru) ;;
  mlp) [ -n "$HIDDEN" ] || { echo "[gru] STUDENT=mlp needs HIDDEN, e.g. \"160 160\". STOP."; exit 1; } ;;
  *)   echo "[gru] STUDENT=$STUDENT; expected gru or mlp. STOP."; exit 1 ;;
esac

[ -f "$POOL" ] || { echo "[gru] missing pool $POOL. STOP."; exit 1; }

# Check GPU memory - nvidia-smi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu \
             --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
    || echo "[gru] no compute processes reported"
  FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${FREE_MIB:-0}" -lt "$MIN_FREE_MIB" ]; then
    echo "[gru] ${FREE_MIB} MiB free, need ${MIN_FREE_MIB}. Something still holds the"
    echo "[gru] card (usually sglang). Stop it, confirm with nvidia-smi, rerun. STOP."
    exit 1
  fi
  echo "[gru] ${FREE_MIB} MiB free: card available"
else
  echo "[gru] no nvidia-smi in this container; cannot verify the card. STOP."
  [ "$MIN_FREE_MIB" -gt 0 ] && exit 1
fi

# Check TensorFlow version, GPU execution, and the decoding libraries
"$PY" - <<'PYENV'
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
assert tf.__version__.startswith('2.15'), f'need TF 2.15.x (Keras 2); got {tf.__version__}'
gpus = tf.config.list_physical_devices('GPU')
print(f"[gru] TensorFlow {tf.__version__}  GPUs: {gpus}")
if not gpus:
    raise SystemExit("[gru] no GPU visible. Check the container was started with "
                     "--device nvidia.com/gpu=all --security-opt=label=disable. STOP.")
with tf.device('/GPU:0'):                    # run an op; enumeration alone proves little
    a = tf.fill((1024, 1024), 0.001)         # deterministic input; random ops need a seed
    assert float(tf.reduce_sum(tf.matmul(a, a))) > 0.0
print("[gru] GPU matmul ok")
import pymatching, stim, numpy as np
print(f"[gru] stim {stim.__version__}  numpy {np.__version__}  pymatching ok")
PYENV

# Check partition bounds against the pool's real length
"$PY" - "$POOL" "$NTRAIN" "$VAL_START" "$VAL_N" "$EVAL_START" "$EVAL_N" "$SEALED_START" \
<<'PYCHECK'
import sys, numpy as np
pool, ntr, v0, vn, e0, en, sealed = sys.argv[1], *map(int, sys.argv[2:])
z = np.load(pool, mmap_mode='r')
N = z['det_evts'].shape[0]
v1, e1 = v0 + vn, e0 + en
print(f"[gru] pool {pool}")
print(f"[gru] shots={N:,}  det_evts{z['det_evts'].shape}  "
      f"measurements{z['measurements'].shape}")
assert N >= sealed, f"pool holds {N:,} shots, sealed block starts at {sealed:,}"
assert ntr <= v0, f"training prefix [0, {ntr:,}) overlaps validation at {v0:,}"
assert v1 <= e0, f"validation [{v0:,}, {v1:,}) overlaps evaluation at {e0:,}"
assert e1 <= sealed, f"evaluation [{e0:,}, {e1:,}) runs into sealed test at {sealed:,}"
print(f"[gru] train [0, {ntr:,})  val [{v0:,}, {v1:,})  eval [{e0:,}, {e1:,})  "
      f"sealed [{sealed:,}, {N:,})  all disjoint")
PYCHECK

# Pool provenance. The fingerprint written at generation carries the generation seed and
# the flips SHA; mismatched geometry means this is a different experiment.
"$PY" - "$POOL" "$D" "$ROUNDS" "$P" "$OUT/pool_provenance.json" <<'PYPROV'
import json, os, sys
pool, d, r = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
p, out = float(sys.argv[4]), sys.argv[5]
fp = pool.replace('.npz', '.fingerprint.json')
if not os.path.exists(fp):
    raise SystemExit(f"[gru] no fingerprint beside {pool}; provenance is required. STOP.")
m = json.load(open(fp))
if (m['d'], m['rounds'], float(m['p'])) != (d, r, p):
    raise SystemExit(f"[gru] fingerprint says d={m['d']} r={m['rounds']} p={m['p']}, "
                     f"this run asks for d={d} r={r} p={p}. STOP.")
print(f"[gru] pool gen_seed={m['gen_seed']}  flips_sha={m['flips_sha256'][:16]}  "
      f"n_total={m['n_total']:,}")
json.dump(m, open(out, 'w'), indent=2)
PYPROV

# Print the measured layout and parameter count
"$PY" - "$STUDENT" "$UNITS" "$D" "$ROUNDS" "$P" $HIDDEN <<'PYPARAM'
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''      # shape check, no GPU needed
student, units = sys.argv[1], int(sys.argv[2])
d, r, p = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
hidden = tuple(int(h) for h in sys.argv[6:])
from StudentModels import build_student, detector_sequence_layout, input_width
if student == 'gru':
    n_t, n_pos, _ = detector_sequence_layout(d, r, p)
    print(f"[gru] sequence layout: {n_t} timesteps x {n_pos} positions")
else:
    print(f"[gru] flat input width: {input_width(d, r, 'evts')}")
m = build_student(student, d=d, rounds=r, inputs='evts', hidden=hidden, units=units)
print(f"[gru] {student} units={units} hidden={hidden}  params={m.count_params():,}")
PYPARAM

PROV=$("$PY" -c "import json;m=json.load(open('$OUT/pool_provenance.json'));\
print(f\"gen_seed={m['gen_seed']} flips_sha={m['flips_sha256']} n_total={m['n_total']}\")")
POOL_N=$("$PY" -c "import json;print(json.load(open('$OUT/pool_provenance.json'))['n_total'])")

# The four blocks as literal shot indices, printed before every run and again per seed.
VAL_END=$((VAL_START + VAL_N))
EVAL_END=$((EVAL_START + EVAL_N))
printf -v GEOMETRY '%s\n%s\n%s\n%s' \
  "train        : [0, $NTRAIN)" \
  "validation   : [$VAL_START, $VAL_END)" \
  "evaluation   : [$EVAL_START, $EVAL_END)" \
  "sealed test  : [$SEALED_START, $POOL_N)   not read"

{
  echo "experiment   : $STUDENT distance scaling, d=$D r=$ROUNDS p=$P (grace1 GH200)"
  echo "launched     : $(date -u '+%Y-%m-%dT%H:%M:%SZ')  on $(hostname)"
  echo "pool         : $POOL"
  echo "pool prov    : $PROV"
  echo "$GEOMETRY"
  echo "recipe       : $STUDENT units=$UNITS hidden=($HIDDEN) inputs=evts alpha=1.0 hard labels"
  echo "             : batch=$BATCH lr=$LR epochs=$EPOCHS early_stop=$EARLY_STOP"
  echo "gpu cap      : ${GPU_MEM_MIB:-none} MiB per process"
  echo "quantization : weights=${W_BITS:-float32} activations=${A_BITS:-float32}"
  echo "selection    : min val_loss checkpoint, scored once on the evaluation block"
  if [ "$STUDENT" = "mlp" ]; then
    echo "reading      : size-matched feed-forward comparison against the GRU at this"
    echo "             : distance. A gap measures that the architectures degrade"
    echo "             : differently with distance; the cause is a separate question."
  fi
  echo "seeds        : $SEEDS"
  echo "determinism  : TF_DETERMINISTIC_OPS=$TF_DETERMINISTIC_OPS "\
       "TF_CUDNN_DETERMINISTIC=$TF_CUDNN_DETERMINISTIC"
  echo "code sha     : $(sha256sum train_student.py | cut -c1-16)  train_student.py"
  echo "             : $(sha256sum StudentModels.py | cut -c1-16)  StudentModels.py"
  echo "             : $(sha256sum "$0" | cut -c1-16)  $(basename "$0")"
} | tee "$OUT/MANIFEST.txt"
echo

QUIET='cuda_|Unable to register|^Total number|^Number of unique'

# Architecture flags, identical between the training call and the scoring call so the
# model rebuilt for scoring is the one that was trained.
if [ "$STUDENT" = "gru" ]; then
  ARCH=(--student gru --inputs evts --hidden --units "$UNITS")
else
  ARCH=(--student mlp --inputs evts --hidden $HIDDEN)
fi

if [ -n "$GPU_MEM_MIB" ]; then
  ARCH+=(--gpu-mem-mib "$GPU_MEM_MIB")
fi

if [ -n "$W_BITS" ]; then ARCH+=(--weight-bits "$W_BITS"); fi
if [ -n "$A_BITS" ]; then ARCH+=(--act-bits "$A_BITS"); fi

if [ "$EARLY_STOP" = "1" ]; then
  STOPPING=(--patience "$PATIENCE")
else
  STOPPING=(--no-early-stopping)
fi

# Train one seed, then score it on the evaluation block. Each seed owns its directory in
# both modes, so sequential and concurrent runs produce the same layout.
run_seed () {
  local SEED="$1"
  local SIZE; [ "$STUDENT" = "gru" ] && SIZE="u${UNITS}" || SIZE="h$(echo $HIDDEN | tr ' ' '-')"
  # Float runs keep their original tag, so finished results still match the skip guard.
  local QTAG=""
  [ -n "$W_BITS$A_BITS" ] && QTAG="_w${W_BITS:-f32}_a${A_BITS:-f32}"
  local TAG="${STUDENT}_d${D}_p004_${SIZE}${QTAG}_hard_seed${SEED}_ntr${NTRAIN}_b${BATCH}"
  local SEEDDIR="$OUT/seed${SEED}"
  local SRUNS="$SEEDDIR/runs"
  local SCKPT="$SEEDDIR/ckpt"
  local SLOG="$SEEDDIR/run.log"
  mkdir -p "$SRUNS" "$SCKPT"

  if [ -f "$SRUNS/$TAG.csv" ] && [ -z "${ALLOW_OVERWRITE:-}" ]; then
    echo "[gru] seed $SEED: result row exists, skipping"   # results are append-only
    return 0
  fi

  {
    echo "=============== d=$D r=$ROUNDS seed $SEED  ($TAG) ==============="
    echo "$GEOMETRY"
    echo "pool         : $POOL"
    echo "pool prov    : $PROV"
    date -u '+%Y-%m-%dT%H:%M:%SZ start'
    # --tail-diagnostics stays off; it scores the evaluation block every epoch.
    "$PY" train_student.py "${ARCH[@]}" "${STOPPING[@]}" \
      --d "$D" --p "$P" --rounds "$ROUNDS" \
      --pool "$POOL" \
      --n-train "$NTRAIN" \
      --val-start "$VAL_START" --val-n "$VAL_N" \
      --eval-start "$EVAL_START" --n-test "$EVAL_N" \
      --alpha 1.0 --temperature 1.0 \
      --batch-size "$BATCH" --lr "$LR" \
      --epochs "$EPOCHS" \
      --seed "$SEED" --require-determinism \
      --ckpt-dir "$SCKPT" --out-dir "$SRUNS" --tag "$TAG" --run-tag "$TAG"

    # Ratio of record: MWPM decoded on these shots, paired shot by shot.
    # train_student.py suppresses its own MWPM column under --eval-start.
    # Scores the min-val_loss checkpoint, never the final-epoch weights.
    "$PY" eval_student_on_tail.py "${ARCH[@]}" \
      --weights "$SCKPT/$TAG.best.weights.h5" \
      --d "$D" --p "$P" --rounds "$ROUNDS" \
      --pool "$POOL" --eval-start "$EVAL_START" --n-test "$EVAL_N" \
      --batch-size "$BATCH" \
      --dump-per-shot "$SEEDDIR/per_shot_${TAG}.npz" \
      --out-csv "$SEEDDIR/eval_best_ckpt.csv"
    date -u '+%Y-%m-%dT%H:%M:%SZ end'
  } 2>&1 | grep --line-buffered -Ev "$QUIET" > "$SLOG"
}

T0=$(date +%s)
if [ "$PARALLEL" = "1" ]; then
  echo "[gru] launching seeds concurrently: $SEEDS"
  declare -A PIDS
  for SEED in $SEEDS; do
    run_seed "$SEED" &
    PIDS[$SEED]=$!
    echo "[gru]   seed $SEED -> pid ${PIDS[$SEED]}  log $OUT/seed${SEED}/run.log"
  done
  rc=0
  for SEED in $SEEDS; do
    if wait "${PIDS[$SEED]}"; then
      echo "[gru] seed $SEED finished"
    else
      echo "[gru] seed $SEED FAILED, see $OUT/seed${SEED}/run.log"; rc=1
    fi
  done
else
  for SEED in $SEEDS; do
    echo "[gru] seed $SEED -> log $OUT/seed${SEED}/run.log"
    run_seed "$SEED" || { echo "[gru] seed $SEED failed"; exit 1; }
  done
  rc=0
fi
WALL=$(( $(date +%s) - T0 ))

echo
echo "[gru] done  d=$D r=$ROUNDS  seeds: $SEEDS  parallel=$PARALLEL  wall ${WALL}s"
echo "[gru]   per-seed output -> $OUT/seed<N>/{runs,ckpt,run.log,eval_best_ckpt.csv}"
grep -h '' "$OUT"/seed*/eval_best_ckpt.csv 2>/dev/null | sort -u | head -20
exit $rc
