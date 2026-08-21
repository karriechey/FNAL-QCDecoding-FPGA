#!/usr/bin/env bash
# Created: 2026-08-20
# Last updated: 2026-08-20
#
# Unattended overnight queue on the GH200. Goal: the best float GRU at d=5, plus the
# capacity and data diagnostics at d=7 and d=9.
#
#     bash gh200_overnight_queue.sh --preflight    # one tiny end-to-end job, then stop
#     bash gh200_overnight_queue.sh                # the full queue
#
# Three lanes run at once, each sequential inside itself. A failing job never stops another
# lane, and every job writes its own timestamped directory. Pools mount read-only.
#
#   lane A   d=5, the target: 15M shots, width probes, then one adaptive follow-up
#   lane B   d=7: 15M shots, then the units=280 probe
#   lane C   d=9: units=280 seeds 1/2, gated on lane A's first job finishing
#
# Every decision this script makes is pre-registered below, reads only COMPLETE runs, and
# is logged with the numbers it used.
set -uo pipefail          # no -e: a failing job must not take the queue down

REPO="${REPO:-$HOME/FNAL-QCDecoding-FPGA}"
RT="${RT:-$HOME/rcnn_threshold}"
IMG="${IMG:-qdec:tf215}"
QLOG="${QLOG:-$HOME/overnight_queue.log}"
STATE="${STATE:-$HOME/queue_state}"
mkdir -p "$STATE"

D5DIR="$HOME/pools_d5_p004_19M"; D5POOL=/pools/data_d5_p0.004_r5_FORMAL.npz
D7DIR="$HOME/pools_d7_p004_19M"; D7POOL=/pools/data_d7_p0.004_r7_FORMAL.npz
D9DIR="$HOME/pools_d9_p004_19M"; D9POOL=/pools/data_d9_p0.004_r9_FORMAL.npz

# --- pre-registered numbers -------------------------------------------------------------
# Width gate. sigma_seed = 0.000191 is the three-seed standard deviation of the d=5 GRU
# units=140 10M run of record (0.008499 / 0.008416 / 0.008780) -- the only d=5 GRU config
# with three seeds, same distance, protocol, data volume and evaluation block, so it is the
# right estimate of training-seed variability for this comparison. A probe is one seed and
# the reference is a three-seed mean, so the difference has
#     sigma_diff = sqrt(sigma^2 + sigma^2/3) = 0.000221
# and the gate is two of those below the reference mean 0.008565:
U140_MEAN=0.008565
WIDTH_GATE=0.008124
# MWPM at d=5 on [15.2M, 17.0M), and the flag point at two evaluation standard errors
# above it (SE = sqrt(p(1-p)/1.8e6) = 0.000068 at p ~ 0.0085).
D5_MWPM=0.007571
D5_NEAR_MWPM=0.007707

log () { echo "[queue $(date -u '+%H:%M:%SZ')] $*" | tee -a "$QLOG" >&2; }

# --- one job ----------------------------------------------------------------------------
# Pools are mounted read-only: a training container must not be able to alter a formal pool
# or its fingerprint. Prints only the output directory on stdout, so callers can capture it.
run_job () {          # run_job <name> <pooldir> <pool> <hbm_mib> <env...>
  local name="$1" pooldir="$2" pool="$3" hbm="$4"; shift 4
  local out="/rt/results/${name}_$(date -u +%Y%m%dT%H%M%SZ)"
  log "START $name -> $out"
  local t0=$(date +%s)
  podman run --rm --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$REPO":/work:z -v "$pooldir":/pools:z,ro -v "$RT":/rt:z -w /work "$IMG" \
    env "$@" POOL="$pool" GPU_MEM_MIB="$hbm" MIN_FREE_MIB=8000 OUT="$out" \
    bash gh200_run_student_scaling.sh >> "$HOME/${name}.log" 2>&1
  local rc=$? dt=$(( $(date +%s) - t0 ))
  local host_out="${out/\/rt/$RT}"
  if [ $rc -eq 0 ] && [ -f "$host_out/COMPLETE" ]; then
    log "DONE  $name in ${dt}s"
  else
    log "FAIL  $name rc=$rc after ${dt}s, COMPLETE absent (queue continues)"
  fi
  echo "$out"
}

# Mean p_L over COMPLETE seeds of one run directory, for the exact tag pattern given.
# Never averages unrelated rows: it matches the checkpoint basename exactly.
mean_pL () {          # mean_pL <container_out> <tag_glob>
  local host_out="${1/\/rt/$RT}" tagglob="$2"
  python3 - "$host_out" "$tagglob" <<'PYEOF' 2>/dev/null
import csv, fnmatch, glob, os, sys
root, tagglob = sys.argv[1], sys.argv[2]
vals = []
for seed_dir in sorted(glob.glob(os.path.join(root, 'seed*'))):
    if not os.path.exists(os.path.join(seed_dir, 'COMPLETE')):
        continue                      # incomplete seeds never enter a decision
    ev = os.path.join(seed_dir, 'eval_best_ckpt.csv')
    if not os.path.exists(ev):
        continue
    for r in csv.DictReader(open(ev)):
        w = r['weights']
        if w.endswith('.best.weights.h5') and fnmatch.fnmatch(w[:-len('.best.weights.h5')], tagglob):
            vals.append(float(r['p_L']))
print(f"{sum(vals)/len(vals):.6f} {len(vals)}" if vals else "")
PYEOF
}

# Paired McNemar between two per-shot dumps on their shared evaluation shots.
pair_report () {      # pair_report <out_a> <tag_a> <out_b> <tag_b> <label>
  local a="${1/\/rt/$RT}/seed0/per_shot_${2}.npz"
  local b="${3/\/rt/$RT}/seed0/per_shot_${4}.npz"
  [ -f "$a" ] && [ -f "$b" ] || { log "pair_report $5: per-shot dumps missing"; return; }
  log "paired McNemar $5"
  podman run --rm --security-opt=label=disable -v "$REPO":/work:z -v "$RT":/rt:z \
    -w /work "$IMG" python3 pair_two_decoders.py \
      --a "${a/$RT/\/rt}" --b "${b/$RT/\/rt}" --a-name "$2" --b-name "$4" \
      --out-csv "/rt/results/paired_${5}.csv" 2>&1 | grep -Ev "cuda_|Unable to register" \
      | tee -a "$QLOG"
}

# --- preflight --------------------------------------------------------------------------
# One tiny job down the exact overnight path, then every artifact a later decision depends
# on is checked. Nothing else runs until this passes.
preflight () {
  log "=== pool assertions ==="
  local ok=1
  podman run --rm -v "$REPO":/work:z -v "$D5DIR":/pools:z,ro -w /work "$IMG" \
    python3 preflight_pools.py --pool "$D5POOL" --d 5 --rounds 5 --p 0.004 \
    --baseline /pools/mwpm_eval_block_15p2M_17M.json 2>&1 | tee -a "$QLOG" || ok=0
  podman run --rm -v "$REPO":/work:z -v "$D7DIR":/pools:z,ro -w /work "$IMG" \
    python3 preflight_pools.py --pool "$D7POOL" --d 7 --rounds 7 --p 0.004 \
    --baseline /pools/mwpm_eval_block_15p2M_17M.json 2>&1 | tee -a "$QLOG" || ok=0
  podman run --rm -v "$REPO":/work:z -v "$D9DIR":/pools:z,ro -w /work "$IMG" \
    python3 preflight_pools.py --pool "$D9POOL" --d 9 --rounds 9 --p 0.004 \
    --baseline /pools/mwpm_eval_block_15p2M_17M.json 2>&1 | tee -a "$QLOG" || ok=0

  log "=== learning-rate semantics ==="
  # The 400-epoch branch is only a clean "same training, longer" test if the rate is
  # constant. train_student.py attaches its epoch-indexed scheduler ONLY when --lr is
  # absent, and this launcher always passes LR, so epochs 1-200 are identical either way.
  grep -n "if args.lr is None" -A 2 "$REPO/train_student.py" | tee -a "$QLOG"

  log "=== end-to-end 100k job ==="
  local out
  out=$(run_job preflight_d5_100k "$D5DIR" "$D5POOL" 8000 \
        D=5 ROUNDS=5 STUDENT=gru UNITS=140 SEEDS="0" EPOCHS=3 NTRAIN=100000 \
        EVAL_N=50000 PARALLEL=1 | tail -1)
  local host="${out/\/rt/$RT}"
  python3 - "$host" <<'PYEOF' 2>&1 | tee -a "$QLOG"
import csv, glob, json, os, sys
root = sys.argv[1]
tag = 'gru_d5_p004_u140_hard_seed0_ntr100000_b10000'
checks = {
    'run COMPLETE':        os.path.exists(os.path.join(root, 'COMPLETE')),
    'seed COMPLETE':       os.path.exists(os.path.join(root, 'seed0', 'COMPLETE')),
    'best checkpoint':     os.path.exists(os.path.join(root, 'seed0', 'ckpt', tag + '.best.weights.h5')),
    'history 3 epochs':    False,
    'exact eval row':      False,
    'per-shot dump':       os.path.exists(os.path.join(root, 'seed0', f'per_shot_{tag}.npz')),
    'pool provenance':     os.path.exists(os.path.join(root, 'pool_provenance.json')),
}
h = os.path.join(root, 'seed0', 'runs', tag + '.history.json')
if os.path.exists(h):
    checks['history 3 epochs'] = len(json.load(open(h))['val_loss']) == 3
ev = os.path.join(root, 'seed0', 'eval_best_ckpt.csv')
if os.path.exists(ev):
    rows = [r for r in csv.DictReader(open(ev)) if r['weights'] == tag + '.best.weights.h5']
    checks['exact eval row'] = len(rows) == 1
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
sys.exit(0 if all(checks.values()) else 1)
PYEOF
  local artifacts=$?
  log "=== analyzer sees it ==="
  python3 "$REPO/analyze_gru_histories.py" --results "$RT/results" \
    --pattern "preflight_d5_100k_*" --out "$HOME/gru_diag_preflight" 2>&1 | tail -8 | tee -a "$QLOG"
  local analyzer=$?
  if [ "$ok" = "1" ] && [ $artifacts -eq 0 ] && [ $analyzer -eq 0 ]; then
    log "PREFLIGHT CLEAN"; return 0
  fi
  log "PREFLIGHT FAILED (pools=$ok artifacts=$artifacts analyzer=$analyzer)"; return 1
}

# --- lane A -----------------------------------------------------------------------------
lane_a () {
  local o15 o200 o280
  o15=$(run_job d5_u140_ntr15M "$D5DIR" "$D5POOL" 16000 \
        D=5 ROUNDS=5 STUDENT=gru UNITS=140 SEEDS="0 1 2" EPOCHS=200 NTRAIN=15000000 PARALLEL=1 | tail -1)
  # Sentinel for lane C: written only when that job left a run-level COMPLETE.
  if [ -f "${o15/\/rt/$RT}/COMPLETE" ]; then
    echo "$o15" > "$STATE/A1_COMPLETE"; log "A1_COMPLETE written"
  else
    echo "failed" > "$STATE/A1_FAILED"; log "A1 did not complete; sentinel A1_FAILED written"
  fi
  local p15; p15=$(mean_pL "$o15" 'gru_d5_p004_u140_hard_seed*_ntr15000000_b10000')
  log "d=5 units=140 15M: p_L=${p15:-none}  (10M reference $U140_MEAN, MWPM $D5_MWPM)"

  o200=$(run_job d5_u200_ntr10M "$D5DIR" "$D5POOL" 12000 \
         D=5 ROUNDS=5 STUDENT=gru UNITS=200 SEEDS="0" EPOCHS=200 NTRAIN=10000000 PARALLEL=1 | tail -1)
  o280=$(run_job d5_u280_ntr10M "$D5DIR" "$D5POOL" 14000 \
         D=5 ROUNDS=5 STUDENT=gru UNITS=280 SEEDS="0" EPOCHS=200 NTRAIN=10000000 PARALLEL=1 | tail -1)

  local p200 p280 best_u best_p best_out
  p200=$(mean_pL "$o200" 'gru_d5_p004_u200_hard_seed0_ntr10000000_b10000' | cut -d' ' -f1)
  p280=$(mean_pL "$o280" 'gru_d5_p004_u280_hard_seed0_ntr10000000_b10000' | cut -d' ' -f1)
  log "width probes: units200=${p200:-none} units280=${p280:-none}; gate $WIDTH_GATE "
  log "  (gate = $U140_MEAN - 2*sqrt(sigma^2 + sigma^2/3), sigma=0.000191 from the 3-seed u140 10M run)"

  best_u=""; best_p=""
  if [ -n "$p200" ] && [ -n "$p280" ]; then
    if python3 -c "import sys; sys.exit(0 if $p200 <= $p280 else 1)"; then
      best_u=200; best_p=$p200; best_out=$o200
    else best_u=280; best_p=$p280; best_out=$o280; fi
  elif [ -n "$p200" ]; then best_u=200; best_p=$p200; best_out=$o200
  elif [ -n "$p280" ]; then best_u=280; best_p=$p280; best_out=$o280
  fi
  [ -z "$best_u" ] && { log "no width probe completed; lane A stops"; return; }

  # Paired McNemar against units=140 seed 0 on the shared evaluation shots. Reported for
  # the record; the launch decision rests on the seed-variability gate, because a paired
  # test on one seed each says nothing about training-seed variability.
  local ref
  ref=$(ls -dt "$RT"/results/gh200_gru_d5_e200_2*/ 2>/dev/null | head -1)
  if [ -n "$ref" ]; then
    pair_report "/rt/results/$(basename "$ref")" \
      "gru_d5_p004_u140_hard_seed0_ntr10000000_b10000" \
      "$best_out" "gru_d5_p004_u${best_u}_hard_seed0_ntr10000000_b10000" \
      "u140_vs_u${best_u}_d5"
  fi

  if python3 -c "import sys; sys.exit(0 if $best_p < $WIDTH_GATE else 1)"; then
    log "units=$best_u clears the gate ($best_p < $WIDTH_GATE): replicate, then 15M"
    run_job "d5_u${best_u}_ntr10M_rep" "$D5DIR" "$D5POOL" 14000 \
      D=5 ROUNDS=5 STUDENT=gru UNITS="$best_u" SEEDS="0 1 2" EPOCHS=200 NTRAIN=10000000 PARALLEL=1 >/dev/null
    run_job "d5_u${best_u}_ntr15M" "$D5DIR" "$D5POOL" 20000 \
      D=5 ROUNDS=5 STUDENT=gru UNITS="$best_u" SEEDS="0 1 2" EPOCHS=200 NTRAIN=15000000 PARALLEL=1 >/dev/null
  else
    log "width gains at d=5 are inside the gate; consulting the learning curves instead"
    python3 "$REPO/analyze_gru_histories.py" --results "$RT/results" --out "$HOME/gru_diag" \
      >> "$QLOG" 2>&1
    # The epoch branch runs only when the diagnostics say a d=5 run is epoch-limited.
    if python3 - "$HOME/gru_diag/gru_history_summary.csv" <<'PYEOF'
import csv, os, sys
path = sys.argv[1]
if not os.path.exists(path):
    sys.exit(1)
rows = [r for r in csv.DictReader(open(path)) if r['d'] == '5' and r['verdict'] == 'epoch-limited']
print(f"[queue] d=5 runs classified epoch-limited: {len(rows)}")
sys.exit(0 if rows else 1)
PYEOF
    then
      log "diagnostics say epoch-limited -> 400-epoch run (constant LR, so epochs 1-200 are unchanged)"
      run_job d5_u140_ntr15M_e400 "$D5DIR" "$D5POOL" 16000 \
        D=5 ROUNDS=5 STUDENT=gru UNITS=140 SEEDS="0" EPOCHS=400 NTRAIN=15000000 PARALLEL=1 >/dev/null
    else
      log "no d=5 run is epoch-limited; skipping the epoch branch rather than burning the slot"
    fi
  fi
}

# --- lane B -----------------------------------------------------------------------------
lane_b () {
  run_job d7_u140_ntr15M "$D7DIR" "$D7POOL" 32000 \
    D=7 ROUNDS=7 STUDENT=gru UNITS=140 SEEDS="0 1 2" EPOCHS=200 NTRAIN=15000000 PARALLEL=1 >/dev/null
  run_job d7_u280_ntr10M "$D7DIR" "$D7POOL" 24000 \
    D=7 ROUNDS=7 STUDENT=gru UNITS=280 SEEDS="0" EPOCHS=200 NTRAIN=10000000 PARALLEL=1 >/dev/null
}

# --- lane C -----------------------------------------------------------------------------
# Waits on lane A's sentinel rather than on the clock. If A1 fails, lane C still runs, but
# only once the card actually has room -- A1 failing frees memory, so waiting forever would
# idle the GPU for no reason.
lane_c () {
  local waited=0 max=14400
  while [ ! -f "$STATE/A1_COMPLETE" ] && [ ! -f "$STATE/A1_FAILED" ] && [ $waited -lt $max ]; do
    sleep 60; waited=$((waited + 60))
    [ $((waited % 600)) -eq 0 ] && log "lane C waiting on A1 (${waited}s)"
  done
  if [ -f "$STATE/A1_COMPLETE" ]; then
    log "lane C: A1 complete"
  else
    log "lane C: no A1 sentinel after ${waited}s; continuing to the memory check"
  fi

  # Two d=9 seeds hold ~67 GB combined (33.4 GB each, set by the 29.8 GiB training array
  # that Keras copies to the device; the 40 GB cap is a ceiling, not a reservation). The
  # threshold is 80 GB rather than 67 so an unattended start keeps ~13 GB of margin.
  # This memory test is the launch condition -- the A1 sentinel only releases lane C into
  # it, and says nothing about lane A being idle, since lane A runs its width probes on.
  local need=80000 free waited2=0
  while [ $waited2 -lt 14400 ]; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "${free:-0}" -ge $need ] && break
    sleep 120; waited2=$((waited2 + 120))
    [ $((waited2 % 600)) -eq 0 ] && log "lane C waiting for ${need} MiB, ${free} MiB free"
  done
  if [ "${free:-0}" -lt $need ]; then
    log "lane C: only ${free} MiB free after ${waited2}s; running one seed instead of two"
    run_job d9_u280_ntr10M_seed1 "$D9DIR" "$D9POOL" 40000 \
      D=9 ROUNDS=9 STUDENT=gru UNITS=280 SEEDS="1" EPOCHS=200 NTRAIN=10000000 PARALLEL=1 >/dev/null
    return
  fi
  log "lane C: ${free} MiB free, starting both seeds"
  run_job d9_u280_ntr10M_rep "$D9DIR" "$D9POOL" 40000 \
    D=9 ROUNDS=9 STUDENT=gru UNITS=280 SEEDS="1 2" EPOCHS=200 NTRAIN=10000000 PARALLEL=1 >/dev/null
}

# --- main -------------------------------------------------------------------------------
if [ "${1:-}" = "--preflight" ]; then
  preflight; exit $?
fi

if [ ! -f "$STATE/PREFLIGHT_OK" ]; then
  log "no $STATE/PREFLIGHT_OK; run --preflight first, then touch that file. STOP."
  exit 1
fi

rm -f "$STATE/A1_COMPLETE" "$STATE/A1_FAILED"
log "queue starting; repo $REPO; results under $RT/results"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$QLOG"

lane_a & A=$!
lane_b & B=$!
lane_c & C=$!
wait $A $B $C
log "queue empty"

python3 "$REPO/analyze_gru_histories.py" --results "$RT/results" --out "$HOME/gru_diag" \
  2>&1 | tail -30 | tee -a "$QLOG"
log "sealed block [17M, 19M) untouched; final scoring is a manual step tomorrow"
