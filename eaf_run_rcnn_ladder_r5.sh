#!/usr/bin/env bash
# Created: 2026-08-04
# Last modified: 2026-08-04
#
# RCNN data-volume ladder at d=5, r=5. Run on EAF.
#
# The 10M rung is already done (out_r5_teacher/seed{0,1,2}); this covers the rest.
# Every run uses the formal pool's partitions: gradient updates from [0, n_train),
# validation [20.0M, 20.2M), sealed test [20.2M, 20.4M) never read.
#
# Split across servers with RUNGS and SEEDS, so each machine takes a slice of the work
# and writes to its own per-rung, per-seed directory. Nothing is shared but the pool.
#
#   RUNGS="20000000" SEEDS="0 1"          bash eaf_run_rcnn_ladder_r5.sh
#   RUNGS="100000 300000 800000 2000000 5000000" SEEDS="0 1 2" bash eaf_run_rcnn_ladder_r5.sh
#
# Cost, extrapolated from the 10M rung's ~4.7 h: roughly linear in n_train, so 20M is the
# long pole at ~9 h per seed and everything below 5M is under an hour.
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
POOL="$RT/pools_r5/data_d5_p0.010_r5_FORMAL.npz"
VAL_START=20000000
VAL_N=200000
OUTROOT="${OUTROOT:-$RT/out_r5_ladder}"
RUNGS="${RUNGS:-100000 300000 800000 2000000 5000000 20000000}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-5}"
PY="${PY:-python}"

[ -f "$POOL" ] || { echo "MISSING $POOL"; exit 1; }

n=0; for r in $RUNGS; do for s in $SEEDS; do n=$((n+1)); done; done
echo "RCNN r=5 ladder: $n runs"
echo "  rungs=$RUNGS"
echo "  seeds=$SEEDS"
echo "  pool=$POOL"
echo "  validation [$VAL_START, $((VAL_START+VAL_N)))   sealed test not read"
echo "  out=$OUTROOT"
echo

i=0
for ntr in $RUNGS; do
  for s in $SEEDS; do
    i=$((i+1))
    tag="r5_ladder_ntr${ntr}_seed${s}"
    out="$OUTROOT/ntr${ntr}/seed${s}"
    echo "=============== [$i/$n] $tag ==============="
    mkdir -p "$out/ckpt"
    $PY train_one.py --d 5 --p 0.010 --rounds 5 --seed "$s" \
      --pool "$POOL" \
      --n-train "$ntr" --n-test "$VAL_N" \
      --val-start "$VAL_START" --val-n "$VAL_N" --seal-test \
      --epochs "$EPOCHS" --batch-size 10000 --patience "$PATIENCE" --save-weights \
      --out-dir "$out" --ckpt-dir "$out/ckpt" --run-tag "$tag" 2>&1 \
      | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique"

    # Score the .best checkpoint explicitly. The run's own p_L comes from whatever weights
    # were in memory when fit() returned, which is the restored best only if early
    # stopping fired -- so it is not comparable across runs. This is.
    $PY eval_on_tail.py --weights "$out/ckpt/${tag}.best.weights.h5" \
      --d 5 --p 0.010 --rounds 5 --n-test "$VAL_N" --eval-start "$VAL_START" \
      --pool "$POOL" --out-csv "$OUTROOT/val_scores_best_ckpt.csv" 2>&1 \
      | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique"
    echo
  done
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SUMMARY="$RT/rcnn_r5_ladder_${STAMP}.csv"
echo "=============================================================="
$PY - "$OUTROOT" "$SUMMARY" <<'PYEOF'
import csv, glob, os, re, sys, statistics as st, collections
root, out = sys.argv[1], sys.argv[2]
# Read the re-scored .best results, not the per-run training CSVs.
rows = []
for r in csv.DictReader(open(os.path.join(root, 'val_scores_best_ckpt.csv'))):
    m = re.search(r'ntr(\d+)_seed(\d)', r['weights'])
    if m:
        r['n_train'], r['seed'] = m.group(1), m.group(2)
        rows.append(r)
if rows:
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
g = collections.defaultdict(list)
for r in rows:
    g[int(r['n_train'])].append(float(r['p_L']))
MW = 0.084215   # MWPM on this pool's validation partition
print(f"  {'n_train':>11} {'seeds':>6} {'val p_L':>9} {'std':>8} {'xMWPM':>7}")
for k in sorted(g):
    v = g[k]
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f"  {k:>11,} {len(v):>6} {st.mean(v):>9.5f} {sd:>8.5f} {st.mean(v)/MW:>7.3f}")
print(f"\n  MWPM on validation = {MW}.  Every value is the .best checkpoint re-scored on")
print("  the validation partition. All three seeds are kept at every rung: report mean and")
print("  spread, not a per-rung median. Seed 2 is the fixed teacher for distillation only.")
print(f"  {len(rows)} runs -> {out}")
PYEOF
