#!/usr/bin/env bash
# Created: 2026-07-31
# Last modified: 2026-07-31
#
# Re-score every saved RCNN ladder checkpoint on the SAME tail the students use, so all
# architectures can appear on one axis against one MWPM line.
# RUN ON EAF.
#
# Why: the ladder rungs were each scored against their own tail's MWPM (10M vs ~0.04875,
# 20M vs 0.04909 on the 20.2M pool's own tail, students vs 0.049405 on TAIL200k). Those
# are individually correct but not mutually comparable, so a single overlaid p_L-vs-N
# figure needs every point re-measured on one tail.
#
# Scoring needs only the checkpoint and the tail pool -- not the pool a model trained on.
# The 20M checkpoints re-score fine even though pools_20M is now empty.
#
# Inference only. No retraining. Every row carries mwpm_source=redecoded_on_tail.
set -euo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
TAIL_POOL="$RT/pools/data_d5_p0.010_r3_TAIL200k.npz"
OUT="$RT/rescored_ladder_TAIL200k_$(date -u +%Y%m%dT%H%M%SZ).csv"
NTE=200000
PY="${PY:-python}"

[ -f "$TAIL_POOL" ] || { echo "MISSING $TAIL_POOL"; exit 1; }
# Output carries a UTC stamp; nothing is ever removed or appended to in place.

# Every directory that may hold RCNN ladder checkpoints. out_t200k_w holds the 10M
# retrains, out_20M the 20M runs, out the lower rungs if they were saved.
mapfile -t WEIGHTS < <(ls "$RT"/out/*.weights.h5 \
                          "$RT"/out_t200k_w/*.weights.h5 \
                          "$RT"/out_20M/*.weights.h5 2>/dev/null | sort -u)

if [ "${#WEIGHTS[@]}" -eq 0 ]; then
  echo "No .weights.h5 found. train_one.py defaults to --no-save-weights, so the lower"
  echo "rungs may never have been saved; those rungs cannot be re-scored without retraining."
  exit 1
fi

echo "Re-scoring ${#WEIGHTS[@]} checkpoints on TAIL200k (MWPM re-decoded per run)."
echo

for w in "${WEIGHTS[@]}"; do
  echo "--- $(basename "$w") ---"
  $PY eval_on_tail.py \
    --weights "$w" \
    --d 5 --p 0.010 --rounds 3 --n-test "$NTE" \
    --pool "$TAIL_POOL" --mcnemar --out-csv "$OUT" \
    2>&1 | grep -Ev "cuda_|Unable to register|^Total number|^Number of unique"
done

echo
echo "=============================================================="
echo "All rungs on one tail:"
echo "=============================================================="
$PY - "$OUT" <<'PYEOF'
import csv, re, sys, collections, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
by_n = collections.defaultdict(list)
for r in rows:
    m = re.search(r'ntr(\d+)', r['weights'])
    if m:
        by_n[int(m.group(1))].append(float(r['p_L']))
mwpm = float(rows[0]['mwpm_p_L'])
print(f"  MWPM on TAIL200k (re-decoded): {mwpm:.6f}\n")
print(f"  {'n_train':>10} {'seeds':>6} {'mean p_L':>10} {'std':>9} {'xMWPM':>7}")
for n in sorted(by_n):
    v = by_n[n]
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f"  {n:>10,} {len(v):>6} {st.mean(v):>10.5f} {sd:>9.5f} {st.mean(v)/mwpm:>7.3f}")
print(f"\n  Written to {sys.argv[1]}")
print("  Every row has mwpm_source=redecoded_on_tail; these supersede any xMWPM computed")
print("  against a stored baseline.")
PYEOF
