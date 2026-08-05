#!/usr/bin/env bash
# Created: 2026-08-05
# Last modified: 2026-08-05
#
# One-screen status for the EAF pods. Run it on any server.
#
# There is no cross-node view of the GPUs: each pod sees only its own device, and
# nvidia-smi's memory fields are permission-blocked, so "is that other server busy" cannot
# be answered directly. But $HOME is shared, so a log file that is still growing means its
# job is alive on whichever node launched it. That is the signal this uses.
#
# Sections:
#   THIS SERVER   hostname, free GPU memory, and the jobs running locally
#   ALL LOGS      every r5 log on shared home, how long since it was written, last line
#   INVENTORY     which (rung, seed) runs actually exist, per architecture
#
# Free GPU memory is the number that has repeatedly decided whether a run succeeds:
# ~38566 MB means the slice is free, ~1056 MB means something else already holds it and a
# launch will die with "Dst tensor is not initialized".
set -uo pipefail

RT="${RT:-$HOME/rcnn_threshold}"
PY="${PY:-$HOME/QuantumDecoderQKeras/.venv/bin/python}"
FRESH_SEC="${FRESH_SEC:-180}"   # a log touched within this many seconds counts as live

echo "=============================================================="
echo " THIS SERVER: $(hostname)    $(date -u '+%Y-%m-%d %H:%M:%SZ')"
echo "=============================================================="

if [ -x "$PY" ]; then
  mem=$("$PY" -c "import tensorflow as tf; tf.random.normal([10,10])" 2>&1 \
        | grep -o "with [0-9]* MB memory" | grep -o "[0-9]*")
  if [ -n "${mem:-}" ]; then
    if [ "$mem" -gt 30000 ]; then
      echo "  GPU free: ${mem} MB   <- slice reads FREE"
      echo "                             (snapshot: a job still loading data has not"
      echo "                              grabbed the GPU yet -- check jobs below too)"
    else
      echo "  GPU free: ${mem} MB   <- BUSY. Something holds this GPU; a launch will OOM."
    fi
  else
    echo "  GPU free: (could not read -- is TF importable?)"
  fi
else
  echo "  GPU free: (no venv at $PY)"
fi

echo
echo "  jobs on THIS node:"
ps -u "$USER" -o pid,etime,rss,cmd --sort=-rss 2>/dev/null \
  | grep -E "[t]rain_one\.py|[t]rain_student\.py" \
  | awk '{printf "    pid=%-8s up=%-10s rss=%6.1fGB  ", $1, $2, $3/1048576;
          for(i=4;i<=NF;i++) printf "%s ", $i; print ""}' \
  | sed 's/--pool [^ ]*//' \
  || echo "    (none)"
ps -u "$USER" -o pid= -o cmd= 2>/dev/null | grep -qE "[t]rain_one\.py|[t]rain_student\.py" \
  || echo "    (none)"

echo
echo "=============================================================="
echo " ALL r5 LOGS ON SHARED HOME  (live = written in last ${FRESH_SEC}s)"
echo "=============================================================="
now=$(date +%s)
shopt -s nullglob
for f in "$HOME"/r5_*.log; do
  age=$(( now - $(stat -c %Y "$f" 2>/dev/null || echo "$now") ))
  if [ "$age" -lt "$FRESH_SEC" ]; then state="LIVE "; else state="idle "; fi
  last=$(grep -E "^\[(train|gru|mlp)\]|Epoch [0-9]+/|Error|Traceback" "$f" 2>/dev/null \
         | tail -1 | cut -c1-88)
  printf "  %s %-34s %6ss ago  %s\n" "$state" "$(basename "$f")" "$age" "$last"
done

echo
echo "=============================================================="
echo " RUN INVENTORY"
echo "=============================================================="
echo "  teacher (rung_seed present in val_scores_best_ckpt.csv):"
if [ -f "$RT/out_r5_ladder/val_scores_best_ckpt.csv" ]; then
  grep -o 'ntr[0-9]*_seed[0-9]' "$RT/out_r5_ladder/val_scores_best_ckpt.csv" \
    | sort -u | sed 's/^/    /' | paste -sd' ' -
  echo "    unique: $(grep -o 'ntr[0-9]*_seed[0-9]' \
    "$RT/out_r5_ladder/val_scores_best_ckpt.csv" | sort -u | wc -l) / 18  (6 rungs x 3 seeds; the 10M rung lives in out_r5_teacher/)"
else
  echo "    (no val_scores_best_ckpt.csv yet)"
fi

echo
echo "  students (per-run csv files):"
# find, not ls: `shopt -s nullglob` makes an unmatched glob expand to nothing, and
# `ls` with no arguments then lists the whole directory -- which reported 108 files as
# "108 runs at 20M". find takes a pattern and returns nothing when nothing matches.
for arch in gru mlp; do
  n=$(find "$RT/out_student_r5_hard" -maxdepth 1 -name "r5hard_${arch}_ntr*_lr*.csv" 2>/dev/null | wc -l)
  n20=$(find "$RT/out_student_r5_hard" -maxdepth 1 -name "r5hard_${arch}_ntr20000000_*.csv" 2>/dev/null | wc -l)
  printf "    %-4s %2d runs total (want 21),  %d at 20M (want 3)\n" "${arch}:" "$n" "$n20"
done
echo
