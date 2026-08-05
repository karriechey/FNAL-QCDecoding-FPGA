#!/usr/bin/env python3
# Created: 2026-08-04
# Last modified: 2026-08-04
"""Parse the r=5 teacher run summary out of EAF nohup logs into a CSV.

Why this exists
The r=5 teacher runs live on EAF; what came back to the Mac was the raw training logs,
not the per-run .csv files train_one.py writes. Plotting from a log means either
hand-copying numbers into a notebook, which leaves them untraceable, or this: parse the
authoritative line once, write it to a CSV with the source file recorded, and have the
notebook read the CSV. Every number in the figure then traces to a file, as the repo's
documentation discipline requires.

What it reads
train_one.py's final summary line, e.g.

  [train] rcnn_d5_p0.010_r5_seed0_ntr10000000  RCNN p_L=0.09136  MWPM=0.08910
          gap=+0.00226  base_rate=0.355  (44 ep, 16639s, 69485 params)

plus the partition line and the checkpoint-identity line, which are needed to interpret
the result and are therefore carried into the CSV rather than dropped.

The MWPM value in that line is DELIBERATELY NOT COPIED. train_one.py's lookup_mwpm reads
the canonical mwpm_baseline.csv for (d, p, rounds), which for r=5 is 0.08910 -- computed
on the OLD r=5 pool's 10k tail. These runs scored on the FORMAL pool's validation
partition [20.0M, 20.2M), where MWPM is 0.084215. Baselines are not portable across
tails; carrying the printed value forward would silently attach the wrong comparator, the
same class of error already documented for the r=3 200k-vs-10k tails. The correct value is
recorded as mwpm_validation and the printed one is kept only as mwpm_printed_STALE so the
discrepancy stays visible instead of being quietly dropped.

  python parse_r5_teacher_logs.py ~/Downloads/r5_teacher_seed*.txt \
      --out rcnn_threshold/r5_teacher_summary.csv
"""
import argparse
import csv
import os
import re

# MWPM on the formal r=5 pool's validation partition [20.0M, 20.2M). Matches
# analyze_r5_ladder.py:27, eaf_run_rcnn_ladder_r5.sh:90, eaf_run_student_ladder_r5.sh:73.
MWPM_VALIDATION = 0.084215

SUMMARY = re.compile(
    r'\[train\]\s+(?P<tag>rcnn_d(?P<d>\d+)_p(?P<p>[\d.]+)_r(?P<rounds>\d+)'
    r'_seed(?P<seed>\d+)_ntr(?P<ntr>\d+))\s+'
    r'RCNN p_L=(?P<pL>[\d.]+)\s+'
    r'MWPM=(?P<mwpm>[\d.]+)\s+'
    r'gap=(?P<gap>[+-][\d.]+)\s+'
    r'base_rate=(?P<base>[\d.]+)\s+'
    r'\((?P<epochs>\d+) ep, (?P<secs>[\d.]+)s, (?P<params>\d+) params\)')

PARTITION = re.compile(
    r'\[train\] partitions: train \[(?P<tr0>[\d,]+), (?P<tr1>[\d,]+)\)\s+'
    r'validation \[(?P<va0>[\d,]+), (?P<va1>[\d,]+)\)')

CKPT = re.compile(
    r'\[train\] checkpoint identity: best==best_restored (?P<a>\w+)\s+'
    r'best==lastepoch (?P<b>\w+)')


def parse_log(path):
    """Pull one run's record out of a log, or None if it holds no completed run."""
    text = open(path, errors='replace').read()
    m = SUMMARY.search(text)
    if not m:
        return None
    g = m.groupdict()

    part = PARTITION.search(text)
    ck = CKPT.search(text)

    def _int(s):
        return int(s.replace(',', ''))

    pL = float(g['pL'])
    rec = {
        'tag': g['tag'],
        'd': int(g['d']),
        'p': float(g['p']),
        'rounds': int(g['rounds']),
        'seed': int(g['seed']),
        'n_train': int(g['ntr']),
        'p_L': pL,
        # The comparator these runs should actually be measured against.
        'mwpm_validation': MWPM_VALIDATION,
        'ratio_vs_mwpm': round(pL / MWPM_VALIDATION, 4),
        # Kept only to make the stale-baseline discrepancy visible; do not plot this.
        'mwpm_printed_STALE': float(g['mwpm']),
        'base_rate': float(g['base']),
        'n_params': int(g['params']),
        'epochs_ran': int(g['epochs']),
        'train_time_s': float(g['secs']),
        'train_lo': _int(part.group('tr0')) if part else '',
        'train_hi': _int(part.group('tr1')) if part else '',
        'val_lo': _int(part.group('va0')) if part else '',
        'val_hi': _int(part.group('va1')) if part else '',
        # best==best_restored False means the scored model is NOT the best-validation
        # checkpoint (with --no-early-stopping there is no restore, so the scored model is
        # the last epoch). Carried through because it qualifies how the p_L should be read.
        'ckpt_best_is_restored': (ck.group('a') if ck else ''),
        'ckpt_best_is_lastepoch': (ck.group('b') if ck else ''),
        'source_log': os.path.basename(path),
    }
    return rec


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+', help='EAF nohup log files')
    ap.add_argument('--out', required=True, help='output CSV path')
    args = ap.parse_args()

    rows = []
    for path in args.logs:
        rec = parse_log(path)
        if rec is None:
            print(f"[parse] no completed run in {os.path.basename(path)} -- skipped")
            continue
        rows.append(rec)
        print(f"[parse] {rec['tag']}  p_L={rec['p_L']:.5f}  "
              f"vs MWPM_val {rec['mwpm_validation']:.6f} = {rec['ratio_vs_mwpm']:.4f}x  "
              f"(log printed the stale {rec['mwpm_printed_STALE']:.5f})")
        if rec['ckpt_best_is_lastepoch'] == 'False':
            print(f"          note: scored model is NOT the best-validation checkpoint")

    if not rows:
        raise SystemExit("[parse] nothing parsed")

    rows.sort(key=lambda r: (r['n_train'], r['seed']))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[parse] wrote {len(rows)} rows -> {args.out}")


if __name__ == '__main__':
    run()
