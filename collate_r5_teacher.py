#!/usr/bin/env python3
# Created: 2026-08-04
# Last modified: 2026-08-04
"""Collate the r=5 teacher's re-scored .best checkpoint results into one ladder CSV.

Why the re-scored values and not the training CSVs
Each run's own p_L comes from whatever weights were in memory when fit() returned, which
is the restored-best only if early stopping fired -- so those are not comparable across
runs (eaf_run_rcnn_ladder_r5.sh:59-61). eval_on_tail.py re-scores each run's .best
checkpoint on the validation partition and appends to val_scores_best_ckpt.csv; those are
the authoritative numbers and the ones this script reports.

Why it cannot just read val_scores_best_ckpt.csv alone
That file identifies a run only by checkpoint filename (r5_teacher_seed0.best.weights.h5),
which carries the seed but not n_train -- the driver's inline collation regex expects an
`ntr<N>_seed<S>` tag and finds nothing here. The per-seed training CSV next to each
checkpoint does carry n_train, n_params and epochs_ran, so the two are joined on seed.

MWPM is not taken from either file. eval_on_tail.py's stored lookup describes the pool's
own tail, not the [20.0M, 20.2M) validation partition these runs scored on, and the column
comes through empty in any case. The correct comparator is pinned here, matching
analyze_r5_ladder.py:27 and both eaf_*_r5.sh drivers.

  python collate_r5_teacher.py <out_r5_teacher dir> \
      --out rcnn_threshold/rcnn_r5_ladder_$(date -u +%Y%m%dT%H%M%SZ).csv
"""
import argparse
import csv
import glob
import os
import re
import statistics as st

MWPM_VALIDATION = 0.084215   # MWPM on the formal pool's validation partition


def load_training_csvs(root):
    """seed -> the run's own training-CSV row, for n_train / n_params / epochs_ran."""
    out = {}
    for f in glob.glob(os.path.join(root, 'seed*', '*.csv')):
        for r in csv.DictReader(open(f)):
            if 'seed' in r and 'n_train' in r:
                out[int(r['seed'])] = r
    return out


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='the out_r5_teacher directory')
    ap.add_argument('--out', required=True)
    ap.add_argument('--scores', default=None,
                    help='override path to val_scores_best_ckpt.csv')
    args = ap.parse_args()

    scores_path = args.scores or os.path.join(args.root, 'val_scores_best_ckpt.csv')
    if not os.path.exists(scores_path):
        raise SystemExit(f"[collate] MISSING {scores_path}")
    train_rows = load_training_csvs(args.root)

    rows = []
    for r in csv.DictReader(open(scores_path)):
        m = re.search(r'seed(\d+)', r['weights'])
        if not m:
            print(f"[collate] cannot read a seed from {r['weights']!r} -- skipped")
            continue
        seed = int(m.group(1))
        tr = train_rows.get(seed, {})
        pL = float(r['p_L'])
        # Flag when the re-scored .best differs from what the run reported for itself.
        # Equal values mean the run happened to end on its best checkpoint; a difference
        # is the reason this collation exists at all.
        own = float(tr['p_L']) if tr.get('p_L') else None
        rows.append({
            'source': 'best_ckpt_rescored',
            'seed': seed,
            'n_train': int(tr.get('n_train', 0)) or '',
            'n_test': int(r['n_test']),
            'p_L': pL,
            'mwpm_validation': MWPM_VALIDATION,
            'ratio_vs_mwpm': round(pL / MWPM_VALIDATION, 4),
            'p_L_run_reported': own if own is not None else '',
            'rescore_changed_p_L': ('' if own is None else int(abs(own - pL) > 1e-9)),
            'n_params': tr.get('n_params', ''),
            'epochs_ran': tr.get('epochs_ran', ''),
            'base_rate': float(r['base_rate']),
            'pool': r.get('pool', ''),
            'weights': r['weights'],
        })

    if not rows:
        raise SystemExit('[collate] nothing collated')
    rows.sort(key=lambda x: (x['n_train'], x['seed']))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"[collate] {len(rows)} runs -> {args.out}\n")
    print(f"  {'seed':>5} {'n_train':>11} {'p_L (.best)':>12} {'xMWPM':>7} "
          f"{'run-reported':>13} {'changed':>8}")
    for r in rows:
        print(f"  {r['seed']:>5} {r['n_train']:>11,} {r['p_L']:>12.6f} "
              f"{r['ratio_vs_mwpm']:>7.4f} {str(r['p_L_run_reported']):>13} "
              f"{r['rescore_changed_p_L']:>8}")
    vals = [r['p_L'] for r in rows]
    print(f"\n  n={len(vals)}  mean={st.mean(vals):.6f}  "
          f"sd={st.stdev(vals) if len(vals) > 1 else 0:.6f}  "
          f"range=[{min(vals):.6f}, {max(vals):.6f}]")
    print(f"  mean / MWPM = {st.mean(vals) / MWPM_VALIDATION:.4f}x   "
          f"(MWPM = {MWPM_VALIDATION} on the validation partition)")


if __name__ == '__main__':
    run()
