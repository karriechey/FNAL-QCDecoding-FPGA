#!/usr/bin/env python3
# Created: 2026-07-31
# Last modified: 2026-07-31
"""Inventory every result CSV under a results root and print one row per run.

Reads the files rather than any hand-maintained list, so it cannot drift from what is
actually on disk. Run it on EAF (or on a synced copy) to produce the table that goes into
the paper's bookkeeping.

Handles the several CSV shapes this project has produced -- the ladder runs from
train_one.py, the quantization sweep from train_one_quantized.py, the eval_on_tail
re-scores, and the student runs -- by pulling whichever of the known columns exist and
inferring the rest from the filename.

Usage:
  python inventory_runs.py ~/rcnn_threshold
  python inventory_runs.py ~/rcnn_threshold --csv inventory.csv
"""
import argparse
import csv
import glob
import os
import re


# Directory -> experiment. Anything unlisted is reported as unknown rather than guessed.
EXPERIMENT_BY_DIR = {
    'diag_eaf': 'Diagnostics (determinism, smoke tests) -- not results',
    'out': 'RCNN data-volume ladder (10k tail)',
    'out_t200k': 'RCNN data-volume ladder (200k tail)',
    'out_t200k_w': 'RCNN 10M retrain, weights saved',
    'out_20M': 'RCNN 20M point (Experiment 5)',
    'out_q': 'Teacher weight-only QAT sweep',
    'out_q_mcnemar': 'Teacher QAT McNemar',
    'out_q_phase2a': 'Teacher activation sweep (phase 2a)',
    'out_q_phase2a_relu5': 'Teacher activation sweep, ReLU I=5',
    'teacher': 'Teacher caches / tail scores',
    'out_student_grid': 'Student hard-vs-soft grid (pools, 1M)',
    'out_student_grid_pools': 'Student hard-vs-soft grid (pools, 1M)',
    'out_student_grid_pools_t200k': 'Student hard-vs-soft grid (pools_t200k)',
    'out_mlp': 'MLP student runs',
    'out_gru': 'GRU student runs',
    'out_mlp_ladder': 'MLP student data-volume ladder',
    'out_gru_ladder': 'GRU student data-volume ladder',
}


def is_inventory_output(path):
    """True if this file is a previous run of this script.

    Scanning a tree that an inventory was written into would otherwise re-ingest it and
    report every row twice, attributed to the inventory rather than to the run that
    produced it.
    """
    if os.path.basename(path).startswith('inventory'):
        return True
    try:
        with open(path) as f:
            return (f.readline().split(',')[:3] == ['experiment', 'dir', 'file'])
    except Exception:
        return False


def classify(path, root):
    """Experiment label and run parameters, from the directory and filename."""
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    parts = rel.split(os.sep)[:-1]
    # A known experiment directory anywhere in the path wins, so the same tree classifies
    # identically whether scanned on EAF (~/rcnn_threshold/out_q/...) or from a nested
    # local snapshot (results_snapshot/eaf/rcnn_threshold/out_q/...).
    d = next((p for p in reversed(parts) if p in EXPERIMENT_BY_DIR),
             parts[0] if parts else '')
    label = EXPERIMENT_BY_DIR.get(d, f'unknown ({d})' if d else 'loose file at root')
    name = os.path.basename(path)
    got = {}
    for key, pat in (('n_train', r'ntr(\d+)'), ('seed', r'seed(\d+)'),
                     ('w_bits', r'_w(\d+)'), ('a_bits', r'_a(\d+)'),
                     ('alpha', r'alpha([\d.]+)')):
        m = re.search(pat, name)
        if m:
            got[key] = m.group(1)
    return d, label, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='results root, e.g. ~/rcnn_threshold')
    ap.add_argument('--csv', default=None, help='also write the table here')
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    rows = []
    for path in sorted(glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)):
        if os.path.basename(path) == 'mwpm_baseline.csv' or is_inventory_output(path):
            continue
        d, label, got = classify(path, root)
        try:
            recs = list(csv.DictReader(open(path)))
        except Exception as e:
            rows.append(dict(experiment=label, dir=d, file=os.path.basename(path),
                             note=f'unreadable: {e}'))
            continue
        for rec in recs:
            rows.append(dict(
                experiment=label,
                dir=d,
                file=os.path.basename(path),
                arch=rec.get('architecture') or rec.get('student', ''),
                n_train=rec.get('n_train', got.get('n_train', '')),
                n_test=rec.get('n_test', ''),
                seed=rec.get('seed', got.get('seed', '')),
                w_bits=rec.get('weight_bits', got.get('w_bits', '')),
                a_bits=rec.get('act_bits', got.get('a_bits', '')),
                alpha=rec.get('alpha', got.get('alpha', '')),
                p_L=rec.get('p_L', ''),
                mwpm_p_L=rec.get('mwpm_p_L', ''),
                # blank on older rows: eval_on_tail only began stamping this once the
                # stored-vs-redecoded baseline distinction was made explicit
                mwpm_source=rec.get('mwpm_source', ''),
                test_pool=rec.get('test_pool', rec.get('pool', '')),
                train_pool_dir=rec.get('train_pool_dir', ''),
                run_utc=rec.get('run_utc', ''),
                mtime=__import__('datetime').datetime.fromtimestamp(
                    os.path.getmtime(path)).strftime('%Y-%m-%d'),
            ))

    cols = ['experiment', 'dir', 'file', 'arch', 'n_train', 'n_test', 'seed', 'w_bits',
            'a_bits', 'alpha', 'p_L', 'mwpm_p_L', 'mwpm_source', 'test_pool',
            'train_pool_dir', 'run_utc', 'mtime']

    by_exp = {}
    for r in rows:
        by_exp.setdefault(r['experiment'], []).append(r)
    print(f"{len(rows)} result rows across {len(by_exp)} experiments under {root}\n")
    for exp in sorted(by_exp):
        rs = by_exp[exp]
        pls = [r['p_L'] for r in rs if r.get('p_L')]
        print(f"  {exp:42s} {len(rs):4d} rows  {len(set(r['file'] for r in rs)):3d} files"
              f"  p_L present: {len(pls)}")

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {args.csv}")


if __name__ == '__main__':
    main()
