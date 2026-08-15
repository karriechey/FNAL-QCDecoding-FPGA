#!/usr/bin/env python3
# Created: 2026-08-04
# Last modified: 2026-08-04
"""Summarise the r=5 RCNN ladder and decide whether the 20M rung is justified.

Reads the re-scored .best validation results plus each run's history JSON:

  per-rung validation p_L for every seed, with mean, std and range
  whether seed 0 underperforms systematically or only at 10M
  best epoch and whether validation loss was still improving when training ended
  whether the curve is monotonic in n_train

  python analyze_r5_ladder.py ~/rcnn_threshold/out_r5_ladder
  python analyze_r5_ladder.py <dir> --teacher-dir ~/rcnn_threshold/out_r5_teacher
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st

MWPM_VAL = 0.084215   # MWPM on the formal pool's validation partition


def load_scores(root):
    """(n_train, seed) -> validation p_L, from the re-scored .best checkpoints."""
    out = {}
    for f in glob.glob(os.path.join(root, '**', 'val_scores_best_ckpt.csv'), recursive=True):
        for r in csv.DictReader(open(f)):
            m = re.search(r'ntr(\d+)_seed(\d)', r['weights'])
            if m:
                out[(int(m.group(1)), int(m.group(2)))] = float(r['p_L'])
    return out


def load_histories(root):
    """(n_train, seed) -> dict of per-epoch arrays."""
    out = {}
    for f in glob.glob(os.path.join(root, '**', '*.history.json'), recursive=True):
        m = re.search(r'ntr(\d+)', f) or re.search(r'ntr(\d+)', os.path.basename(f))
        s = re.search(r'seed(\d)', f)
        if m and s:
            out[(int(m.group(1)), int(s.group(1)))] = json.load(open(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--teacher-dir', default=None,
                    help='also fold in the existing 10M teacher runs')
    args = ap.parse_args()

    scores = load_scores(os.path.expanduser(args.root))
    hists = load_histories(os.path.expanduser(args.root))
    if args.teacher_dir:
        td = os.path.expanduser(args.teacher_dir)
        for f in glob.glob(os.path.join(td, 'val_scores_best_ckpt.csv')):
            for r in csv.DictReader(open(f)):
                m = re.search(r'seed(\d)', r['weights'])
                if m:
                    scores[(10_000_000, int(m.group(1)))] = float(r['p_L'])
        hists.update({(10_000_000, k[1]): v for k, v in load_histories(td).items()})

    rungs = sorted({n for n, _ in scores})
    seeds = sorted({s for _, s in scores})
    if not rungs:
        raise SystemExit(f'no val_scores_best_ckpt.csv found under {args.root}')

    print(f"validation p_L by rung and seed   (MWPM on validation = {MWPM_VAL})\n")
    hdr = ' '.join(f'{"seed "+str(s):>10}' for s in seeds)
    print(f"  {'n_train':>11} {hdr} {'mean':>10} {'std':>9} {'range':>9} {'xMWPM':>7}")
    means = {}
    for n in rungs:
        v = [scores.get((n, s)) for s in seeds]
        have = [x for x in v if x is not None]
        cells = ' '.join(f'{x:>10.5f}' if x is not None else f'{"-":>10}' for x in v)
        if len(have) > 1:
            m, sd, rg = st.mean(have), st.stdev(have), max(have) - min(have)
        else:
            m, sd, rg = (have[0], 0.0, 0.0) if have else (float('nan'),) * 3
        means[n] = m
        print(f"  {n:>11,} {cells} {m:>10.5f} {sd:>9.5f} {rg:>9.5f} {m/MWPM_VAL:>7.3f}")

    # Is seed 0 systematically worst, or only at 10M?
    print("\nseed rank per rung (1 = best):")
    worst_counts = {s: 0 for s in seeds}
    for n in rungs:
        v = {s: scores.get((n, s)) for s in seeds if scores.get((n, s)) is not None}
        if len(v) < 2:
            continue
        order = sorted(v, key=lambda s: v[s])
        ranks = {s: i + 1 for i, s in enumerate(order)}
        worst_counts[order[-1]] += 1
        print(f"  {n:>11,}  " + "  ".join(f"seed{s}={ranks[s]}" for s in seeds))
    print("\n  times each seed was worst: " +
          "  ".join(f"seed{s}={c}" for s, c in worst_counts.items()))
    n_rungs = sum(1 for n in rungs if len([1 for s in seeds if (n, s) in scores]) > 1)
    if worst_counts.get(0, 0) >= max(2, n_rungs - 1):
        print("  -> seed 0 is worst at nearly every rung: systematic, not a single bad draw.")
    elif worst_counts.get(0, 0) <= 1:
        print("  -> seed 0 is not systematically worst; the 10M result looks like one "
              "unfavourable basin.")

    # Training behaviour: did runs stop while still improving?
    if hists:
        print("\ntraining behaviour:")
        print(f"  {'n_train':>11} {'seed':>5} {'epochs':>7} {'best_ep':>8} "
              f"{'best_val_loss':>14} {'last_val_loss':>14} {'still improving':>16}")
        for n in rungs:
            for s in seeds:
                h = hists.get((n, s))
                if not h or 'val_loss' not in h:
                    continue
                vl = h['val_loss']
                b = min(range(len(vl)), key=lambda i: vl[i])
                # "still improving" = best was in the final fifth of the run
                improving = b >= len(vl) - max(2, len(vl) // 5)
                print(f"  {n:>11,} {s:>5} {len(vl):>7} {b+1:>8} {vl[b]:>14.5f} "
                      f"{vl[-1]:>14.5f} {str(improving):>16}")

    # Monotonicity: more data should not make it worse
    print("\ncurve shape:")
    ok = True
    for a, b in zip(rungs, rungs[1:]):
        if means[b] > means[a]:
            ok = False
            print(f"  NOT monotonic: {a:,} -> {b:,} rises {means[a]:.5f} -> {means[b]:.5f}")
    if ok:
        print("  monotonic decreasing across every rung.")
    print("\n  Decision: proceed to 20M if seed 0 is anomalous only at 10M and the curve is")
    print("  monotonic. Stop and investigate if seed 0 lags consistently.")


if __name__ == '__main__':
    main()
