#!/usr/bin/env python3
# Created: 2026-08-17
# Last modified: 2026-08-17
"""Plot logical error rate against GRU parameter count, d=5, r=5, p=0.004.

Reads only the tidy summary CSV from collate_gru_compression_p004.py, so the figure and
the table cannot disagree. Nothing is recomputed here.

  x  measured GRU parameters, log scale (~6k, ~10k, ~20k, full 69,441)
  y  logical error rate on the evaluation block [15.2M, 17.0M)
     one curve for hard-label students, one for GRU-teacher distilled students,
     each point a 3-seed mean with a standard-deviation error bar
     MWPM as a horizontal reference line with its binomial band
     the full-size GRU as its own 3-seed reference point

The full-size point is drawn once, from the three teacher runs, and is deliberately not
attached to either curve: only one of those seeds became the distillation teacher, and
none of the three was itself distilled. Both curves are drawn to it with a dotted
connector so the reader can see where compression starts from without the marker implying
membership in either arm.

Adding the RCNN-teacher arm later means one more row group in the summary CSV with
mode='distill_rcnn'; CURVES below is where it gets registered, and nothing else changes.

  python plot_gru_compression_p004.py \
      --summary <sweep>/gru_compression_p004_summary.csv \
      --out figures/gru_compression_p004.png
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # no display on the pod, and none needed to write a file
import matplotlib.pyplot as plt

# Which supervision arms become curves, in legend order: mode -> (label, colour, marker).
# The RCNN-teacher arm is pre-registered so the later run needs no code change; a mode
# absent from the CSV is skipped silently rather than drawing an empty legend entry.
CURVES = [
    ('hard',         'Hard labels (Stim ground truth)',  '#1f77b4', 'o'),
    ('distill',      'Distilled from full-size GRU',     '#d62728', 's'),
    ('distill_rcnn', 'Distilled from RCNN',              '#2ca02c', '^'),
]


def load_summary(path):
    """Group the tidy CSV by supervision arm, and pull out the full-size reference."""
    by_mode = defaultdict(list)
    reference = None
    for r in csv.DictReader(open(path)):
        rec = {
            'params': int(r['params']),
            'units': int(r['units']),
            'mode': r['mode'],
            'role': r['role'],
            'n_seeds': int(r['n_seeds']),
            'mean': float(r['mean_p_L']),
            'std': float(r['std_p_L']) if r['std_p_L'] else 0.0,
            'mwpm': float(r['mwpm_p_L']),
        }
        if rec['role'] == 'teacher':
            reference = rec
        else:
            by_mode[rec['mode']].append(rec)
    for recs in by_mode.values():
        recs.sort(key=lambda d: d['params'])
    return by_mode, reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True,
                    help='gru_compression_p004_summary.csv from the collator')
    ap.add_argument('--out', required=True, help='output image path (.png or .pdf)')
    ap.add_argument('--mwpm-err', type=float, default=0.000065,
                    help='binomial standard error on the MWPM baseline over the 1.8M '
                         'evaluation block (exp13 table of record)')
    ap.add_argument('--title', default='GRU compression at d=5, r=5, p=0.004')
    ap.add_argument('--logy', action='store_true',
                    help='log y-axis. Worth using if one width degrades toward the 0.194 '
                         'base rate, which on a linear axis flattens the MWPM line and '
                         'every point that stayed near it into one band.')
    args = ap.parse_args()

    by_mode, reference = load_summary(args.summary)
    if not by_mode:
        raise SystemExit(f"[plot] no student rows in {args.summary}")

    # Every row carries the same baseline (the collator writes it from one constant), so
    # any row will do.
    mwpm = next(iter(by_mode.values()))[0]['mwpm']

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # MWPM first, so the curves draw over it rather than under.
    ax.axhline(mwpm, color='0.35', linestyle='--', linewidth=1.2, zorder=1,
               label=f'MWPM  ({mwpm:.5f})')
    ax.axhspan(mwpm - args.mwpm_err, mwpm + args.mwpm_err, color='0.35', alpha=0.15,
               zorder=0)

    for mode, label, colour, marker in CURVES:
        recs = by_mode.get(mode)
        if not recs:
            continue
        xs = [r['params'] for r in recs]
        ys = [r['mean'] for r in recs]
        es = [r['std'] for r in recs]
        # Dotted connector from the last reduced point up to the full-size reference: the
        # trend continues there, but that point belongs to neither arm.
        if reference is not None:
            ax.plot([xs[-1], reference['params']], [ys[-1], reference['mean']],
                    color=colour, linestyle=':', linewidth=1.2, zorder=2)
        ax.errorbar(xs, ys, yerr=es, color=colour, marker=marker, markersize=7,
                    linewidth=1.8, capsize=4, label=label, zorder=3)

    if reference is not None:
        ax.errorbar([reference['params']], [reference['mean']], yerr=[reference['std']],
                    color='black', marker='*', markersize=15, linewidth=0, capsize=4,
                    zorder=4,
                    label=f"Full-size GRU ({reference['params']:,} params, "
                          f"{reference['n_seeds']}-seed mean)")

    ax.set_xscale('log')
    if args.logy:
        ax.set_yscale('log')
    ax.set_xlabel('GRU trainable parameters')
    ax.set_ylabel('Logical error rate $p_L$  (evaluation block, 1.8M shots)')
    ax.set_title(args.title)

    # Ticks at the actual measured capacities, not decade defaults: the whole point of the
    # x-axis is that these are the four sizes that were built.
    ticks = sorted({r['params'] for recs in by_mode.values() for r in recs} |
                   ({reference['params']} if reference else set()))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t / 1000:.1f}k' for t in ticks])
    # Only the x minor ticks go: a log x-axis would otherwise stamp decade subdivisions
    # between the four measured capacities. The y minor ticks are left alone so --logy
    # keeps its decade structure readable.
    ax.tick_params(axis='x', which='minor', bottom=False)

    ax.grid(True, which='major', alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"[plot] wrote {args.out}")

    # Console echo of exactly what was drawn, so a figure in a message can always be
    # traced back to numbers without reopening the CSV.
    for mode, label, _c, _m in CURVES:
        for r in by_mode.get(mode, []):
            print(f"  {label:<38} {r['params']:>7,} params  "
                  f"p_L={r['mean']:.6f} +/- {r['std']:.6f}  "
                  f"({r['mean'] / mwpm:.3f}x MWPM, n={r['n_seeds']})")
    if reference is not None:
        print(f"  {'Full-size GRU reference':<38} {reference['params']:>7,} params  "
              f"p_L={reference['mean']:.6f} +/- {reference['std']:.6f}  "
              f"({reference['mean'] / mwpm:.3f}x MWPM, n={reference['n_seeds']})")


if __name__ == '__main__':
    main()
