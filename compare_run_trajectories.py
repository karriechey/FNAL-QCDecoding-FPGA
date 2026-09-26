#!/usr/bin/env python3
# Created: 2026-08-19
# Last updated: 2026-08-19
"""Compare training runs at matched epochs and matched optimizer updates.

Reads runs that are still training. Every epoch train_one.py writes
`{ckpt-dir}/{tag}.resume.json` holding the run identity and the full per-epoch history,
and at completion it writes `{out-dir}/{tag}.history.json`. Neither file is removed, so a
run in progress and a finished run are read the same way. Nothing here writes into a run
directory or touches a checkpoint.

Two alignments, because they answer different questions:

  matched epochs   compares schedules at equal passes over the data. The right axis when
                   the runs share a batch size, since the learning-rate schedule is
                   indexed by epoch.
  matched updates  compares at equal numbers of optimizer steps. The right axis when
                   batch sizes differ, since batch 7,500 takes 1,334 steps per epoch
                   against 2,000 at batch 5,000 and would otherwise look slower per epoch
                   for a reason that has nothing to do with learning.

The monitoring validation block is 200k shots, shared by every run in this study, so
val_accuracy converts directly to a logical error rate: p_L = 1 - val_accuracy, with a
binomial standard error of sqrt(p_L (1 - p_L) / 200000). At p_L near 0.075 that is about
6e-4, so differences below roughly 1.2e-3 are inside noise on this block. The 1.8M
evaluation block resolves about 3x finer and is what final numbers come from.

Usage:

  python compare_run_trajectories.py \
      original=~/rcnn_threshold/out_d9_p004_ladder/ntr10000000_seed0 \
      slow=~/rcnn_threshold/out_d9_p004_ladder_slow/ntr10000000_seed0

  python compare_run_trajectories.py --out-csv traj.csv --every 5 label=<dir> ...

Each argument is `label=run_directory`, where the directory is the run's --out-dir (the
one containing ckpt/). Labels name the runs in the output and are yours to choose.
"""

import argparse
import json
import os
import sys

VAL_N = 200000          # monitoring validation block, shared by every run in this study


def load_run(run_dir):
    """Return (identity, history) for one run directory.

    History comes from history.json when the run has finished and from the per-epoch
    resume checkpoint while it is still training. Identity always comes from the resume
    JSON, which is the only file recording batch size and schedule.
    """
    run_dir = os.path.expanduser(run_dir)
    ckpt_dir = os.path.join(run_dir, 'ckpt')

    resume_files = ([os.path.join(ckpt_dir, f) for f in sorted(os.listdir(ckpt_dir))
                     if f.endswith('.resume.json')] if os.path.isdir(ckpt_dir) else [])
    if not resume_files:
        raise SystemExit(f"no .resume.json under {ckpt_dir} -- is this a run --out-dir?")
    with open(resume_files[0]) as fh:
        resume = json.load(fh)
    identity = dict(resume['identity'])
    identity['last_epoch'] = resume['epoch']            # epochs completed at last write
    identity['early_best'] = resume.get('early_best')
    identity['early_wait'] = resume.get('early_wait')

    hist_files = [os.path.join(run_dir, f) for f in sorted(os.listdir(run_dir))
                  if f.endswith('.history.json')]
    if hist_files:                                      # run finished; authoritative
        with open(hist_files[0]) as fh:
            history = json.load(fh)
        identity['state'] = 'finished'
    else:                                               # still training
        history = resume['history']
        identity['state'] = 'running'
    return identity, history


def steps_per_epoch(identity):
    """Optimizer updates in one epoch: ceil(n_train / batch_size)."""
    n, b = identity['n_train'], identity['batch_size']
    return -(-n // b)


def p_L_from_accuracy(acc):
    """Logical error rate and its binomial standard error on the 200k monitoring block."""
    p = 1.0 - acc
    return p, (p * (1.0 - p) / VAL_N) ** 0.5


def interpolate(xs, ys, x):
    """Linear interpolation of ys(xs) at x. Returns None outside the measured range."""
    if x < xs[0] or x > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if xs[i] >= x:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', metavar='label=dir',
                    help='one or more label=run_directory pairs')
    ap.add_argument('--every', type=int, default=1,
                    help='print every Nth epoch (default 1)')
    ap.add_argument('--out-csv', default=None,
                    help='also write the matched-epoch table here (appends, never overwrites)')
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        if '=' not in spec:
            raise SystemExit(f"expected label=dir, got {spec!r}")
        label, run_dir = spec.split('=', 1)
        identity, history = load_run(run_dir)
        runs.append(dict(label=label, dir=run_dir, identity=identity, history=history,
                         spe=steps_per_epoch(identity)))

    # ---------------------------------------------------------------- run summary
    print("runs")
    for r in runs:
        i, h = r['identity'], r['history']
        n_ep = len(h['loss'])
        best_i = min(range(n_ep), key=lambda k: h['val_loss'][k])
        pl, se = p_L_from_accuracy(h['val_accuracy'][best_i])
        print(f"  {r['label']:<12} d={i['d']} p={i['p']} r={i['rounds']} seed={i['seed']}"
              f"  n_train={i['n_train']:,}  batch={i['batch_size']}  lr={i['lr_schedule']}"
              f"  [{i['state']}]")
        print(f"  {'':<12} {r['spe']:,} steps/epoch   {n_ep} epochs done"
              f"   best val_loss {h['val_loss'][best_i]:.5f} at epoch {best_i + 1}"
              f"   p_L {pl:.5f} +/- {se:.5f}")

    # Comparing runs that differ in more than the intended variable would be misleading,
    # so name every field that differs across the set.
    keys = ('d', 'p', 'rounds', 'seed', 'n_train', 'batch_size', 'lr_schedule', 'epochs')
    differing = [k for k in keys if len({r['identity'][k] for r in runs}) > 1]
    print(f"\nfields that differ across these runs: {', '.join(differing) or 'none'}")

    # ---------------------------------------------------------------- matched epochs
    print("\nmatched epochs -- val_loss, and p_L = 1 - val_accuracy on the 200k block")
    header = f"{'epoch':>6}" + ''.join(f"{r['label']:>22}" for r in runs)
    print(header)
    print(f"{'':>6}" + ''.join(f"{'val_loss    p_L':>22}" for _ in runs))
    n_max = max(len(r['history']['loss']) for r in runs)
    rows = []
    for e in range(0, n_max, args.every):
        cells, row = '', {'epoch': e + 1}
        for r in runs:
            h = r['history']
            if e < len(h['loss']):
                pl, _ = p_L_from_accuracy(h['val_accuracy'][e])
                cells += f"{h['val_loss'][e]:>13.5f}{pl:>9.5f}"
                row[f"{r['label']}_val_loss"] = h['val_loss'][e]
                row[f"{r['label']}_p_L"] = pl
            else:
                cells += f"{'':>22}"
        print(f"{e + 1:>6}" + cells)
        rows.append(row)

    # Paired difference, only meaningful for exactly two runs.
    if len(runs) == 2:
        a, b = runs
        n_common = min(len(a['history']['loss']), len(b['history']['loss']))
        print(f"\n{b['label']} minus {a['label']}, matched epochs "
              f"(negative = {b['label']} better)")
        for e in range(0, n_common, args.every):
            d_loss = b['history']['val_loss'][e] - a['history']['val_loss'][e]
            pa, _ = p_L_from_accuracy(a['history']['val_accuracy'][e])
            pb, _ = p_L_from_accuracy(b['history']['val_accuracy'][e])
            # Difference of two independent proportions on the same block size.
            se = ((pa * (1 - pa) + pb * (1 - pb)) / VAL_N) ** 0.5
            flag = '  *' if abs(pb - pa) > 2 * se else ''
            print(f"  epoch {e + 1:>3}   val_loss {d_loss:+.5f}   "
                  f"p_L {pb - pa:+.5f} +/- {se:.5f}{flag}")
        print("  * marks a p_L gap beyond 2 standard errors on the 200k block")

        # Epochs 0-9 share a schedule and a seed, so under --require-determinism the two
        # arms must agree there. Divergence points at the runs, not the schedules.
        if a['identity']['batch_size'] == b['identity']['batch_size']:
            warm = min(10, n_common)
            drift = max(abs(a['history']['val_loss'][e] - b['history']['val_loss'][e])
                        for e in range(warm))
            verdict = 'agree' if drift < 1e-6 else f"DIVERGE by {drift:.2e}"
            print(f"\n  shared warm-up check, epochs 1-{warm}: {verdict}")

    # ---------------------------------------------------------------- matched updates
    if len({r['spe'] for r in runs}) > 1:
        print("\nmatched optimizer updates -- val_loss interpolated onto a common grid")
        grids = [[(e + 1) * r['spe'] for e in range(len(r['history']['loss']))] for r in runs]
        lo = max(g[0] for g in grids)
        hi = min(g[-1] for g in grids)
        print(f"{'updates':>10}" + ''.join(f"{r['label']:>14}" for r in runs))
        for k in range(11):
            x = lo + (hi - lo) * k / 10
            cells = ''
            for r, g in zip(runs, grids):
                y = interpolate(g, r['history']['val_loss'], x)
                cells += f"{y:>14.5f}" if y is not None else f"{'':>14}"
            print(f"{int(x):>10,}" + cells)
    else:
        print("\nall runs share a step count per epoch, so matched epochs and matched "
              "updates are the same axis")

    if args.out_csv:
        import csv
        path = os.path.expanduser(args.out_csv)
        fields = sorted({k for row in rows for k in row})
        exists = os.path.exists(path)
        with open(path, 'a', newline='') as fh:      # append; results are never overwritten
            w = csv.DictWriter(fh, fieldnames=fields)
            if not exists:
                w.writeheader()
            w.writerows(rows)
        print(f"\nappended {len(rows)} rows -> {path}")


if __name__ == '__main__':
    sys.exit(main())
