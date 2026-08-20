#!/usr/bin/env python3
# Created: 2026-08-20
# Last updated: 2026-08-20
"""Learning-curve diagnostics across every finished student run.

Reads each run's history.json and eval_best_ckpt.csv and answers one question: is the
200-epoch budget the binding constraint, or is something else?

  python analyze_gru_histories.py --results ~/rcnn_threshold/results --out ~/gru_diag

Writes gru_history_summary.csv, one PNG per run directory, and prints a verdict per run:

  epoch-limited      best epoch in the last 10%, val_loss still descending -> more epochs
  plateaued          train and val both flat -> width or architecture, not epochs
  overfitting        train still falling, val flat or rising -> more data
  converged          val flat for a long stretch, train close behind
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_baselines(pool_dirs):
    """MWPM per distance, read from the saved block JSONs rather than hard-coded.

    Each file must describe exactly [15.2M, 17.0M); a file covering different shots is not
    a baseline for these runs and is rejected rather than silently used.
    """
    out = {}
    for d, path in pool_dirs.items():
        if not os.path.exists(path):
            print(f"[diag] WARNING no baseline for d={d} at {path}")
            continue
        b = json.load(open(path))
        if (int(b['block_start']), int(b['block_stop']), int(b['n'])) != (15200000, 17000000, 1800000):
            print(f"[diag] WARNING baseline for d={d} covers "
                  f"[{b['block_start']}, {b['block_stop']}) n={b['n']}; ignoring")
            continue
        out[d] = float(b['mwpm_p_L'])
        print(f"[diag] d={d} MWPM {out[d]:.6f} from {int(b['residual_errors']):,}/"
              f"{int(b['n']):,}  sha {str(b.get('flips_sha256'))[:16]}")
    return out


def tail_slope(y, tail):
    """Least-squares slope over the last `tail` epochs, as a fraction of the tail mean.

    Normalised so one threshold applies across runs whose losses differ by an order of
    magnitude, and fitted rather than taken endpoint-to-endpoint, so a single noisy final
    epoch cannot flip a verdict.
    """
    seg = np.asarray(y[-tail:], dtype=float)
    x = np.arange(len(seg), dtype=float)
    slope = np.polyfit(x, seg, 1)[0]
    return slope * len(seg) / max(seg.mean(), 1e-12)


def verdict(train, val, best_epoch, n_epochs):
    """Classify a run from its loss curves. Relative thresholds, so they travel.

    Order matters: a rising validation loss is decided first, because a run can have its
    best epoch late and still be overfitting on the tail.
    """
    tail = max(5, n_epochs // 10)
    val_slope = tail_slope(val, tail)          # negative = still improving
    train_slope = tail_slope(train, tail)
    near_end = best_epoch >= n_epochs - tail

    if val_slope > 0.005:
        return ('overfitting',
                f'val_loss rising over the last {tail} epochs ({val_slope:+.3f} of its mean)')
    if train_slope < -0.02 and val_slope > -0.005:
        return ('overfitting',
                f'train improving ({train_slope:+.3f}) while val_loss is flat ({val_slope:+.3f})')
    if near_end and val_slope < -0.01:
        return ('epoch-limited',
                f'best epoch {best_epoch} of {n_epochs} and val_loss still falling ({val_slope:+.3f})')
    if abs(train_slope) < 0.02 and abs(val_slope) < 0.01:
        return ('plateaued',
                f'train {train_slope:+.3f} and val {val_slope:+.3f} both flat over the tail')
    return ('converged',
            f'val flat over the tail ({val_slope:+.3f}) with train close behind')


def parse_tag(tag):
    """Pull (d, units/hidden, seed, n_train) out of a run tag."""
    out = {'d': None, 'units': None, 'seed': None, 'n_train': None}
    for part in tag.split('_'):
        if part.startswith('d') and part[1:].isdigit():
            out['d'] = int(part[1:])
        elif part.startswith('u') and part[1:].isdigit():
            out['units'] = int(part[1:])
        elif part.startswith('seed') and part[4:].isdigit():
            out['seed'] = int(part[4:])
        elif part.startswith('ntr') and part[3:].isdigit():
            out['n_train'] = int(part[3:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.expanduser('~/rcnn_threshold/results'))
    ap.add_argument('--out', default=os.path.expanduser('~/gru_diag'))
    ap.add_argument('--pattern', default='gh200_*', help='result directories to scan')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    baselines = load_baselines({
        5: os.path.expanduser('~/pools_d5_p004_19M/mwpm_eval_block_15p2M_17M.json'),
        7: os.path.expanduser('~/pools_d7_p004_19M/mwpm_eval_block_15p2M_17M.json'),
        9: os.path.expanduser('~/pools_d9_p004_19M/mwpm_eval_block_15p2M_17M.json'),
    })

    rows, skipped = [], []
    for hist_path in sorted(glob.glob(os.path.join(
            args.results, args.pattern, 'seed*', 'runs', '*.history.json'))):
        run_dir = os.path.dirname(os.path.dirname(hist_path))          # .../seedN
        tag = os.path.basename(hist_path).replace('.history.json', '')
        # A run without COMPLETE is truncated or still in flight; it must never enter a
        # summary a decision is made from.
        if not os.path.exists(os.path.join(run_dir, 'COMPLETE')):
            skipped.append(tag)
            continue
        h = json.load(open(hist_path))
        if 'val_loss' not in h:
            continue
        train, val = np.array(h['loss']), np.array(h['val_loss'])
        acc = np.array(h.get('hard_accuracy', h.get('accuracy', [])))
        vacc = np.array(h.get('val_hard_accuracy', h.get('val_accuracy', [])))
        lr = np.array(h.get('lr', []))
        best = int(np.argmin(val))
        meta = parse_tag(tag)

        # the scored result, if the run got that far
        p_L = ratio = ''
        ev = os.path.join(run_dir, 'eval_best_ckpt.csv')
        want = tag + '.best.weights.h5'          # exact checkpoint, never a prefix match
        if os.path.exists(ev):
            for r in csv.DictReader(open(ev)):
                if r['weights'] == want:
                    p_L, ratio = float(r['p_L']), float(r['ratio'])

        v, why = verdict(train, val, best, len(val))
        rows.append(dict(
            run=os.path.basename(os.path.dirname(run_dir)), tag=tag,
            d=meta['d'], n_train=meta['n_train'], units=meta['units'], seed=meta['seed'],
            epochs=len(val), best_epoch=best + 1,
            min_val_loss=round(float(val[best]), 6),
            val_acc_at_best=round(float(vacc[best]), 6) if len(vacc) else '',
            final_train_loss=round(float(train[-1]), 6),
            final_val_loss=round(float(val[-1]), 6),
            train_val_gap=round(float(val[-1] - train[-1]), 6),
            p_L=p_L, ratio_vs_mwpm=ratio, verdict=v, reason=why))

        fig, axes = plt.subplots(1, 3 if len(lr) else 2, figsize=(14, 4.2))
        ax = axes[0]
        ax.plot(train, color='#eb6834', lw=1.8, label='train')
        ax.plot(val, color='#2a78d6', lw=1.8, label='validation')
        ax.axvline(best, color='#6b6a63', ls='--', lw=1.2)
        ax.annotate(f'best ep {best + 1}\n{val[best]:.5f}', xy=(best, val[best]),
                    xytext=(6, 12), textcoords='offset points', fontsize=8, color='#55534e')
        ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.set_yscale('log')
        ax.legend(frameon=False, fontsize=9)
        if len(acc):
            axes[1].plot(acc, color='#eb6834', lw=1.8, label='train')
            axes[1].plot(vacc, color='#2a78d6', lw=1.8, label='validation')
            axes[1].axvline(best, color='#6b6a63', ls='--', lw=1.2)
            axes[1].set_xlabel('epoch'); axes[1].set_ylabel('accuracy')
            axes[1].legend(frameon=False, fontsize=9)
        if len(lr):
            axes[2].plot(lr, color='#1baf7a', lw=1.8)
            axes[2].set_xlabel('epoch'); axes[2].set_ylabel('learning rate')
            axes[2].set_yscale('log')
        fig.suptitle(f'{tag}   —   {v}: {why}', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f'{tag}.png'), dpi=140, bbox_inches='tight')
        plt.close(fig)

    if skipped:
        print(f"[diag] skipped {len(skipped)} run(s) without a COMPLETE marker: "
              f"{', '.join(sorted(skipped)[:6])}{' …' if len(skipped) > 6 else ''}")
    if not rows:
        raise SystemExit(f'[diag] no COMPLETE histories under {args.results}/{args.pattern}')

    rows.sort(key=lambda r: (r['d'] or 0, r['n_train'] or 0, r['units'] or 0, r['seed'] or 0))
    csv_path = os.path.join(args.out, 'gru_history_summary.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"[diag] {len(rows)} runs -> {csv_path}, plots in {args.out}")
    print(f"{'tag':62s} {'best':>5s} {'minval':>9s} {'gap':>8s} {'xMWPM':>7s}  verdict")
    for r in rows:
        print(f"{r['tag'][:62]:62s} {r['best_epoch']:5d} {r['min_val_loss']:9.5f} "
              f"{r['train_val_gap']:8.5f} {str(r['ratio_vs_mwpm']):>7s}  {r['verdict']}")

    counts = {}
    for r in rows:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    print('[diag] ' + '  '.join(f'{k}={v}' for k, v in sorted(counts.items())))


if __name__ == '__main__':
    main()
