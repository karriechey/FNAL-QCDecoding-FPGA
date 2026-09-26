#!/usr/bin/env python3
# Created: 2026-09-22
# Last modified: 2026-09-22
"""Validation-loss comparison: parameter-matched Mamba against the Experiment 16 GRU, d=7.

Two panels, one figure:
  left   every curve on the same axes -- the three GRU seeds and the Mamba run -- so the
         separation is visible without reading numbers off a table
  right  Mamba's train and validation loss together, which is the diagnostic that says
         the gap is underfitting rather than overfitting

The GRU curves come from the histories pulled off grace1 into
rcnn_threshold/results_meta/gh200_gru_d7_e200_20260819T192524Z/seed*/runs/*.history.json
(Experiment 16, run 2: 3 seeds, 200 epochs, batch 10,000, LR 3e-3, 79,521 params).
The Mamba curve comes from two whitespace-separated "epoch value" files scraped from the
live EAF driver log while the run was still going, so the figure can be rebuilt before
the run writes its own history.json. Once that file exists, pass --mamba-history instead
and the scrape is no longer needed.

  .venv/bin/python plot_mamba_vs_gru_d7.py
  .venv/bin/python plot_mamba_vs_gru_d7.py --mamba-history <tag>.history.json
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GRU_DIR = os.path.join(HERE, 'rcnn_threshold', 'results_meta',
                       'gh200_gru_d7_e200_20260819T192524Z')

# Categorical colours consistent with Experiments 6-16: the GRU keeps its hue across the
# whole log, so a reader who knows those figures reads this one without a new legend.
GRU_C = '#1f77b4'
MAMBA_C = '#d62728'
TRAIN_C = '#ff9896'


def read_two_col(path):
    """Read an 'epoch value' text file scraped from the driver log."""
    a = np.loadtxt(path)
    return a[:, 0].astype(int), a[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mamba-val', default=os.path.join(HERE, 'mamba_lr003_curve.txt'))
    ap.add_argument('--mamba-train', default=os.path.join(HERE, 'mamba_lr003_trainloss.txt'))
    ap.add_argument('--mamba-history', default=None,
                    help='train_mamba.py history.json; overrides the two scraped files')
    ap.add_argument('--out', default=os.path.join(HERE, 'figures', 'exp18_mamba_vs_gru_d7.png'))
    args = ap.parse_args()

    if args.mamba_history:
        h = json.load(open(args.mamba_history))
        m_val = np.asarray(h['val_loss'])
        m_train = np.asarray(h['loss'])
        m_ep = np.arange(1, len(m_val) + 1)
    else:
        m_ep, m_val = read_two_col(args.mamba_val)
        _, m_train = read_two_col(args.mamba_train)

    gru = {}
    for f in sorted(glob.glob(os.path.join(GRU_DIR, 'seed*', 'runs', '*.history.json'))):
        seed = f.split(os.sep)[-3]
        gru[seed] = np.asarray(json.load(open(f))['val_loss'])
    if not gru:
        raise SystemExit(f'no GRU histories under {GRU_DIR}')

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # --- left: every curve, log y (the two families differ by ~3x) --------------------
    for i, (seed, v) in enumerate(sorted(gru.items())):
        ax.plot(np.arange(1, len(v) + 1), v, color=GRU_C, lw=1.4, alpha=0.85,
                label='GRU, 79,521 params (3 seeds)' if i == 0 else None)
    ax.plot(m_ep, m_val, color=MAMBA_C, lw=2.2,
            label=f'Mamba, 79,001 params (seed 0, {len(m_ep)} ep)')

    gru_best = min(v.min() for v in gru.values())
    gru_worst = max(v.min() for v in gru.values())
    ax.axhspan(gru_best, gru_worst, color=GRU_C, alpha=0.10, zorder=0)
    ax.annotate(f'GRU best checkpoints\n{gru_best:.4f}–{gru_worst:.4f}',
                xy=(178, gru_worst), xytext=(58, gru_worst * 1.30),
                fontsize=8.5, color=GRU_C,
                arrowprops=dict(arrowstyle='->', color=GRU_C, lw=1))
    ax.annotate(f'Mamba best {m_val.min():.4f}\n= {m_val.min() / gru_best:.1f}× the GRU',
                xy=(m_ep[-1], m_val[-1]), xytext=(72, m_val.min() * 1.38),
                fontsize=8.5, color=MAMBA_C,
                arrowprops=dict(arrowstyle='->', color=MAMBA_C, lw=1))

    ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('validation loss  (BCE, 200k shots [15.0M, 15.2M))')
    ax.set_title('Same data, same input, matched parameters:\nonly the recurrent layer differs',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25, which='both')
    ax.set_xlim(0, 205)
    ax.set_ylim(0.028, 0.32)

    # --- right: Mamba train vs val ----------------------------------------------------
    ax2.plot(m_ep, m_train, color=TRAIN_C, lw=2.0, label='Mamba training loss')
    ax2.plot(m_ep, m_val, color=MAMBA_C, lw=2.0, label='Mamba validation loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.set_title(f'Train and validation coincide ({m_train[-1]:.4f} vs {m_val[-1]:.4f}):\n'
                  'the model underfits, it does not overfit', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.set_xlim(0, 205)

    # Deceleration, stated in numbers rather than left to the eye.
    marks = [e for e in (50, 100, 150, len(m_ep)) if e <= len(m_ep)]
    txt = '  '.join(f'ep{e}: {m_val[e - 1]:.4f}' for e in marks)
    ax2.text(0.02, -0.19, txt, transform=ax2.transAxes, fontsize=8.5, color='#444444')

    # Epoch-to-epoch stability, the second thing the left panel shows. The GRU's
    # validation curve oscillates and, on seed 0, rises steeply after epoch 140; the
    # Mamba curve is monotone. Quantified as the mean absolute epoch-to-epoch step over
    # the second half, where both families have settled.
    def jitter(v):
        h = np.asarray(v)[len(v) // 2:]
        return float(np.abs(np.diff(h)).mean())
    j_m = jitter(m_val)
    j_g = np.mean([jitter(v) for v in gru.values()])
    ax.text(0.02, -0.19,
            f'mean |Δ val_loss| per epoch, second half:  Mamba {j_m:.5f},  GRU {j_g:.5f}',
            transform=ax.transAxes, fontsize=8.5, color='#444444')

    fig.suptitle('d=7, r=7, p=0.004 — parameter-matched Mamba vs GRU, 10M training shots, '
                 'Adam 3e-3, batch 10,000', fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160)
    fig.savefig(args.out.replace('.png', '.pdf'))
    print(f'wrote {args.out}')

    # Console summary, so the numbers in the write-up are never typed from memory.
    print(f'\nMamba  epochs={len(m_ep)}  best val_loss={m_val.min():.4f} '
          f'@ epoch {int(m_ep[m_val.argmin()])}  final train={m_train[-1]:.4f}')
    for seed, v in sorted(gru.items()):
        print(f'GRU {seed}  epochs={len(v)}  best val_loss={v.min():.4f} '
              f'@ epoch {int(v.argmin()) + 1}  final={v[-1]:.4f}')
    print(f'stability (mean |delta val_loss|/epoch, 2nd half): '
          f'mamba {j_m:.5f}  gru {j_g:.5f}')
    for e in (10, 30, 60, 100, 150, len(m_ep)):
        if e <= len(m_ep):
            g = np.mean([v[e - 1] for v in gru.values()])
            print(f'  epoch {e:>3}:  mamba {m_val[e - 1]:.4f}   gru mean {g:.4f}   '
                  f'ratio {m_val[e - 1] / g:.2f}x')


if __name__ == '__main__':
    main()
