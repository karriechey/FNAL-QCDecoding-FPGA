#!/usr/bin/env python3
# Created: 2026-08-21
# An animated view of the same runs, for a talk: every completed run's validation loss draws
# itself epoch by epoch on the left, while the right panel fills in the scored result of each
# run as its training finishes. Written as a GIF so it drops into slides with no codec.
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_data import MWPM, SEALED, rows, load_curves, load_rcnn_curves

C = {5: '#eb6834', 7: '#2a78d6', 9: '#1baf7a'}
INK, MUTED, GREY, GRID = '#1a1a1a', '#55534e', '#8a8782', '#e6e4e0'
META = os.environ.get('RESULTS_META', '../rcnn_threshold/results_meta')
OUT = os.environ.get('FIGDIR', '../figures')

C_RCNN = '#8a5fb0'
_all = load_curves(META) + load_rcnn_curves(
    os.environ.get('RESULTS_META_EAF', os.path.join(META, '..', 'results_meta_eaf')))
curves = [(c['d'], c['arch'], np.array(c['val'])) for c in _all]

R = sorted([r for r in rows() if r['arch'] == 'gru' and r['label'] == 'GRU u140'],
           key=lambda r: (r['d'], r['shots']))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.0, 6.4))
fig.patch.set_facecolor('white')


def style(ax, xlab, ylab, title):
    ax.set_facecolor('white')
    ax.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#d8d5d0')
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlab, fontsize=11, color=INK)
    ax.set_ylabel(ylab, fontsize=11, color=INK)
    ax.set_title(title, fontsize=11.5, color=INK, loc='left', pad=8)


axL.set_xlim(0, 205); axL.set_yscale('log'); axL.set_ylim(0.018, 0.42)
axL.yaxis.set_major_locator(FixedLocator([0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.4]))
axL.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
axL.yaxis.set_minor_formatter(NullFormatter())
style(axL, 'epoch', 'validation loss', f'training — {len(curves)} runs, one line per seed')

axR.set_xscale('log'); axR.set_yscale('log')
axR.set_xlim(1.6e6, 34e6); axR.set_ylim(0.0018, 0.09)
axR.xaxis.set_major_locator(FixedLocator([2e6, 5e6, 10e6, 20e6]))
axR.xaxis.set_minor_locator(NullLocator())
axR.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1e6:g}M'))
axR.yaxis.set_major_locator(FixedLocator([0.002, 0.003, 0.005, 0.008, 0.0125, 0.02, 0.03, 0.05]))
axR.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
axR.yaxis.set_minor_formatter(NullFormatter())
style(axR, 'training shots', 'logical error rate  p_L', 'what each finished run scored')
for d in (5, 7, 9):
    axR.axhline(MWPM[d], color=C[d], ls='--', lw=1.3, alpha=0.55)
    axR.annotate(f'MWPM d={d}', xy=(1.75e6, MWPM[d]), xytext=(0, 4),
                 textcoords='offset points', fontsize=8.5, color=C[d])

lines = [axL.plot([], [], color=C_RCNN if a == 'rcnn' else C[d],
                  lw=2.2 if a == 'rcnn' else 1.5,
                  alpha=0.95 if a == 'rcnn' else (0.8 if a == 'gru' else 0.45),
                  ls={'gru': '-', 'mlp': (0, (3, 2)), 'rcnn': (0, (1, 1.6))}[a])[0]
         for d, a, _ in curves]
ladders = {d: axR.plot([], [], color=C[d], lw=2.3, marker='o', markersize=7,
                       markeredgecolor='white', markeredgewidth=1.4)[0] for d in (5, 7, 9)}
star, = axR.plot([], [], marker='*', markersize=18, color=C[5],
                 markeredgecolor='white', markeredgewidth=1.2, ls='none')
# The curves are unlabelled otherwise: colour is the only thing distinguishing distance.
handles = [plt.Line2D([], [], color=C[d], lw=2, label=f'd = {d}') for d in (5, 7, 9)]
handles.append(plt.Line2D([], [], color=MUTED, lw=1.6, ls=(0, (3, 2)), alpha=0.7,
                          label='MLP'))
handles.append(plt.Line2D([], [], color=C_RCNN, lw=2.2, ls=(0, (1, 1.6)), label='RCNN'))
axL.legend(handles=handles, frameon=False, fontsize=10, loc='upper right',
           bbox_to_anchor=(1.0, 0.90), ncol=2, labelcolor=MUTED)

epoch_txt = axL.text(0.97, 0.96, '', transform=axL.transAxes, ha='right', fontsize=12,
                     color=MUTED, fontweight='medium')
note = axR.text(0.98, 0.93, '', transform=axR.transAxes, ha='right', fontsize=10.5,
                color=C[5], fontweight='medium')

FRAMES, HOLD = 100, 12


def update(f):
    e = min(200, int((f + 1) / (FRAMES - HOLD) * 200)) if f < FRAMES - HOLD else 200
    for ln, (_, _, v) in zip(lines, curves):
        k = min(e, len(v))
        ln.set_data(np.arange(1, k + 1), v[:k])
    epoch_txt.set_text(f'epoch {e}')

    # a run appears on the right once its training would have finished
    frac = e / 200
    for d in (5, 7, 9):
        lad = [r for r in R if r['d'] == d]
        n = int(round(frac * len(lad)))
        ladders[d].set_data([r['shots'] for r in lad[:n]], [r['p_L'] for r in lad[:n]])
    if e >= 200:
        star.set_data([SEALED['shots']], [SEALED['p_L']])
        note.set_text('d=5 at 20M shots: 0.95× MWPM\nconfirmed on the sealed block')
    return lines + list(ladders.values()) + [star, epoch_txt, note]


fig.suptitle(f'Experiment 16 — {len(curves)} training runs, three distances, one decoder that beats matching',
             fontsize=13.5, color=INK, x=0.008, ha='left')
fig.tight_layout(rect=[0, 0, 1, 0.94])
anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, 'exp16_training.gif')
anim.save(path, writer=PillowWriter(fps=12))
print(f'wrote {path}  ({os.path.getsize(path)/1024/1024:.1f} MB, {len(curves)} curves)')
