#!/usr/bin/env python3
# Created: 2026-08-21
# Last updated: 2026-08-21
# Error rate against training volume, two views of the same runs.
#
# Left: absolute p_L with each distance's MWPM as a dashed line. Right: the same runs
# divided by their own MWPM, so all three distances share one axis and 1.0 is matching.
#
# Reads results_data.py, so adding a run there updates this figure.
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_data import MWPM, SEALED, rows

C = {5: '#eb6834', 7: '#2a78d6', 9: '#1baf7a'}
C_RCNN = '#8a5fb0'
INK, MUTED, GREY = '#1a1a1a', '#55534e', '#8a8782'
R = rows()

# RCNN, Experiment 13 protocol, carried as a reference line with its own MWPM per point.
RCNN = {5: {10e6: (0.008905, 0.007572), 15e6: (0.011098, 0.007572)},
        9: {10e6: (0.074773, 0.002303)}}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.4, 5.6))
for ax in (axL, axR):
    ax.set_facecolor('white')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('training shots', fontsize=11, color=INK)
    ax.grid(True, which='major', color='#e6e4e0', lw=0.9)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#d8d5d0')
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_xlim(1.6e6, 32e6)
    ax.xaxis.set_major_locator(FixedLocator([2e6, 5e6, 10e6, 20e6]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1e6:g}M'))

def ladder(d):
    """The protocol ladder at one distance: units=140, the standard rate, converged."""
    return sorted([r for r in R if r['d'] == d and r['label'] == 'GRU u140'
                   and not r['diverged']], key=lambda r: r['shots'])

for d in (5, 7, 9):
    lad = ladder(d)
    xs = [r['shots'] for r in lad]
    axL.errorbar(xs, [r['p_L'] for r in lad], yerr=[r['sd'] for r in lad], color=C[d],
                 marker='o', markersize=8, lw=2.2, capsize=4, markeredgecolor='white',
                 markeredgewidth=1.5, zorder=4)
    axL.axhline(MWPM[d], color=C[d], ls='--', lw=1.3, alpha=0.55, zorder=2)
    axL.annotate(f'MWPM d={d}', xy=(1.75e6, MWPM[d]), xytext=(0, 4),
                 textcoords='offset points', fontsize=8.5, color=C[d])
    axL.annotate(f'GRU d={d}', xy=(xs[-1], lad[-1]['p_L']), xytext=(9, 2),
                 textcoords='offset points', fontsize=10, color=C[d], fontweight='medium')
    axR.errorbar(xs, [r['p_L'] / r['mwpm'] for r in lad],
                 yerr=[r['sd'] / r['mwpm'] for r in lad], color=C[d], marker='o',
                 markersize=8, lw=2.2, capsize=4, markeredgecolor='white',
                 markeredgewidth=1.5, zorder=4)

    # off-protocol runs at this distance: wider models, the MLP, LR diagnostics, divergences
    for r in [x for x in R if x['d'] == d and x not in lad]:
        m = {'gru': '^', 'mlp': 's', 'rcnn': 'D'}[r['arch']]
        if r['label'] == 'GRU u140':          # same width, so it is a rate or a failure
            m = 'X' if r['diverged'] else 'v'
        axR.plot([r['shots']], [r['p_L'] / r['mwpm']], marker=m, color=C[d], markersize=8,
                 markerfacecolor='white', markeredgewidth=1.6, zorder=3)

for d, pts in RCNN.items():
    xs = sorted(pts)
    axL.plot(xs, [pts[x][0] for x in xs], color=C_RCNN, ls=':', lw=1.8, marker='D',
             markersize=7, markerfacecolor='white', markeredgewidth=1.6, zorder=5)
    axR.plot(xs, [pts[x][0] / pts[x][1] for x in xs], color=C_RCNN, ls=':', lw=1.8,
             marker='D', markersize=7, markerfacecolor='white', markeredgewidth=1.6, zorder=5)
    axL.annotate(f'RCNN d={d}', xy=(xs[-1], pts[xs[-1]][0]),
                 xytext=(13 if d == 5 else 9, -3 if d == 5 else 1),
                 textcoords='offset points', fontsize=9, color=C_RCNN)

axL.plot([SEALED['shots']], [SEALED['p_L']], marker='*', markersize=17, color=C[5],
         markeredgecolor='white', markeredgewidth=1.2, zorder=6)
axL.annotate('sealed block\n0.95x MWPM', xy=(SEALED['shots'], SEALED['p_L']),
             xytext=(4, -32), textcoords='offset points', fontsize=8.5, color=C[5])

axR.axhline(1.0, color=GREY, ls='--', lw=1.6, zorder=2)
axR.annotate('matches MWPM', xy=(1.75e6, 1.0), xytext=(0, 7), textcoords='offset points',
             fontsize=9, color=MUTED)

axL.set_ylabel('logical error rate  p_L', fontsize=11, color=INK)
axR.set_ylabel('p_L / MWPM p_L', fontsize=11, color=INK)
axL.yaxis.set_major_locator(FixedLocator(
    [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.0125, 0.02, 0.03, 0.05, 0.075]))
axR.yaxis.set_major_locator(FixedLocator([0.9, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30]))
for ax in (axL, axR):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())

h = [plt.Line2D([], [], marker='^', ls='none', color=MUTED, markerfacecolor='white',
                markeredgewidth=1.6, markersize=8, label='wider GRU (200/280)'),
     plt.Line2D([], [], marker='s', ls='none', color=MUTED, markerfacecolor='white',
                markeredgewidth=1.6, markersize=7, label='size-matched MLP'),
     plt.Line2D([], [], marker='v', ls='none', color=MUTED, markerfacecolor='white',
                markeredgewidth=1.6, markersize=8, label='LR 1e-3 diagnostic'),
     plt.Line2D([], [], marker='X', ls='none', color=MUTED, markerfacecolor='white',
                markeredgewidth=1.6, markersize=9, label='diverged'),
     plt.Line2D([], [], marker='D', ls=':', color=C_RCNN, markerfacecolor='white',
                markeredgewidth=1.6, markersize=7, label='RCNN (Exp 13 protocol)')]
axR.legend(handles=h, frameon=False, fontsize=9, loc='lower left', bbox_to_anchor=(0.0, 0.02))

fig.suptitle('Experiment 16 — Logical Error Rate vs Training Set Size at p = 0.004, r = d',
             fontsize=14, color=INK, x=0.008, ha='left')
cap = ('Lines are the protocol ladder: GRU units=140, batch 10,000, constant LR 3e-3, 200 fixed epochs, '
       'minimum-validation-loss checkpoint, scored on [15.2M, 17.0M) with MWPM decoded on those '
       'shots.\nError bars are the standard deviation over seeds. Open markers are off-protocol runs at '
       'the same distance: wider models, the size-matched MLP, LR 1e-3 diagnostics, and the d=9 15M run '
       'that diverged.\nThe d=5 20M and d=7 20M points are augmented sets (15M primary + 5M independently '
       'generated), not prefixes. RCNN follows Experiment 13\'s protocol and its own pools.')
fig.text(0.008, -0.02, cap, fontsize=8.4, color=MUTED, va='top')
fig.tight_layout(rect=[0, 0, 1, 0.95])

OUT = os.environ.get('FIGDIR', os.path.join('..', 'figures'))
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'exp16_data_scaling_all.{ext}'), dpi=200,
                bbox_inches='tight', facecolor='white')
print(f'wrote {OUT}/exp16_data_scaling_all.png and .pdf')
