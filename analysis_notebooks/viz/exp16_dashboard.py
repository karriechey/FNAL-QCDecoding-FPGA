#!/usr/bin/env python3
# Created: 2026-08-21
# Experiment 16 at a glance: one figure, five panels, every scored configuration.
#
#   A  logical error rate against training shots, per distance, with each MWPM
#   B  every configuration as a ratio to its own MWPM, grouped by distance
#   C  validation-loss curves for every completed run (needs results_meta/)
#   D  parameters against ratio -- does spending parameters buy accuracy
#   E  training minutes against ratio -- what each result cost
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_data import MWPM, SEALED, rows, load_curves, load_rcnn_curves

C = {5: '#eb6834', 7: '#2a78d6', 9: '#1baf7a'}
M = {'gru': 'o', 'mlp': 's', 'rcnn': 'D'}
INK, MUTED, GREY, GRID = '#1a1a1a', '#55534e', '#8a8782', '#e6e4e0'
R = rows()

META = os.path.join(os.path.dirname(__file__), '..', '..', 'results_meta')
META = os.environ.get('RESULTS_META', META)

fig = plt.figure(figsize=(17.5, 14.5))
gs = fig.add_gridspec(3, 6, height_ratios=[1.05, 1.45, 0.95], hspace=0.38, wspace=1.15)
axA = fig.add_subplot(gs[0, 0:3])
axB = fig.add_subplot(gs[0, 3:6])
axC = fig.add_subplot(gs[1, 0:6])
axD = fig.add_subplot(gs[2, 0:3])
axE = fig.add_subplot(gs[2, 3:6])


def style(ax, xlab, ylab):
    ax.set_facecolor('white')
    ax.grid(True, which='major', color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#d8d5d0')
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlab, fontsize=10, color=INK)
    ax.set_ylabel(ylab, fontsize=10, color=INK)


# --- A: the data ladders ----------------------------------------------------------------
for d in (5, 7, 9):
    lad = sorted([r for r in R if r['d'] == d and r['arch'] == 'gru'
                  and r['label'] == 'GRU u140'], key=lambda r: r['shots'])
    axA.errorbar([r['shots'] for r in lad], [r['p_L'] for r in lad],
                 yerr=[r['sd'] for r in lad], color=C[d], marker='o', markersize=7,
                 lw=2.2, capsize=3, markeredgecolor='white', markeredgewidth=1.4, zorder=4)
    axA.axhline(MWPM[d], color=C[d], ls='--', lw=1.2, alpha=0.5, zorder=2)
    axA.annotate(f'MWPM d={d}', xy=(1.75e6, MWPM[d]), xytext=(0, 3),
                 textcoords='offset points', fontsize=8, color=C[d], alpha=0.9)
axA.plot([SEALED['shots']], [SEALED['p_L']], marker='*', markersize=17, color=C[5],
         markeredgecolor='white', markeredgewidth=1.2, zorder=6)
axA.annotate('sealed block\n0.95x MWPM', xy=(SEALED['shots'], SEALED['p_L']),
             xytext=(-6, -32), textcoords='offset points', ha='right', fontsize=8.5,
             color=C[5], fontweight='medium')
axA.set_xscale('log'); axA.set_yscale('log'); axA.set_xlim(1.6e6, 38e6)
axA.xaxis.set_major_locator(FixedLocator([2e6, 5e6, 10e6, 20e6]))
axA.xaxis.set_minor_locator(NullLocator())
axA.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1e6:g}M'))
axA.yaxis.set_major_locator(FixedLocator([0.002, 0.003, 0.005, 0.008, 0.0125, 0.02, 0.03, 0.05]))
axA.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
axA.yaxis.set_minor_formatter(NullFormatter())
style(axA, 'training shots', 'logical error rate  p_L')
for d in (5, 7, 9):
    lad = sorted([r for r in R if r['d'] == d and r['label'] == 'GRU u140'],
                 key=lambda r: r['shots'])
    axA.annotate(f'd={d}', xy=(lad[-1]['shots'], lad[-1]['p_L']), xytext=(9, 3),
                 textcoords='offset points', fontsize=10, color=C[d], fontweight='medium')
axA.set_title('A   more data, lower error — d=5 crosses MWPM at 20M',
              fontsize=10.5, color=INK, loc='left', pad=8)

# --- B: everything as a ratio, grouped by distance ---------------------------------------
order = sorted(R, key=lambda r: (r['d'], r['p_L'] / r['mwpm']))
ypos, ylabels, colors = [], [], []
for i, r in enumerate(order):
    ypos.append(i)
    ylabels.append(f"{r['label'].replace('GRU u', 'GRU ')}  {r['shots']/1e6:g}M")
    colors.append(C[r['d']])
ratios = [r['p_L'] / r['mwpm'] for r in order]
axB.barh(ypos, ratios, color=colors, height=0.66, alpha=0.85,
         edgecolor='white', linewidth=1.0, zorder=3)
axB.axvline(1.0, color=GREY, ls='--', lw=1.6, zorder=4)
axB.annotate('matches MWPM', xy=(1.0, len(order) - 0.2), xytext=(4, 0),
             textcoords='offset points', fontsize=8.5, color=MUTED)
for y, v, r in zip(ypos, ratios, order):
    # labels sit outside the bar, and clear of the 1x rule for the near-parity rows
    dx = 6 if v > 1.05 else 10
    axB.annotate(f'{v:.2f}x', xy=(v, y), xytext=(dx, 0), textcoords='offset points',
                 va='center', fontsize=8,
                 color=INK if v < 1 else MUTED,
                 fontweight='bold' if v < 1 else 'normal')
axB.set_yticks(ypos); axB.set_yticklabels(ylabels, fontsize=7.6)
axB.set_xscale('log'); axB.set_xlim(0.8, 60)
axB.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 30]))
axB.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}x'))
axB.xaxis.set_minor_formatter(NullFormatter())
style(axB, 'p_L / MWPM p_L', '')
axB.set_title('B   every configuration against its own MWPM',
              fontsize=10.5, color=INK, loc='left', pad=8)

# --- C: validation curves for every completed run ----------------------------------------
curves = load_curves(META) + load_rcnn_curves(
    os.environ.get('RESULTS_META_EAF', os.path.join(META, '..', 'results_meta_eaf')))
C_RCNN = '#8a5fb0'
for c in curves:
    col = C_RCNN if c['arch'] == 'rcnn' else C[c['d']]
    ls = {'gru': '-', 'mlp': (0, (3, 2)), 'rcnn': (0, (1, 1.6))}[c['arch']]
    axC.plot(range(1, len(c['val']) + 1), c['val'], color=col,
             lw=1.9 if c['arch'] == 'rcnn' else 1.1,
             alpha=0.9 if c['arch'] == 'rcnn' else (0.7 if c['arch'] == 'gru' else 0.5),
             ls=ls, zorder=4 if c['arch'] == 'rcnn' else 3)
    axC.plot([c['best'] + 1], [c['val'][c['best']]], marker='o',
             markersize=5.5 if c['arch'] == 'rcnn' else 4, color=col,
             markeredgecolor='white', markeredgewidth=1.0, zorder=6)
    if c['arch'] == 'rcnn':
        axC.annotate(f"RCNN d={c['d']}", xy=(len(c['val']), c['val'][-1]), xytext=(7, -2),
                     textcoords='offset points', fontsize=8.5, color=C_RCNN)
drawn = len(curves)
hC = [plt.Line2D([], [], color=C[d], lw=2, label=f'd = {d}') for d in (5, 7, 9)]
hC.append(plt.Line2D([], [], color=MUTED, lw=1.6, ls=(0, (3, 2)), label='MLP'))
hC.append(plt.Line2D([], [], color=C_RCNN, lw=1.9, ls=(0, (1, 1.6)), label='RCNN'))
axC.legend(handles=hC, frameon=False, fontsize=10, loc='upper right', ncol=5, labelcolor=MUTED)
axC.set_yscale('log')
axC.yaxis.set_major_locator(FixedLocator([0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.45]))
axC.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
axC.yaxis.set_minor_formatter(NullFormatter())
style(axC, 'epoch', 'validation loss')
axC.set_title(f'C   validation loss, {drawn} runs — one line per seed per configuration '
              '(dot = the scored epoch; RCNN stops where early stopping fired)',
              fontsize=10.5, color=INK, loc='left', pad=8)
if drawn == 0:
    axC.text(0.5, 0.5, 'run the rsync from grace1 to populate results_meta/',
             transform=axC.transAxes, ha='center', fontsize=10, color=MUTED)

# --- D and E: what accuracy cost ----------------------------------------------------------
for ax, key, xlab, title in (
        (axD, 'params', 'parameters', 'D   parameters bought little — the wins came from data'),
        (axE, 'minutes', 'training minutes (1 seed)', 'E   and cost minutes, not hours')):
    for r in R:
        ax.plot([r[key]], [r['p_L'] / r['mwpm']], marker=M[r['arch']], markersize=8,
                color=C[r['d']], markerfacecolor=C[r['d']] if r['arch'] == 'gru' else 'white',
                markeredgewidth=1.6, markeredgecolor=C[r['d']], zorder=4, alpha=0.9)
    ax.axhline(1.0, color=GREY, ls='--', lw=1.5, zorder=2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 30]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}x'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    style(ax, xlab, 'p_L / MWPM p_L')
    ax.set_title(title, fontsize=10.5, color=INK, loc='left', pad=8)
axD.xaxis.set_major_locator(FixedLocator([70000, 100000, 200000, 300000]))
axD.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1000:g}k'))
axD.xaxis.set_minor_formatter(NullFormatter())
axE.xaxis.set_major_locator(FixedLocator([10, 30, 60, 120, 360, 1200, 6000]))
axE.xaxis.set_major_formatter(FuncFormatter(
    lambda v, _: f'{v:g} min' if v < 100 else f'{round(v/60)} h'))
axE.xaxis.set_minor_formatter(NullFormatter())

hand = [plt.Line2D([], [], marker='o', ls='none', color=MUTED, markersize=8, label='GRU'),
        plt.Line2D([], [], marker='s', ls='none', color=MUTED, markerfacecolor='white',
                   markeredgewidth=1.6, markersize=8, label='MLP'),
        plt.Line2D([], [], marker='D', ls='none', color=MUTED, markerfacecolor='white',
                   markeredgewidth=1.6, markersize=8, label='RCNN')]
axE.legend(handles=hand, frameon=False, fontsize=9, loc='upper right')

fig.suptitle('Experiment 16 — surface-code decoding at p=0.004: 17 configurations across '
             'three distances', fontsize=15, color=INK, x=0.008, ha='left', y=0.985)
fig.text(0.008, 0.952, 'A 69,441-parameter GRU reaches 0.95x MWPM at d=5 given 20M training '
         'shots, confirmed on a sealed block. Colour is distance throughout; marker is '
         'architecture.', fontsize=10, color=MUTED)

OUT = os.environ.get('FIGDIR', os.path.join('..', 'figures'))
os.makedirs(OUT, exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'exp16_dashboard.{ext}'), dpi=190,
                bbox_inches='tight', facecolor='white')
print(f'wrote {OUT}/exp16_dashboard.png and .pdf   ({drawn} learning curves)')
