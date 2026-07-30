#!/usr/bin/env python3
# Created: 2026-07-25
# Last modified: 2026-07-25
"""Rebuild ONLY the Phase-1 McNemar knee figure, as PNG + vector PDF.

Why this exists separately from build_quant_notebook.py: that builder emits the whole
QUANTIZATION_EXPERIMENTS.ipynb, and executing that notebook end-to-end on the laptop kills
the kernel (later cells pull in the TF/QKeras stack, which conda base can't load). This
script carries only the cells the knee figure actually needs -- pandas + matplotlib over
the result CSVs on disk -- so the paper figure can be regenerated without a live kernel.

The plotting code is a copy of the notebook cell in build_quant_notebook.py; if you change
the figure, change it in BOTH places (or delete the notebook copy) so they can't drift.

    python make_mcnemar_knee_figure.py [--out figures/mcnemar_knee.png]

Writes the .png (screen) and the matching .pdf (vector, for the paper) next to each other.
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Resolve everything relative to THIS file, not the working directory, so the script gives
# the same answer whether it is run from the repo root or anywhere else.
REPO = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(REPO, 'rcnn_threshold')
OUT_Q = os.path.join(BASE, 'out_q')            # Step-2 Pareto CSVs + fp32 anchor
OUT_MC = os.path.join(BASE, 'out_q_mcnemar')   # Phase-1 McNemar rows


def load_pareto():
    """Return (g, mwpm, fa): the seed-aggregated p_L table, the MWPM baseline, the anchor rows.

    Mirrors the notebook's Pareto cell: one CSV per (bits, seed) run, plus fp32_anchor.csv
    standing in for bits=32 (that point reuses the existing 10M weights, it was never
    retrained). Duplicated (bits, seed) rows keep the LAST one -- reruns append.
    """
    rows = [pd.read_csv(f) for f in
            sorted(glob.glob(OUT_Q + '/rcnn_d5_p0.010_r3_w*_seed*_ntr*.csv'))]
    fa = pd.read_csv(os.path.join(OUT_Q, 'fp32_anchor.csv')).drop_duplicates('weights', keep='last')
    npar = int(rows[0]['n_params'].iloc[0]) if rows else np.nan
    for _, x in fa.iterrows():
        sd = int(str(x['weights']).split('seed')[1].split('_')[0])
        rows.append(pd.DataFrame([dict(weight_bits=32, seed=sd, p_L=float(x['p_L']),
                                       n_params=npar, n_test=int(x['n_test']))]))

    # The MWPM baseline comes from the anchor's mwpm_p_L, which eval_on_tail.py --mcnemar
    # overwrote with a real PyMatching decode of THESE tail shots. Do not read the baseline
    # out of a sweep CSV column -- that one is a (d, p, rounds) lookup blind to the tail.
    m = pd.to_numeric(fa['mwpm_p_L'], errors='coerce').dropna()
    mwpm = float(m.mean()) if len(m) else None

    df = pd.concat(rows, ignore_index=True).drop_duplicates(['weight_bits', 'seed'], keep='last')
    g = df.groupby('weight_bits')['p_L'].agg(['mean', 'std', 'count'])
    return g, mwpm, fa


def load_mcnemar(fa):
    """Phase-1 McNemar rows for bits={6,8}, concatenated with the bits=32 anchor rows."""
    mc = pd.read_csv(os.path.join(OUT_MC, 'mcnemar_knee.csv')).drop_duplicates('weights', keep='last')
    mc['bits'] = mc['weights'].str.extract(r'_w(\d+)_').astype(int)
    mc['seed'] = mc['weights'].str.extract(r'seed(\d+)_').astype(int)
    return pd.concat([
        fa.assign(bits=32, seed=fa['weights'].str.extract(r'seed(\d+)_')[0].astype(int)),
        mc,
    ], ignore_index=True).drop_duplicates(['bits', 'seed'], keep='last')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'figures', 'mcnemar_knee.png'),
                    help='PNG path; the PDF is written alongside with the same stem')
    a = ap.parse_args()

    g, mwpm, fa = load_pareto()
    mc_all = load_mcnemar(fa)
    print('MWPM (fresh tail) =', mwpm)

    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.5, wspace=0.35)

    # top: p_L vs bits, mean +/- std across seeds, MWPM reference line
    ax0 = fig.add_subplot(gs[0, :])
    ax0.errorbar(g.index, g['mean'], yerr=g['std'], marker='o', capsize=4,
                 color='tab:blue', label='RCNN p_L (mean +/- std, n=3 seeds)')
    if mwpm:
        ax0.axhline(mwpm, color='tab:red', ls='--', label=f'MWPM p_L = {mwpm:.5f}')
    ax0.set_xlabel('weight bits'); ax0.set_ylabel('logical error rate p_L')
    ax0.set_title('RCNN vs MWPM across weight bit-width (d=5, r=3, fresh 200k-shot tail)')
    ax0.legend(); ax0.grid(alpha=0.3)

    # bottom: 2x2 McNemar contingency per bit-width, seed-averaged, fraction of n_test
    for i, b in enumerate(sorted(mc_all['bits'].unique())):
        ax = fig.add_subplot(gs[1, i])
        sub = mc_all[mc_all['bits'] == b]
        nte = sub['n_test'].iloc[0]
        table = np.array([[sub['both_right'].mean(), sub['rcnn_only'].mean()],
                          [sub['mwpm_only'].mean(), sub['both_wrong'].mean()]]) / nte
        ax.imshow(table, cmap='Blues', vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f'{table[r, c]*100:.2f}%', ha='center', va='center',
                        color='white' if table[r, c] > 0.5 else 'black', fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['MWPM right', 'MWPM wrong'], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(['RCNN right', 'RCNN wrong'], fontsize=8)
        p_worst = sub['p_exact'].max()  # worst (least significant) seed at this bit-width
        ax.set_title(f'{b}-bit\nworst p={p_worst:.1e}, n={len(sub)} seeds', fontsize=9)

    fig.suptitle('Phase 1 knee: paired McNemar, RCNN vs MWPM, shot-by-shot on shared tail', y=1.02)

    out_png = a.out
    out_pdf = os.path.splitext(out_png)[0] + '.pdf'
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')   # screen / notebook inline
    fig.savefig(out_pdf, bbox_inches='tight')            # vector, for the paper
    print('saved ->', out_png)
    print('saved ->', out_pdf)


if __name__ == '__main__':
    main()
