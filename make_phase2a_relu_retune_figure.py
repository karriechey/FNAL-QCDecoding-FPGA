#!/usr/bin/env python3
# Created: 2026-07-25
# Last modified: 2026-07-25
"""Rebuild ONLY the Phase 2a-ReLU retune figure, as PNG + vector PDF.

Companion to make_mcnemar_knee_figure.py, and it exists for the same reason: the figure
normally lives in a cell of QUANTIZATION_EXPERIMENTS.ipynb, but executing that notebook on
the laptop kills the kernel (later cells import the TF/QKeras stack, which conda base
cannot load). This script needs only pandas + matplotlib over the result CSVs on disk.

The plotting code is a copy of the notebook cell in build_quant_notebook.py; if you change
the figure, change it in BOTH places (or delete the notebook copy) so they can't drift.

    python make_phase2a_relu_retune_figure.py [--out figures/phase2a_relu_retune.png]

What it shows: the within-seed activation-quantization cost at B=6, under the two decoder
ReLU integer widths (abs-max I=6/F=0 vs retuned I=5/F=1). Each bar differences a B=6 run
against that same seed's own act32 control, so the comparison is paired within seed.
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Resolve paths from THIS file, not the working directory, so the script behaves the same
# no matter where it is invoked from.
REPO = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(REPO, 'rcnn_threshold')
OUT_2A = os.path.join(BASE, 'out_q_phase2a')              # relu I=6 (abs-max), primary sweep
OUT_2A_RELU5 = os.path.join(BASE, 'out_q_phase2a_relu5')  # relu I=5 (profiled), the retune

SEEDS = [0, 1, 2]


def _pl(d):
    """{(act_bits, seed): p_L} from the 10M-shot CSVs in dir d.

    Each result dir carries its OWN act32 controls (they were copied into the relu5 dir on
    purpose), so each B=6 run is differenced against a control trained in the same setting.
    """
    out = {}
    for f in glob.glob(os.path.join(d, '*_ntr10000000.csv')):
        r = pd.read_csv(f).iloc[0]
        out[(int(r['act_bits']), int(r['seed']))] = float(r['p_L'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'figures', 'phase2a_relu_retune.png'),
                    help='PNG path; the PDF is written alongside with the same stem')
    a = ap.parse_args()

    absmax = _pl(OUT_2A)
    retune = _pl(OUT_2A_RELU5)

    # within-seed cost = p_L(B=6) - p_L(that seed's own act32 control)
    d_abs = [absmax[(6, s)] - absmax[(32, s)] for s in SEEDS]
    d_ret = [retune[(6, s)] - retune[(32, s)] for s in SEEDS]
    m_abs, m_ret = np.mean(d_abs), np.mean(d_ret)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = np.arange(len(SEEDS)); w = 0.36
    ax.bar(x - w/2, np.array(d_abs)*1e3, w, label='abs-max ReLU I=6 (F=0)', color='#4C78A8')
    ax.bar(x + w/2, np.array(d_ret)*1e3, w, label='retuned ReLU I=5 (F=1)', color='#F58518')
    # mean lines (in the same 1e-3 units as the bars)
    ax.axhline(m_abs*1e3, color='#4C78A8', ls='--', lw=1.3, label=f'mean abs-max = {m_abs:+.5f}')
    ax.axhline(m_ret*1e3, color='#F58518', ls='--', lw=1.3, label=f'mean retuned = {m_ret:+.5f}')
    ax.axhline(0, color='0.4', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f'seed {s}' for s in SEEDS])
    ax.set_ylabel(r'within-seed cost  $p_L(B{=}6) - p_L(\mathrm{control})$   [$\times10^{-3}$]')
    ax.set_title('Phase 2a-ReLU: B=6 activation cost vs decoder-ReLU integer width\n'
                 'lowering I=6->5 (frac 0->1) did NOT reduce the cost -- seeds move both ways, '
                 'mean unchanged', fontsize=9)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)

    out_png = a.out
    out_pdf = os.path.splitext(out_png)[0] + '.pdf'
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')   # screen / notebook inline
    fig.savefig(out_pdf, bbox_inches='tight')            # vector, for the paper
    print('saved ->', out_png)
    print('saved ->', out_pdf)
    print(f'abs-max deltas: {[round(v, 5) for v in d_abs]}  mean {m_abs:+.5f}')
    print(f'retuned deltas: {[round(v, 5) for v in d_ret]}  mean {m_ret:+.5f}')


if __name__ == '__main__':
    main()
