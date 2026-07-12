#!/usr/bin/env python3
"""Collate the QAT per-run CSVs into the weight-bit-width Pareto (Step 2 deliverable).

Reads OUT_DIR/rcnn_d{d}_p{p}_r{r}_w{bits}_seed{seed}_ntr{ntr}.csv (from train_one_quantized)
plus OUT_DIR/fp32_anchor.csv (eval_on_tail rows for the 32-bit point). Per bit-width:
  mean p_L, combined error = sqrt(SEM_seed^2 + p(1-p)/n_test), size KB = n_params*bits/8/1024,
  ratio to MWPM. Pareto plot (x = size KB, y = p_L) in the plot_pl_vs_n_gpu_v2 honest style:
faint per-seed scatter, mean + combined error bars, MWPM band, KNEE annotated (smallest bits
still at parity = mean within one combined error of MWPM, or below).

MWPM anchor: the FRESH-tail MWPM p_L (NOT mwpm_baseline.csv's 10k value). Pass --mwpm; else
read it from fp32_anchor.csv's mwpm_p_L column.
"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt

BLUE, GREEN, RED = '#1f77b4', '#2ca02c', '#d62728'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.expanduser('~/rcnn_threshold/out_q'))
    ap.add_argument('--d', type=int, default=5); ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--mwpm', type=float, default=None, help='fresh-tail MWPM p_L (overrides fp32 csv)')
    ap.add_argument('--mwpm-ntest', type=int, default=200000)
    ap.add_argument('--plot-out', default='./plots/rcnn_d5_r3_qat_pareto.png')
    args = ap.parse_args()
    d, p, r = args.d, args.p, args.rounds

    pat = os.path.join(args.out_dir, f'rcnn_d{d}_p{p:.3f}_r{r}_w*_seed*_ntr*.csv')
    files = sorted(glob.glob(pat))
    print(f'[collate] {len(files)} QAT per-run CSVs from {pat}')
    rows = [pd.read_csv(f) for f in files]

    # FP32 anchor rows (eval_on_tail schema) -> normalize to the same columns
    fp32 = os.path.join(args.out_dir, 'fp32_anchor.csv')
    mwpm = args.mwpm
    if os.path.exists(fp32):
        fa = pd.read_csv(fp32).drop_duplicates(subset=['weights'], keep='last')
        # need n_params for size: reuse a QAT row's n_params (same architecture)
        npar = int(rows[0]['n_params'].iloc[0]) if rows else None
        for _, x in fa.iterrows():
            seed = int(str(x['weights']).split('seed')[1].split('_')[0])
            rows.append(pd.DataFrame([dict(weight_bits=32, seed=seed, p_L=float(x['p_L']),
                n_params=npar, n_test=int(x['n_test']), mwpm_p_L=x.get('mwpm_p_L', ''))]))
        if mwpm is None:
            m = pd.to_numeric(fa['mwpm_p_L'], errors='coerce').dropna()
            if len(m): mwpm = float(m.mean())
    if not rows:
        raise SystemExit('[collate] no data')
    df = pd.concat(rows, ignore_index=True)
    df = df.drop_duplicates(subset=['weight_bits', 'seed'], keep='last')
    if mwpm is None:
        raise SystemExit('[collate] no MWPM anchor -- pass --mwpm (fresh-tail value)')
    n_test = int(df['n_test'].dropna().mode().iloc[0])
    mwpm_band = float(np.sqrt(mwpm * (1 - mwpm) / args.mwpm_ntest))

    agg = []
    for bits, g in df.groupby('weight_bits'):
        pl = g['p_L'].to_numpy(float); ns = len(pl); m = float(pl.mean())
        sem = float(pl.std(ddof=1) / np.sqrt(ns)) if ns > 1 else 0.0
        binom = float(np.sqrt(m * (1 - m) / n_test))
        comb = float(np.sqrt(sem**2 + binom**2))
        npar = int(g['n_params'].dropna().iloc[0])
        agg.append(dict(bits=int(bits), n=ns, mean_pL=m, sem=sem, binom=binom, comb=comb,
                        size_kb=npar * int(bits) / 8 / 1024, ratio=m / mwpm))
    agg = pd.DataFrame(agg).sort_values('size_kb').reset_index(drop=True)

    # knee = smallest bits at parity (mean within one combined err of MWPM, or below)
    parity = agg[(agg.mean_pL < mwpm) | ((agg.mean_pL - mwpm).abs() <= agg.comb)]
    knee = parity.sort_values('bits').iloc[0] if len(parity) else None

    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 160, 'font.size': 11,
                         'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True})
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for _, a in agg.iterrows():
        g = df[df.weight_bits == a.bits]
        xs = [a.size_kb] * len(g)
        ax.scatter(xs, g.p_L, color=BLUE, alpha=0.28, s=26, zorder=2,
                   label='per-seed runs' if a.bits == agg.bits.iloc[0] else None)
    ax.axhline(mwpm, color=GREEN, ls='--', lw=2, zorder=3,
               label=f'MWPM = {mwpm:.4f} (fresh tail, target)')
    ax.fill_between([agg.size_kb.min() * 0.8, agg.size_kb.max() * 1.2],
                    mwpm - mwpm_band, mwpm + mwpm_band, color=GREEN, alpha=0.15, zorder=1)
    ax.errorbar(agg.size_kb, agg.mean_pL, yerr=agg.comb, color=BLUE, lw=2, marker='o',
                ms=6, capsize=4, zorder=4,
                label='QAT mean ($\\pm\\sqrt{\\mathrm{SEM}_{seed}^2+p(1-p)/n_{test}}$)')
    for _, a in agg.iterrows():
        ax.annotate(f'{a.bits}b\n{a.ratio:.2f}×', (a.size_kb, a.mean_pL),
                    textcoords='offset points', xytext=(0, 12), ha='center', fontsize=8.5,
                    color=BLUE, bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.7))
    if knee is not None:
        ax.scatter([knee.size_kb], [knee.mean_pL], s=180, facecolors='none',
                   edgecolors=RED, lw=2, zorder=5,
                   label=f'knee: {int(knee.bits)}-bit still at parity ({knee.size_kb:.1f} KB)')

    ax.set_xscale('log'); ax.set_xlabel('model size (KB)  = params × bits / 8   (log)')
    ax.set_ylabel('logical error rate  $p_L$  (lower = better)')
    fig.suptitle('FullRCNNModel QAT weight-bit-width Pareto', fontsize=13, fontweight='bold', y=0.98)
    ax.set_title(f'surface code d={d}, p={p}, r={r} · 10M shots · fresh {n_test:,} tail', fontsize=10)
    ax.legend(loc='upper right', framealpha=0.9)
    os.makedirs(os.path.dirname(args.plot_out), exist_ok=True)
    fig.savefig(args.plot_out, bbox_inches='tight')
    print(f'[collate] wrote {args.plot_out}')

    print('\n bits   size_KB   mean_pL   comb_err   xMWPM   n')
    for _, a in agg.iterrows():
        print(f' {a.bits:>4}  {a.size_kb:>7.1f}   {a.mean_pL:.5f}   {a.comb:.5f}   {a.ratio:.3f}   {int(a.n)}')
    print(f' MWPM={mwpm:.5f} ±{mwpm_band:.5f}  (n_test={args.mwpm_ntest})')
    if knee is not None:
        print(f' KNEE: {int(knee.bits)}-bit ({knee.size_kb:.1f} KB) still statistically at MWPM parity')


if __name__ == '__main__':
    main()
