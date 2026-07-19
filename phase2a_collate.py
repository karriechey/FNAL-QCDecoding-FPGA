#!/usr/bin/env python3
# Created: 2026-07-16
# Last modified: 2026-07-16
"""Collate the Phase 2a activation-precision sweep into a table + plot.

Reads the per-run CSVs in out_q_phase2a/ (weights fixed at 6 bits; activation word length B in
{32, 8, 6, 4}, where B=32 is the activation-quant-OFF control). Produces:

  (1) a per-(B, seed) table of the logical error rate p_L;
  (2) the WITHIN-SEED table: p_L(B) - p_L(control) for each seed, meaned across seeds. This is
      the honest measure of what activation precision costs, because it subtracts each seed's
      own control -- so the run-to-run baseline scatter (seed 2's control sits ~0.011 high) does
      NOT contaminate the precision comparison;
  (3) a plot of p_L vs B with per-seed points, the across-seed mean, the control band, and the
      MWPM reference.

IMPORTANT -- MWPM value. The per-run CSV's `mwpm_p_L` column is 0.0451 (a stale 10k-tail lookup);
the CORRECT MWPM on the fresh 200k tail is 0.049405 (from the Phase-1 McNemar run on that tail).
We use 0.049405 here and IGNORE the CSV column. (See the MWPM-baseline note in the run log.)

Robust to partial data: if only the control (B=32) runs are present, it shows those; it fills in
8/6/4 as they land.

  python phase2a_collate.py [--dir rcnn_threshold/out_q_phase2a]
"""
import argparse, csv, glob, os
import numpy as np

# Fresh-200k-tail references (from Phase 1). Do NOT read these from the sweep CSVs.
MWPM_FRESH_TAIL = 0.049405
ANCHOR = {0: 0.046675, 1: 0.047715, 2: 0.045580}   # 6-bit-weight / float-activation, per seed


def load(dir_):
    """Return {(act_bits, seed): p_L} from the 10M-shot CSVs in dir_."""
    out = {}
    for f in glob.glob(os.path.join(dir_, '*_ntr10000000.csv')):
        r = next(csv.DictReader(open(f)))
        out[(int(r['act_bits']), int(r['seed']))] = float(r['p_L'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='rcnn_threshold/out_q_phase2a')
    ap.add_argument('--plot-out', default='plots/phase2a_activation_sweep.png')
    a = ap.parse_args()

    data = load(a.dir)
    if not data:
        raise SystemExit(f'no *_ntr10000000.csv in {a.dir}')
    bits_present = sorted({b for (b, s) in data}, reverse=True)   # e.g. [32, 8, 6, 4]
    seeds = sorted({s for (b, s) in data})
    print(f'activation bit-widths present: {bits_present}   seeds: {seeds}')
    print(f'MWPM (fresh 200k tail) = {MWPM_FRESH_TAIL}\n')

    # (1) raw per-(B, seed) p_L, with mean +/- standard error across seeds
    print('=== (1) p_L by activation bit-width B (B=32 is the control, activations off) ===')
    print(f'{"B":>4}  ' + '  '.join(f'seed{s}' for s in seeds) + '   mean      SE       xMWPM')
    means = {}
    for b in bits_present:
        vals = [data.get((b, s)) for s in seeds]
        have = [v for v in vals if v is not None]
        m = float(np.mean(have)); se = float(np.std(have, ddof=1) / np.sqrt(len(have))) if len(have) > 1 else 0.0
        means[b] = (m, se)
        cells = '  '.join(f'{v:.5f}' if v is not None else '  --  ' for v in vals)
        tag = '  (control)' if b >= 32 else ''
        print(f'{b:>4}  {cells}   {m:.5f}  {se:.5f}  {m/MWPM_FRESH_TAIL:.3f}{tag}')

    # (2) WITHIN-SEED cost: p_L(B) - p_L(control), so each seed's baseline cancels
    if 32 in bits_present:
        print('\n=== (2) activation-precision COST, within-seed: p_L(B) - p_L(control_B32) ===')
        print('    (subtracts each seed\'s own control, so run-to-run baseline scatter cancels)')
        print(f'{"B":>4}  ' + '  '.join(f'seed{s}' for s in seeds) + '   mean_delta  SE')
        for b in bits_present:
            if b >= 32:
                continue
            deltas = []
            cells = []
            for s in seeds:
                vb, v32 = data.get((b, s)), data.get((32, s))
                if vb is not None and v32 is not None:
                    d = vb - v32; deltas.append(d); cells.append(f'{d:+.5f}')
                else:
                    cells.append('   --   ')
            if deltas:
                md = float(np.mean(deltas)); se = float(np.std(deltas, ddof=1)/np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
                print(f'{b:>4}  ' + '  '.join(cells) + f'   {md:+.5f}   {se:.5f}')
        print('  read: a delta within ~+/-0.0005 (binomial noise at 200k) = activation quant is free at that B.')

    # control vs anchor sanity (reproduction check)
    if 32 in bits_present:
        print('\n=== control (B=32) vs Phase-1 anchor, per seed (reproduction check) ===')
        for s in seeds:
            c = data.get((32, s))
            if c is not None:
                print(f'  seed{s}: control {c:.5f}  anchor {ANCHOR[s]:.5f}  delta {c-ANCHOR[s]:+.5f}')
        print('  (seed 2 control sits high -- run-to-run nondeterminism; the within-seed table above'
              ' is immune to it.)')

    # (3) plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except Exception:
        print('\n[plot] matplotlib not available; table only.'); return
    xs = list(range(len(bits_present)))                 # categorical x, high bits -> low
    labels = ['32 (off)' if b >= 32 else str(b) for b in bits_present]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for s in seeds:                                     # per-seed points
        ys = [data.get((b, s)) for b in bits_present]
        ax.scatter(xs, [y if y is not None else np.nan for y in ys], alpha=0.5, s=42,
                   label=f'seed {s}')
    mvals = [means[b][0] for b in bits_present]; merr = [means[b][1] for b in bits_present]
    ax.errorbar(xs, mvals, yerr=merr, color='black', lw=2, marker='o', ms=7, capsize=4,
                label='mean +/- SE', zorder=5)
    ax.axhline(MWPM_FRESH_TAIL, color='#d62728', ls='--', lw=2, label=f'MWPM = {MWPM_FRESH_TAIL:.4f}')
    if 32 in means:
        ax.axhline(means[32][0], color='#2ca02c', ls=':', lw=1.5, label='control (act off) mean')
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel('activation word length B (bits)  --  6-bit weights fixed')
    ax.set_ylabel('logical error rate  $p_L$  (lower = better)')
    ax.set_title('Phase 2a: decoder accuracy vs activation precision\n'
                 'surface code d=5, p=0.010, r=3 -- 10M shots, fresh 200k tail', fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='best')
    os.makedirs(os.path.dirname(a.plot_out), exist_ok=True)
    fig.savefig(a.plot_out, bbox_inches='tight', dpi=150)
    print(f'\n[plot] wrote {a.plot_out}')


if __name__ == '__main__':
    main()
