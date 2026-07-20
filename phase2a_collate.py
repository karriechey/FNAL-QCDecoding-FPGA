#!/usr/bin/env python3
# Created: 2026-07-16
# Last modified: 2026-07-20
"""Collate the Phase 2a activation-precision sweep into a table + plot.

Reads the per-run CSVs in out_q_phase2a/ (weights fixed at 6 bits; activation word length B in
{32, 8, 6, 4}, where B=32 is the activation-quant-OFF control). Produces:

  (1) a per-(B, seed) table of the logical error rate p_L;
  (2) the WITHIN-SEED table: p_L(B) - p_L(control) for each seed. This subtracts each seed's own
      control, so seed-to-seed convergence scatter does not contaminate the precision comparison.
      (This mattered more before the determinism fix, when seed 2's control landed ~0.011 high on
      a spiked run; with TF_DETERMINISTIC_OPS on, the three controls now sit at 0.04659 / 0.04731 /
      0.046565, all matching their Phase-1 anchors. Within-seed differencing is still the right
      reading -- it is just no longer compensating for a broken control.)
      NOTE these differences are descriptive, NOT a significance test: at n_test=200k the binomial
      SE on one p_L is ~4.7e-4, so a difference of two runs carries SE ~6.7e-4 before training
      variance. For the actual paired test on the same shots, run phase2a_mcnemar.py;
  (3) a plot of p_L vs B with per-seed points, the across-seed mean, the control band, and the
      MWPM reference.

IMPORTANT -- MWPM value. The per-run CSV's `mwpm_p_L` column is 0.0451 and MUST be ignored. It
comes from train_one.lookup_mwpm(), which reads pools/mwpm_baseline.csv keyed only on
(d, p, rounds) -- that lookup has no idea WHICH tail is being evaluated, so it returns a value
belonging to a different shot pool. Any figure of ours quoting 0.0451, or the 0.0518 that the same
lookup produced elsewhere, is quoting the wrong pool.

The correct MWPM on this fresh 200k tail is 0.049405. Provenance, verified 2026-07-20 rather than
assumed: eval_on_tail.py --mcnemar does NOT look the number up -- it builds the 4-channel
rotated_memory_z circuit, takes its detector error model, and decodes these exact shots with
pymatching.Matching.decode_batch, then overwrites the looked-up value with that decode. Every row
of out_q_mcnemar/mcnemar_knee.csv and out_q/fp32_anchor.csv carries mwpm_p_L = 0.049405 from that
path, and phase2a_mcnemar.py re-decodes it independently. This number is the denominator of every
"beats MWPM" claim in the paper, so it is a decode, not a lookup.

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
    ap.add_argument('--plot-exclude-bits', default='4',
                    help='comma-separated B values to leave OUT OF THE PLOT ONLY (tables still '
                         'show them). Defaults to 4: that run collapsed to the base rate 0.2826 '
                         'because its fixed-point format is infeasible (negative fractional '
                         'width), not because 4-bit activations are too coarse to decode. On a '
                         'shared axis it is ~6x every other point, flattens the 0.046-0.050 band '
                         'where the actual result lives, and reads as "4-bit fails" -- the one '
                         'conclusion the data does NOT support. Pass an empty string to include it.')
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
        print('  read: these deltas are DESCRIPTIVE only. A delta inside ~+/-0.0007 is the scale of'
              ' noise\n        for a difference of two independent runs -- but that is the WRONG'
              ' test here, because\n        both runs decode the SAME 200k shots. Use'
              ' phase2a_mcnemar.py for the paired test;\n        it can resolve a real effect that'
              ' this table calls noise (it did for B=8).')

    # control vs anchor sanity (reproduction check)
    if 32 in bits_present:
        print('\n=== control (B=32) vs Phase-1 anchor, per seed (reproduction check) ===')
        for s in seeds:
            c = data.get((32, s))
            if c is not None:
                print(f'  seed{s}: control {c:.5f}  anchor {ANCHOR[s]:.5f}  delta {c-ANCHOR[s]:+.5f}')
        print('  (all three controls reproduce their anchors within ~0.001 now that the determinism'
              '\n   flags are on -- the earlier seed-2 outlier was a spiked, nondeterministic run'
              ' and is gone.)')

    # (3) plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except Exception:
        print('\n[plot] matplotlib not available; table only.'); return
    excl = {int(x) for x in a.plot_exclude_bits.split(',') if x.strip()}
    plot_bits = [b for b in bits_present if b not in excl]
    if not plot_bits:
        print('\n[plot] every bit-width excluded; nothing to draw.'); return

    xs = list(range(len(plot_bits)))                    # categorical x, high bits -> low
    labels = ['32 (off)' if b >= 32 else str(b) for b in plot_bits]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    # Per-seed points, not just the mean. At B=8 the seed spread IS the result: two seeds are
    # individually significant against their own controls in OPPOSITE directions (see
    # phase2a_mcnemar.py), so a mean-only plot would hide the very thing that makes B=8 readable
    # as "no systematic cost" rather than "no difference".
    for s in seeds:
        ys = [data.get((b, s)) for b in plot_bits]
        ax.scatter(xs, [y if y is not None else np.nan for y in ys], alpha=0.6, s=52,
                   label=f'seed {s}', zorder=4)
    mvals = [means[b][0] for b in plot_bits]; merr = [means[b][1] for b in plot_bits]
    ax.errorbar(xs, mvals, yerr=merr, color='black', lw=2, marker='o', ms=7, capsize=4,
                label='mean +/- SE', zorder=5)
    ax.axhline(MWPM_FRESH_TAIL, color='#d62728', ls='--', lw=2, label=f'MWPM = {MWPM_FRESH_TAIL:.4f}')
    if 32 in means:
        ax.axhline(means[32][0], color='#2ca02c', ls=':', lw=1.5, label='control (act off) mean')
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel('activation word length B (bits)  --  6-bit weights fixed')
    ax.set_ylabel('logical error rate  $p_L$  (lower = better)')
    title = ('Phase 2a: decoder accuracy vs activation precision\n'
             'surface code d=5, p=0.010, r=3 -- 10M shots, fresh 200k tail')
    ax.set_title(title, fontsize=10)
    # Legend outside the axes. The plot band is only ~0.004 wide and every corner holds a real
    # point: inside-left covers the seed-0/2 controls, inside-top covers the MWPM line. Putting it
    # to the right keeps all six data points and both reference lines unobstructed.
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.95)
    if excl:
        # State WHY the excluded point is absent, on the figure itself -- otherwise the omission
        # looks like cherry-picking, and the reason is the actual finding.
        shown = ', '.join(f'B={b}' for b in sorted(excl))
        ax.text(0.5, -0.16, f'{shown} omitted: infeasible fixed-point format (negative fractional '
                            f'width), not a precision limit.\nIts run collapsed to the base rate '
                            f'0.2826; the minimum viable B under the current integer widths is 6.',
                transform=ax.transAxes, ha='center', va='top', fontsize=7.5, color='#555555')
    os.makedirs(os.path.dirname(a.plot_out), exist_ok=True)
    fig.savefig(a.plot_out, bbox_inches='tight', dpi=150)
    print(f'\n[plot] wrote {a.plot_out}')


if __name__ == '__main__':
    main()
