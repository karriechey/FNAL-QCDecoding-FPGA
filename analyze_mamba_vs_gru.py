#!/usr/bin/env python3
# Created: 2026-09-21
# Last modified: 2026-09-21
"""Paired analysis of the d=7 Mamba-vs-GRU comparison from per-shot dumps.

Inputs are the per-shot npz files written on the GPU host:
  Mamba  eval_mamba_on_tail.py   --dump-per-shot   (columns mamba_pred / mamba_correct)
  GRU    eval_student_on_tail.py --dump-per-shot   (columns student_pred / student_correct)
  MWPM   either the mwpm_correct column both of the above carry, or a separate dump from
         mwpm_on_block.py --dump-per-shot

For every dump: error count, p_L, 95% binomial confidence interval (Clopper-Pearson,
exact), and p_L / MWPM. Per architecture: mean and sample standard deviation over seeds.
Pairwise, per seed: McNemar on identical shots for Mamba vs GRU, Mamba vs MWPM and GRU vs
MWPM, reporting b, c, n_discordant, exact and chi-square p-values, and the absolute
difference in p_L.

p-values are NOT combined across seeds. Each seed is its own paired test on the same
shots; the seed-level spread is reported as mean +/- sd of p_L, and the per-seed tests
stand on their own.

Alignment is by absolute shot index and is strict: every dump must cover exactly the same
set of shot indices, and the truth labels must agree shot by shot, or the script exits
with an error naming the offending file. Comparing row i of one file with row i of another
is never done.

  python analyze_mamba_vs_gru.py \
      --mamba per_shot_mamba_seed0.npz per_shot_mamba_seed1.npz per_shot_mamba_seed2.npz \
      --gru   per_shot_gru_seed0.npz   per_shot_gru_seed1.npz   per_shot_gru_seed2.npz \
      --mwpm  mwpm_sealed_test_per_shot.npz \
      --out-csv mamba_vs_gru_d7_sealed.csv --out-md mamba_vs_gru_d7_sealed.md
"""
import argparse
import csv
import os
import sys

import numpy as np

# The 2x2 is computed here rather than imported from eval_on_tail so this script runs
# in any environment with numpy + scipy (no TensorFlow, no stim), e.g. on the Mac after
# the dumps are pulled back.


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided binomial confidence interval for k successes in n trials."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def mcnemar(a_correct, b_correct):
    """Paired 2x2 between decoders A and B on identical shots.

    b = A right, B wrong  (A wins the shot)
    c = A wrong, B right  (B wins the shot)
    Returns the four cells, discordant count, chi-square with continuity correction and
    its p-value, and the exact two-sided binomial p-value on (b, b + c) at 0.5.
    """
    from scipy.stats import chi2, binomtest
    a = np.asarray(a_correct, dtype=bool)
    b_ = np.asarray(b_correct, dtype=bool)
    both = int((a & b_).sum())
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    neither = int((~a & ~b_).sum())
    n_disc = b + c
    chi2_cc = ((abs(b - c) - 1) ** 2 / n_disc) if n_disc else 0.0
    p_chi2 = float(chi2.sf(chi2_cc, 1)) if n_disc else 1.0
    p_exact = float(binomtest(b, n_disc, 0.5, alternative='two-sided').pvalue) if n_disc else 1.0
    return dict(both_right=both, b=b, c=c, both_wrong=neither, n_discordant=n_disc,
                net_a_wins=b - c, chi2_cc=chi2_cc, p_chi2=p_chi2, p_exact=p_exact)


def load_dump(path, kind):
    """Return dict(idx, truth, correct, mwpm_correct or None, label) for one dump.

    kind selects the decoder column: 'mamba' -> mamba_correct, 'gru' -> student_correct,
    'mwpm' -> mwpm_correct. Anything missing is a hard error naming the file and its keys.
    """
    if not os.path.exists(path):
        sys.exit(f"[analyze] MISSING {path}")
    z = np.load(path, allow_pickle=False)
    keys = set(z.files)
    col = {'mamba': 'mamba_correct', 'gru': 'student_correct', 'mwpm': 'mwpm_correct'}[kind]
    if col not in keys:
        sys.exit(f"[analyze] {os.path.basename(path)} has no '{col}' column (kind={kind}); "
                 f"keys are {sorted(keys)}")
    idx_key = 'shot_idx' if 'shot_idx' in keys else ('tail_idx' if 'tail_idx' in keys else None)
    if idx_key is None:
        sys.exit(f"[analyze] {os.path.basename(path)} has no shot index column; "
                 f"keys are {sorted(keys)}")
    if 'truth' not in keys:
        sys.exit(f"[analyze] {os.path.basename(path)} has no 'truth' column")
    idx = np.asarray(z[idx_key]).reshape(-1).astype(np.int64)
    if len(np.unique(idx)) != len(idx):
        sys.exit(f"[analyze] {os.path.basename(path)} has duplicate shot indices")
    return dict(
        path=path, name=os.path.basename(path), idx=idx,
        truth=np.asarray(z['truth']).reshape(-1).astype(np.int8),
        correct=np.asarray(z[col]).reshape(-1).astype(bool),
        mwpm_correct=(np.asarray(z['mwpm_correct']).reshape(-1).astype(bool)
                      if 'mwpm_correct' in keys and kind != 'mwpm' else None),
    )


def align_all(dumps):
    """Sort every dump by shot index and assert they cover the identical index set with
    identical truth labels. Returns the shared sorted index."""
    ref = dumps[0]
    ref_order = np.argsort(ref['idx'])
    ref_idx = ref['idx'][ref_order]
    for dmp in dumps:
        order = np.argsort(dmp['idx'])
        for k in ('idx', 'truth', 'correct', 'mwpm_correct'):
            if dmp[k] is not None:
                dmp[k] = dmp[k][order]
        if len(dmp['idx']) != len(ref_idx) or not np.array_equal(dmp['idx'], ref_idx):
            only_a = np.setdiff1d(ref_idx, dmp['idx']).size
            only_b = np.setdiff1d(dmp['idx'], ref_idx).size
            sys.exit(f"[analyze] shot-index mismatch: {dmp['name']} covers "
                     f"[{dmp['idx'].min():,}, {dmp['idx'].max() + 1:,}) n={len(dmp['idx']):,} "
                     f"but {ref['name']} covers [{ref_idx.min():,}, {ref_idx.max() + 1:,}) "
                     f"n={len(ref_idx):,}  ({only_a:,} shots only in the reference, "
                     f"{only_b:,} only in this file). Every dump must score the SAME block.")
        if not np.array_equal(dmp['truth'], ref['truth']):
            n_bad = int((dmp['truth'] != ref['truth']).sum())
            sys.exit(f"[analyze] truth labels disagree on {n_bad:,} shots between "
                     f"{dmp['name']} and {ref['name']}: same indices, different data "
                     f"(different pool?).")
    return ref_idx


def summarize(correct, mwpm_pL):
    n = len(correct)
    k = int((~correct).sum())
    pL = k / n
    lo, hi = clopper_pearson(k, n)
    return dict(n=n, errors=k, p_L=pL, ci_lo=lo, ci_hi=hi,
                ratio=(pL / mwpm_pL if mwpm_pL else float('nan')))


def fmt_p(p):
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def run():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mamba', nargs='+', required=True, help='Mamba per-shot npz, one per seed')
    ap.add_argument('--gru', nargs='+', required=True, help='GRU per-shot npz, one per seed')
    ap.add_argument('--mwpm', default=None,
                    help='mwpm_on_block.py per-shot npz. Optional: without it the '
                         'mwpm_correct column carried by the model dumps is used, and it '
                         'is asserted identical across every dump.')
    ap.add_argument('--seeds', type=int, nargs='*', default=None,
                    help='seed labels for the dumps, in the order given (default 0,1,..). '
                         'Mamba seed i is paired with GRU seed i.')
    ap.add_argument('--out-csv', default=None)
    ap.add_argument('--out-md', default=None)
    args = ap.parse_args()

    mamba = [load_dump(f, 'mamba') for f in args.mamba]
    gru = [load_dump(f, 'gru') for f in args.gru]
    seeds = args.seeds or list(range(max(len(mamba), len(gru))))
    dumps = mamba + gru
    mw = None
    if args.mwpm:
        mw = load_dump(args.mwpm, 'mwpm')
        dumps.append(mw)
    idx = align_all(dumps)
    n = len(idx)
    truth = mamba[0]['truth']
    print(f"[analyze] {n:,} shots aligned on shot_idx [{idx.min():,}, {idx.max() + 1:,}); "
          f"truth agrees across all {len(dumps)} dumps; base rate {truth.mean():.5f}")

    # One MWPM correctness vector for the block. From --mwpm when given, otherwise from
    # the model dumps, which must all carry the same one (same shots, same decoder).
    if mw is not None:
        mwpm_correct = mw['correct']
        for dmp in mamba + gru:
            if dmp['mwpm_correct'] is not None and not np.array_equal(dmp['mwpm_correct'],
                                                                      mwpm_correct):
                sys.exit(f"[analyze] mwpm_correct inside {dmp['name']} differs from "
                         f"{mw['name']}: MWPM was not decoded identically.")
    else:
        carriers = [dmp for dmp in mamba + gru if dmp['mwpm_correct'] is not None]
        if not carriers:
            sys.exit("[analyze] no MWPM: pass --mwpm, or dump the models with MWPM enabled")
        mwpm_correct = carriers[0]['mwpm_correct']
        for dmp in carriers[1:]:
            if not np.array_equal(dmp['mwpm_correct'], mwpm_correct):
                sys.exit(f"[analyze] mwpm_correct differs between {carriers[0]['name']} "
                         f"and {dmp['name']} on the same shots.")
    mw_sum = summarize(mwpm_correct, None)
    mwpm_pL = mw_sum['p_L']

    lines = []
    rows = []

    def out(s=''):
        lines.append(s)
        print(s)

    out(f"\n## Block: {n:,} shots, shot_idx [{idx.min():,}, {idx.max() + 1:,})")
    out(f"MWPM: {mw_sum['errors']:,} errors, p_L = {mwpm_pL:.6f} "
        f"[{mw_sum['ci_lo']:.6f}, {mw_sum['ci_hi']:.6f}] 95% CI")

    out("\n### Per run")
    out("| model | seed | errors | p_L | 95% CI | p_L / MWPM |")
    out("|---|---:|---:|---|---|---:|")
    per_arch = {}
    for arch, dl in (('Mamba', mamba), ('GRU', gru)):
        for s, dmp in zip(seeds, dl):
            sm = summarize(dmp['correct'], mwpm_pL)
            per_arch.setdefault(arch, []).append(sm['p_L'])
            out(f"| {arch} | {s} | {sm['errors']:,} | {sm['p_L']:.6f} | "
                f"[{sm['ci_lo']:.6f}, {sm['ci_hi']:.6f}] | {sm['ratio']:.4f} |")
            rows.append(dict(kind='run', model=arch, seed=s, file=dmp['name'], n=n,
                             errors=sm['errors'], p_L=sm['p_L'], ci_lo=sm['ci_lo'],
                             ci_hi=sm['ci_hi'], mwpm_p_L=mwpm_pL, ratio_vs_mwpm=sm['ratio']))

    out("\n### Seed mean and sd")
    out("| model | seeds | mean p_L | sd p_L | mean ratio vs MWPM |")
    out("|---|---:|---|---|---:|")
    for arch, vals in per_arch.items():
        v = np.asarray(vals)
        sd = float(v.std(ddof=1)) if len(v) > 1 else float('nan')
        out(f"| {arch} | {len(v)} | {v.mean():.6f} | {sd:.6f} | {v.mean() / mwpm_pL:.4f} |")
        rows.append(dict(kind='seed_summary', model=arch, n_seeds=len(v),
                         mean_p_L=float(v.mean()), sd_p_L=sd, mwpm_p_L=mwpm_pL,
                         mean_ratio_vs_mwpm=float(v.mean() / mwpm_pL)))

    out("\n### McNemar, paired on identical shots (b = first model right, second wrong)")
    out("| comparison | seed | b | c | n_discordant | net wins (b-c) | p exact | p chi2(cc) "
        "| delta p_L (first - second) |")
    out("|---|---:|---:|---:|---:|---:|---|---|---|")

    def pair(name, a, b_, s):
        mc = mcnemar(a, b_)
        d_pL = float((~a).mean() - (~b_).mean())
        out(f"| {name} | {s} | {mc['b']:,} | {mc['c']:,} | {mc['n_discordant']:,} | "
            f"{mc['net_a_wins']:+,} | {fmt_p(mc['p_exact'])} | {fmt_p(mc['p_chi2'])} | "
            f"{d_pL:+.6f} |")
        rows.append(dict(kind='mcnemar', comparison=name, seed=s, n=n, **mc,
                         delta_p_L=d_pL))

    for s, dm in zip(seeds, mamba):
        pair('Mamba vs MWPM', dm['correct'], mwpm_correct, s)
    for s, dg in zip(seeds, gru):
        pair('GRU vs MWPM', dg['correct'], mwpm_correct, s)
    for s, dm, dg in zip(seeds, mamba, gru):
        pair('Mamba vs GRU', dm['correct'], dg['correct'], s)
    out("\np-values are per seed and are not combined across seeds.")

    if args.out_csv:
        keys = []
        for r_ in rows:
            for k in r_:
                if k not in keys:
                    keys.append(k)
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r_ in rows:
                w.writerow(r_)
        print(f"[analyze] wrote {args.out_csv}")
    if args.out_md:
        with open(args.out_md, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"[analyze] wrote {args.out_md}")


if __name__ == '__main__':
    run()
