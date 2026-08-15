#!/usr/bin/env python3
# Created: 2026-08-06
# Last modified: 2026-08-06
"""Paired McNemar between any two decoders, from their per-shot dumps.

Why this exists
eval_on_tail.py and eval_student_on_tail.py each compare ONE model against MWPM. Two
separate comparisons against a common third model do not give the comparison between the
two: "0.933x MWPM" and "0.966x MWPM" are means over different runs and say nothing about
how often one is right where the other is wrong. That is a paired question and needs the
2x2 on shared shots.

The specific claim it settles here: at n_train=20M the hard-label GRU (0.07854) beats the
RCNN at the same n_train (0.08134) at matched capacity, 69,441 vs 69,485 parameters, with
no teacher involved. That is an architecture result, independent of the distillation study,
and it is the only entry in the r=5 table comfortably below MWPM.

No decoding happens here. Both dumps already contain per-shot correctness, so this aligns
them on absolute shot index and counts. Alignment is by index, never by position: the two
dumps can cover different ranges, and assuming row i of one is row i of the other is
exactly the silent-misalignment failure that would produce a plausible but meaningless 2x2.

  python pair_two_decoders.py \
      --a rcnn_20M_pershot.npz --a-name "RCNN @20M" \
      --b gru_20M_pershot.npz  --b-name "GRU hard @20M" \
      --out-csv ~/rcnn_threshold/paired_rcnn_vs_gru_r5.csv
"""
import argparse
import csv
import os
import numpy as np

from eval_on_tail import mcnemar_from_correct   # one implementation of the 2x2


# Each dump names its columns after the model that produced it, so there is no single
# key to read. Try the known spellings and say plainly what was found.
IDX_KEYS = ('shot_idx', 'tail_idx')
CORRECT_KEYS = ('rcnn_correct', 'student_correct', 'mwpm_correct')


def load_dump(path, which=None):
    """Return (shot_idx, correct, label) from a per-shot npz.

    `which` picks the column when a dump holds more than one decoder (both eval scripts
    also store mwpm_correct alongside their own model).
    """
    if not os.path.exists(path):
        raise SystemExit(f"[pair] MISSING {path}")
    z = np.load(path, allow_pickle=False)

    idx_key = next((k for k in IDX_KEYS if k in z.files), None)
    if idx_key is None:
        raise SystemExit(f"[pair] {os.path.basename(path)} has no shot index "
                         f"(looked for {IDX_KEYS}); keys are {sorted(z.files)}")

    if which:
        if which not in z.files:
            raise SystemExit(f"[pair] {os.path.basename(path)} has no '{which}'; "
                             f"keys are {sorted(z.files)}")
        col = which
    else:
        # Prefer the dump's own model over the mwpm_correct it also carries.
        found = [k for k in CORRECT_KEYS if k in z.files and k != 'mwpm_correct']
        if not found:
            found = [k for k in CORRECT_KEYS if k in z.files]
        if not found:
            raise SystemExit(f"[pair] {os.path.basename(path)} has no correctness column; "
                             f"keys are {sorted(z.files)}")
        col = found[0]

    return (np.asarray(z[idx_key]).reshape(-1).astype(np.int64),
            np.asarray(z[col]).reshape(-1).astype(bool),
            col,
            np.asarray(z['truth']).reshape(-1).astype(np.int8) if 'truth' in z.files else None)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='per-shot npz for decoder A')
    ap.add_argument('--b', required=True, help='per-shot npz for decoder B')
    ap.add_argument('--a-name', default='A')
    ap.add_argument('--b-name', default='B')
    ap.add_argument('--a-column', default=None,
                    help="which correctness column to read from A's dump, when it holds "
                         "more than one (e.g. mwpm_correct)")
    ap.add_argument('--b-column', default=None)
    ap.add_argument('--out-csv', default=None)
    args = ap.parse_args()

    ia, ca, cola, ta = load_dump(args.a, args.a_column)
    ib, cb, colb, tb = load_dump(args.b, args.b_column)
    print(f"[pair] A = {args.a_name:<18} {os.path.basename(args.a)}  column={cola}  "
          f"{len(ia):,} shots [{ia.min():,}, {ia.max() + 1:,})", flush=True)
    print(f"[pair] B = {args.b_name:<18} {os.path.basename(args.b)}  column={colb}  "
          f"{len(ib):,} shots [{ib.min():,}, {ib.max() + 1:,})", flush=True)

    # Intersect on absolute shot index. Positional alignment would silently compare
    # different shots whenever the two dumps cover different ranges.
    shared, pa, pb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
    if len(shared) == 0:
        raise SystemExit("[pair] the two dumps share no shots -- different partitions?")
    if len(shared) < len(ia) or len(shared) < len(ib):
        print(f"[pair] note: comparing the {len(shared):,} shared shots "
              f"({len(ia):,} in A, {len(ib):,} in B)", flush=True)

    a_ok, b_ok = ca[pa], cb[pb]

    # The truth labels come from the pool in both cases, so they must agree shot by shot.
    # If they do not, one dump does not describe the shots its indices claim.
    if ta is not None and tb is not None:
        if not np.array_equal(ta[pa], tb[pb]):
            raise SystemExit("[pair] truth labels disagree on the shared shots -- the two "
                             "dumps do not describe the same data.")
        print("[pair] truth labels agree on all shared shots.", flush=True)

    n = len(shared)
    pL_a, pL_b = float((~a_ok).mean()), float((~b_ok).mean())
    # mcnemar_from_correct is written as (rcnn, mwpm); here it is (B, A), so 'rcnn_only'
    # reads as B-only-correct -- i.e. B's wins. Named explicitly on output to keep that
    # straight.
    mc = mcnemar_from_correct(b_ok, a_ok)

    print(f"\n[pair] {args.a_name}: p_L = {pL_a:.6f}")
    print(f"[pair] {args.b_name}: p_L = {pL_b:.6f}   ({pL_b / pL_a:.4f}x of "
          f"{args.a_name})")
    print(f"\n[pair] 2x2 on {n:,} shared shots:")
    print(f"[pair]   both right              = {mc['both_right']:,}")
    print(f"[pair]   {args.b_name} only right{'':<4} = {mc['rcnn_only']:,}")
    print(f"[pair]   {args.a_name} only right{'':<4} = {mc['mwpm_only']:,}")
    print(f"[pair]   both wrong              = {mc['both_wrong']:,}")
    print(f"[pair]   discordant              = {mc['n_discordant']:,} "
          f"({mc['n_discordant'] / n:.2%} of shots)")
    print(f"[pair]   net {args.b_name} wins = {mc['net_rcnn_wins']:+,}  "
          f"(= n x (p_L_A - p_L_B) = {n * (pL_a - pL_b):+.0f})")
    print(f"[pair]   chi2(cc) = {mc['mcnemar_chi2_cc']:.1f}   p_chi2 = {mc['p_chi2']:.3e}   "
          f"p_exact = {mc['p_exact']:.3e}")

    if args.out_csv:
        cols = ['a_name', 'b_name', 'a_dump', 'b_dump', 'n_shared', 'p_L_a', 'p_L_b',
                'ratio_b_over_a', 'both_right', 'b_only', 'a_only', 'both_wrong',
                'n_discordant', 'net_b_wins', 'mcnemar_chi2_cc', 'p_chi2', 'p_exact']
        new = not os.path.exists(args.out_csv)
        with open(args.out_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(cols)
            w.writerow([args.a_name, args.b_name, os.path.basename(args.a),
                        os.path.basename(args.b), n, round(pL_a, 6), round(pL_b, 6),
                        round(pL_b / pL_a, 4), mc['both_right'], mc['rcnn_only'],
                        mc['mwpm_only'], mc['both_wrong'], mc['n_discordant'],
                        mc['net_rcnn_wins'], round(mc['mcnemar_chi2_cc'], 3),
                        mc['p_chi2'], mc['p_exact']])
        print(f"\n[pair] appended -> {args.out_csv}")


if __name__ == '__main__':
    run()
