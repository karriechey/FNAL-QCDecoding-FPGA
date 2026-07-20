#!/usr/bin/env python3
# Created: 2026-07-20
# Last modified: 2026-07-20
"""Phase 2a: paired McNemar tests on the activation-precision sweep.

WHY THIS EXISTS. phase2a_collate.py compares activation word lengths by differencing aggregate
p_L: p_L(B) - p_L(control) within each seed. That difference is not a significance test. At
n_test = 200k the binomial standard error on a single p_L is ~4.7e-4, so the difference of two
runs carries SE ~6.7e-4 before any training variance is added -- and the observed B=8 deltas
(+0.00198, -0.00102, +0.00058) straddle zero and are non-monotonic against B=6 on seed 0. That
pattern is what noise-dominated differences look like. Claiming "8-bit activations are free"
needs the paired test, not the difference of two aggregates.

The right test is McNemar on the SAME 200k shots. The quantized model and its control both decode
the identical tail, so each shot is a matched pair: the informative quantity is the discordant
count -- shots where exactly one of the two is right (b = quantized-only-right,
c = control-only-right) -- not the marginal error rates. This is the same convention as Phase 1
(phase1_mcnemar.py / eval_on_tail.py --mcnemar), which reports
both_right / rcnn_only / mwpm_only / both_wrong and an exact binomial p-value.

WHY IT IS A SEPARATE SCRIPT (not eval_on_tail.py --mcnemar). eval_on_tail.py has no
activation-quantization support: it accepts --weight-bits but never touches ActQuant, so it would
load an 8-bit-activation checkpoint and run it with FLOAT activations. That is a different model
from the one that was trained, and its p_L would not be the sweep's p_L. Rather than extend a tool
that Phase 1's published numbers depend on, this script rebuilds each model exactly as
train_one_quantized.py did (same build_quantized_rcnn call, same weight AND activation bits) and
evaluates it itself.

WHAT IT REPORTS, per (B, seed):
  (a) B vs its OWN seed's control (act_bits=32)   <- the precision-cost test, the primary result
  (b) B vs MWPM decoded on the same tail          <- the "still beats MWPM" claim
MWPM is decoded ONCE with PyMatching on the tail and reused, since it does not depend on B.

STATISTICAL CAVEAT -- do not combine the three seeds' p-values. The three seeds share the same
200k tail, so their tests are correlated, not independent; Fisher-combining them would overstate
significance. Report the three seeds side by side and require the effect to be consistent in sign
and magnitude across them. That per-seed consistency is the evidence, not a pooled p-value.

Usage:
  python phase2a_mcnemar.py                       # all B, all seeds found in --dir
  python phase2a_mcnemar.py --acts 8,6 --seeds 0  # subset
"""
import argparse
import csv
import os

import numpy as np

# Weight width is fixed at 6 bits for the whole Phase-2a sweep; only activations are swept.
WEIGHT_BITS = 6
CONTROL_ACT = 32          # act_bits >= 32 means activation quantization is OFF (the control)
D, P, ROUNDS, KERNEL = 5, 0.010, 3, 3
N_TRAIN = 10_000_000
N_TEST = 200_000
HIDDEN, HIDDEN_LAYERS, NPOL = 100, 2, 2


def tag(act_bits, seed):
    """Filename stem written by train_one_quantized.py for this sweep point."""
    return (f'rcnn_d{D}_p{P:.3f}_r{ROUNDS}_w{WEIGHT_BITS}'
            f'_a{act_bits}_seed{seed}_ntr{N_TRAIN}')


def mcnemar(a_correct, b_correct, a_name, b_name):
    """Exact paired McNemar test between two per-shot correctness vectors on the same shots.

    Returns the 2x2 table plus a two-sided exact binomial p-value on the discordant pairs. Only
    the discordant counts carry information: under the null (both decoders equally accurate on
    this tail) each discordant shot is a fair coin, so b ~ Binomial(b + c, 0.5).
    """
    from scipy import stats

    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    both_right = int((a & b).sum())
    a_only = int((a & ~b).sum())        # b in the classic notation: A right, B wrong
    b_only = int((~a & b).sum())        # c: B right, A wrong
    both_wrong = int((~a & ~b).sum())
    n_disc = a_only + b_only

    if n_disc == 0:
        p_exact = 1.0
    else:
        p_exact = float(stats.binomtest(a_only, n_disc, 0.5).pvalue)

    return dict(both_right=both_right, a_only=a_only, b_only=b_only, both_wrong=both_wrong,
                n_discordant=n_disc, net_a_wins=a_only - b_only, p_exact=p_exact,
                a_name=a_name, b_name=b_name)


def fmt(mc):
    """One-line summary of a McNemar result, in the Phase-1 reporting convention."""
    return (f"both-right={mc['both_right']:,}  {mc['a_name']}-only={mc['a_only']:,}  "
            f"{mc['b_name']}-only={mc['b_only']:,}  both-wrong={mc['both_wrong']:,}  "
            f"n_disc={mc['n_discordant']:,}  net={mc['net_a_wins']:+,}  "
            f"p_exact={mc['p_exact']:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.expanduser('~/rcnn_threshold/out_q_phase2a'),
                    help='directory holding the sweep .weights.h5 checkpoints')
    ap.add_argument('--pool', default=os.path.expanduser(
        '~/rcnn_threshold/pools/data_d5_p0.010_r3_TAIL200k.npz'),
        help='the FRESH disjoint 200k tail -- must be the same pool the sweep tested on')
    # B=4 is deliberately NOT in the default set. Its checkpoints exist, but they are the collapsed
    # constant-predictor runs (p_L = base rate) caused by negative fractional widths, and
    # ActQuant.set_bits now refuses to build that configuration at all. There is no precision
    # question to test there -- the format is infeasible. Pass --acts 4 only to demonstrate the
    # guard firing.
    ap.add_argument('--acts', default='8,6', help='comma-separated activation widths to test')
    ap.add_argument('--seeds', default='0,1,2', help='comma-separated seeds')
    ap.add_argument('--out-csv', default=None, help='append per-(B,seed) rows here')
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    acts = [int(x) for x in args.acts.split(',') if x.strip()]
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    # Same determinism flags as the trainer. Inference is deterministic anyway, but this keeps the
    # environment identical to the run that produced the checkpoints -- one less difference to
    # explain if a reproduced p_L ever disagrees with the sweep CSV.
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')
    import tensorflow as tf
    assert tf.__version__.startswith('2.15'), (
        f'need TF 2.15.x (Keras 2) -- Keras 3 breaks the custom layers; got TF {tf.__version__}.')
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')

    from types_cfg import get_types
    from circuit_partition import split_measurements
    from CNNModel_quantized import build_quantized_rcnn

    bt, tt, it, pt = get_types(D, ROUNDS, KERNEL)

    # ---- load the tail once -------------------------------------------------------------
    z = np.load(args.pool)
    db, _, _ = split_measurements(z['measurements'].astype(bt), D, it)
    de = z['det_evts'].astype(bt)
    fl = z['flips'].astype(bt)
    N = db.shape[0]
    te = slice(N - N_TEST, N)                     # identical slice convention to the trainer
    # flips is stored (N, 1). It MUST be flattened to (N,) here: every downstream comparison is
    # elementwise against a (N,) prediction vector, and NumPy would broadcast a (N,) against a
    # (N, 1) into an (N, N) matrix -- 40 GB at n_test=200k, and a meaningless "error rate" of
    # 2*p*(1-p) ~ 0.405 if it fits. Same .reshape(-1) that eval_on_tail.py applies.
    Xte, truth = [db[te], de[te]], fl[te].astype(np.int8).reshape(-1)
    print(f'[mcnemar] tail: {args.pool}  n_test={N_TEST:,}  '
          f'base_rate={float(truth.mean()):.5f}', flush=True)

    # ---- decode MWPM once ---------------------------------------------------------------
    # MWPM does not depend on B, so decoding it per (B, seed) would repeat the expensive step for
    # no reason. This is a real PyMatching decode on these exact shots -- NOT lookup_mwpm(), whose
    # CSV is keyed only on (d, p, rounds) and does not know which tail it is being asked about.
    import pymatching
    from circuit_generators import get_builtin_circuit

    # Same 4-channel noise model as generate_pools / eval_on_tail.build_circuit. All four channels
    # must match the circuit the pool was sampled from, or the DEM -- and therefore MWPM -- is
    # decoding a different experiment than the one on disk.
    circuit = get_builtin_circuit(
        'surface_code:rotated_memory_z', distance=D, rounds=ROUNDS,
        before_round_data_depolarization=P, after_reset_flip_probability=P,
        after_clifford_depolarization=P, before_measure_flip_probability=P)
    dem = circuit.detector_error_model(decompose_errors=True)
    pym = pymatching.Matching.from_detector_error_model(dem)
    mwpm_pred = pym.decode_batch(de[te], bit_packed_predictions=False,
                                 bit_packed_shots=False).astype(np.int8).reshape(-1)
    mwpm_correct = (mwpm_pred == truth)
    mwpm_pL = float((~mwpm_correct).mean())
    print(f'[mcnemar] MWPM decoded on this tail: p_L = {mwpm_pL:.6f}', flush=True)

    # ---- evaluate every checkpoint ------------------------------------------------------
    def predict(act_bits, seed):
        """Rebuild the model EXACTLY as train_one_quantized.py did, load its weights, predict.

        The activation width must be passed to build_quantized_rcnn: the quantizers live in the
        forward pass, so a model loaded without them is a different function of the same weights.
        """
        w = os.path.join(args.dir, tag(act_bits, seed) + '.weights.h5')
        if not os.path.exists(w):
            return None
        abits = None if act_bits >= 32 else act_bits
        model = build_quantized_rcnn(
            WEIGHT_BITS, 'ZL', D, KERNEL, ROUNDS, [HIDDEN for _ in range(HIDDEN_LAYERS)],
            act_bits=abits, npol=NPOL, stop_round=None, has_nonuniform_response=False,
            do_all_data_qubits=False, return_all_rounds=False)
        _ = model([Xte[0][0:1], Xte[1][0:1]])          # force build before load_weights
        model.load_weights(w)
        pred = model.predict(Xte, batch_size=args.batch_size, verbose=0)
        correct = (truth == (pred > 0.5).astype(bt).reshape(-1))
        return correct

    rows = []
    for seed in seeds:
        ctrl = predict(CONTROL_ACT, seed)
        if ctrl is None:
            print(f'[mcnemar] seed {seed}: control checkpoint missing, skipping seed', flush=True)
            continue
        ctrl_pL = float((~ctrl).mean())
        print(f'\n=== seed {seed} ===  control (B=32) p_L = {ctrl_pL:.6f}', flush=True)

        for B in acts:
            q = predict(B, seed)
            if q is None:
                print(f'  B={B}: checkpoint missing', flush=True)
                continue
            q_pL = float((~q).mean())

            # (a) PRIMARY: quantized vs its own seed's control, paired on the same shots.
            vs_ctrl = mcnemar(q, ctrl, f'B{B}', 'ctrl')
            # (b) quantized vs MWPM on the same shots -- the "beats MWPM" claim.
            vs_mwpm = mcnemar(q, mwpm_correct, f'B{B}', 'mwpm')

            print(f'  B={B}: p_L={q_pL:.6f}  delta_vs_control={q_pL - ctrl_pL:+.6f}', flush=True)
            print(f'      vs control: {fmt(vs_ctrl)}', flush=True)
            print(f'      vs MWPM   : {fmt(vs_mwpm)}', flush=True)

            rows.append(dict(
                act_bits=B, seed=seed, weight_bits=WEIGHT_BITS, n_test=N_TEST,
                p_L=round(q_pL, 6), control_p_L=round(ctrl_pL, 6),
                delta_vs_control=round(q_pL - ctrl_pL, 6),
                mwpm_p_L=round(mwpm_pL, 6), ratio_vs_mwpm=round(q_pL / mwpm_pL, 4),
                ctrl_both_right=vs_ctrl['both_right'], ctrl_q_only=vs_ctrl['a_only'],
                ctrl_ctrl_only=vs_ctrl['b_only'], ctrl_both_wrong=vs_ctrl['both_wrong'],
                ctrl_n_discordant=vs_ctrl['n_discordant'], ctrl_p_exact=vs_ctrl['p_exact'],
                mwpm_both_right=vs_mwpm['both_right'], mwpm_q_only=vs_mwpm['a_only'],
                mwpm_mwpm_only=vs_mwpm['b_only'], mwpm_both_wrong=vs_mwpm['both_wrong'],
                mwpm_n_discordant=vs_mwpm['n_discordant'], mwpm_p_exact=vs_mwpm['p_exact'],
            ))

    if rows and args.out_csv:
        new = not os.path.exists(args.out_csv)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, 'a', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                wr.writeheader()
            wr.writerows(rows)
        print(f'\n[mcnemar] wrote {len(rows)} rows -> {args.out_csv}', flush=True)

    # Reminder printed with the results so it travels with the numbers, not just this docstring.
    print('\n[mcnemar] NOTE: the three seeds share one 200k tail, so their tests are correlated. '
          'Read them as three consistent (or inconsistent) readings -- do NOT Fisher-combine the '
          'p-values into a single significance claim.', flush=True)


if __name__ == '__main__':
    main()
