#!/usr/bin/env python3
# Created: 2026-07-12
# Last modified: 2026-07-15
"""Re-evaluate a SAVED FullRCNNModel on a fixed test tail WITHOUT retraining,
optionally with a paired McNemar test vs MWPM on the SAME shots.

This is the payoff of train_one.py's --save-weights: once a model's weights are on
disk, measuring it on a different tail (bigger n_test, a sanity re-check, etc.) is
just load + inference (seconds), instead of a full retrain (hours).

--mcnemar adds the reviewer-proof part: on the shared tail it decodes MWPM too, builds
the 2x2 contingency (both-right / RCNN-only / MWPM-only / both-wrong), and reports the
paired McNemar test. The DISCORDANT cells (RCNN-only vs MWPM-only) are the scientific
content -- they show HOW the RCNN wins, not just that it does -- so we store the raw
four counts, not just the p-value. Note: p_L_MWPM - p_L_RCNN == (RCNN-only - MWPM-only)
/ n_test exactly, so the discordant difference IS the p_L gap, shot-by-shot.

Reconstruct the architecture from the SAME constructor args used in train_one.py (implied
by the .weights.h5 filename; keep --kernel/--hidden/--hidden-layers/--npol identical).
"""
import argparse
import csv
import os
import numpy as np


def rcnn_pred_and_correct(pred, truth):
    """Single source of the RCNN decision conventions: output-shape flatten, 0.5 threshold,
    `flips` as truth. Imported by profile_ranges.py so its p_L wrap-check uses THIS logic,
    not a reimplementation (a divergent copy is exactly the class of bug we keep catching)."""
    rcnn_pred = (np.asarray(pred) > 0.5).astype(np.int8).reshape(-1)
    return rcnn_pred, (rcnn_pred == truth)


def lookup_mwpm(data_dir, d, p, rounds):
    path = os.path.join(data_dir, "mwpm_baseline.csv")
    if not os.path.exists(path):
        return None
    for row in csv.DictReader(open(path)):
        if (int(row["d"]) == d and abs(float(row["p"]) - p) < 1e-9
                and int(row["rounds"]) == rounds):
            return float(row["mwpm_p_L"])
    return None


def build_circuit(d, p, rounds):
    """4-channel rotated_memory_z circuit, identical to generate_datasets/mwpm_on_pool."""
    from circuit_generators import get_builtin_circuit
    return get_builtin_circuit(
        'surface_code:rotated_memory_z', distance=d, rounds=rounds,
        before_round_data_depolarization=p, after_reset_flip_probability=p,
        after_clifford_depolarization=p, before_measure_flip_probability=p)


def mcnemar_from_correct(rcnn_correct, mwpm_correct):
    """Paired 2x2 + McNemar from per-shot boolean correctness arrays.
    Returns dict with the four raw counts and the test stats. b = RCNN-only-correct
    (RCNN wins the shot), c = MWPM-only-correct (MWPM wins)."""
    rc = np.asarray(rcnn_correct, dtype=bool)
    mc = np.asarray(mwpm_correct, dtype=bool)
    both_right = int((rc & mc).sum())
    rcnn_only = int((rc & ~mc).sum())    # b: RCNN right, MWPM wrong -> RCNN wins
    mwpm_only = int((~rc & mc).sum())    # c: MWPM right, RCNN wrong -> MWPM wins
    both_wrong = int((~rc & ~mc).sum())
    b, c = rcnn_only, mwpm_only
    n_disc = b + c
    # McNemar with continuity correction -> chi-square, 1 dof
    if n_disc > 0:
        chi2_cc = (abs(b - c) - 1) ** 2 / n_disc
    else:
        chi2_cc = 0.0
    # p-values: chi2 approx (safe, no version issues) + exact binomial if scipy has it
    try:
        from scipy.stats import chi2 as _chi2
        p_chi2 = float(_chi2.sf(chi2_cc, 1))
    except Exception:
        p_chi2 = float('nan')
    p_exact = float('nan')
    try:
        from scipy.stats import binomtest
        p_exact = float(binomtest(b, n_disc, 0.5, alternative='two-sided').pvalue) if n_disc else 1.0
    except Exception:
        try:
            from scipy.stats import binom_test
            p_exact = float(binom_test(b, n_disc, 0.5)) if n_disc else 1.0
        except Exception:
            pass
    return dict(both_right=both_right, rcnn_only=rcnn_only, mwpm_only=mwpm_only,
                both_wrong=both_wrong, n_discordant=n_disc, net_rcnn_wins=b - c,
                mcnemar_chi2_cc=chi2_cc, p_chi2=p_chi2, p_exact=p_exact)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='path to a .weights.h5 from train_one')
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--rounds', type=int, required=True)
    ap.add_argument('--n-test', type=int, required=True, help='tail size to score on')
    ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--hidden', type=int, default=100)
    ap.add_argument('--hidden-layers', type=int, default=2)
    ap.add_argument('--npol', type=int, default=2)
    ap.add_argument('--weight-bits', type=int, default=None,
                    help='if set, rebuild the QUANTIZED arch (StateDecoder=QDense, point-of-use '
                         'quantized_bits) so quantized .weights.h5 load AND inference matches the '
                         'trained model. Must equal the value train_one_quantized used. None=FP32.')
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--data-dir', default=os.path.expanduser('~/rcnn_threshold/pools'))
    ap.add_argument('--pool', default=None,
                    help='explicit pool npz (overrides --data-dir/name); e.g. a fresh '
                         'disjoint tail. te=slice(N-n_test, N) over THIS file.')
    ap.add_argument('--out-csv', default=None, help='append a result row here')
    ap.add_argument('--mcnemar', action='store_true',
                    help='also decode MWPM on the same tail and run the paired McNemar test')
    ap.add_argument('--dump-per-shot', default=None,
                    help='save an .npz of per-shot arrays on the tail (for failure-case '
                         'analysis). Includes MWPM per-shot correctness only if --mcnemar is set.')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')

    from types_cfg import get_types
    from circuit_partition import split_measurements

    d, p, r, k, nte = args.d, args.p, args.rounds, args.kernel, args.n_test
    binary_t, time_t, idx_t, packed_t = get_types(d, r, k)

    fn = args.pool or os.path.join(args.data_dir, f'data_d{d}_p{p:.3f}_r{r}.npz')
    if not os.path.exists(fn):
        raise SystemExit(f"[eval] MISSING pool {fn}")
    if not os.path.exists(args.weights):
        raise SystemExit(f"[eval] MISSING weights {args.weights}")
    z = np.load(fn)
    measurements = z['measurements'].astype(binary_t)
    det_evts = z['det_evts'].astype(binary_t)
    flips = z['flips'].astype(binary_t)
    det_bits, _, _ = split_measurements(measurements, d, idx_t)
    N = measurements.shape[0]
    te = slice(N - nte, N)

    hidden = [args.hidden for _ in range(args.hidden_layers)]
    if args.weight_bits is None or args.weight_bits >= 32:
        from CNNModel import FullRCNNModel
        model = FullRCNNModel(
            'ZL', d, k, r, hidden, npol=args.npol, stop_round=None,
            has_nonuniform_response=False, do_all_data_qubits=False, return_all_rounds=False)
    else:
        # rebuild the SAME quantized arch training saved (QDense StateDecoder + point-of-use
        # quantized_bits) so the checkpoint layout matches AND forward inference is quantized.
        from CNNModel_quantized import build_quantized_rcnn
        model = build_quantized_rcnn(
            args.weight_bits, 'ZL', d, k, r, hidden, npol=args.npol, stop_round=None,
            has_nonuniform_response=False, do_all_data_qubits=False, return_all_rounds=False)
    _ = model([det_bits[0:1], det_evts[0:1]])  # build
    model.load_weights(args.weights)

    pred = model.predict([det_bits[te], det_evts[te]], batch_size=args.batch_size, verbose=0)
    truth = flips[te].astype(np.int8).reshape(-1)
    rcnn_pred, rcnn_correct = rcnn_pred_and_correct(pred, truth)
    pL = float((~rcnn_correct).mean())
    base_rate = float(truth.mean())
    mwpm = lookup_mwpm(args.data_dir, d, p, r)
    mwpm_tail = mwpm  # overwritten with the freshly-decoded value when --mcnemar

    gap = '' if mwpm is None else f'  MWPM={mwpm:.5f}  gap={pL - mwpm:+.5f}  ratio={pL/mwpm:.3f}x'
    print(f"[eval] {os.path.basename(args.weights)}  n_test={nte:,}  "
          f"RCNN p_L={pL:.5f}{gap}  base_rate={base_rate:.3f}", flush=True)

    mc = None
    if args.mcnemar:
        import pymatching
        circ = build_circuit(d, p, r)
        dem = circ.detector_error_model(decompose_errors=True)
        pym = pymatching.Matching.from_detector_error_model(dem)
        mwpm_pred = pym.decode_batch(det_evts[te], bit_packed_predictions=False,
                                     bit_packed_shots=False).astype(np.int8).reshape(-1)
        mwpm_correct = (mwpm_pred == truth)
        mwpm_pL = float((~mwpm_correct).mean())
        mwpm_tail = mwpm_pL  # fresh-tail MWPM -> this is the anchor for THIS pool
        mc = mcnemar_from_correct(rcnn_correct, mwpm_correct)
        print(f"[mcnemar] 2x2 on {nte:,} shared shots:", flush=True)
        print(f"[mcnemar]   both-right = {mc['both_right']:,}", flush=True)
        print(f"[mcnemar]   RCNN-only  = {mc['rcnn_only']:,}   (RCNN wins these)", flush=True)
        print(f"[mcnemar]   MWPM-only  = {mc['mwpm_only']:,}   (MWPM wins these)", flush=True)
        print(f"[mcnemar]   both-wrong = {mc['both_wrong']:,}", flush=True)
        print(f"[mcnemar]   net RCNN wins = {mc['net_rcnn_wins']:+,}  "
              f"(= n_test x (p_L_MWPM - p_L_RCNN) = {nte*(mwpm_pL - pL):+.0f})", flush=True)
        print(f"[mcnemar]   McNemar chi2(cc)={mc['mcnemar_chi2_cc']:.1f}  "
              f"p_chi2={mc['p_chi2']:.2e}  p_exact={mc['p_exact']:.2e}  "
              f"(MWPM p_L on this tail = {mwpm_pL:.5f})", flush=True)

    if args.dump_per_shot:
        tail_idx = np.arange(N - nte, N)              # shot indices into the pool
        dump = dict(
            tail_idx=tail_idx,
            truth=truth.astype(np.int8),              # true logical flip, per shot
            rcnn_pred=rcnn_pred.astype(np.int8),
            rcnn_correct=rcnn_correct.astype(bool),
            rcnn_prob=pred.reshape(-1).astype(np.float32),   # raw sigmoid, for confidence
            det_evts=det_evts[te].astype(np.int8),    # per-shot detector-event pattern
            d=d, p=p, rounds=r, n_test=nte,
        )
        if mc is not None:
            dump['mwpm_pred'] = mwpm_pred.astype(np.int8)
            dump['mwpm_correct'] = mwpm_correct.astype(bool)
        np.savez_compressed(args.dump_per_shot, **dump)
        keys = ', '.join(sorted(dump.keys()))
        print(f"[eval] dumped per-shot -> {args.dump_per_shot}  (keys: {keys})", flush=True)

    if args.out_csv:
        base_cols = ['weights', 'd', 'p', 'rounds', 'n_test', 'p_L', 'mwpm_p_L', 'ratio',
                     'base_rate']
        mc_cols = ['both_right', 'rcnn_only', 'mwpm_only', 'both_wrong', 'n_discordant',
                   'net_rcnn_wins', 'mcnemar_chi2_cc', 'p_chi2', 'p_exact']
        new = not os.path.exists(args.out_csv)
        with open(args.out_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(base_cols + mc_cols)
            base = [os.path.basename(args.weights), d, p, r, nte, round(pL, 6),
                    '' if mwpm_tail is None else round(mwpm_tail, 6),
                    '' if mwpm_tail is None else round(pL / mwpm_tail, 4), round(base_rate, 5)]
            extra = ([mc['both_right'], mc['rcnn_only'], mc['mwpm_only'], mc['both_wrong'],
                      mc['n_discordant'], mc['net_rcnn_wins'], round(mc['mcnemar_chi2_cc'], 3),
                      mc['p_chi2'], mc['p_exact']] if mc else [''] * len(mc_cols))
            w.writerow(base + extra)
        print(f"[eval] appended -> {args.out_csv}", flush=True)


if __name__ == '__main__':
    run()
