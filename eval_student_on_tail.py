#!/usr/bin/env python3
# Created: 2026-08-05
# Last modified: 2026-08-05
"""Score a saved student on a fixed partition and run the paired McNemar test vs MWPM.

Why a separate script from eval_on_tail.py
eval_on_tail.py reconstructs a FullRCNNModel, takes [det_bits, det_evts] as two inputs,
and thresholds a sigmoid at 0.5. A student is a different architecture, takes one
assembled feature array, and emits a LOGIT (threshold 0). Rather than branch that file on
architecture -- which would put two decision conventions in one place, the exact thing the
repo has been bitten by -- this reuses its statistics helpers and supplies the student's
own build and decision path.

What it answers
The r=5 hard-label GRU reaches p_L = 0.08145 against MWPM 0.084215 on the validation
partition. That is a comparison of means over 3 seeds. The claim "the student beats MWPM"
needs the PAIRED test on the same shots: how often does the student decode a shot right
where MWPM gets it wrong, and vice versa. Those discordant counts are the scientific
content -- they show how the win is made, not just that the means differ.

MWPM is decoded here, on these exact shots, never read from mwpm_baseline.csv. That file
describes whichever tail it was built on and is not portable across partitions.

  python eval_student_on_tail.py \
      --weights ~/rcnn_threshold/out_student_r5_hard/r5hard_gru_ntr10000000_seed0_lr0.003.weights.h5 \
      --student gru --units 140 --d 5 --p 0.010 --rounds 5 \
      --pool ~/rcnn_threshold/pools_r5/data_d5_p0.010_r5_FORMAL.npz \
      --eval-start 20000000 --n-test 200000 \
      --out-csv ~/rcnn_threshold/student_mcnemar_r5.csv
"""
import argparse
import csv
import os
import numpy as np

# Reuse the statistics and circuit construction, so there is exactly one implementation of
# the 2x2 and one definition of the noise model.
from eval_on_tail import mcnemar_from_correct, build_circuit


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True,
                    help='.weights.h5 written by train_student.py')
    ap.add_argument('--student', choices=['mlp', 'gru'], required=True)
    ap.add_argument('--inputs', choices=['evts', 'evts+bits'], default='evts')
    ap.add_argument('--hidden', type=int, nargs='*', default=[],
                    help='MLP: the Dense stack (e.g. 209 209). GRU: post-GRU Dense layers.')
    ap.add_argument('--units', type=int, default=140, help='GRU hidden state size')
    ap.add_argument('--weight-bits', type=int, default=None)
    ap.add_argument('--act-bits', type=int, default=None)
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--eval-start', type=int, required=True,
                    help='first shot of the partition to score (e.g. 20000000)')
    ap.add_argument('--n-test', type=int, default=200000)
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--out-csv', default=None)
    ap.add_argument('--dump-per-shot', default=None,
                    help='npz of per-shot arrays, for failure-case analysis')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--gpu-mem-mib', type=int, default=None,
                    help='cap this process to N MiB of GPU memory')
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    assert tf.__version__.startswith('2.15'), (
        f'need TF 2.15.x (Keras 2); got TF {tf.__version__}.')
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')
    elif args.gpu_mem_mib:
        from train_student import cap_gpu_memory
        cap_gpu_memory(tf, args.gpu_mem_mib)

    from types_cfg import get_types
    from circuit_partition import split_measurements
    from StudentModels import build_student, build_gru_student, assemble_features, \
        student_pred_and_correct

    d, p, r, k = args.d, args.p, args.rounds, args.kernel
    binary_t, _t, idx_t, _pk = get_types(d, r, k)

    if not os.path.exists(args.pool):
        raise SystemExit(f"[student-eval] MISSING pool {args.pool}")
    if not os.path.exists(args.weights):
        raise SystemExit(f"[student-eval] MISSING weights {args.weights}")

    z = np.load(args.pool)
    N = z['measurements'].shape[0]
    lo, hi = args.eval_start, args.eval_start + args.n_test
    if hi > N:
        raise SystemExit(f"[student-eval] partition [{lo}, {hi}) exceeds pool size {N}")
    te = slice(lo, hi)
    print(f"[student-eval] scoring shots [{lo:,}, {hi:,}) of {os.path.basename(args.pool)}",
          flush=True)

    m_te = z['measurements'][te].astype(binary_t)
    e_te = z['det_evts'][te].astype(binary_t)
    truth = z['flips'][te].astype(np.int8).reshape(-1)
    b_te, _, _ = split_measurements(m_te, d, idx_t)
    x_te = assemble_features(b_te, e_te, args.inputs, student=args.student,
                             d=d, rounds=r, p=p)
    det_evts_te = e_te            # kept for the MWPM decode below
    del m_te, b_te

    # Rebuild the SAME architecture the weights were trained with, then load. A mismatch
    # in units/hidden raises on load rather than silently producing a wrong score.
    if args.student == 'gru':
        model = build_gru_student(d=d, rounds=r, inputs=args.inputs, units=args.units,
                                  hidden=tuple(args.hidden), p=p,
                                  weight_bits=args.weight_bits, act_bits=args.act_bits)
    else:
        model = build_student('mlp', d=d, rounds=r, inputs=args.inputs,
                              hidden=tuple(args.hidden) or (209, 209),
                              weight_bits=args.weight_bits, act_bits=args.act_bits)
    model.load_weights(args.weights)
    print(f"[student-eval] {args.student} params={model.count_params():,}", flush=True)

    logits = model.predict(x_te, batch_size=args.batch_size, verbose=0)
    s_pred, s_correct = student_pred_and_correct(logits, truth)
    pL = float((~s_correct).mean())
    base_rate = float(truth.mean())

    # MWPM decoded on THESE shots. Never from the stored baseline: that file belongs to
    # whichever tail it was computed on, and mixing the two has already caused one real
    # bug in this repo.
    import pymatching
    circ = build_circuit(d, p, r)
    dem = circ.detector_error_model(decompose_errors=True)
    pym = pymatching.Matching.from_detector_error_model(dem)
    m_pred = pym.decode_batch(det_evts_te, bit_packed_predictions=False,
                              bit_packed_shots=False).astype(np.int8).reshape(-1)
    m_correct = (m_pred == truth)
    mwpm_pL = float((~m_correct).mean())

    mc = mcnemar_from_correct(s_correct, m_correct)
    # mcnemar_from_correct names its cells for the RCNN; here b = student-only-correct.
    print(f"[student-eval] student p_L={pL:.6f}  MWPM p_L={mwpm_pL:.6f}  "
          f"ratio={pL / mwpm_pL:.4f}x  base_rate={base_rate:.5f}", flush=True)
    print(f"[mcnemar] 2x2 on {args.n_test:,} shared shots:", flush=True)
    print(f"[mcnemar]   both-right   = {mc['both_right']:,}", flush=True)
    print(f"[mcnemar]   student-only = {mc['rcnn_only']:,}   (student wins these)", flush=True)
    print(f"[mcnemar]   MWPM-only    = {mc['mwpm_only']:,}   (MWPM wins these)", flush=True)
    print(f"[mcnemar]   both-wrong   = {mc['both_wrong']:,}", flush=True)
    print(f"[mcnemar]   net student wins = {mc['net_rcnn_wins']:+,}  "
          f"(= n_test x (p_L_MWPM - p_L_student) = {args.n_test * (mwpm_pL - pL):+.0f})",
          flush=True)
    print(f"[mcnemar]   chi2(cc)={mc['mcnemar_chi2_cc']:.1f}  p_chi2={mc['p_chi2']:.3e}  "
          f"p_exact={mc['p_exact']:.3e}", flush=True)

    if args.dump_per_shot:
        np.savez_compressed(
            args.dump_per_shot,
            shot_idx=np.arange(lo, hi, dtype=np.int64), truth=truth,
            student_pred=s_pred, student_correct=s_correct,
            student_logit=np.asarray(logits).reshape(-1).astype(np.float32),
            mwpm_pred=m_pred, mwpm_correct=m_correct,
            det_evts=det_evts_te.astype(np.int8),
            d=d, p=p, rounds=r, n_test=args.n_test)
        print(f"[student-eval] dumped per-shot -> {args.dump_per_shot}", flush=True)

    if args.out_csv:
        cols = ['weights', 'student', 'd', 'p', 'rounds', 'eval_start', 'n_test',
                'n_params', 'p_L', 'mwpm_p_L', 'ratio', 'base_rate',
                'both_right', 'student_only', 'mwpm_only', 'both_wrong', 'n_discordant',
                'net_student_wins', 'mcnemar_chi2_cc', 'p_chi2', 'p_exact']
        new = not os.path.exists(args.out_csv)
        with open(args.out_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(cols)
            w.writerow([os.path.basename(args.weights), args.student, d, p, r,
                        lo, args.n_test, int(model.count_params()),
                        round(pL, 6), round(mwpm_pL, 6), round(pL / mwpm_pL, 4),
                        round(base_rate, 5),
                        mc['both_right'], mc['rcnn_only'], mc['mwpm_only'],
                        mc['both_wrong'], mc['n_discordant'], mc['net_rcnn_wins'],
                        round(mc['mcnemar_chi2_cc'], 3), mc['p_chi2'], mc['p_exact']])
        print(f"[student-eval] appended -> {args.out_csv}", flush=True)


if __name__ == '__main__':
    run()
