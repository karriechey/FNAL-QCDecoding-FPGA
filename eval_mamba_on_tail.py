#!/usr/bin/env python3
# Created: 2026-09-21
# Last modified: 2026-09-22
"""Score a saved Mamba decoder on one explicit shot block, decode MWPM on the same shots,
and dump per-shot outcomes keyed by absolute shot index.

Mirrors eval_student_on_tail.py so the two decoders produce the same artefacts:
  - a per-shot npz with shot_idx / truth / <model>_pred / <model>_correct / <model>_logit
    / mwpm_pred / mwpm_correct, readable by pair_two_decoders.py and
    analyze_mamba_vs_gru.py (alignment is always by shot_idx, never by row position)
  - one appended CSV row with p_L, MWPM p_L, the ratio, and the paired McNemar 2x2

MWPM is decoded HERE on these exact shots, never read from a stored baseline. The
architecture is rebuilt from the .config.json train_mamba.py wrote next to its results,
so the flags cannot drift between training and scoring.

  python eval_mamba_on_tail.py \
      --weights <ckpt>/<tag>.best.weights.h5 --config <out>/<tag>.config.json \
      --pool <pool.npz> --eval-start 17000000 --n-test 2000000 \
      --dump-per-shot <dir>/per_shot_<tag>_test.npz --out-csv <dir>/eval_test.csv
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from eval_on_tail import mcnemar_from_correct, build_circuit   # one 2x2, one circuit


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='.best.weights.h5 from train_mamba.py')
    ap.add_argument('--config', required=True, help='.config.json from train_mamba.py')
    ap.add_argument('--pool', required=True)
    ap.add_argument('--eval-start', type=int, required=True,
                    help='first shot of the block to score, absolute in the pool')
    ap.add_argument('--n-test', type=int, required=True)
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--label', default='',
                    help="free-text block label written into the CSV, e.g. 'eval' or "
                         "'sealed_test'")
    ap.add_argument('--out-csv', default=None)
    ap.add_argument('--dump-per-shot', default=None)
    ap.add_argument('--skip-mwpm', action='store_true',
                    help='skip the MWPM decode (a 2M-shot d=7 decode takes minutes); the '
                         'dump then has no mwpm_* columns')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--gpu-mem-mib', type=int, default=None)
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

    from StudentModels import to_sequence
    from train_mamba import to_sequence_int8
    from MambaModel import build_mamba_decoder, mamba_pred_and_correct

    for f in (args.weights, args.config, args.pool):
        if not os.path.exists(f):
            raise SystemExit(f"[mamba-eval] MISSING {f}")
    cfg = json.load(open(args.config))
    d, p, r = cfg['d'], cfg['p'], cfg['rounds']

    z = np.load(args.pool)
    N = z['det_evts'].shape[0]
    lo, hi = args.eval_start, args.eval_start + args.n_test
    if hi > N:
        raise SystemExit(f"[mamba-eval] block [{lo}, {hi}) exceeds pool size {N}")
    print(f"[mamba-eval] scoring shots [{lo:,}, {hi:,}) of {os.path.basename(args.pool)}"
          f"  d={d} r={r} p={p}", flush=True)

    det = z['det_evts'][lo:hi].astype(np.int8)
    truth = z['flips'][lo:hi].astype(np.int8).reshape(-1)
    in_dtype = cfg.get('input_dtype', 'float32')
    x_te = to_sequence_int8(det, d, r, p) if in_dtype == 'int8' else to_sequence(det, d, r, p)

    # Rebuild exactly the trained architecture from the saved config, then load.
    model = build_mamba_decoder(cfg['n_timesteps'], cfg['n_positions'],
                                input_dtype=in_dtype, **cfg['model'])
    model.load_weights(args.weights)
    n_params = int(model.count_params())
    if n_params != cfg['n_params']:
        raise SystemExit(f"[mamba-eval] rebuilt model has {n_params} params, config says "
                         f"{cfg['n_params']} -- architecture drift")
    print(f"[mamba-eval] {model.name} params={n_params:,}  weights "
          f"{os.path.basename(args.weights)}", flush=True)

    t0 = time.time()
    logits = model.predict(x_te, batch_size=args.batch_size, verbose=0).reshape(-1)
    m_pred, m_correct = mamba_pred_and_correct(logits, truth)
    pL = float((~m_correct).mean())
    base_rate = float(truth.mean())
    print(f"[mamba-eval] p_L={pL:.6f}  errors={int((~m_correct).sum()):,}/{args.n_test:,}"
          f"  base_rate={base_rate:.5f}  ({time.time() - t0:.0f}s)", flush=True)

    mwpm_pL, w_pred, w_correct, mc = None, None, None, None
    if not args.skip_mwpm:
        import pymatching
        circ = build_circuit(d, p, r)
        if circ.num_detectors != det.shape[1]:
            raise SystemExit(f"[mamba-eval] circuit has {circ.num_detectors} detectors, "
                             f"pool stores {det.shape[1]}")
        dem = circ.detector_error_model(decompose_errors=True)
        pym = pymatching.Matching.from_detector_error_model(dem)
        t0 = time.time()
        w_pred = pym.decode_batch(det, bit_packed_predictions=False,
                                  bit_packed_shots=False).astype(np.int8).reshape(-1)
        w_correct = (w_pred == truth)
        mwpm_pL = float((~w_correct).mean())
        mc = mcnemar_from_correct(m_correct, w_correct)   # b = mamba-only-correct
        print(f"[mamba-eval] MWPM p_L={mwpm_pL:.6f}  ratio={pL / mwpm_pL:.4f}x  "
              f"(decode {(time.time() - t0) / 60:.1f} min)", flush=True)
        print(f"[mcnemar] both-right={mc['both_right']:,}  mamba-only={mc['rcnn_only']:,}  "
              f"MWPM-only={mc['mwpm_only']:,}  both-wrong={mc['both_wrong']:,}  "
              f"net mamba wins={mc['net_rcnn_wins']:+,}  p_exact={mc['p_exact']:.3e}",
              flush=True)

    if args.dump_per_shot:
        arrays = dict(shot_idx=np.arange(lo, hi, dtype=np.int64), truth=truth,
                      mamba_pred=m_pred, mamba_correct=m_correct,
                      mamba_logit=logits.astype(np.float32),
                      d=d, p=p, rounds=r, n_test=args.n_test,
                      weights=os.path.basename(args.weights),
                      pool=os.path.basename(args.pool))
        if w_pred is not None:
            arrays.update(mwpm_pred=w_pred, mwpm_correct=w_correct)
        np.savez_compressed(args.dump_per_shot, **arrays)
        print(f"[mamba-eval] dumped per-shot -> {args.dump_per_shot}", flush=True)

    if args.out_csv:
        cols = ['weights', 'label', 'd', 'p', 'rounds', 'eval_start', 'n_test', 'n_params',
                'errors', 'p_L', 'mwpm_errors', 'mwpm_p_L', 'ratio', 'base_rate',
                'both_right', 'mamba_only', 'mwpm_only', 'both_wrong', 'n_discordant',
                'net_mamba_wins', 'mcnemar_chi2_cc', 'p_chi2', 'p_exact']
        new = not os.path.exists(args.out_csv)
        with open(args.out_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(cols)
            if mc is None:
                w.writerow([os.path.basename(args.weights), args.label, d, p, r, lo,
                            args.n_test, n_params, int((~m_correct).sum()), round(pL, 6)]
                           + [''] * 13)
            else:
                w.writerow([os.path.basename(args.weights), args.label, d, p, r, lo,
                            args.n_test, n_params, int((~m_correct).sum()), round(pL, 6),
                            int((~w_correct).sum()), round(mwpm_pL, 6),
                            round(pL / mwpm_pL, 4), round(base_rate, 5),
                            mc['both_right'], mc['rcnn_only'], mc['mwpm_only'],
                            mc['both_wrong'], mc['n_discordant'], mc['net_rcnn_wins'],
                            round(mc['mcnemar_chi2_cc'], 3), mc['p_chi2'], mc['p_exact']])
        print(f"[mamba-eval] appended -> {args.out_csv}", flush=True)


if __name__ == '__main__':
    run()
