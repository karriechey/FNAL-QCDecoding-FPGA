#!/usr/bin/env python3
# Created: 2026-07-16
# Last modified: 2026-07-16
"""Phase 2a identity check for ActQuant. Downstream sweep numbers are only meaningful after
this passes.

Part 1 (identity check): with activation quantization disabled (act_bits=None), the 6-bit-weight
model must reproduce the 6-bit-weight / float-activation anchor p_L exactly --
0.046675 / 0.047715 / 0.045580 for seeds 0/1/2, agreeing to 1e-6. This is the same check that
validated the profiler: if disabling activation quantization does not recover the anchor
value, the ActQuant.qa() calls have been inserted in the wrong places and every later sweep
number is wrong.

Part 2 (quantization-active check): with act_bits=8, p_L must change relative to the anchor,
confirming the quantizers actually run.

Both parts use model.predict (graph execution), which matches how the anchor p_L was originally
recorded (eval_on_tail), so the identity comparison is exact to the anchor CSV's 6-decimal
rounding and is not affected by graph-vs-eager threshold-boundary differences.

  .venv/bin/python phase2a_identity_check.py            # runs both parts, all 3 seeds
"""
import argparse, os
import numpy as np

ANCHOR = {0: 0.046675, 1: 0.047715, 2: 0.045580}   # w6/act-FP32, from Phase 1 mcnemar_knee.csv


def _base(repo_local='rcnn_threshold'):
    return repo_local if os.path.isdir(repo_local) else os.path.expanduser('~/rcnn_threshold')


def eval_pL(weight_bits, act_bits, weights, db, de, truth, d, k, r, hidden, hlayers, npol, bs):
    from CNNModel_quantized import build_quantized_rcnn, restore_originals
    from eval_on_tail import rcnn_pred_and_correct
    restore_originals()                                  # clean slate each build (process-global config)
    model = build_quantized_rcnn(
        weight_bits, 'ZL', d, k, r, [hidden for _ in range(hlayers)], act_bits=act_bits,
        npol=npol, stop_round=None, has_nonuniform_response=False,
        do_all_data_qubits=False, return_all_rounds=False)
    _ = model([db[0:1], de[0:1]])
    model.load_weights(weights)
    pred = model.predict([db, de], batch_size=bs, verbose=0)
    _, correct = rcnn_pred_and_correct(pred, truth)
    return float((~correct).mean())


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weight-bits', type=int, default=6)
    ap.add_argument('--d', type=int, default=5); ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--rounds', type=int, default=3); ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--hidden', type=int, default=100); ap.add_argument('--hidden-layers', type=int, default=2)
    ap.add_argument('--npol', type=int, default=2)
    ap.add_argument('--n-test', type=int, default=200000); ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--tol', type=float, default=1e-6)
    ap.add_argument('--pool', default=None)
    ap.add_argument('--weights-dir', default=None)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    base = _base()
    pool = args.pool or os.path.join(base, 'pools', f'data_d{args.d}_p{args.p:.3f}_r{args.rounds}_TAIL200k.npz')
    wdir = args.weights_dir or os.path.join(base, 'out_q_mcnemar')

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')
    from types_cfg import get_types
    from circuit_partition import split_measurements
    d, r, k, nte = args.d, args.rounds, args.kernel, args.n_test
    binary_t, _, idx_t, _ = get_types(d, r, k)
    if not os.path.exists(pool):
        raise SystemExit(f'MISSING pool {pool}')
    z = np.load(pool)
    meas = z['measurements'].astype(binary_t); de = z['det_evts'].astype(binary_t)
    flips = z['flips'].astype(binary_t)
    db, _, _ = split_measurements(meas, d, idx_t)
    N = meas.shape[0]; te = slice(N - nte, N)
    db, de, truth = db[te], de[te], flips[te].astype(np.int8).reshape(-1)

    def wpath(seed):
        return os.path.join(wdir, f'rcnn_d{d}_p{args.p:.3f}_r{r}_w{args.weight_bits}_seed{seed}_ntr10000000.weights.h5')

    # Part 1: with activation quantization disabled, the model must reproduce the anchor exactly.
    print('=== identity check: act_bits=None must reproduce the 6-bit-weight/float-activation anchor ===')
    ok = True
    for seed, exp in ANCHOR.items():
        w = wpath(seed)
        if not os.path.exists(w):
            print(f'  seed{seed}: missing weights {w} -- skipped'); continue
        pL = eval_pL(args.weight_bits, None, w, db, de, truth, d, k, r,
                     args.hidden, args.hidden_layers, args.npol, args.batch_size)
        diff = abs(pL - exp); good = diff <= args.tol
        ok = ok and good
        print(f'  seed{seed}: p_L={pL:.6f}  anchor={exp:.6f}  |diff|={diff:.2e}  '
              f'{"match" if good else "MISMATCH -- fails"}')
    print(f'  identity check: {"passed (ActQuant is an exact no-op when disabled)" if ok else "FAILED -- quantizers are misplaced, stop here"}')
    assert ok, 'identity check failed: act_bits=None does not reproduce the anchor p_L'

    # Part 2: with activation quantization on, the output must change (confirms the quantizers run).
    print('\n=== quantization-active check: act_bits=8 must change p_L relative to the anchor ===')
    w0 = wpath(0)
    if os.path.exists(w0):
        pL8 = eval_pL(args.weight_bits, 8, w0, db, de, truth, d, k, r,
                      args.hidden, args.hidden_layers, args.npol, args.batch_size)
        changed = abs(pL8 - ANCHOR[0]) > args.tol
        print(f'  seed0 with act_bits=8: p_L={pL8:.6f}  vs anchor {ANCHOR[0]:.6f}  '
              f'{"changed (quantizers are running)" if changed else "UNCHANGED -- quantizers are not running"}')
        assert changed, 'quantization-active check failed: act_bits=8 did not change p_L'
    print('\n=== both checks passed. The Phase 2a activation-precision sweep is now meaningful. ===')


if __name__ == '__main__':
    run()
