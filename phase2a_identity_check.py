#!/usr/bin/env python3
# Created: 2026-07-16
# Last modified: 2026-07-16
"""Phase 2a GATE: ActQuant identity check. NOTHING downstream counts until this passes.

With activation quant DISABLED (act_bits=None), the w6 model MUST reproduce the
w6/act-FP32 anchor p_L exactly -- 0.046675 / 0.047715 / 0.045580 for seeds 0/1/2, to 1e-6.
Same proof that made the profiler trustworthy: if disabling activation quant does not recover
the anchor bit-for-bit, the ActQuant.qa() calls are in the wrong places and every sweep number
is garbage.

Then a BITE check: with act_bits=8 the p_L must CHANGE (quantizers actually fire). Both use
model.predict (graph), matching how the anchor was recorded (eval_on_tail), so the identity
comparison is exact to the CSV's 6-dp rounding, not subject to graph/eager boundary flips.

  .venv/bin/python phase2a_identity_check.py            # gate all 3 seeds + bite check
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
        raise SystemExit(f'[gate] MISSING pool {pool}')
    z = np.load(pool)
    meas = z['measurements'].astype(binary_t); de = z['det_evts'].astype(binary_t)
    flips = z['flips'].astype(binary_t)
    db, _, _ = split_measurements(meas, d, idx_t)
    N = meas.shape[0]; te = slice(N - nte, N)
    db, de, truth = db[te], de[te], flips[te].astype(np.int8).reshape(-1)

    def wpath(seed):
        return os.path.join(wdir, f'rcnn_d{d}_p{args.p:.3f}_r{r}_w{args.weight_bits}_seed{seed}_ntr10000000.weights.h5')

    print('=== GATE: act_bits=None must reproduce the w6/act-FP32 anchor exactly ===')
    ok = True
    for seed, exp in ANCHOR.items():
        w = wpath(seed)
        if not os.path.exists(w):
            print(f'  seed{seed}: MISSING {w} -- skip'); continue
        pL = eval_pL(args.weight_bits, None, w, db, de, truth, d, k, r,
                     args.hidden, args.hidden_layers, args.npol, args.batch_size)
        d_ = abs(pL - exp); good = d_ <= args.tol
        ok = ok and good
        print(f'  seed{seed}: p_L={pL:.6f}  anchor={exp:.6f}  |d|={d_:.2e}  {"PASS" if good else "FAIL <<<"}')
    print(f'  IDENTITY GATE: {"PASS -- ActQuant is a clean no-op when disabled" if ok else "FAIL -- quantizers misplaced, STOP"}')
    assert ok, 'identity gate failed: act_bits=None does not reproduce the anchor'

    print('\n=== BITE: act_bits=8 must CHANGE p_L (quantizers actually fire) ===')
    w0 = wpath(0)
    if os.path.exists(w0):
        pL8 = eval_pL(args.weight_bits, 8, w0, db, de, truth, d, k, r,
                      args.hidden, args.hidden_layers, args.npol, args.batch_size)
        changed = abs(pL8 - ANCHOR[0]) > args.tol
        print(f'  seed0 act8 p_L={pL8:.6f}  vs anchor {ANCHOR[0]:.6f}  '
              f'{"BITES (changed)" if changed else "NO CHANGE <<< quantizers not firing"}')
        assert changed, 'bite check failed: act_bits=8 did not change p_L -- quantizers inert'
    print('\n=== GATE + BITE PASSED. Phase 2a sweep is now meaningful. ===')


if __name__ == '__main__':
    run()
