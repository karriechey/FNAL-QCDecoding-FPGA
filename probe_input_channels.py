#!/usr/bin/env python3
# Created: 2026-08-04
# Last modified: 2026-08-04
"""Measure how much a trained RCNN relies on each of its two input channels.

Feeds the frozen teacher its normal inputs, then each channel zeroed in turn, and reports
p_L for each. Nothing here modifies CNNModel.py -- the lesion is applied to the input
arrays, outside the model.

What this does and does not tell you. The checkpoint was trained with both channels, so
zeroing one at inference measures dependence on it, not what an events-only model could
reach if trained that way. A large degradation means the channel is load-bearing for this
model; a small one means an events-only variant is worth building. Either way the number
is a lower bound on what retraining would recover.

Scores on the validation partition, never the sealed tail: this is a diagnostic that
informs a design decision, which makes it selection-adjacent.

  python probe_input_channels.py --weights <ckpt> --pool <pool> --n-train 10000000
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, nargs='+', help='one or more checkpoints')
    ap.add_argument('--pool', required=True)
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--hidden', type=int, default=100)
    ap.add_argument('--hidden-layers', type=int, default=2)
    ap.add_argument('--npol', type=int, default=2)
    ap.add_argument('--n-train', type=int, default=10000000,
                    help="the checkpoint's training prefix, used to locate the validation "
                         "partition")
    ap.add_argument('--val-split', type=float, default=0.2)
    ap.add_argument('--n-eval', type=int, default=200000,
                    help='shots to score, taken from the end of the validation partition')
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    assert tf.__version__.startswith('2.15'), (
        f'need TF 2.15.x (Keras 2); got TF {tf.__version__}. '
        f'Activate the uv venv: source .venv/bin/activate.')
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')

    from types_cfg import get_types
    from circuit_partition import split_measurements
    from CNNModel import FullRCNNModel

    d, p, r, k = args.d, args.p, args.rounds, args.kernel
    binary_t, _t, idx_t, _pk = get_types(d, r, k)

    # validation partition = last --val-split of the training prefix, as Keras carves it
    val_start = int(args.n_train * (1 - args.val_split))
    lo = args.n_train - args.n_eval
    if lo < val_start:
        raise SystemExit(f"[probe] --n-eval {args.n_eval} exceeds the validation partition "
                         f"[{val_start}, {args.n_train})")
    print(f"[probe] scoring shots [{lo:,}, {args.n_train:,}) -- inside the validation "
          f"partition [{val_start:,}, {args.n_train:,}); the tail is untouched", flush=True)

    z = np.load(args.pool)
    meas = z['measurements'][lo:args.n_train].astype(binary_t)
    evts = z['det_evts'][lo:args.n_train].astype(binary_t)
    truth = z['flips'][lo:args.n_train].astype(np.int8).reshape(-1)
    bits, _, _ = split_measurements(meas, d, idx_t)
    del meas

    zeros_b = np.zeros_like(bits)
    zeros_e = np.zeros_like(evts)
    base_rate = float(truth.mean())
    print(f"[probe] {len(truth):,} shots, base rate {base_rate:.5f}\n", flush=True)

    hidden = [args.hidden for _ in range(args.hidden_layers)]
    print(f"  {'checkpoint':>46}  {'both':>9} {'evts only':>10} {'bits only':>10}")
    for wpath in args.weights:
        model = FullRCNNModel('ZL', d, k, r, hidden, npol=args.npol, stop_round=None,
                              has_nonuniform_response=False, do_all_data_qubits=False,
                              return_all_rounds=False)
        _ = model([bits[0:1], evts[0:1]])
        model.load_weights(wpath)

        def p_L(b, e):
            pred = model.predict([b, e], batch_size=args.batch_size, verbose=0)
            return float(((np.asarray(pred).reshape(-1) > 0.5).astype(np.int8) != truth).mean())

        both = p_L(bits, evts)
        evts_only = p_L(zeros_b, evts)      # det_bits lesioned
        bits_only = p_L(bits, zeros_e)      # det_evts lesioned
        print(f"  {os.path.basename(wpath):>46}  {both:>9.5f} {evts_only:>10.5f} "
              f"{bits_only:>10.5f}", flush=True)

    print(f"\n  base rate {base_rate:.5f} is the all-zero predictor; a lesioned p_L at or "
          f"above it means that channel carried everything.")


if __name__ == '__main__':
    main()
