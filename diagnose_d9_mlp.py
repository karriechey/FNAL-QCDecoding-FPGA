#!/usr/bin/env python3
# Created: 2026-08-10
# Last updated: 2026-08-10
"""Why does the d=9 MLP reach 0.454 on training data and 0.495 on validation?

A 194,789-parameter model holds about 6 Mbit; the training block carries 10M labels.
It cannot memorise its way to a 4-point train/validation gap, and for iid shots from one Stim circuit any real feature it finds
in shots [0, 10M) must also hold in [10M, 10.2M). One of those assumptions is wrong.

MWPM reaches 0.122 on this pool, against a 0.489 base rate, so the syndrome carries
plenty of decodable signal. This is not a dynamic-range problem.

Five checks, cheapest and most decisive first:

  1  MWPM on a training slice vs the validation block. MWPM never trains, so if both
     land at ~0.122 the two blocks are the same distribution AND det_evts/flips are
     row-aligned in both. That single result clears most of the plumbing at once.
  2  Per-block statistics: base rate, detector firing rate, per-detector means. Catches
     a distribution shift that MWPM's aggregate p_L could average over.
  3  Tiny-subset overfit: fit 10k shots hard and score those same shots. A model that
     cannot overfit 10k has a fitting or plumbing fault, not a generalisation one.
  4  Logistic regression on the same 720 inputs, as a linear reference. Separates
     "the MLP extracts only low-order signal" from "the MLP extracts nothing".
  5  Layer shapes and where the parameters sit, d=9 against d=5.

Run on the pod, pinned interpreter:

  $PY diagnose_d9_mlp.py --pool $POOL
  $PY diagnose_d9_mlp.py --pool $POOL --skip mwpm     # if MWPM was already run
"""
import argparse
import os
import time

import numpy as np

BLOCK = 200000
TRAIN_LO = 0
VAL_LO = 10000000


def banner(s):
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}", flush=True)


def binom_err(p, n):
    return (p * (1 - p) / n) ** 0.5


def check_mwpm(evts, flips, d, p, rounds):
    """MWPM on a training slice and on the validation block.

    The strongest single check available: MWPM has no learned parameters, so a
    difference between blocks cannot come from training. It also fails loudly if
    det_evts and flips are misaligned, since a shuffled pairing would push p_L to the
    base rate.
    """
    banner("1  MWPM on train slice vs validation block")
    import pymatching
    from eval_on_tail import build_circuit
    circ = build_circuit(d, p, rounds)
    dem = circ.detector_error_model(decompose_errors=True)
    pym = pymatching.Matching.from_detector_error_model(dem)

    out = {}
    for name, lo in (('train[0:200k]', TRAIN_LO), ('val[10.0M:10.2M]', VAL_LO)):
        sl = slice(lo, lo + BLOCK)
        t0 = time.time()
        pred = pym.decode_batch(evts[sl], bit_packed_predictions=False,
                                bit_packed_shots=False).astype(np.int8).reshape(-1)
        truth = np.asarray(flips[sl]).reshape(-1)
        pL = float((pred != truth).mean())
        out[name] = pL
        print(f"  {name:>20}  p_L={pL:.5f} +/- {binom_err(pL, BLOCK):.5f}  "
              f"base={truth.mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    diff = abs(out['train[0:200k]'] - out['val[10.0M:10.2M]'])
    sigma = (binom_err(out['train[0:200k]'], BLOCK) ** 2
             + binom_err(out['val[10.0M:10.2M]'], BLOCK) ** 2) ** 0.5
    print(f"  difference {diff:.5f} = {diff / sigma:.1f} sigma")
    print("  -> blocks agree; pool and row alignment are fine" if diff < 3 * sigma
          else "  -> blocks DIFFER; investigate pool generation before anything else")
    return out


def check_stats(evts, flips):
    """Per-block statistics, in case an aggregate p_L hides a shift."""
    banner("2  Block statistics")
    print(f"  {'block':>20} {'base rate':>10} {'det mean':>10} {'det std':>9}")
    cols = {}
    for name, lo in (('train[0:200k]', TRAIN_LO), ('train[5.0M]', 5000000),
                     ('train[9.8M]', 9800000), ('val[10.0M]', VAL_LO)):
        sl = slice(lo, lo + BLOCK)
        e = evts[sl]
        t = np.asarray(flips[sl]).reshape(-1)
        cm = e.mean(axis=0)
        cols[name] = cm
        print(f"  {name:>20} {t.mean():>10.4f} {e.mean():>10.5f} {cm.std():>9.5f}")
    a, b = cols['train[0:200k]'], cols['val[10.0M]']
    print(f"  per-detector mean, train vs val: max|diff|={np.abs(a - b).max():.5f}  "
          f"corr={np.corrcoef(a, b)[0, 1]:.5f}")
    print("  (max|diff| should be within sampling noise ~0.003 at 200k shots)")


def build_mlp(d, rounds):
    from StudentModels import build_student
    return build_student(d=d, rounds=rounds, inputs='evts', student='mlp',
                         hidden=(209, 209))


def check_overfit(evts, flips, d, rounds, n=10000, epochs=300):
    """Can the model fit 10k shots at all?

    Scored on the same shots it trained on. This asks whether the model, loss and
    input/label wiring CAN fit the target, with generalisation deliberately removed
    from the question. A model that cannot overfit 10k examples has a fitting fault.
    """
    banner(f"3  Overfit {n:,} shots ({epochs} epochs), scored on those same shots")
    import tensorflow as tf
    x = np.asarray(evts[0:n], dtype=np.float32)
    y = np.asarray(flips[0:n], dtype=np.float32).reshape(-1, 1)
    m = build_mlp(d, rounds)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    h = m.fit(x, y, batch_size=256, epochs=epochs, verbose=0)
    pred = m.predict(x, batch_size=10000, verbose=0).reshape(-1)
    pL = float(((pred > 0) != (y.reshape(-1) > 0)).mean())
    print(f"  loss {h.history['loss'][0]:.5f} -> {h.history['loss'][-1]:.5f}")
    print(f"  p_L on the training shots themselves: {pL:.5f}  (base {y.mean():.4f})")
    print("  -> can fit; treat this as a generalisation question" if pL < 0.25
          else "  -> CANNOT fit 10k shots; this is a fitting/plumbing fault")
    return pL


def check_logreg(evts, flips, d, rounds, n_train=1000000, epochs=20):
    """Linear reference on the same 720 inputs.

    A single dense layer with a logit output is logistic regression, trained the same
    way as the MLP so the comparison isolates depth rather than the optimiser.
    """
    banner(f"4  Logistic regression on the same inputs ({n_train:,} shots)")
    import tensorflow as tf
    from tensorflow.keras import layers as KL
    n_feat = evts.shape[1]
    lin = tf.keras.Sequential([KL.Input((n_feat,)), KL.Dense(1)])
    lin.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    x = np.asarray(evts[0:n_train], dtype=np.float32)
    y = np.asarray(flips[0:n_train], dtype=np.float32).reshape(-1, 1)
    lin.fit(x, y, batch_size=10000, epochs=epochs, verbose=0)
    del x, y

    out = {}
    for name, lo in (('train[0:200k]', TRAIN_LO), ('val[10.0M:10.2M]', VAL_LO)):
        sl = slice(lo, lo + BLOCK)
        pred = lin.predict(np.asarray(evts[sl], dtype=np.float32),
                           batch_size=10000, verbose=0).reshape(-1)
        truth = np.asarray(flips[sl]).reshape(-1)
        out[name] = float(((pred > 0) != (truth > 0)).mean())
        print(f"  {name:>20}  p_L={out[name]:.5f}")
    return out


def check_shapes(d, rounds):
    """Where the parameters sit, d=9 against d=5."""
    banner("5  Layer shapes and parameter placement")
    for dd, rr in ((5, 5), (d, rounds)):
        m = build_mlp(dd, rr)
        total = m.count_params()
        print(f"  d={dd}, r={rr}: input {m.input_shape}, total {total:,}")
        for lyr in m.layers:
            n = int(sum(np.prod(w.shape) for w in lyr.weights)) if lyr.weights else 0
            if n:
                print(f"      {lyr.name:<12} {str(lyr.output_shape):>14} "
                      f"{n:>9,}  ({100 * n / total:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--d', type=int, default=9)
    ap.add_argument('--rounds', type=int, default=9)
    ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--skip', nargs='*', default=[],
                    help='any of: mwpm stats overfit logreg shapes')
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    print(f"[diag] pool {args.pool}", flush=True)
    z = np.load(args.pool)
    evts, flips = z['det_evts'], z['flips']
    print(f"[diag] det_evts {evts.shape} {evts.dtype}   flips {flips.shape} {flips.dtype}")

    if 'mwpm' not in args.skip:
        check_mwpm(evts, flips, args.d, args.p, args.rounds)
    if 'stats' not in args.skip:
        check_stats(evts, flips)
    if 'shapes' not in args.skip:
        check_shapes(args.d, args.rounds)
    if 'overfit' not in args.skip:
        check_overfit(evts, flips, args.d, args.rounds)
    if 'logreg' not in args.skip:
        check_logreg(evts, flips, args.d, args.rounds)


if __name__ == '__main__':
    main()
