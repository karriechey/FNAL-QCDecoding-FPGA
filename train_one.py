#!/usr/bin/env python3
"""Train ONE RCNN point: a single (d, p, seed, n_train) on a fixed held-out tail.

This is the per-job unit of the convergence ladder and the HTCondor fan-out. Unlike the
committed benchmark_rcnn.py (which sets te = slice(ntr, ntr+nte) -- a test block that
MOVES as ntr grows), this uses the FIXED tail te = slice(N-nte, N) for every rung, so the
p_L-vs-N learning curve is measured on identical shots and nested front prefixes never
touch the test set. This is the trustworthy structure verified in the MWPM-trust goal.

Reads  {data-dir}/data_d{d}_p{p:.3f}_r{rounds}.npz  (measurements, det_evts, flips).
Writes {out-dir}/rcnn_d{d}_p{p:.3f}_r{rounds}_seed{seed}_ntr{ntr}.csv      (one result row)
       {out-dir}/rcnn_..._ntr{ntr}.history.json                           (per-epoch arrays)
Clobber-safe: the filename carries d, p, rounds, seed AND ntr, so parallel Condor jobs
never collide. Default --no-save-weights.

Recipe fidelity: the reference architecture's LR schedule + Adam + binary_crossentropy, validation_split
carved inside the train prefix (its last val-split fraction, before shuffling -> disjoint
from the tail). No class weighting / focal loss / resampling (positives are 8-45%, not
rare -- so none is needed; introducing any would be flagged here).
"""
import argparse
import csv
import json
import os
import time
import numpy as np

LR_FLOOR = 0.0


def set_seeds(seed):
    import random
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def learning_rate_scheduler(epoch, lr):
    """the reference architecture's schedule (surface_code_d5_r3_RCNN.ipynb): high at epoch 0 to move the
    slow-start zero-init state correlator, then decay."""
    if epoch < 10:
        sched = 0.001 * (10 - epoch)
    elif epoch < 20:
        sched = lr * 0.9
    elif epoch < 30:
        sched = lr * 0.8
    else:
        sched = lr * 0.65
    return max(sched, LR_FLOOR)


def lookup_mwpm(data_dir, d, p, rounds):
    """Read the canonical once-per-(d,p,rounds) MWPM p_L, so the result row carries the
    target the RCNN is chasing. Returns None if unavailable.

    NOTE: matches on rounds too. The same (d,p) can have multiple pools at different
    rounds (e.g. d5/p0.010 exists at BOTH r5=0.0891 and r3~0.048); matching on (d,p)
    alone silently attaches the wrong-rounds MWPM to a run."""
    path = os.path.join(data_dir, "mwpm_baseline.csv")
    if not os.path.exists(path):
        return None
    for row in csv.DictReader(open(path)):
        if (int(row["d"]) == d and abs(float(row["p"]) - p) < 1e-9
                and int(row["rounds"]) == rounds):
            return float(row["mwpm_p_L"])
    return None


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--rounds', type=int, required=True)
    ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--n-train', type=int, required=True)
    ap.add_argument('--n-test', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--val-split', type=float, default=0.2)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=100)
    ap.add_argument('--hidden-layers', type=int, default=2)
    ap.add_argument('--npol', type=int, default=2)
    ap.add_argument('--data-dir', default=os.path.expanduser('~/rcnn_threshold/pools'))
    ap.add_argument('--out-dir', default=os.path.expanduser('~/rcnn_threshold/out'))
    ap.add_argument('--no-save-weights', action='store_true', default=True)
    ap.add_argument('--save-weights', dest='no_save_weights', action='store_false')
    ap.add_argument('--no-early-stopping', action='store_true',
                    help='fixed --epochs, no EarlyStopping (ladder wants a fixed budget).')
    ap.add_argument('--cpu', action='store_true', help='hide GPU, run on CPU.')
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')
    gpus = tf.config.list_physical_devices('GPU')
    print(f"[train] TF {tf.__version__}  GPUs visible: {gpus}", flush=True)

    from types_cfg import get_types
    from circuit_generators import get_builtin_circuit  # noqa: F401  (kept for parity)
    from circuit_partition import split_measurements
    from CNNModel import FullRCNNModel

    d, p, rounds, k, seed = args.d, args.p, args.rounds, args.kernel, args.seed
    ntr, nte = args.n_train, args.n_test
    binary_t, time_t, idx_t, packed_t = get_types(d, rounds, k)

    fn = os.path.join(args.data_dir, f'data_d{d}_p{p:.3f}_r{rounds}.npz')
    if not os.path.exists(fn):
        raise SystemExit(f"[train] MISSING {fn} -- run generate_pools.py first. STOP.")
    z = np.load(fn)
    measurements = z['measurements'].astype(binary_t)
    det_evts = z['det_evts'].astype(binary_t)
    flips = z['flips'].astype(binary_t)
    det_bits, _, _ = split_measurements(measurements, d, idx_t)

    N = measurements.shape[0]
    # FIXED tail (same shots for every rung) + nested front prefix; provably disjoint.
    te = slice(N - nte, N)
    tr = slice(0, ntr)
    assert ntr <= N - nte, f"train/test overlap: ntr={ntr} > N-nte={N - nte}"

    set_seeds(seed)
    model = FullRCNNModel(
        'ZL', d, k, rounds, [args.hidden for _ in range(args.hidden_layers)],
        npol=args.npol, stop_round=None, has_nonuniform_response=False,
        do_all_data_qubits=False, return_all_rounds=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    _ = model([det_bits[0:1], det_evts[0:1]])  # build
    n_params = int(model.count_params())

    callbacks = [tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)]
    if not args.no_early_stopping:
        callbacks.insert(0, tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.patience, restore_best_weights=True))

    t0 = time.time()
    hist = model.fit(
        x=[det_bits[tr], det_evts[tr]], y=flips[tr],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_split=args.val_split, shuffle=True, verbose=2, callbacks=callbacks)
    train_time = time.time() - t0
    epochs_ran = len(hist.history['loss'])
    best_val = float(min(hist.history['val_loss']))

    pred = model.predict([det_bits[te], det_evts[te]], batch_size=args.batch_size, verbose=0)
    truth = flips[te]
    pL = float((truth != (pred > 0.5).astype(binary_t)).mean())
    base_rate = float(truth.mean())                 # all-zero predictor's error on the tail
    mwpm = lookup_mwpm(args.data_dir, d, p, rounds)

    # CLASS-COLLAPSE CHECK (inverted, generous): the model must beat the all-zero base rate.
    beats_base = pL < base_rate
    collapse_flag = "" if beats_base else " <-- FAIL: p_L >= base rate (class collapse)"

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f'rcnn_d{d}_p{p:.3f}_r{rounds}_seed{seed}_ntr{ntr}'
    fields = ['architecture', 'd', 'p', 'rounds', 'kernel', 'seed', 'n_train', 'n_test',
              'epochs', 'epochs_ran', 'batch_size', 'n_params', 'p_L', 'mwpm_p_L',
              'base_rate', 'beats_base_rate', 'best_val_loss', 'train_time_s']
    with open(os.path.join(args.out_dir, tag + '.csv'), 'w', newline='') as cf:
        wri = csv.DictWriter(cf, fieldnames=fields)
        wri.writeheader()
        wri.writerow(dict(
            architecture='FullRCNNModel', d=d, p=p, rounds=rounds, kernel=k, seed=seed,
            n_train=ntr, n_test=nte, epochs=args.epochs, epochs_ran=epochs_ran,
            batch_size=args.batch_size, n_params=n_params, p_L=round(pL, 6),
            mwpm_p_L=('' if mwpm is None else round(mwpm, 6)), base_rate=round(base_rate, 5),
            beats_base_rate=int(beats_base), best_val_loss=round(best_val, 5),
            train_time_s=round(train_time, 1)))
    with open(os.path.join(args.out_dir, tag + '.history.json'), 'w') as hf:
        json.dump({k2: [float(x) for x in v] for k2, v in hist.history.items()}, hf)

    # Save the trained weights so the model can be RE-EVALUATED on a different tail
    # (bigger n_test, another p, a sanity re-check) via eval_on_tail.py WITHOUT retraining.
    # Opt-in with --save-weights; the .weights.h5 name carries the full config so a loader
    # can reconstruct the identical architecture. (This is the thing whose absence forced
    # the ~17 GPU-hr retrain for the 200k-tail re-measurement.)
    if not args.no_save_weights:
        wpath = os.path.join(args.out_dir, tag + '.weights.h5')
        model.save_weights(wpath)
        print(f"[train] saved weights -> {wpath}", flush=True)

    gap = '' if mwpm is None else f'  MWPM={mwpm:.5f}  gap={pL - mwpm:+.5f}'
    print(f"[train] {tag}  RCNN p_L={pL:.5f}{gap}  base_rate={base_rate:.3f}"
          f"{collapse_flag}  ({epochs_ran} ep, {train_time:.0f}s, {n_params} params)",
          flush=True)


if __name__ == '__main__':
    run()
