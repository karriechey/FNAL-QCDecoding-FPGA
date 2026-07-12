#!/usr/bin/env python3
"""Train ONE QAT point: (d,p,seed,n_train,weight_bits) on a DISJOINT tail.

Path B, Experiment 1 per-job unit. Mirrors train_one.py's recipe exactly (the reference architecture's
LR schedule, Adam, BCE, 50 epochs, batch 10k) but builds the model via
CNNModel_quantized.build_quantized_rcnn(weight_bits, ...) so weights are quantized in
place. weight_bits None/>=32 => FP32 baseline (no-op quantizer).

Held-out integrity: the tail comes from a SEPARATE --test-pool (fresh shots, disjoint
by construction) when given; otherwise from the SAME pool with the train_one disjoint
assert (ntr <= N - nte). This exists because eval_on_tail.py has no such guard and a
200k tail on the 10.01M pool would overlap a 10M-train set by 190k shots (in-sample).

Reads {data-dir}/data_d{d}_p{p:.3f}_r{rounds}.npz for train (+ optional --test-pool).
Writes {out-dir}/rcnn_d{d}_p{p:.3f}_r{rounds}_w{wbits}_seed{seed}_ntr{ntr}.csv (+ .history.json).
"""
import argparse, csv, json, os
import numpy as np
from train_one import set_seeds, learning_rate_scheduler, lookup_mwpm  # reuse recipe


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--rounds', type=int, required=True)
    ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--n-train', type=int, required=True)
    ap.add_argument('--n-test', type=int, required=True)
    ap.add_argument('--weight-bits', type=int, default=None,
                    help='QAT weight bit-width; None or >=32 => FP32 baseline.')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--val-split', type=float, default=0.2)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=100)
    ap.add_argument('--hidden-layers', type=int, default=2)
    ap.add_argument('--npol', type=int, default=2)
    ap.add_argument('--data-dir', default=os.path.expanduser('~/rcnn_threshold/pools'))
    ap.add_argument('--test-pool', default=None,
                    help='separate npz for a FRESH disjoint tail (measurements,det_evts,flips).')
    ap.add_argument('--out-dir', default=os.path.expanduser('~/rcnn_threshold/out_q'))
    ap.add_argument('--init-weights', default=None,
                    help='optional FP32 .weights.h5 to warm-start QAT from (must match arch).')
    ap.add_argument('--no-save-weights', action='store_true', default=True)
    ap.add_argument('--save-weights', dest='no_save_weights', action='store_false')
    ap.add_argument('--no-early-stopping', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')
    print(f"[qtrain] TF {tf.__version__}  GPUs: {tf.config.list_physical_devices('GPU')}", flush=True)

    from types_cfg import get_types
    from circuit_partition import split_measurements
    from CNNModel_quantized import build_quantized_rcnn

    d, p, r, k, seed = args.d, args.p, args.rounds, args.kernel, args.seed
    ntr, nte = args.n_train, args.n_test
    wbits = args.weight_bits
    bt, tt, it, pt = get_types(d, r, k)

    def load(fn):
        if not os.path.exists(fn):
            raise SystemExit(f"[qtrain] MISSING {fn}")
        z = np.load(fn)
        db, _, _ = split_measurements(z['measurements'].astype(bt), d, it)
        return db, z['det_evts'].astype(bt), z['flips'].astype(bt)

    fn = os.path.join(args.data_dir, f'data_d{d}_p{p:.3f}_r{r}.npz')
    tr_db, tr_de, tr_fl = load(fn)
    Ntr_pool = tr_db.shape[0]

    if args.test_pool:                       # fresh disjoint tail
        te_db, te_de, te_fl = load(args.test_pool)
        te = slice(te_db.shape[0] - nte, te_db.shape[0])
        Xte, Yte = [te_db[te], te_de[te]], te_fl[te]
        assert ntr <= Ntr_pool, f"ntr={ntr} > train pool {Ntr_pool}"
    else:                                    # same-pool tail: MUST be disjoint
        assert ntr <= Ntr_pool - nte, (
            f"train/test overlap: ntr={ntr} > N-nte={Ntr_pool - nte}. "
            f"Use --test-pool for a fresh disjoint {nte}-shot tail.")
        te = slice(Ntr_pool - nte, Ntr_pool)
        te_db, te_de, te_fl = tr_db, tr_de, tr_fl
        Xte, Yte = [te_db[te], te_de[te]], te_fl[te]
    Xtr, Ytr = [tr_db[0:ntr], tr_de[0:ntr]], tr_fl[0:ntr]

    set_seeds(seed)
    # QAT: build the model with quantization already in the graph, then train it.
    # build_quantized_rcnn(wbits, ...) installs quantized_bits(wbits,1) at every weight's point of use, so the forward pass is quantized before training
    model = build_quantized_rcnn(
        wbits, 'ZL', d, k, r, [args.hidden for _ in range(args.hidden_layers)],
        npol=args.npol, stop_round=None, has_nonuniform_response=False,
        do_all_data_qubits=False, return_all_rounds=False)
    # Configure the quantized model for training.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Run one tiny forward pass to force TensorFlow/Keras to build the model weights
    _ = model([Xtr[0][0:1], Xtr[1][0:1]])

    # trained fresh from a seeded init (no FP32 warm-start unless --init-weights);
    # each bit-width is its own run.
    if args.init_weights:
        model.load_weights(args.init_weights)
    # Count the number of params in the model, useful for reporting model size in the paper
    n_params = int(model.count_params())

    cbs = [tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)]
    if not args.no_early_stopping:
        cbs.insert(0, tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.patience, restore_best_weights=True))

    import time
    t0 = time.time()
    # Gradient descent runs with the quantizer live in the forward pass, so master weights (FP32) are updated under quantization
    # nothing quantizes an already-trained model after the fact
    hist = model.fit(x=Xtr, y=Ytr, batch_size=args.batch_size, epochs=args.epochs,
                     validation_split=args.val_split, shuffle=True, verbose=2, callbacks=cbs)
    train_time = time.time() - t0
    epochs_ran = len(hist.history['loss'])
    best_val = float(min(hist.history['val_loss']))

    pred = model.predict(Xte, batch_size=args.batch_size, verbose=0)
    truth = Yte
    pL = float((truth != (pred > 0.5).astype(bt)).mean())
    base_rate = float(truth.mean())
    mwpm = lookup_mwpm(args.data_dir, d, p, r)
    eff_bits = 32 if (wbits is None or wbits >= 32) else wbits
    size_kb = round(n_params * eff_bits / 8 / 1024, 2)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f'rcnn_d{d}_p{p:.3f}_r{r}_w{eff_bits}_seed{seed}_ntr{ntr}'
    fields = ['architecture', 'd', 'p', 'rounds', 'kernel', 'seed', 'n_train', 'n_test',
              'weight_bits', 'size_kb', 'epochs', 'epochs_ran', 'batch_size', 'n_params',
              'p_L', 'mwpm_p_L', 'base_rate', 'beats_base_rate', 'best_val_loss',
              'train_time_s', 'test_pool']
    with open(os.path.join(args.out_dir, tag + '.csv'), 'w', newline='') as cf:
        w = csv.DictWriter(cf, fieldnames=fields); w.writeheader()
        w.writerow(dict(
            architecture='FullRCNNModel_QAT', d=d, p=p, rounds=r, kernel=k, seed=seed,
            n_train=ntr, n_test=nte, weight_bits=eff_bits, size_kb=size_kb,
            epochs=args.epochs, epochs_ran=epochs_ran, batch_size=args.batch_size,
            n_params=n_params, p_L=round(pL, 6),
            mwpm_p_L=('' if mwpm is None else round(mwpm, 6)), base_rate=round(base_rate, 5),
            beats_base_rate=int(pL < base_rate), best_val_loss=round(best_val, 5),
            train_time_s=round(train_time, 1),
            test_pool=(os.path.basename(args.test_pool) if args.test_pool else 'same-pool-tail')))
    with open(os.path.join(args.out_dir, tag + '.history.json'), 'w') as hf:
        json.dump({k2: [float(x) for x in v] for k2, v in hist.history.items()}, hf)
    if not args.no_save_weights:
        model.save_weights(os.path.join(args.out_dir, tag + '.weights.h5'))

    gap = '' if mwpm is None else f'  MWPM={mwpm:.5f}  ratio={pL/mwpm:.3f}x'
    print(f"[qtrain] {tag}  p_L={pL:.5f}{gap}  size={size_kb}KB  "
          f"({epochs_ran}ep {train_time:.0f}s {n_params}p)", flush=True)


if __name__ == '__main__':
    run()
