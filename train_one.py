#!/usr/bin/env python3
# Last updated: 2026-08-12
"""Train ONE RCNN point: a single (d, p, seed, n_train) on a fixed held-out tail.

This is the per-job unit of the convergence ladder. Unlike the
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


def learning_rate_scheduler_slow(epoch, lr):
    """Same warm-up, gentler decay.

    The original schedule multiplies by 0.65 every epoch past 30, so the rate falls from
    3.7e-5 at epoch 30 to 6.8e-9 at epoch 50 -- the last ~15 epochs change nothing. This
    keeps the identical epoch<10 ramp, which exists to move the zero-init state
    correlator, then decays by 0.93 per epoch and holds a 1e-5 floor so late epochs still
    make progress.
    """
    if epoch < 10:
        sched = 0.001 * (10 - epoch)
    else:
        sched = lr * 0.93
    return max(sched, 1e-5)


LR_SCHEDULES = {}   # filled below; keys are the --lr-schedule choices


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


def learning_rate_scheduler_flat(epoch, lr):
    """Constant 0.003, with no warm-up ramp.

    Diagnostic schedule for the d=9 investigation. Both 'original' and 'slow' open at
    0.010 (epoch 0 gives 0.001 * 10), a value inherited from the d=5, r=3 reference
    notebook. At d=9 the model unrolls 49 kernel positions over 9 rounds instead of 9
    positions over 3, and a d=9, r=9 run at 10M shots sat at chance accuracy (0.525)
    through 13 epochs while the d=5 reference reached 0.797 in its first epoch. An
    opening rate that is too large for the deeper unroll would produce exactly that:
    the model is pushed into a flat region in the first few steps and the decaying rate
    never lets it back out.

    0.003 is the rate the hard-label MLP and GRU students used on this same d=9 pool,
    where the GRU did learn (p_L 0.328 against a 0.489 base rate). Using the same value
    keeps the comparison honest.

    A run using this schedule is a diagnostic. Comparing it against a d=5 number
    requires re-running d=5 on the same schedule, since the learning rate is one of the
    quantities the distance-scaling experiment holds fixed.
    """
    return 0.003


LR_SCHEDULES.update(original=learning_rate_scheduler, slow=learning_rate_scheduler_slow,
                    flat=learning_rate_scheduler_flat)


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
    ap.add_argument('--lr-schedule', choices=['original', 'slow', 'flat'],
                    default='original',
                    help="'original' is the schedule every r=3 run used. 'slow' keeps the "
                         "same warm-up but decays by 0.93/epoch with a 1e-5 floor, so late "
                         "epochs still move. 'flat' holds 0.003 throughout and skips the "
                         "warm-up ramp; it is a diagnostic for the d=9 investigation. "
                         "Every rung of a ladder must use the same one.")
    ap.add_argument('--no-early-stopping', action='store_true',
                    help='fixed --epochs, no EarlyStopping (ladder wants a fixed budget).')
    # --- explicit validation (Option A). Without these the original validation_split
    # behaviour is unchanged, so every earlier r=3 run reproduces exactly. ---
    ap.add_argument('--pool', default=None,
                    help='explicit pool npz, overriding the --data-dir name convention')
    ap.add_argument('--val-start', type=int, default=None,
                    help='first shot of a DEDICATED validation block, disjoint from the '
                         'training prefix. Set with --val-n to pass validation_data '
                         'explicitly and disable validation_split.')
    ap.add_argument('--val-n', type=int, default=None, help='size of that block')
    ap.add_argument('--seal-test', action='store_true',
                    help='do not evaluate the final tail. Under the Option A layout the '
                         'last --n-test shots ARE the sealed test partition, so the '
                         'post-training evaluation would read it. Report validation p_L '
                         'instead and score the test set once, later, after selection.')
    ap.add_argument('--resume', action='store_true',
                    help='continue from the resume checkpoint in --ckpt-dir when one '
                         'matches this run. Restores weights, Adam slots, learning rate, '
                         'EarlyStopping and ModelCheckpoint state, and the epoch counter. '
                         'Note: a resumed run reshuffles from a fresh RNG stream, so it '
                         'differs from an uninterrupted one even under '
                         '--require-determinism; resumed_from_epoch records it.')
    ap.add_argument('--ckpt-dir', default=None,
                    help='write best-validation and true final-epoch checkpoints here')
    ap.add_argument('--run-tag', default=None, help='prefix for checkpoint filenames')
    ap.add_argument('--cpu', action='store_true', help='hide GPU, run on CPU.')
    # Added 2026-08-06 for the parameter-reduction study. Off by default, so every
    # earlier run reproduces unchanged; the new launcher passes it so a run started
    # without the determinism variables dies instead of quietly being nondeterministic.
    ap.add_argument('--require-determinism', action='store_true',
                    help='hard-fail unless TF_DETERMINISTIC_OPS=1 and '
                         'TF_CUDNN_DETERMINISTIC=1 were exported before this process, '
                         'and TensorFlow is 2.15.x. Both variables are read by '
                         'TensorFlow at import time, so setting them inside the process '
                         'would have no effect.')
    args = ap.parse_args()

    if args.require_determinism:
        from slice_guard_r5 import assert_deterministic_env
        assert_deterministic_env()          # must precede the TensorFlow import

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    if args.require_determinism:
        from slice_guard_r5 import assert_tf_version
        assert_tf_version(tf)
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

    fn = args.pool or os.path.join(args.data_dir, f'data_d{d}_p{p:.3f}_r{rounds}.npz')
    if not os.path.exists(fn):
        raise SystemExit(f"[train] MISSING {fn} -- run generate_pools.py first. STOP.")
    z = np.load(fn)
    # copy=False: the pools are already int8, so a plain .astype() duplicates every
    # array for nothing. At d=9 that is an extra 8.3 GB for `measurements` alone.
    measurements = z['measurements'].astype(binary_t, copy=False)
    det_evts = z['det_evts'].astype(binary_t, copy=False)
    flips = z['flips'].astype(binary_t, copy=False)
    det_bits, _, _ = split_measurements(measurements, d, idx_t)

    N = measurements.shape[0]
    # `measurements` is dead after this point -- det_bits is a fresh array (np.delete
    # copies) and N is the only other thing read from it. Freeing it returns 8.3 GB at
    # d=9, r=9, which matters because the EAF pod's cgroup caps the process at 90 GB and
    # a d=9 run needs ~71 GB for the training graph alone.
    del measurements
    z.close()
    # FIXED tail (same shots for every rung) + nested front prefix; provably disjoint.
    te = slice(N - nte, N)
    tr = slice(0, ntr)
    assert ntr <= N - nte, f"train/test overlap: ntr={ntr} > N-nte={N - nte}"

    # Explicit validation block, disjoint from both the training prefix and the tail.
    if args.seal_test and (args.val_start is None or args.val_n is None):
        raise SystemExit('[train] --seal-test requires --val-start and --val-n: with no '
                         'explicit validation block there is nothing to report but the '
                         'sealed partition itself')
    val_data = None
    if args.val_start is not None:
        if args.val_n is None:
            raise SystemExit('[train] --val-start requires --val-n')
        v0, v1 = args.val_start, args.val_start + args.val_n
        if v0 < ntr:
            raise SystemExit(f'[train] validation [{v0}, {v1}) overlaps the training '
                             f'prefix [0, {ntr}) -- not disjoint')
        if v1 > N:
            raise SystemExit(f'[train] validation [{v0}, {v1}) exceeds pool size {N}')
        if args.seal_test and v1 > N - nte:
            raise SystemExit(
                f'[train] validation [{v0}, {v1}) overlaps the sealed test block '
                f'[{N - nte}, {N}) -- the partitions must be disjoint')
        val_data = ([det_bits[v0:v1], det_evts[v0:v1]], flips[v0:v1])
        print(f"[train] partitions: train [0, {ntr:,})  validation [{v0:,}, {v1:,})  "
              f"(validation_split disabled)", flush=True)

    set_seeds(seed)
    model = FullRCNNModel(
        'ZL', d, k, rounds, [args.hidden for _ in range(args.hidden_layers)],
        npol=args.npol, stop_round=None, has_nonuniform_response=False,
        do_all_data_qubits=False, return_all_rounds=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    _ = model([det_bits[0:1], det_evts[0:1]])  # build
    n_params = int(model.count_params())

    # Keras runs on_epoch_end in list order, and
    # EarlyStopping(restore_best_weights=True) swaps the best weights back in during its
    # own on_epoch_end on the stopping epoch. Anything that must observe the true final
    # epoch has to come BEFORE it, so the list is assembled front-to-back.
    save_last_cb = None
    if args.ckpt_dir:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        ck = os.path.join(args.ckpt_dir, (args.run_tag or 'run'))

        class _SaveLastEpoch(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                self.model.save_weights(ck + '.lastepoch.weights.h5')
        save_last_cb = _SaveLastEpoch()

    # Resume Mechanism
    # Restarts continue from the last epoch. Model weights and Adam slots go through
    # tf.train.Checkpoint; learning rate, EarlyStopping and ModelCheckpoint state go
    # through a JSON, both written every epoch.
    resume_state = None
    resume_path = (ck + '.resume') if args.ckpt_dir else None      # weights + optimizer
    resume_json = (ck + '.resume.json') if args.ckpt_dir else None  # lr + callback state
    # Resume is valid only for the same configuration; change any of these and it refuses.
    run_identity = dict(d=d, p=p, rounds=rounds, seed=seed, n_train=ntr,
                        batch_size=args.batch_size, lr_schedule=args.lr_schedule,
                        epochs=args.epochs)

    if args.resume and resume_json and os.path.exists(resume_json):
        with open(resume_json) as fh:
            cand = json.load(fh)
        if cand.get('identity') != run_identity:
            raise SystemExit(
                f"[train] resume checkpoint at {resume_json} belongs to a different run:\n"
                f"    stored:  {cand.get('identity')}\n"
                f"    current: {run_identity}\n"
                "Delete it or point --ckpt-dir elsewhere. STOP.")
        resume_state = cand
        print(f"[train] resuming from epoch {resume_state['epoch']} "
              f"(lr {resume_state['lr']:.6g}, best val_loss {resume_state['early_best']:.6f})",
              flush=True)
    elif args.resume:
        print("[train] --resume given, no resume checkpoint found; starting from epoch 0",
              flush=True)

    initial_epoch = resume_state['epoch'] if resume_state else 0
    prior_history = resume_state['history'] if resume_state else None

    class _ResumableEarlyStopping(tf.keras.callbacks.EarlyStopping):
        """EarlyStopping carrying its best score and patience counter across a restart."""

        def __init__(self, *a, resume_best=None, resume_wait=0, **kw):
            super().__init__(*a, **kw)
            self._resume_best = resume_best
            self._resume_wait = resume_wait

        def on_train_begin(self, logs=None):
            super().on_train_begin(logs)
            if self._resume_best is not None:
                self.best = self._resume_best
                self.wait = self._resume_wait

    early_cb = None
    if not args.no_early_stopping:
        early_cb = _ResumableEarlyStopping(
            monitor='val_loss', patience=args.patience, restore_best_weights=True,
            resume_best=(resume_state['early_best'] if resume_state else None),
            resume_wait=(resume_state['early_wait'] if resume_state else 0))

    ckpt_cb = None
    if args.ckpt_dir:
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            ck + '.best.weights.h5', monitor='val_loss', mode='min',
            save_best_only=True, save_weights_only=True, verbose=0)
        if resume_state:
            ckpt_cb.best = resume_state['ckpt_best']   # keeps a worse epoch from overwriting

    resume_cb = None
    if args.ckpt_dir:
        tf_ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)

        class _WriteResumeState(tf.keras.callbacks.Callback):
            """Write the resume checkpoint once per epoch."""

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                for k, v in (logs.items()):
                    hist_accum.setdefault(k, []).append(float(v))
                tf_ckpt.write(resume_path)
                state = dict(
                    identity=run_identity,
                    epoch=epoch + 1,
                    lr=float(tf.keras.backend.get_value(self.model.optimizer.lr)),
                    early_best=(float(early_cb.best) if early_cb is not None
                                else float('inf')),
                    early_wait=(int(early_cb.wait) if early_cb is not None else 0),
                    ckpt_best=(float(ckpt_cb.best) if ckpt_cb is not None
                               else float('inf')),
                    history=hist_accum,
                )
                tmp = resume_json + '.tmp'
                with open(tmp, 'w') as fh:
                    json.dump(state, fh)
                os.replace(tmp, resume_json)           # rename last, so a kill mid-write keeps the old state

        resume_cb = _WriteResumeState()

    hist_accum = {k: list(v) for k, v in prior_history.items()} if prior_history else {}  # spans restarts

    callbacks = []
    if save_last_cb is not None:
        callbacks.append(save_last_cb)                 # must precede EarlyStopping
    if early_cb is not None:
        callbacks.append(early_cb)
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(
        LR_SCHEDULES[args.lr_schedule]))
    if ckpt_cb is not None:
        callbacks.append(ckpt_cb)
    if resume_cb is not None:
        callbacks.append(resume_cb)                    # must follow the callbacks it records
    print(f"[train] lr schedule: {args.lr_schedule}", flush=True)
    print("[train] callback order: " +
          " -> ".join(type(c).__name__ for c in callbacks), flush=True)

    if resume_state:
        # Adam slots are created on the first gradient step, so this restore is deferred
        # until then; expect_partial() silences the not-yet-consumed warning.
        tf_ckpt.restore(resume_path).expect_partial()
        tf.keras.backend.set_value(model.optimizer.lr, resume_state['lr'])  # schedule is multiplicative
        print(f"[train] restored weights, optimizer and lr; continuing at epoch "
              f"{initial_epoch} of {args.epochs}", flush=True)

    t0 = time.time()
    fit_kw = (dict(validation_data=val_data) if val_data is not None
              else dict(validation_split=args.val_split))
    hist = model.fit(
        x=[det_bits[tr], det_evts[tr]], y=flips[tr],
        batch_size=args.batch_size, epochs=args.epochs, initial_epoch=initial_epoch,
        shuffle=True, verbose=2, callbacks=callbacks, **fit_kw)
    train_time = time.time() - t0

    # hist_accum spans restarts; hist.history covers this process only (no --ckpt-dir).
    full_history = hist_accum if hist_accum.get('loss') else dict(hist.history)
    epochs_ran = len(full_history['loss'])
    best_val = float(min(full_history['val_loss']))
    resumed_from = initial_epoch if resume_state else 0

    if args.seal_test:
        # score the dedicated validation block; the sealed test partition is never read
        v0, v1 = args.val_start, args.val_start + args.val_n
        ev = slice(v0, v1)
        print(f"[train] --seal-test: reporting p_L on validation [{v0:,}, {v1:,}); "
              f"the test partition is not read", flush=True)
    else:
        ev = te
    pred = model.predict([det_bits[ev], det_evts[ev]], batch_size=args.batch_size, verbose=0)
    # flatten both sides: comparing (n,1) against (n,) broadcasts to (n,n) silently
    truth = np.asarray(flips[ev]).reshape(-1).astype(np.int8)
    pred_bits = (np.asarray(pred).reshape(-1) > 0.5).astype(np.int8)
    assert pred_bits.shape == truth.shape, (pred_bits.shape, truth.shape)
    pL = float((truth != pred_bits).mean())
    base_rate = float(truth.mean())                 # all-zero predictor's error on the tail
    mwpm = lookup_mwpm(args.data_dir, d, p, rounds)
    # The stored lookup belongs to whatever pool mwpm_baseline.csv was built on. Under an
    # explicit --pool, or when scoring the validation slice, it describes different shots,
    # so neither the CSV nor the console may carry it.
    mwpm_stale = bool(args.pool) or bool(args.seal_test)

    # CLASS-COLLAPSE CHECK (inverted, generous): the model must beat the all-zero base rate.
    beats_base = pL < base_rate
    collapse_flag = "" if beats_base else " <-- FAIL: p_L >= base rate (class collapse)"

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f'rcnn_d{d}_p{p:.3f}_r{rounds}_seed{seed}_ntr{ntr}'
    fields = ['architecture', 'd', 'p', 'rounds', 'kernel', 'seed', 'n_train', 'n_test',
              'epochs', 'epochs_ran', 'batch_size', 'n_params', 'p_L', 'mwpm_p_L',
              'base_rate', 'beats_base_rate', 'best_val_loss', 'train_time_s',
              # which shots p_L was measured on, and where mwpm_p_L came from
              'eval_start', 'eval_stop', 'eval_partition', 'mwpm_source',
              # 0 for an uninterrupted run; the epoch a restart picked up from
              # otherwise. A resumed run reshuffles from a fresh RNG stream, so it
              # is not bit-identical to one that ran straight through.
              'resumed_from_epoch']
    with open(os.path.join(args.out_dir, tag + '.csv'), 'w', newline='') as cf:
        wri = csv.DictWriter(cf, fieldnames=fields)
        wri.writeheader()
        wri.writerow(dict(
            architecture='FullRCNNModel', d=d, p=p, rounds=rounds, kernel=k, seed=seed,
            n_train=ntr, n_test=nte, epochs=args.epochs, epochs_ran=epochs_ran,
            batch_size=args.batch_size, n_params=n_params, p_L=round(pL, 6),
            mwpm_p_L=('' if (mwpm is None or mwpm_stale) else round(mwpm, 6)),
            eval_start=ev.start, eval_stop=ev.stop,
            eval_partition=('validation' if args.seal_test else 'pool-tail'),
            mwpm_source=('suppressed_different_shots' if mwpm_stale
                         else ('stored_baseline' if mwpm is not None else 'none')),
            base_rate=round(base_rate, 5),
            beats_base_rate=int(beats_base), best_val_loss=round(best_val, 5),
            train_time_s=round(train_time, 1), resumed_from_epoch=resumed_from))
    with open(os.path.join(args.out_dir, tag + '.history.json'), 'w') as hf:
        json.dump({k2: [float(x) for x in v] for k2, v in full_history.items()}, hf)

    # Save the trained weights so the model can be RE-EVALUATED on a different tail
    # (bigger n_test, another p, a sanity re-check) via eval_on_tail.py WITHOUT retraining.
    # Opt-in with --save-weights; the .weights.h5 name carries the full config so a loader
    # can reconstruct the identical architecture. (This is the thing whose absence forced
    # the ~17 GPU-hr retrain for the 200k-tail re-measurement.)
    if args.ckpt_dir:
        # After fit(), restore_best_weights has already put the best weights back, so this
        # is the best-validation state. Saved under its own name and byte-compared against
        # the checkpoint the callback wrote, so the two files cannot be silently confused.
        ck = os.path.join(args.ckpt_dir, (args.run_tag or 'run'))
        model.save_weights(ck + '.best_restored.weights.h5')

        # Compare weight VALUES, not file bytes: two HDF5 files holding identical arrays
        # differ byte-wise, so hashing them would prove nothing either way.
        def _vals(path):
            probe = FullRCNNModel(
                'ZL', d, k, rounds, [args.hidden for _ in range(args.hidden_layers)],
                npol=args.npol, stop_round=None, has_nonuniform_response=False,
                do_all_data_qubits=False, return_all_rounds=False)
            _ = probe([det_bits[0:1], det_evts[0:1]])
            probe.load_weights(path)
            return [w.numpy() for w in probe.weights]

        def _same(a, b):
            return all(np.array_equal(x, y) for x, y in zip(a, b))

        best_w = _vals(ck + '.best.weights.h5')
        last_w = _vals(ck + '.lastepoch.weights.h5')
        rest_w = _vals(ck + '.best_restored.weights.h5')
        print(f"[train] checkpoint identity: best==best_restored {_same(best_w, rest_w)}  "
              f"best==lastepoch {_same(best_w, last_w)}", flush=True)
        if not _same(best_w, rest_w):
            print(f"[train]   note: early stopping did not fire (ran {epochs_ran}/"
                  f"{args.epochs}), so no restore happened and .best_restored holds the "
                  f"FINAL epoch. Select on .best.", flush=True)

    if not args.no_save_weights:
        wpath = os.path.join(args.out_dir, tag + '.weights.h5')
        model.save_weights(wpath)
        print(f"[train] saved weights -> {wpath}", flush=True)

    if mwpm_stale:
        gap = ('  MWPM=suppressed (explicit --pool or --seal-test: the stored baseline '
               'describes different shots)')
    else:
        gap = '' if mwpm is None else f'  MWPM={mwpm:.5f}  gap={pL - mwpm:+.5f}'
    print(f"[train] {tag}  RCNN p_L={pL:.5f}{gap}  base_rate={base_rate:.3f}"
          f"{collapse_flag}  ({epochs_ran} ep, {train_time:.0f}s, {n_params} params)",
          flush=True)


if __name__ == '__main__':
    run()
