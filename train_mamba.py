#!/usr/bin/env python3
# Created: 2026-09-21
# Last modified: 2026-09-22
"""Train one parameter-matched Mamba decoder on hard logical-flip labels.

Companion to train_student.py, restricted to what the GRU-vs-Mamba comparison needs:
hard labels only (alpha=1), one architecture, explicit shot-index partitions, a fixed
epoch budget with minimum-val_loss checkpoint selection, and NO scoring of any
evaluation or test block. Scoring is done separately by eval_mamba_on_tail.py, so a
learning-rate sweep run with this script touches only the training prefix and the
validation block, and the test tail cannot leak into tuning by accident.

Recipe of record (Experiment 16, gh200_run_student_scaling.sh, the d=7 GRU it is
compared against):
    labels        hard Stim logical flips
    inputs        det_evts only, scattered to [8 timesteps, 48 positions]
    optimizer     Adam, constant learning rate (3e-3 for the GRU; swept here)
    batch         10,000
    epochs        200, no early stopping
    selection     minimum val_loss checkpoint on [15.0M, 15.2M)
    seeds         0, 1, 2
    determinism   TF_DETERMINISTIC_OPS=1, TF_CUDNN_DETERMINISTIC=1, exported by the
                  launcher before Python starts; --require-determinism enforces it

Partitions (Experiment 13 layout, 19M pool):
    training    [0, n_train)
    validation  [val_start, val_start + val_n)
    evaluation  [15.2M, 17.0M)   scored later by eval_mamba_on_tail.py
    test        [17.0M, 19.0M)   sealed; scored once at the very end

Reads  the pool npz (det_evts, flips). `measurements` is never read: the sequence
       input is built from detector events alone, as for the GRU.
Writes {out-dir}/{tag}.csv           one row: config, params, best_val_loss, timing,
                                      provenance
       {out-dir}/{tag}.history.json  per-epoch loss / val_loss / hard_accuracy
       {ckpt-dir}/{tag}.best.weights.h5       minimum-val_loss weights
       {ckpt-dir}/{tag}.lastepoch.weights.h5  true final-epoch weights
       {out-dir}/{tag}.config.json   the exact model kwargs, so eval rebuilds the
                                      same architecture without re-typing flags
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

LOG = '[mamba]'


def sha16(path):
    """First 16 hex chars of a file's SHA-256, or '' if it is not there."""
    if not path or not os.path.exists(path):
        return ''
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()[:16]


def pool_sidecar(pool_path):
    """The generator's .fingerprint.json beside the pool, or {} when absent."""
    fp = pool_path.replace('.npz', '.fingerprint.json')
    if not os.path.exists(fp):
        return {}
    try:
        return json.load(open(fp))
    except (ValueError, OSError):
        return {}


def make_bce_on_logits():
    """Plain binary cross-entropy on a logit output, the alpha=1 arm of
    train_student.make_distillation_loss with the teacher column removed."""
    import tensorflow as tf
    bce = tf.keras.losses.binary_crossentropy

    def loss(y_true, y_pred):
        return bce(y_true, y_pred, from_logits=True)

    return loss


def make_hard_accuracy():
    import tensorflow as tf

    def hard_accuracy(y_true, y_pred):
        pred = tf.cast(y_pred > 0.0, y_true.dtype)      # logit > 0 == prob > 0.5
        return tf.reduce_mean(tf.cast(tf.equal(pred, y_true), tf.float32))

    return hard_accuracy


def to_sequence_int8(det_evts, d, rounds, p):
    """StudentModels.to_sequence() with an int8 result instead of float32.

    Same scatter, same index map, same zero fill; only the storage type differs. The
    model casts to float32 on the device. Used so the d=7 10M-shot array fits a pod
    with limited host RAM and a 20 GB MIG slice.
    """
    from StudentModels import detector_sequence_layout
    n_t, n_pos, index_map = detector_sequence_layout(d, rounds, p)
    ev = np.asarray(det_evts)
    out = np.zeros((ev.shape[0], n_t, n_pos), dtype=np.int8)
    for t in range(n_t):
        cols = index_map[t]
        present = cols >= 0
        out[:, t, present] = ev[:, cols[present]]
    return out


def add_model_args(ap):
    """Model flags shared with eval_mamba_on_tail.py. Defaults are the parameter-matched
    configuration chosen on 2026-09-21 by MambaModel.find_matched_config():
    79,001 parameters against the GRU's 79,521 (-0.65%)."""
    ap.add_argument('--d-model', type=int, default=100)
    ap.add_argument('--n-layers', type=int, default=1)
    ap.add_argument('--d-state', type=int, default=16)
    ap.add_argument('--d-conv', type=int, default=4)
    ap.add_argument('--expand', type=int, default=2)
    ap.add_argument('--dt-rank', default='auto',
                    help="'auto' = ceil(d_model/16), or an integer")


def model_kwargs(args):
    dt_rank = args.dt_rank if args.dt_rank == 'auto' else int(args.dt_rank)
    return dict(d_model=args.d_model, n_layers=args.n_layers, d_state=args.d_state,
                d_conv=args.d_conv, expand=args.expand, dt_rank=dt_rank)


def run():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- problem geometry ---
    ap.add_argument('--d', type=int, default=7)
    ap.add_argument('--p', type=float, default=0.004)
    ap.add_argument('--rounds', type=int, default=7)
    # --- data / partitions ---
    ap.add_argument('--pool', required=True, help='training pool npz (det_evts, flips)')
    ap.add_argument('--n-train', type=int, required=True, help='training prefix [0, N)')
    ap.add_argument('--val-start', type=int, required=True)
    ap.add_argument('--val-n', type=int, required=True)
    ap.add_argument('--sealed-start', type=int, default=None,
                    help='first shot of the sealed test block; asserted to lie at or '
                         'after the end of validation, so nothing here can read it')
    # --- model ---
    add_model_args(ap)
    # --- recipe ---
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--lr', type=float, required=True, help='constant Adam learning rate')
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=None,
                    help='early stopping patience. Default None = off, the Experiment 16 '
                         'protocol (fixed budget, min-val_loss checkpoint).')
    ap.add_argument('--jit', action='store_true',
                    help='compile the train/predict steps with XLA (jit_compile=True). '
                         'The unrolled selective scan is many small elementwise ops that '
                         'XLA fuses; measured on the EAF smoke as the difference between '
                         '250 ms/step and whatever this reports. Same setting must be '
                         'used for every run in a comparison.')
    ap.add_argument('--require-determinism', action='store_true',
                    help='hard-fail unless TF_DETERMINISTIC_OPS=1 and '
                         'TF_CUDNN_DETERMINISTIC=1 were exported before this process')
    # --- io ---
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tag', required=True, help='filename stem for every output')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--gpu-mem-mib', type=int, default=None,
                    help='cap this process to N MiB of GPU memory')
    args = ap.parse_args()

    if args.require_determinism:
        # Must run before TensorFlow is imported; the variables are read at import.
        from slice_guard_r5 import assert_deterministic_env
        assert_deterministic_env()
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')
    import tensorflow as tf
    assert tf.__version__.startswith('2.15'), (
        f'need TF 2.15.x (Keras 2); got TF {tf.__version__}. Activate the pinned env.')
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')
    elif args.gpu_mem_mib:
        from train_student import cap_gpu_memory
        cap_gpu_memory(tf, args.gpu_mem_mib)
    print(f"{LOG} TF {tf.__version__}  GPUs: {tf.config.list_physical_devices('GPU')}",
          flush=True)

    from train_one import set_seeds
    from StudentModels import detector_sequence_layout
    from MambaModel import build_mamba_decoder

    d, p, r = args.d, args.p, args.rounds
    ntr = args.n_train
    v0, v1 = args.val_start, args.val_start + args.val_n

    # --- partition sanity, before any array is read -------------------------------
    if not os.path.exists(args.pool):
        raise SystemExit(f"{LOG} MISSING pool {args.pool}")
    z = np.load(args.pool)
    N = z['det_evts'].shape[0]
    n_det = z['det_evts'].shape[1]
    if ntr > v0:
        raise SystemExit(f"{LOG} training [0, {ntr}) overlaps validation at {v0}")
    if v1 > N:
        raise SystemExit(f"{LOG} validation [{v0}, {v1}) exceeds pool size {N}")
    if args.sealed_start is not None and v1 > args.sealed_start:
        raise SystemExit(f"{LOG} validation [{v0}, {v1}) runs into the sealed block at "
                         f"{args.sealed_start}")
    n_t, n_pos, _ = detector_sequence_layout(d, r, p)
    if n_det != r * (d * d - 1):
        raise SystemExit(f"{LOG} pool has {n_det} detectors, d={d} r={r} needs "
                         f"{r * (d * d - 1)} -- not the same experiment")
    print(f"{LOG} pool {args.pool}  shots={N:,}  det_evts width={n_det}", flush=True)
    print(f"{LOG} sequence layout {n_t} timesteps x {n_pos} positions", flush=True)
    print(f"{LOG} partitions: train [0, {ntr:,})  validation [{v0:,}, {v1:,})"
          + (f"  sealed from {args.sealed_start:,} (never read)"
             if args.sealed_start is not None else ''), flush=True)

    side = pool_sidecar(args.pool)
    if side and (side.get('d'), side.get('rounds')) != (d, r):
        raise SystemExit(f"{LOG} fingerprint says d={side.get('d')} r={side.get('rounds')}, "
                         f"run asks for d={d} r={r}")

    # --- features -------------------------------------------------------------------
    # Same scatter as the GRU: det_evts -> [N, 8, 48], zero at the unmeasured positions
    # of the first and last timesteps, stored int8 (the model casts). measurements is
    # never touched. z['det_evts'] is read ONCE: each access to an npz member re-reads
    # the whole 6.4 GB array, and two live copies plus the scatter is what OOM-kills a
    # small pod.
    t0 = time.time()
    det_all = z['det_evts']
    x_tr = to_sequence_int8(det_all[0:ntr], d, r, p)
    x_va = to_sequence_int8(det_all[v0:v1], d, r, p)
    del det_all
    flips_all = z['flips']
    y_tr = flips_all[0:ntr].astype(np.float32).reshape(-1, 1)
    y_va = flips_all[v0:v1].astype(np.float32).reshape(-1, 1)
    del flips_all
    print(f"{LOG} features ready in {time.time() - t0:.0f}s: x_tr {x_tr.shape} "
          f"({x_tr.nbytes / 2**30:.1f} GiB)  x_va {x_va.shape}  "
          f"train base rate {y_tr.mean():.5f}  val base rate {y_va.mean():.5f}", flush=True)

    # --- build, compile, fit --------------------------------------------------------
    # set_seeds also seeds numpy, which MambaBlock.build() uses for the dt bias init,
    # so the initial model is a function of --seed alone.
    set_seeds(args.seed)
    cfg = model_kwargs(args)
    model = build_mamba_decoder(n_t, n_pos, input_dtype='int8', **cfg)
    n_params = int(model.count_params())
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss=make_bce_on_logits(), metrics=[make_hard_accuracy()],
                  jit_compile=bool(args.jit))
    print(f"{LOG} jit_compile={bool(args.jit)}", flush=True)
    print(f"{LOG} config {cfg}  params={n_params:,}", flush=True)
    model.summary(print_fn=lambda s: print(f"{LOG} {s}"))

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    ck = os.path.join(args.ckpt_dir, args.tag)

    class _SaveLastEpoch(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            self.model.save_weights(ck + '.lastepoch.weights.h5')

    # Order: last-epoch saver first (must see the true final weights before any
    # EarlyStopping restore), best-checkpoint saver last. Same rule as train_student.py.
    callbacks = [_SaveLastEpoch()]
    if args.patience is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.patience, restore_best_weights=True))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        ck + '.best.weights.h5', monitor='val_loss', mode='min',
        save_best_only=True, save_weights_only=True, verbose=0))
    print(f"{LOG} callback order: {[type(c).__name__ for c in callbacks]}", flush=True)
    print(f"{LOG} checkpoints -> {ck}.best.weights.h5 and {ck}.lastepoch.weights.h5",
          flush=True)

    steps = int(np.ceil(ntr / args.batch_size))
    print(f"{LOG} {steps} steps/epoch x {args.epochs} epochs = {steps * args.epochs:,} "
          f"optimizer updates at lr={args.lr:g}", flush=True)

    # Never overwrite a finished result: archive rather than clobber.
    csv_path = os.path.join(args.out_dir, args.tag + '.csv')
    if os.path.exists(csv_path):
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        for ext in ('.csv', '.history.json', '.config.json'):
            old = os.path.join(args.out_dir, args.tag + ext)
            if os.path.exists(old):
                os.rename(old, os.path.join(args.out_dir,
                                            f'{args.tag}.superseded_{stamp}{ext}'))
        print(f"{LOG} existing results archived as {args.tag}.superseded_{stamp}.*",
              flush=True)

    t0 = time.time()
    hist = model.fit(x=x_tr, y=y_tr, batch_size=args.batch_size, epochs=args.epochs,
                     shuffle=True, verbose=2, callbacks=callbacks,
                     validation_data=(x_va, y_va))
    train_time = time.time() - t0
    epochs_ran = len(hist.history['loss'])
    val_losses = hist.history['val_loss']
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val = float(min(val_losses))
    sec_per_epoch = train_time / max(epochs_ran, 1)

    # --- write results --------------------------------------------------------------
    prov = dict(
        run_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        train_pool=os.path.basename(args.pool),
        train_pool_dir=os.path.basename(os.path.dirname(os.path.abspath(args.pool))),
        pool_gen_seed=side.get('gen_seed', ''),
        pool_flips_sha=str(side.get('flips_sha256', ''))[:16],
        code_sha=sha16(os.path.abspath(__file__)),
        model_sha=sha16(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'MambaModel.py')),
        host=os.uname().nodename,
        tf_version=tf.__version__,
        gpus=len(tf.config.list_physical_devices('GPU')),
    )
    fields = ['architecture', 'd', 'p', 'rounds', 'seed', 'n_train', 'val_start', 'val_n',
              'lr', 'batch_size', 'epochs', 'epochs_ran', 'patience', 'jit',
              'd_model', 'n_layers', 'd_state', 'd_conv', 'expand', 'dt_rank', 'n_params',
              'best_val_loss', 'best_epoch', 'final_val_loss',
              'train_time_s', 'sec_per_epoch'] + list(prov)
    row = dict(architecture=model.name, d=d, p=p, rounds=r, seed=args.seed, n_train=ntr,
               val_start=v0, val_n=args.val_n, lr=args.lr, batch_size=args.batch_size,
               epochs=args.epochs, epochs_ran=epochs_ran,
               patience=('' if args.patience is None else args.patience),
               jit=int(bool(args.jit)), **cfg, n_params=n_params,
               best_val_loss=round(best_val, 6), best_epoch=best_epoch,
               final_val_loss=round(float(val_losses[-1]), 6),
               train_time_s=round(train_time, 1), sec_per_epoch=round(sec_per_epoch, 2),
               **prov)
    with open(csv_path, 'w', newline='') as cf:
        w = csv.DictWriter(cf, fieldnames=fields)
        w.writeheader()
        w.writerow(row)
    with open(os.path.join(args.out_dir, args.tag + '.history.json'), 'w') as hf:
        json.dump({k: [float(x) for x in v] for k, v in hist.history.items()}, hf)
    with open(os.path.join(args.out_dir, args.tag + '.config.json'), 'w') as cfh:
        json.dump(dict(model=cfg, input_dtype='int8',
                       n_timesteps=n_t, n_positions=n_pos, d=d, p=p, rounds=r,
                       n_params=n_params, seed=args.seed, lr=args.lr, jit=int(bool(args.jit)),
                       batch_size=args.batch_size, epochs=args.epochs), cfh, indent=2)

    print(f"{LOG} {args.tag}  best val_loss={best_val:.6f} at epoch {best_epoch}/"
          f"{epochs_ran}  ({train_time:.0f}s, {sec_per_epoch:.1f} s/epoch)", flush=True)
    print(f"{LOG} wrote -> {csv_path}  (+ history, config)", flush=True)


if __name__ == '__main__':
    run()
