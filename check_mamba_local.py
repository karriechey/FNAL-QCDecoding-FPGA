#!/usr/bin/env python3
# Created: 2026-09-21
# Last modified: 2026-09-21
"""Local, data-free checks of the Mamba decoder before it is sent to the GPU host.

Runs on CPU with synthetic tensors of the d=7, r=7 shape. Nothing here is a scientific
result; every check is a structural property that must hold for the comparison to be
valid at all:

  1  import + construction of the pinned config, and of the d=7 GRU it is matched to
  2  parameter counts of both, and the percentage difference
  3  forward pass shape [B, 1] on a synthetic [B, 8, 48] batch
  4  backward pass: every trainable weight receives a finite, non-zero gradient
     (catches a disconnected A_log / D / dt bias, which would train silently as a
     smaller model than the count claims)
  5  loss computation with BCE-on-logits against synthetic labels
  6  causality: perturbing timestep t changes the block output at times >= t only
     (the scan and the left-padded conv must both be causal or the "recurrence"
     framing is false)
  7  checkpoint round trip: save_weights / load_weights into a fresh model gives
     bit-identical logits
  8  seed determinism: two models built under the same seed give identical logits
  9  per-shot export: predictions keyed by shot index round-trip through npz and
     re-align by index after a shuffle

Exit status is non-zero on the first failure.

  CUDA_VISIBLE_DEVICES= .venv/bin/python check_mamba_local.py
"""
import os
import sys
import tempfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf

from train_one import set_seeds
from StudentModels import build_student, detector_sequence_layout
from MambaModel import (build_mamba_decoder, MambaBlock, mamba_pred_and_correct,
                        GRU_D7_U140_PARAMS)

D, R, P = 7, 7, 0.004
CFG = dict(d_model=100, n_layers=1, d_state=16, d_conv=4, expand=2, dt_rank='auto')
B = 64
FAILS = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def main():
    print(f"TF {tf.__version__}  devices {tf.config.list_physical_devices('GPU') or 'CPU only'}")
    n_t, n_pos, index_map = detector_sequence_layout(D, R, P)
    check('d=7 layout is 8 x 48', (n_t, n_pos) == (8, 48), f'{n_t} x {n_pos}')
    check('336 detectors mapped', int((index_map >= 0).sum()) == R * (D * D - 1))

    # 1-2 construction and parameter counts
    set_seeds(0)
    mamba = build_mamba_decoder(n_t, n_pos, **CFG)
    gru = build_student('gru', d=D, rounds=R, inputs='evts', units=140, hidden=())
    n_m, n_g = int(mamba.count_params()), int(gru.count_params())
    diff = 100.0 * (n_m - n_g) / n_g
    print(f"\n  Mamba params : {n_m:,}")
    print(f"  GRU   params : {n_g:,}")
    print(f"  difference   : {diff:+.2f}%  (Mamba - GRU) / GRU")
    check('GRU count matches Experiment 16', n_g == GRU_D7_U140_PARAMS)
    check('parameter match within 2%', abs(diff) < 2.0, f'{diff:+.2f}%')

    # 3 forward
    rng = np.random.default_rng(0)
    x = (rng.random((B, n_t, n_pos)) < 0.05).astype(np.float32)
    x[:, 0, index_map[0] < 0] = 0.0           # respect the unmeasured slots
    x[:, -1, index_map[-1] < 0] = 0.0
    y = (rng.random((B, 1)) < 0.3).astype(np.float32)
    out = mamba(x, training=False)
    check('forward output shape [B, 1]', tuple(out.shape) == (B, 1), str(tuple(out.shape)))
    check('forward output finite', bool(np.all(np.isfinite(out.numpy()))))

    # 4-5 loss and gradients
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    with tf.GradientTape() as tape:
        logits = mamba(x, training=True)
        loss = bce(y, logits)
    grads = tape.gradient(loss, mamba.trainable_variables)
    check('loss is finite scalar', np.isfinite(float(loss)), f'loss={float(loss):.4f}')
    dead = [v.name for v, g in zip(mamba.trainable_variables, grads)
            if g is None or not np.all(np.isfinite(g.numpy())) or float(tf.reduce_max(tf.abs(g))) == 0.0]
    check('every trainable weight gets a finite non-zero gradient', not dead,
          ', '.join(dead) if dead else f'{len(grads)} tensors')
    # one optimizer step must change the loss
    opt = tf.keras.optimizers.Adam(1e-3)
    opt.apply_gradients(zip(grads, mamba.trainable_variables))
    loss2 = float(bce(y, mamba(x, training=True)))
    check('one Adam step changes the loss', loss2 != float(loss), f'{float(loss):.5f} -> {loss2:.5f}')

    # 6 causality on the block output (all timesteps)
    blk_in = tf.keras.Input((n_t, CFG['d_model']))
    blk = MambaBlock(CFG['d_model'], d_state=CFG['d_state'], d_conv=CFG['d_conv'],
                     expand=CFG['expand'], dt_rank=CFG['dt_rank'])(blk_in)
    block_model = tf.keras.Model(blk_in, blk)
    h = rng.standard_normal((4, n_t, CFG['d_model'])).astype(np.float32)
    base = block_model(h).numpy()
    causal_ok = True
    for t in range(n_t):
        h2 = h.copy()
        h2[:, t, :] += 1.0
        delta = np.abs(block_model(h2).numpy() - base).max(axis=(0, 2))   # per timestep
        if t > 0 and delta[:t].max() > 0:
            causal_ok = False
        if delta[t] == 0:
            causal_ok = False
    check('block is causal (perturb t affects only >= t)', causal_ok)

    # 7 checkpoint round trip
    with tempfile.TemporaryDirectory() as td:
        wpath = os.path.join(td, 'm.weights.h5')
        mamba.save_weights(wpath)
        fresh = build_mamba_decoder(n_t, n_pos, **CFG)
        fresh.load_weights(wpath)
        a, b_ = mamba(x, training=False).numpy(), fresh(x, training=False).numpy()
        check('save/load gives bit-identical logits', np.array_equal(a, b_),
              f'max |diff| = {np.abs(a - b_).max():.3g}')

    # 8 seed determinism at construction
    set_seeds(3)
    m1 = build_mamba_decoder(n_t, n_pos, **CFG)
    set_seeds(3)
    m2 = build_mamba_decoder(n_t, n_pos, **CFG)
    check('same seed -> identical initial logits',
          np.array_equal(m1(x, training=False).numpy(), m2(x, training=False).numpy()))
    set_seeds(4)
    m3 = build_mamba_decoder(n_t, n_pos, **CFG)
    check('different seed -> different initial logits',
          not np.array_equal(m1(x, training=False).numpy(), m3(x, training=False).numpy()))

    # 9 per-shot export keyed by index
    logits = mamba(x, training=False).numpy().reshape(-1)
    truth = y.reshape(-1).astype(np.int8)
    pred, correct = mamba_pred_and_correct(logits, truth)
    check('decision rule: pred == (logit > 0)', np.array_equal(pred, (logits > 0).astype(np.int8)))
    idx = np.arange(17_000_000, 17_000_000 + B, dtype=np.int64)
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, 'ps.npz')
        perm = rng.permutation(B)
        np.savez_compressed(f, shot_idx=idx[perm], truth=truth[perm], mamba_pred=pred[perm],
                            mamba_correct=correct[perm], mamba_logit=logits[perm])
        z = np.load(f)
        order = np.argsort(z['shot_idx'])
        check('per-shot export re-aligns by shot_idx after shuffle',
              np.array_equal(z['shot_idx'][order], idx)
              and np.array_equal(z['mamba_correct'][order], correct)
              and np.array_equal(z['mamba_logit'][order], logits))

    print()
    if FAILS:
        print(f"FAILED: {FAILS}")
        sys.exit(1)
    print("all checks passed")


if __name__ == '__main__':
    main()
