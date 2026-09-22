#!/usr/bin/env python3
# Created: 2026-09-21
# Last modified: 2026-09-22
"""Selective state-space (Mamba) decoder, parameter-matched to the d=7 GRU student.

Why this file exists
Experiment 16 measured the GRU student at d=7, r=7, p=0.004 (units=140, 79,521
parameters, 2.87x MWPM at 10M shots). The question this model serves is narrow: holding
the data, the dense temporal input, the labels, the training recipe and the parameter
budget fixed, does a selective SSM recurrence decode better than a gated RNN recurrence?
Everything that is not the recurrence is therefore copied from StudentModels.py, and
nothing new is added to the input (no defect tokens, no cumulative-XOR channel).

Why it is written in TensorFlow / Keras 2 rather than the reference PyTorch `mamba_ssm`
The GRU it is compared against is a Keras 2 model on the pinned TF 2.15 stack. Keeping
the SSM in the same framework means the same data pipeline (StudentModels.to_sequence),
the same seeding (train_one.set_seeds), the same loss (BCE on logits), the same
determinism flags and the same result/per-shot file formats, so the paired McNemar in
pair_two_decoders.py / analyze_mamba_vs_gru.py reads both decoders without adapters.
The reference CUDA selective-scan kernel is a throughput optimisation for long
sequences; here the sequence is 8 detector timesteps, so the scan is unrolled in plain
TensorFlow ops and is exact.

Input
The same [n_timesteps, n_positions] sequence the GRU takes: for d=7, r=7 that is
(8, 48), produced by StudentModels.to_sequence() from det_evts. Timesteps 0 and 7 carry
24 detectors each (the first round detects only Z stabilisers; the last comes from the
data-qubit readout) and are zero-filled at the unmeasured positions, exactly as for the
GRU. The model never sees the raw measurement bits.

Architecture (Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective
State Spaces", Algorithm 2), one block:

    u  = x @ W_in                        [B, T, 2E]  ->  split into x_b, z   (each E)
    x_b = SiLU(causal_depthwise_conv1d(x_b, k))
    (dt_raw, Bm, Cm) = x_b @ W_x         [B, T, R + N + N]
    dt = softplus(dt_raw @ W_dt + b_dt)  [B, T, E]      the selective step size
    A  = -exp(A_log)                     [E, N]         learned, negative real
    h_t = exp(dt_t * A) * h_{t-1} + (dt_t * Bm_t) * x_b,t        (ZOH on A, Euler on B)
    y_t = <h_t, Cm_t> + D * x_b,t
    y  = y * SiLU(z)
    out = y @ W_out                      [B, T, D_model]

Blocks are pre-norm residual: x <- x + Block(RMSNorm(x)). A Dense embedding lifts the
48-wide detector vector to d_model before the first block; after the last block the
final timestep's vector (the SSM is causal, so it has seen every round) goes through a
final RMSNorm and a linear head to ONE logit. The decision is logit > 0, the same
convention as the GRU student (StudentModels.student_pred_and_correct).

What "parameter-matched" means here
`count_params()` on the built model, compared against the GRU's 79,521, the same way
Experiment 16 matched the MLP to the GRU. `find_matched_config()` searches a small grid
of (d_model, expand, d_state, n_layers) for the count closest to the target and reports
the percentage difference, so the match is measured on the built model rather than
estimated from a formula.

Output is a logit, not a probability, for the same reasons given in StudentModels.py.
"""
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import Model


# The GRU count this model is matched against: StudentModels.build_gru_student at
# d=7, r=7, units=140, no post-GRU Dense layers. Verified locally 2026-09-21 with
# count_params(); it is also the value in Experiment 16's table and MANIFEST files.
GRU_D7_U140_PARAMS = 79_521


class RMSNorm(KL.Layer):
    """Root-mean-square layer norm (no mean subtraction, no bias), as used in Mamba."""

    def __init__(self, eps=1e-5, **kw):
        super().__init__(**kw)
        self.eps = eps

    def build(self, input_shape):
        self.scale = self.add_weight('scale', shape=(int(input_shape[-1]),),
                                     initializer='ones', trainable=True)

    def call(self, x):
        var = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        return x * tf.math.rsqrt(var + self.eps) * self.scale

    def get_config(self):
        return {**super().get_config(), 'eps': self.eps}


class MambaBlock(KL.Layer):
    """One selective-SSM mixer (the inner part of a Mamba block, without the residual).

    Parameters
      d_model   width of the residual stream in and out of the block
      d_state   N, size of the per-channel SSM state
      d_conv    kernel of the causal depthwise conv on the input branch
      expand    E = expand * d_model, the inner width the scan runs over
      dt_rank   R, rank of the low-rank projection producing the step size dt

    The scan is unrolled over the (static) sequence length in a Python loop of TensorFlow
    ops. For T=8 this is a handful of elementwise multiplies on [B, E, N] tensors per
    step; there is no custom kernel and nothing non-deterministic in it.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank='auto',
                 dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4, **kw):
        super().__init__(**kw)
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.expand * self.d_model
        self.dt_rank = (math.ceil(self.d_model / 16) if dt_rank == 'auto' else int(dt_rank))
        self.dt_min, self.dt_max, self.dt_init_floor = dt_min, dt_max, dt_init_floor

    def build(self, input_shape):
        E, N, R = self.d_inner, self.d_state, self.dt_rank
        # x and z branches in one matmul, no bias (as in the reference implementation).
        self.in_proj = KL.Dense(2 * E, use_bias=False, name='in_proj')
        # Depthwise conv over time. Keras DepthwiseConv1D does not accept
        # padding='causal', so call() left-pads by k-1 zeros and runs it 'valid': timestep
        # t then only sees t-k+1..t and the recurrence stays causal.
        self.conv1d = KL.DepthwiseConv1D(self.d_conv, padding='valid', use_bias=True,
                                         name='conv1d')
        # From the inner activation to (dt_raw, B, C): the "selective" part, since B and
        # C now depend on the input rather than being fixed matrices.
        self.x_proj = KL.Dense(R + 2 * N, use_bias=False, name='x_proj')
        # Low-rank step-size projection with a bias initialised so softplus(bias) lands
        # in [dt_min, dt_max], log-uniformly, as in the reference init. Without this the
        # initial dt is ~0.7 for every channel and the state forgets nothing.
        dt = np.exp(np.random.uniform(np.log(self.dt_min), np.log(self.dt_max), size=E))
        dt = np.maximum(dt, self.dt_init_floor)
        inv_softplus = dt + np.log(-np.expm1(-dt))          # softplus^-1(dt)
        self.dt_proj = KL.Dense(E, use_bias=True, name='dt_proj',
                                bias_initializer=tf.constant_initializer(inv_softplus))
        # S4D-real initialisation: A_n = -(n+1) for n = 0..N-1 on every channel, stored
        # as log(-A) so A stays negative through training.
        a_init = np.tile(np.log(np.arange(1, N + 1, dtype=np.float32)), (E, 1))
        self.A_log = self.add_weight('A_log', shape=(E, N),
                                     initializer=tf.constant_initializer(a_init),
                                     trainable=True)
        # Skip connection D * x, one scalar per channel.
        self.D = self.add_weight('D', shape=(E,), initializer='ones', trainable=True)
        self.out_proj = KL.Dense(self.d_model, use_bias=False, name='out_proj')
        super().build(input_shape)

    def call(self, x):
        # x: [B, T, d_model]
        E, N, R = self.d_inner, self.d_state, self.dt_rank
        xz = self.in_proj(x)                                  # [B, T, 2E]
        xb, z = tf.split(xz, 2, axis=-1)                      # each [B, T, E]
        xb = tf.pad(xb, [[0, 0], [self.d_conv - 1, 0], [0, 0]])   # left-pad: causal
        xb = tf.nn.silu(self.conv1d(xb))                      # conv + SiLU, [B, T, E]

        proj = self.x_proj(xb)                                # [B, T, R + 2N]
        dt_raw, Bm, Cm = tf.split(proj, [R, N, N], axis=-1)
        dt = tf.nn.softplus(self.dt_proj(dt_raw))             # [B, T, E], positive
        A = -tf.exp(self.A_log)                               # [E, N], negative

        # Discretise per timestep and scan. Shapes inside the loop:
        #   dA_t = exp(dt_t[:, :, None] * A[None])            [B, E, N]
        #   dBx_t = dt_t[:, :, None] * Bm_t[:, None, :] * xb_t[:, :, None]   [B, E, N]
        #   h_t = dA_t * h_{t-1} + dBx_t
        #   y_t = sum_n h_t * Cm_t[:, None, :] + D * xb_t     [B, E]
        T = xb.shape[1]
        if T is None:
            raise ValueError('MambaBlock needs a static sequence length; the d=7 '
                             'input is (8, 48).')
        h = tf.zeros([tf.shape(xb)[0], E, N], dtype=xb.dtype)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :]                                 # [B, E]
            dA = tf.exp(dt_t[:, :, None] * A[None, :, :])      # [B, E, N]
            dBx = dt_t[:, :, None] * Bm[:, t, None, :] * xb[:, t, :, None]
            h = dA * h + dBx
            y_t = tf.reduce_sum(h * Cm[:, t, None, :], axis=-1) + self.D * xb[:, t, :]
            ys.append(y_t)
        y = tf.stack(ys, axis=1)                              # [B, T, E]
        y = y * tf.nn.silu(z)                                 # output gate
        return self.out_proj(y)                               # [B, T, d_model]

    def get_config(self):
        return {**super().get_config(), 'd_model': self.d_model, 'd_state': self.d_state,
                'd_conv': self.d_conv, 'expand': self.expand, 'dt_rank': self.dt_rank}


def build_mamba_decoder(n_timesteps=8, n_positions=48, d_model=64, n_layers=2,
                        d_state=16, d_conv=4, expand=2, dt_rank='auto',
                        input_dtype='float32', name='mamba_decoder'):
    """Stack of pre-norm residual Mamba blocks over the dense detector sequence.

    Input  [n_timesteps, n_positions], from StudentModels.to_sequence(). The values are
           detector bits (0/1), so `input_dtype='int8'` accepts the packed array and
           casts to float32 as the first op: the 10M-shot d=7 training array is then
           3.6 GiB instead of 14.3 GiB, on the host and on the device. The cast has no
           parameters and no effect on the numbers.
    Output one logit per shot (linear head, no sigmoid).
    """
    seq = KL.Input(shape=(n_timesteps, n_positions), dtype=input_dtype,
                   name='syndrome_sequence')
    x = seq
    if input_dtype != 'float32':
        x = KL.Lambda(lambda t: tf.cast(t, tf.float32), name='to_float')(x)
    # Lift the 48-wide detector vector to the residual width. A bias is kept so the
    # zero-filled unmeasured positions at t=0 and t=T-1 are not forced to map to zero.
    x = KL.Dense(d_model, use_bias=True, name='embed')(x)
    for i in range(n_layers):
        h = RMSNorm(name=f'norm_{i}')(x)
        h = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                       dt_rank=dt_rank, name=f'mamba_{i}')(h)
        x = KL.Add(name=f'residual_{i}')([x, h])
    x = RMSNorm(name='norm_final')(x)
    # Last timestep only, mirroring GRU(return_sequences=False): the recurrence is
    # causal, so the final vector has integrated all n_timesteps detector rounds.
    last = KL.Lambda(lambda t: t[:, -1, :], name='last_timestep')(x)
    out = KL.Dense(1, name='logit')(last)
    return Model(seq, out, name=name)


def count_params_for(n_timesteps, n_positions, **cfg):
    """Build once on CPU and return count_params(); the measured number, not a formula."""
    m = build_mamba_decoder(n_timesteps, n_positions, **cfg)
    return int(m.count_params())


def find_matched_config(target=GRU_D7_U140_PARAMS, n_timesteps=8, n_positions=48,
                        d_models=range(32, 129, 4), n_layers_opts=(1, 2, 3),
                        d_states=(8, 16), expands=(2,), verbose=True):
    """Grid over (d_model, n_layers, d_state, expand); return the config nearest the target.

    Ties are broken toward the smaller model. Every count comes from a built model.
    Called once at design time; the chosen config is then pinned in the run script, so
    the search never runs on the GPU host.
    """
    best = None
    rows = []
    for L in n_layers_opts:
        for N in d_states:
            for ex in expands:
                for D in d_models:
                    n = count_params_for(n_timesteps, n_positions, d_model=D, n_layers=L,
                                         d_state=N, expand=ex)
                    rows.append((abs(n - target), n, dict(d_model=D, n_layers=L,
                                                          d_state=N, expand=ex)))
                    tf.keras.backend.clear_session()
    rows.sort(key=lambda r: (r[0], r[1]))
    best = rows[0]
    if verbose:
        print(f"target (GRU units=140, d=7 r=7) = {target:,}")
        for gap, n, cfg in rows[:8]:
            print(f"  {n:>8,}  ({100.0 * (n - target) / target:+.2f}%)  {cfg}")
    return best[2], best[1]


def mamba_pred_and_correct(logits, truth):
    """Same decision rule as the GRU student: logit > 0. Kept as its own name so the
    per-shot dump column names say which decoder produced them."""
    pred = (np.asarray(logits).reshape(-1) > 0.0).astype(np.int8)
    truth = np.asarray(truth).reshape(-1).astype(np.int8)
    return pred, (pred == truth)


if __name__ == '__main__':
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    cfg, n = find_matched_config()
    print(f"\nchosen: {cfg}  params={n:,}  "
          f"GRU={GRU_D7_U140_PARAMS:,}  diff={100.0 * (n - GRU_D7_U140_PARAMS) / GRU_D7_U140_PARAMS:+.2f}%")
