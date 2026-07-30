#!/usr/bin/env python3
# Created: 2026-07-29
# Last modified: 2026-07-29
"""Cache the FP32 teacher's per-shot output over a pool, for knowledge distillation.

WHY THIS EXISTS
The distillation student is trained on the TEACHER's outputs (soft labels), not only on
the hard `flips` labels. Running the teacher's forward pass inside every student epoch
would be wasteful and slow: the teacher is frozen, so its output for a given shot never
changes across epochs or across student architectures. Computing it ONCE and caching it
to disk means the student trainer is a plain supervised fit over arrays, and the same
cache is reused by every student variant (MLP, GRU, every seed, every bit-width in the
later quantization sweep).

WHAT IT WRITES
An .npz with, for the shots covered:
  p_teacher      float32 [n]  teacher sigmoid output, i.e. P(logical flip) per shot
  logit_teacher  float32 [n]  log(p/(1-p)), the pre-sigmoid score
  flips          int8    [n]  the hard truth label, carried along so the student trainer
                              needs exactly one file
  shot_idx       int64   [n]  index of each shot back into the source pool
Plus scalar metadata (d, p, rounds, kernel, hidden, hidden_layers, npol, weights path,
pool path, n_start, n_shots) so a cache file is self-describing and can never be silently
paired with the wrong pool or the wrong teacher.

WHY BOTH p AND logit
The distillation loss softens the teacher with a temperature T, which is defined in
LOGIT space (sigmoid(z/T)). Recovering z from a stored p as log(p/(1-p)) loses precision
in the saturated tails, exactly where a confident teacher lives. The teacher's final
layer here is a sigmoid, so we recover the logit from p in float64 before downcasting;
that is still far more accurate than doing it later from a float32 p, and storing both
means the student trainer never has to invert anything.

DISJOINTNESS
This script does NOT slice a train/test split -- it just dumps whatever shot range is
asked for. Keeping the split logic in the trainer (where the existing asserts live)
avoids a second, divergent copy of the disjointness rule. Dump the training prefix
[0, n_train) for distillation; dump a tail range separately if you want the teacher's
probabilities on the evaluation tail for gap analysis.

USAGE
  python dump_teacher_probs.py \
      --weights ~/rcnn_threshold/out_t200k_w/rcnn_d5_p0.010_r3_seed0_ntr10000000.weights.h5 \
      --d 5 --p 0.010 --rounds 3 --n-shots 1000000 \
      --out ~/rcnn_threshold/teacher/teacher_seed0_first1M.npz
"""
import argparse
import os
import numpy as np


def recover_logit(p):
    """Invert the teacher's final sigmoid: z = log(p / (1-p)).

    Done in float64 and with the probabilities clipped away from exactly 0 and 1, because
    a fully-saturated prediction would otherwise give +/-inf and poison the distillation
    loss. The clip bound is one float32 epsilon-ish step from the endpoints, which caps
    |z| near 16 -- well beyond any temperature-softened value the student needs to match,
    so the clip changes nothing about a normal shot and only tames the degenerate ones.
    """
    p64 = np.asarray(p, dtype=np.float64).reshape(-1)
    eps = 1e-7
    p64 = np.clip(p64, eps, 1.0 - eps)
    return np.log(p64 / (1.0 - p64))


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True,
                    help='teacher .weights.h5 saved by train_one.py --save-weights')
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--rounds', type=int, required=True)
    ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--hidden', type=int, default=100)
    ap.add_argument('--hidden-layers', type=int, default=2)
    ap.add_argument('--npol', type=int, default=2)
    ap.add_argument('--n-start', type=int, default=0,
                    help='first shot index in the pool to dump (default 0 = training prefix)')
    ap.add_argument('--n-shots', type=int, required=True,
                    help='number of shots to dump starting at --n-start')
    ap.add_argument('--weight-bits', type=int, default=None,
                    help='if set, rebuild the QUANTIZED teacher instead of FP32. Only needed '
                         'if you ever want to distil from a quantized teacher; the plan of '
                         'record distils from the FP32 teacher, so leave this unset.')
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--data-dir', default=os.path.expanduser('~/rcnn_threshold/pools'))
    ap.add_argument('--pool', default=None,
                    help='explicit pool npz; overrides the --data-dir name convention')
    ap.add_argument('--out', required=True, help='output .npz path for the cached outputs')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')

    from types_cfg import get_types
    from circuit_partition import split_measurements

    d, p, r, k = args.d, args.p, args.rounds, args.kernel
    binary_t, _time_t, idx_t, _packed_t = get_types(d, r, k)

    fn = args.pool or os.path.join(args.data_dir, f'data_d{d}_p{p:.3f}_r{r}.npz')
    if not os.path.exists(fn):
        raise SystemExit(f"[teacher] MISSING pool {fn}")
    if not os.path.exists(args.weights):
        raise SystemExit(f"[teacher] MISSING weights {args.weights}")

    z = np.load(fn)
    N = z['measurements'].shape[0]
    lo, hi = args.n_start, args.n_start + args.n_shots
    if hi > N:
        raise SystemExit(f"[teacher] requested shots [{lo}, {hi}) exceed pool size {N}")
    sl = slice(lo, hi)

    # Slice BEFORE the measurement split so we only materialise the shots we asked for.
    # The full 10M pool is ~1 GB per array; dumping a 1M de-risk subset should not pay
    # the memory cost of the whole thing.
    measurements = z['measurements'][sl].astype(binary_t)
    det_evts = z['det_evts'][sl].astype(binary_t)
    flips = z['flips'][sl].astype(binary_t)
    det_bits, _obs_bits, _data_bits = split_measurements(measurements, d, idx_t)

    hidden = [args.hidden for _ in range(args.hidden_layers)]
    if args.weight_bits is None or args.weight_bits >= 32:
        from CNNModel import FullRCNNModel
        model = FullRCNNModel(
            'ZL', d, k, r, hidden, npol=args.npol, stop_round=None,
            has_nonuniform_response=False, do_all_data_qubits=False, return_all_rounds=False)
    else:
        from CNNModel_quantized import build_quantized_rcnn
        model = build_quantized_rcnn(
            args.weight_bits, 'ZL', d, k, r, hidden, npol=args.npol, stop_round=None,
            has_nonuniform_response=False, do_all_data_qubits=False, return_all_rounds=False)
    _ = model([det_bits[0:1], det_evts[0:1]])  # build so the checkpoint layout matches
    model.load_weights(args.weights)

    pred = model.predict([det_bits, det_evts], batch_size=args.batch_size, verbose=0)
    p_teacher = np.asarray(pred, dtype=np.float32).reshape(-1)
    logit_teacher = recover_logit(p_teacher).astype(np.float32)
    truth = np.asarray(flips, dtype=np.int8).reshape(-1)

    # Sanity: the teacher's own p_L on the dumped shots. If --n-start points at the
    # training prefix this is an IN-SAMPLE number and must not be quoted as a result --
    # it is printed only so an obviously broken load (p_L at the base rate, or a constant
    # output) is caught here rather than after a student has been trained on garbage.
    teacher_pred = (p_teacher > 0.5).astype(np.int8)
    pL_insample = float((teacher_pred != truth).mean())
    base_rate = float(truth.mean())
    frac_confident = float(((p_teacher < 0.05) | (p_teacher > 0.95)).mean())
    print(f"[teacher] shots [{lo:,}, {hi:,})  p_L(on these shots)={pL_insample:.5f}  "
          f"base_rate={base_rate:.5f}  mean_p={p_teacher.mean():.5f}  "
          f"frac_confident={frac_confident:.3f}", flush=True)
    if pL_insample >= base_rate:
        print("[teacher] WARNING: teacher does not beat the all-zero base rate on these "
              "shots -- check the weights path and the architecture args.", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    np.savez_compressed(
        args.out,
        p_teacher=p_teacher,
        logit_teacher=logit_teacher,
        flips=truth,
        shot_idx=np.arange(lo, hi, dtype=np.int64),
        # metadata -- makes a cache file self-describing so it cannot be paired with the
        # wrong pool or the wrong teacher checkpoint by accident
        d=d, p=p, rounds=r, kernel=k, hidden=args.hidden,
        hidden_layers=args.hidden_layers, npol=args.npol,
        weight_bits=(-1 if args.weight_bits is None else args.weight_bits),
        n_start=lo, n_shots=args.n_shots,
        weights_path=os.path.abspath(args.weights),
        pool_path=os.path.abspath(fn),
    )
    print(f"[teacher] wrote -> {args.out}", flush=True)


if __name__ == '__main__':
    run()
