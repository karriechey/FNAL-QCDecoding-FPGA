#!/usr/bin/env python3
# Created: 2026-07-29
# Last modified: 2026-08-17
"""Cache the FP32 teacher's per-shot output over a pool, for knowledge distillation.

The teacher is frozen, so its output for a shot never changes across epochs, seeds or
student architectures. Computing it once makes the student trainer a plain supervised fit
and lets every student variant reuse one teacher pass.

Writes an .npz containing, for the shots covered:
  p_teacher      float32 [n]  teacher sigmoid output, P(logical flip) per shot
  logit_teacher  float32 [n]  log(p/(1-p)), recovered in float64 before downcasting
  flips          int8    [n]  truth label, so the trainer needs one file
  shot_idx       int64   [n]  index back into the source pool
plus metadata and fingerprints (see pool_fingerprint) that make a cache self-describing.

Both p and its logit are stored because the distillation temperature acts in logit space
and inverting a float32 p later loses precision in the saturated tails, where a confident
teacher lives.

Dumps whatever shot range is asked for and applies no train/test split; that logic stays
in the trainer so there is only one copy of the disjointness rule.

  python dump_teacher_probs.py \
      --weights ~/rcnn_threshold/out_t200k_w/rcnn_d5_p0.010_r3_seed0_ntr10000000.weights.h5 \
      --d 5 --p 0.010 --rounds 3 --n-shots 1000000 \
      --out ~/rcnn_threshold/teacher/teacher_seed0_first1M.npz

Two teacher architectures
-------------------------
`--teacher-arch rcnn` (default) rebuilds CNNModel.FullRCNNModel, whose head ends in a
sigmoid: it emits a probability, and the stored logit is recovered by inverting that
sigmoid in float64.

`--teacher-arch gru` rebuilds StudentModels.build_gru_student -- the same GRU family the
students come from, used as the teacher for the parameter-reduction study at d=5, r=5,
p=0.004. That model's head is linear by design (see StudentModels' module docstring), so
it emits the logit directly and the stored probability is sigmoid(logit). Running
recover_logit() on a GRU output would be wrong in both directions: it would treat a logit
as a probability, and every value outside (0, 1) would be clipped to the +/-16 rails.

The output .npz layout, fingerprints and metadata are identical for both architectures, so
train_student.py consumes either without knowing which produced it.
"""
import argparse
import hashlib
import os
import numpy as np


def sha256_file(path, chunk=1 << 20):
    """SHA-256 of a file. Fingerprints the teacher checkpoint, which alone determines
    every soft target in the cache; a couple of hundred KB, so hashing it whole is free.
    """
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def pool_fingerprint(pool_npz):
    """Identity fingerprint for a pool: (flips_sha256, measurements_shape_string).

    Hashes `flips` (10 MB of int8, milliseconds) and records the `measurements` shape.
    flips comes from the same generator draw as the measurements, so agreement on 10M
    labels and on the shape means the same draw. The measurement arrays are left alone;
    hashing them would be ~1.7 GB of IO per run for no extra assurance.
    """
    flips = np.ascontiguousarray(pool_npz['flips'])
    digest = hashlib.sha256(flips.tobytes()).hexdigest()
    return digest, str(pool_npz['measurements'].shape)


def recover_logit(p):
    """Invert the teacher's final sigmoid: z = log(p / (1-p)).

    Float64, with p clipped off 0 and 1 so a saturated prediction gives a finite z. The
    clip caps |z| near 16, beyond any value the student needs to match, so it affects only
    degenerate shots.
    """
    p64 = np.asarray(p, dtype=np.float64).reshape(-1)
    eps = 1e-7
    p64 = np.clip(p64, eps, 1.0 - eps)
    return np.log(p64 / (1.0 - p64))


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True,
                    help='teacher .weights.h5 saved by train_one.py --save-weights, or by '
                         'train_student.py when --teacher-arch gru')
    ap.add_argument('--teacher-arch', choices=['rcnn', 'gru'], default='rcnn',
                    help="'rcnn' = CNNModel.FullRCNNModel (sigmoid head, probability out); "
                         "'gru' = StudentModels.build_gru_student (linear head, logit out). "
                         'See the module docstring for why the two are converted differently.')
    ap.add_argument('--units', type=int, default=64,
                    help='GRU hidden-state width; --teacher-arch gru only. Must match the '
                         'width the checkpoint was trained with or load_weights fails.')
    ap.add_argument('--student-hidden', type=int, nargs='*', default=[],
                    help='Dense widths between the GRU and its head; --teacher-arch gru '
                         'only. Empty (the default) matches the studies run so far.')
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
    # Keras 3 rebuilds CNNModel.py's custom layers through a different path and fails deep
    # inside an initializer (list-vs-tuple shape compare), so fail here instead.
    assert tf.__version__.startswith('2.15'), (
        f'need TF 2.15.x (Keras 2); got TF {tf.__version__}. '
        f'Activate the uv venv: source .venv/bin/activate, or run with '
        f'PY=<repo>/.venv/bin/python.')
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

    # Identity fingerprints, computed once here and re-verified by train_student.py. These
    # catch "same filename, different content" -- a re-generated pool or a re-trained
    # teacher checkpoint -- which the runtime flips-agreement check cannot see because it
    # only compares the shots the cache itself carries.
    weights_sha = sha256_file(args.weights)
    pool_flips_sha, pool_meas_shape = pool_fingerprint(z)
    print(f"[teacher] weights sha256={weights_sha[:16]}...  "
          f"pool flips sha256={pool_flips_sha[:16]}...  measurements{pool_meas_shape}",
          flush=True)
    lo, hi = args.n_start, args.n_start + args.n_shots
    if hi > N:
        raise SystemExit(f"[teacher] requested shots [{lo}, {hi}) exceed pool size {N}")
    sl = slice(lo, hi)

    # Slice before the measurement split so only the requested shots are materialised;
    # the arrays are ~1 GB each at 10M.
    det_evts = z['det_evts'][sl].astype(binary_t)
    flips = z['flips'][sl].astype(binary_t)

    hidden = [args.hidden for _ in range(args.hidden_layers)]
    if args.teacher_arch == 'gru':
        # The GRU teacher reads det_evts only, so the measurement array is never sliced --
        # that is ~1.5 GB of IO and host memory avoided per dump at 10M shots. The scatter
        # into [n_timesteps, n_positions] happens in numpy, exactly as in the trainer, so
        # the teacher and the students it supervises share one input pipeline.
        from StudentModels import assemble_features, build_gru_student
        model = build_gru_student(d=d, rounds=r, inputs='evts', units=args.units,
                                  hidden=tuple(args.student_hidden), p=p)
        x = assemble_features(None, det_evts, 'evts', student='gru', d=d, rounds=r, p=p)
        print(f"[teacher] gru teacher units={args.units} "
              f"hidden={tuple(args.student_hidden)} params={model.count_params():,} "
              f"input{x.shape[1:]}", flush=True)
        model.load_weights(args.weights)

        # Linear head: this is the logit. p follows from it, not the other way round.
        pred = model.predict(x, batch_size=args.batch_size, verbose=0)
        logit_teacher = np.asarray(pred, dtype=np.float32).reshape(-1)
        p_teacher = (1.0 / (1.0 + np.exp(-logit_teacher.astype(np.float64)))).astype(np.float32)
        del x
    else:
        measurements = z['measurements'][sl].astype(binary_t)
        det_bits, _obs_bits, _data_bits = split_measurements(measurements, d, idx_t)
        if args.weight_bits is None or args.weight_bits >= 32:
            from CNNModel import FullRCNNModel
            model = FullRCNNModel(
                'ZL', d, k, r, hidden, npol=args.npol, stop_round=None,
                has_nonuniform_response=False, do_all_data_qubits=False,
                return_all_rounds=False)
        else:
            from CNNModel_quantized import build_quantized_rcnn
            model = build_quantized_rcnn(
                args.weight_bits, 'ZL', d, k, r, hidden, npol=args.npol, stop_round=None,
                has_nonuniform_response=False, do_all_data_qubits=False,
                return_all_rounds=False)
        _ = model([det_bits[0:1], det_evts[0:1]])  # build so the checkpoint layout matches
        model.load_weights(args.weights)

        # Sigmoid head: this is a probability, and the logit is recovered by inverting it.
        pred = model.predict([det_bits, det_evts], batch_size=args.batch_size, verbose=0)
        p_teacher = np.asarray(pred, dtype=np.float32).reshape(-1)
        logit_teacher = recover_logit(p_teacher).astype(np.float32)
        del measurements, det_bits

    truth = np.asarray(flips, dtype=np.int8).reshape(-1)

    # The teacher's p_L on the dumped shots. In-sample when --n-start covers the training
    # prefix, so not a result; printed to catch a broken load (p_L at the base rate, or a
    # constant output) before a student trains on it.
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
        # Which architecture produced these outputs, and its width. Recorded so a cache
        # cannot be silently attributed to the wrong teacher family when both an RCNN and
        # a GRU teacher exist for the same pool.
        teacher_arch=args.teacher_arch, units=args.units,
        student_hidden=np.asarray(args.student_hidden, dtype=np.int64),
        weight_bits=(-1 if args.weight_bits is None else args.weight_bits),
        n_start=lo, n_shots=args.n_shots,
        weights_path=os.path.abspath(args.weights),
        pool_path=os.path.abspath(fn),
        # fingerprints -- see pool_fingerprint()/sha256_file() for why these two and not
        # a full hash of the measurement arrays
        weights_sha256=weights_sha,
        pool_flips_sha256=pool_flips_sha,
        pool_measurements_shape=pool_meas_shape,
    )
    print(f"[teacher] wrote -> {args.out}", flush=True)


if __name__ == '__main__':
    run()
