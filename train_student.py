#!/usr/bin/env python3
# Created: 2026-07-29
# Last modified: 2026-07-29
"""Train ONE distillation student (MLP or GRU) against the cached FP32 teacher outputs.

A single point is (student, inputs, seed, n_train, alpha, temperature, weight_bits,
act_bits). This is the Phase-2 workhorse: it trains an hls4ml-native student to imitate
the custom RCNN teacher, so the FPGA flow never has to parse the teacher's unsupported
combiner ops (pow/sqrt/log/gather).

THE HARD-VS-SOFT CONTROL IS ONE FLAG
  --alpha 1.0  -> pure hard labels. The control arm: what the student architecture can do
                  on its own, learning from `flips` exactly like the teacher did.
  --alpha 0.0  -> pure distillation. Learns only from the teacher's soft output.
  0 < a < 1    -> the usual blend.
Same code path for all three, so the comparison is not confounded by two different
trainers. The de-risk experiment is this script run at both ends of alpha at 1M shots.

WHY SOFT LABELS SHOULD HELP HERE
The hard label is one bit: did the logical qubit flip. The teacher's output is a
calibrated probability that also encodes HOW ambiguous the shot's syndrome was. Shots
where the teacher sits near 0.5 are the genuinely undecidable ones, and telling the
student "this one is a coin flip" is information the hard label physically cannot carry.
That extra signal per shot is the entire reason distillation can beat same-architecture
hard-label training, and it is exactly what --alpha measures.

TEMPERATURE
Softening is done in logit space: the soft target is sigmoid(z_teacher / T) and the
student's logit is likewise divided by T. The soft term is multiplied by T^2 so its
gradient magnitude stays comparable to the hard term as T varies (otherwise raising T
would silently shrink the soft learning rate and confound a temperature scan with a
loss-weight scan). T=1 is the untempered teacher distribution.

DISJOINTNESS
Same rule as train_one_quantized.py, and for the same reason: the tail comes from a
SEPARATE --test-pool when given, otherwise from the same pool behind the
`ntr <= N - nte` assert. A 200k tail on the 10.01M pool would overlap a 10M-shot training
prefix by 190k shots. The teacher cache is additionally checked to cover the training
range and to have come from the same pool file.

Reads  the pool npz (measurements, det_evts, flips) + the teacher cache from
       dump_teacher_probs.py.
Writes {out-dir}/student_{tag}.csv, .history.json, and .weights.h5.
"""
import argparse
import csv
import json
import os
import time
import numpy as np

from train_one import set_seeds, learning_rate_scheduler, lookup_mwpm  # reuse the recipe


def make_distillation_loss(alpha, temperature):
    """Blended hard-label + teacher-imitation loss over PACKED targets.

    y_true is packed as two columns, [hard_label, teacher_logit], because Keras hands the
    loss exactly one target tensor. y_pred is the student's raw logit.

    Returned as a closure rather than a Loss subclass so it stays trivially readable and
    needs no custom-object registration when weights are reloaded (we reload weights into
    a freshly built architecture, never a whole serialized model).
    """
    import tensorflow as tf
    bce = tf.keras.losses.binary_crossentropy
    T = float(temperature)
    a = float(alpha)

    def loss(y_true, y_pred):
        y_hard = y_true[:, 0:1]
        z_teacher = y_true[:, 1:2]
        z_student = y_pred

        # Hard term: ordinary BCE against the true logical flip, on logits.
        hard = bce(y_hard, z_student, from_logits=True)

        # Soft term: match the temperature-softened teacher distribution. Both sides are
        # divided by T; the T^2 factor restores the gradient scale (the 1/T inside the
        # softened sigmoid otherwise shrinks the gradient by 1/T^2).
        soft_target = tf.sigmoid(z_teacher / T)
        soft = bce(soft_target, z_student / T, from_logits=True) * (T * T)

        return a * hard + (1.0 - a) * soft

    return loss


def make_hard_accuracy():
    """Accuracy against the HARD label only, ignoring the packed teacher column.

    Without this, Keras's stock accuracy would compare the student's logit against a
    2-column target and report a meaningless number, which is worse than reporting none.
    """
    import tensorflow as tf

    def hard_accuracy(y_true, y_pred):
        y_hard = y_true[:, 0:1]
        pred = tf.cast(y_pred > 0.0, y_hard.dtype)  # logit > 0 == prob > 0.5
        return tf.reduce_mean(tf.cast(tf.equal(pred, y_hard), tf.float32))

    return hard_accuracy


def load_teacher_cache(path, pool_path, lo, hi):
    """Load cached teacher outputs and verify they actually cover [lo, hi) of THIS pool.

    Every failure this checks for is silent otherwise: a cache built from a different
    pool, or covering a shorter range than the requested training prefix, would still
    broadcast into a training run and produce a plausible-looking loss curve fit against
    the wrong shots. Fail loudly instead.
    """
    if not os.path.exists(path):
        raise SystemExit(f"[student] MISSING teacher cache {path} -- run dump_teacher_probs.py")
    tc = np.load(path, allow_pickle=False)

    cached_pool = str(tc['pool_path']) if 'pool_path' in tc.files else ''
    if cached_pool and os.path.abspath(cached_pool) != os.path.abspath(pool_path):
        raise SystemExit(
            f"[student] teacher cache was built from a DIFFERENT pool:\n"
            f"  cache pool: {cached_pool}\n  train pool: {os.path.abspath(pool_path)}")

    idx = tc['shot_idx']
    if idx[0] > lo or idx[-1] < hi - 1:
        raise SystemExit(
            f"[student] teacher cache covers shots [{int(idx[0])}, {int(idx[-1]) + 1}) but "
            f"training needs [{lo}, {hi}). Re-dump with a wider --n-shots.")
    # Map the requested absolute shot range onto positions within the cache.
    off = lo - int(idx[0])
    n = hi - lo
    return (tc['logit_teacher'][off:off + n].astype(np.float32),
            tc['p_teacher'][off:off + n].astype(np.float32),
            tc['flips'][off:off + n].astype(np.int8))


def run():
    ap = argparse.ArgumentParser()
    # --- problem config (must match the teacher's) ---
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--kernel', type=int, default=3)
    # --- student architecture ---
    ap.add_argument('--student', choices=['mlp', 'gru'], required=True)
    ap.add_argument('--inputs', choices=['evts', 'evts+bits'], default='evts',
                    help="'evts' = detector events only (what MWPM sees, smallest FPGA "
                         "input). 'evts+bits' also feeds the raw stabilizer measurements, "
                         "which the teacher sees.")
    ap.add_argument('--hidden', type=int, nargs='*', default=[128, 128],
                    help='MLP: the Dense stack. GRU: optional Dense layers after the GRU '
                         '(pass with no values for none).')
    ap.add_argument('--units', type=int, default=64, help='GRU hidden state size')
    ap.add_argument('--weight-bits', type=int, default=None,
                    help='Phase-3 QAT weight word length; None/>=32 => float weights '
                         '(the Phase-2 distillation setting)')
    ap.add_argument('--act-bits', type=int, default=None,
                    help='Phase-3 activation word length; None/>=32 => float activations')
    # --- distillation ---
    ap.add_argument('--teacher-cache', default=None,
                    help='npz from dump_teacher_probs.py over the TRAINING prefix. '
                         'Required unless --alpha 1.0.')
    ap.add_argument('--teacher-tail-cache', default=None,
                    help='npz from dump_teacher_probs.py over the EVALUATION TAIL. '
                         'Optional; enables the student-vs-teacher agreement column, '
                         'which is the distillation gap measurement.')
    ap.add_argument('--alpha', type=float, default=0.0,
                    help='weight on the HARD-label term. 1.0 = hard labels only (control '
                         'arm, needs no teacher cache); 0.0 = pure distillation.')
    ap.add_argument('--temperature', type=float, default=1.0)
    # --- training recipe (mirrors train_one.py) ---
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--n-train', type=int, required=True)
    ap.add_argument('--n-test', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--val-split', type=float, default=0.2)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--no-early-stopping', action='store_true')
    # --- io ---
    ap.add_argument('--data-dir', default=os.path.expanduser('~/rcnn_threshold/pools'))
    ap.add_argument('--pool', default=None, help='explicit training pool npz')
    ap.add_argument('--test-pool', default=None,
                    help='separate npz for a FRESH disjoint tail; strongly preferred')
    ap.add_argument('--out-dir', default=os.path.expanduser('~/rcnn_threshold/out_student'))
    ap.add_argument('--tag', default=None, help='override the auto-generated output tag')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    if args.alpha < 1.0 and not args.teacher_cache:
        raise SystemExit("[student] --teacher-cache is required unless --alpha 1.0 "
                         "(pure hard-label control arm).")

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    # Same determinism flags as train_one_quantized.py. Without them, GPU float reductions
    # accumulate in a nondeterministic order and the same seed can hit a val_loss spike at
    # large data volume that the fixed LR decay never recovers from (diagnosed 2026-07-19).
    # Must be set before TensorFlow is imported so deterministic kernels are selected.
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')
    import tensorflow as tf
    # Same hard version guard as the rest of the repo: TF is pinned at 2.15.x (Keras 2).
    # The student itself is Keras-3-safe, but the teacher cache and every sibling script
    # are not, and a mixed-version result would be silently incomparable.
    assert tf.__version__.startswith('2.15'), (
        f'need TF 2.15.x (Keras 2); got TF {tf.__version__}. Activate the pinned env.')
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')
    print(f"[student] TF {tf.__version__}  GPUs: {tf.config.list_physical_devices('GPU')}",
          flush=True)

    from types_cfg import get_types
    from circuit_partition import split_measurements
    from StudentModels import build_student, assemble_features, student_pred_and_correct

    d, p, r, k, seed = args.d, args.p, args.rounds, args.kernel, args.seed
    ntr, nte = args.n_train, args.n_test
    binary_t, _time_t, idx_t, _packed_t = get_types(d, r, k)

    train_fn = args.pool or os.path.join(args.data_dir, f'data_d{d}_p{p:.3f}_r{r}.npz')
    if not os.path.exists(train_fn):
        raise SystemExit(f"[student] MISSING pool {train_fn}")
    ztr = np.load(train_fn)
    N = ztr['measurements'].shape[0]

    # --- training features ---------------------------------------------------------
    if args.test_pool:
        # Fresh disjoint tail: the training prefix may use the whole training pool.
        if ntr > N:
            raise SystemExit(f"[student] n_train={ntr} exceeds pool size {N}")
    else:
        # Same-pool split: the tail is the LAST nte shots, so the prefix must stop short.
        assert ntr <= N - nte, f"train/test overlap: ntr={ntr} > N-nte={N - nte}"

    m_tr = ztr['measurements'][0:ntr].astype(binary_t)
    e_tr = ztr['det_evts'][0:ntr].astype(binary_t)
    f_tr = ztr['flips'][0:ntr].astype(binary_t).reshape(-1)
    b_tr, _, _ = split_measurements(m_tr, d, idx_t)
    x_tr = assemble_features(b_tr, e_tr, args.inputs)
    del m_tr, b_tr, e_tr

    # --- targets: pack [hard_label, teacher_logit] ----------------------------------
    if args.alpha >= 1.0 and not args.teacher_cache:
        # Hard-label control arm with no teacher available: the teacher column is unused
        # by the loss (weight 1-alpha == 0), so fill it with zeros rather than requiring
        # a cache the arm does not need.
        z_teach = np.zeros(ntr, dtype=np.float32)
        print("[student] hard-label control arm: teacher column unused (alpha=1).",
              flush=True)
    else:
        z_teach, p_teach, f_cache = load_teacher_cache(args.teacher_cache, train_fn, 0, ntr)
        # The cache carries its own copy of `flips`; if it disagrees with the pool's, the
        # two files are not describing the same shots and every soft target is misaligned.
        if not np.array_equal(f_cache.reshape(-1), f_tr.reshape(-1)):
            raise SystemExit("[student] teacher cache `flips` disagree with the pool's -- "
                             "the cache is misaligned with these shots. STOP.")
        print(f"[student] teacher cache ok: {ntr:,} shots, mean p_teacher={p_teach.mean():.5f}",
              flush=True)

    y_tr = np.stack([f_tr.astype(np.float32), z_teach], axis=1)

    # --- evaluation tail ------------------------------------------------------------
    zte = np.load(args.test_pool) if args.test_pool else ztr
    Nte = zte['measurements'].shape[0]
    te = slice(Nte - nte, Nte)
    m_te = zte['measurements'][te].astype(binary_t)
    e_te = zte['det_evts'][te].astype(binary_t)
    f_te = zte['flips'][te].astype(binary_t).reshape(-1)
    b_te, _, _ = split_measurements(m_te, d, idx_t)
    x_te = assemble_features(b_te, e_te, args.inputs)
    del m_te, b_te, e_te

    # --- build, compile, fit --------------------------------------------------------
    set_seeds(seed)
    model = build_student(args.student, d=d, rounds=r, inputs=args.inputs,
                          hidden=tuple(args.hidden), units=args.units,
                          weight_bits=args.weight_bits, act_bits=args.act_bits)
    model.compile(optimizer='adam',
                  loss=make_distillation_loss(args.alpha, args.temperature),
                  metrics=[make_hard_accuracy()])
    n_params = int(model.count_params())
    print(f"[student] {args.student} inputs={args.inputs} params={n_params:,}", flush=True)

    callbacks = [tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)]
    if not args.no_early_stopping:
        callbacks.insert(0, tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.patience, restore_best_weights=True))

    t0 = time.time()
    hist = model.fit(x=x_tr, y=y_tr, batch_size=args.batch_size, epochs=args.epochs,
                     validation_split=args.val_split, shuffle=True, verbose=2,
                     callbacks=callbacks)
    train_time = time.time() - t0
    epochs_ran = len(hist.history['loss'])
    best_val = float(min(hist.history['val_loss']))

    # --- score on the tail ----------------------------------------------------------
    logits = model.predict(x_te, batch_size=args.batch_size, verbose=0)
    pred, correct = student_pred_and_correct(logits, f_te)
    pL = float((~correct).mean())
    base_rate = float(f_te.mean())
    beats_base = pL < base_rate

    # MWPM baselines are NOT portable across tails. lookup_mwpm returns the baseline that
    # was computed on the --data-dir pool's own tail (locally: 0.0518 on a 10k tail), which
    # is a different number from the 200k-tail anchor. If the student was scored on a
    # SEPARATE --test-pool, that stored baseline describes different shots and any ratio
    # built from it is meaningless -- so refuse to emit one rather than write a plausible
    # wrong number into a results CSV. Decode MWPM on the actual tail (eval_on_tail.py
    # --mcnemar) and compare there. This exact confusion has already caused one real bug.
    if args.test_pool:
        mwpm = None
        print("[student] --test-pool given: suppressing the stored MWPM ratio (baselines "
              "are tail-specific). Decode MWPM on THIS tail to compare.", flush=True)
    else:
        mwpm = lookup_mwpm(args.data_dir, d, p, r)

    # --- distillation gap: how closely does the student track the teacher? ----------
    # Agreement is measured on DECISIONS over the tail, which is the quantity that
    # actually matters for deployment -- a student can differ in probability everywhere
    # and still decode identically. Only computed when a tail teacher cache is supplied.
    agree = ''
    teacher_tail_pL = ''
    if args.teacher_tail_cache:
        if not os.path.exists(args.teacher_tail_cache):
            raise SystemExit(f"[student] MISSING tail cache {args.teacher_tail_cache}")
        tc = np.load(args.teacher_tail_cache)
        if tc['p_teacher'].shape[0] == nte:
            t_pred = (tc['p_teacher'] > 0.5).astype(np.int8)
            agree = round(float((t_pred == pred).mean()), 5)
            teacher_tail_pL = round(float((t_pred != f_te).mean()), 6)
        else:
            print(f"[student] tail teacher cache has {tc['p_teacher'].shape[0]} shots, "
                  f"tail is {nte} -- skipping agreement.", flush=True)

    # --- write results --------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    wb = 'f32' if args.weight_bits is None else f'w{args.weight_bits}'
    ab = 'f32' if args.act_bits is None else f'a{args.act_bits}'
    tag = args.tag or (f'student_{args.student}_{args.inputs.replace("+", "-")}_'
                       f'{wb}_{ab}_alpha{args.alpha:g}_T{args.temperature:g}_'
                       f'seed{seed}_ntr{ntr}')

    fields = ['architecture', 'student', 'inputs', 'd', 'p', 'rounds', 'seed', 'n_train',
              'n_test', 'alpha', 'temperature', 'weight_bits', 'act_bits', 'n_params',
              'epochs', 'epochs_ran', 'batch_size', 'p_L', 'mwpm_p_L', 'ratio_vs_mwpm',
              'base_rate', 'beats_base_rate', 'teacher_tail_p_L', 'agree_with_teacher',
              'best_val_loss', 'train_time_s']
    with open(os.path.join(args.out_dir, tag + '.csv'), 'w', newline='') as cf:
        w = csv.DictWriter(cf, fieldnames=fields)
        w.writeheader()
        w.writerow(dict(
            architecture=model.name, student=args.student, inputs=args.inputs,
            d=d, p=p, rounds=r, seed=seed, n_train=ntr, n_test=nte,
            alpha=args.alpha, temperature=args.temperature,
            weight_bits=('' if args.weight_bits is None else args.weight_bits),
            act_bits=('' if args.act_bits is None else args.act_bits),
            n_params=n_params, epochs=args.epochs, epochs_ran=epochs_ran,
            batch_size=args.batch_size, p_L=round(pL, 6),
            mwpm_p_L=('' if mwpm is None else round(mwpm, 6)),
            ratio_vs_mwpm=('' if mwpm is None else round(pL / mwpm, 4)),
            base_rate=round(base_rate, 5), beats_base_rate=int(beats_base),
            teacher_tail_p_L=teacher_tail_pL, agree_with_teacher=agree,
            best_val_loss=round(best_val, 5), train_time_s=round(train_time, 1)))
    with open(os.path.join(args.out_dir, tag + '.history.json'), 'w') as hf:
        json.dump({k2: [float(x) for x in v] for k2, v in hist.history.items()}, hf)
    wpath = os.path.join(args.out_dir, tag + '.weights.h5')
    model.save_weights(wpath)

    gap = '' if mwpm is None else (f'  MWPM(same-pool tail)={mwpm:.5f}  '
                                   f'ratio={pL / mwpm:.3f}x')
    flag = '' if beats_base else '  <-- FAIL: p_L >= base rate (class collapse)'
    print(f"[student] {tag}  p_L={pL:.5f}{gap}  base_rate={base_rate:.3f}"
          f"  agree_with_teacher={agree}{flag}  ({epochs_ran} ep, {train_time:.0f}s)",
          flush=True)
    print(f"[student] wrote -> {os.path.join(args.out_dir, tag + '.csv')}  (+ history, weights)",
          flush=True)


if __name__ == '__main__':
    run()
