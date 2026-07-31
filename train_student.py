#!/usr/bin/env python3
# Created: 2026-07-29
# Last modified: 2026-07-31
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

# Prefix on every log line; set to '[mlp]' or '[gru]' once args are parsed, so a saved log
# names the architecture that produced it. Module-level so helper functions can use it
# without an args parameter.
LOG = '[student]'


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


class TailDiagnostics:
    """Per-epoch tail diagnostics, built as a Keras callback at call time.

    Exists because the failure mode this de-risk is hunting -- a too-hot learning rate --
    shows up in the SHAPE of the curves, and specifically in signals that appear before
    p_L moves at all. A student sliding toward predicting one class everywhere first shows
    a collapsing logit spread; p_L only follows once the collapse is nearly complete. So
    the logit std is the early tell and is logged alongside the outcome metrics.

    Recorded each epoch:
      tail_p_L         error rate on the evaluation tail
      pred_pos_rate    fraction of tail shots predicted positive (logit > 0)
      base_rate        fraction actually positive -- logged every epoch, not once, so the
                       comparison never has to be reconstructed from another file
      collapsed        1 if |pred_pos_rate - base_rate| < eps AND the logit spread is
                       degenerate; the two together are what class collapse means
      logit_mean/std   distribution of the raw student output
    """

    def __init__(self, x_tail, y_tail, batch_size, eps=0.01, logit_std_floor=0.05):
        self.x, self.y = x_tail, np.asarray(y_tail).reshape(-1)
        self.batch_size = batch_size
        self.eps = eps
        self.logit_std_floor = logit_std_floor
        self.base_rate = float(self.y.mean())

    def build(self):
        import tensorflow as tf
        outer = self

        class _CB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs if logs is not None else {}
                logits = self.model.predict(outer.x, batch_size=outer.batch_size,
                                            verbose=0).reshape(-1)
                pred = (logits > 0.0).astype(np.int8)
                p_L = float((pred != outer.y).mean())
                ppr = float(pred.mean())
                lmean, lstd = float(logits.mean()), float(logits.std())
                collapsed = int(abs(ppr - outer.base_rate) < outer.eps
                                and lstd < outer.logit_std_floor)
                logs['tail_p_L'] = p_L
                logs['pred_pos_rate'] = ppr
                logs['base_rate'] = outer.base_rate
                logs['collapsed'] = collapsed
                logs['logit_mean'] = lmean
                logs['logit_std'] = lstd
                print(f"    [tail] epoch {epoch + 1:3d}  p_L={p_L:.5f}  "
                      f"pred_pos={ppr:.4f} (base={outer.base_rate:.4f})  "
                      f"logit mean={lmean:+.3f} std={lstd:.3f}"
                      f"{'  <-- COLLAPSED' if collapsed else ''}", flush=True)

        return _CB()


def teacher_agreement(student_pred, teacher_prob, truth, band):
    """Student-vs-teacher decision agreement, overall AND restricted to ambiguous shots.

    WHY THE RAW NUMBER IS NOT ENOUGH
    This teacher is highly confident: ~80% of shots land at p<0.05 or p>0.95. On those
    shots almost any competent student agrees, so a raw agreement rate is dominated by the
    easy majority and will read high (and flatteringly stable) no matter how badly the
    student reproduces the teacher where it actually matters. It cannot distinguish a
    student that has genuinely absorbed the teacher's decision function from one that has
    only learned the easy bulk.

    The informative quantity is agreement on the AMBIGUOUS band -- the shots where the
    teacher itself is unsure (band[0] < p_teacher < band[1]). Those are the syndromes
    whose decoding is genuinely contested, they are where the teacher's advantage over a
    hard label lives, and they are where a student that merely learned the bulk will
    visibly diverge. Distillation is supposed to transfer exactly this.

    Returns a dict with the raw rate, the band-restricted rate, the confident-subset rate
    for contrast, the band population, and both models' error rates inside the band (a
    student can agree with the teacher there and both still be wrong).
    """
    p_t = np.asarray(teacher_prob).reshape(-1)
    s_pred = np.asarray(student_pred).reshape(-1).astype(np.int8)
    t_pred = (p_t > 0.5).astype(np.int8)
    truth = np.asarray(truth).reshape(-1).astype(np.int8)

    lo, hi = band
    ambiguous = (p_t > lo) & (p_t < hi)
    confident = ~ambiguous
    n_amb = int(ambiguous.sum())

    out = {
        'agree_all': round(float((s_pred == t_pred).mean()), 5),
        'n_ambiguous': n_amb,
        'frac_ambiguous': round(float(ambiguous.mean()), 5),
        'teacher_tail_p_L': round(float((t_pred != truth).mean()), 6),
    }
    out['agree_confident'] = (round(float((s_pred[confident] == t_pred[confident]).mean()), 5)
                              if confident.any() else '')
    if n_amb:
        out['agree_ambiguous'] = round(float((s_pred[ambiguous] == t_pred[ambiguous]).mean()), 5)
        out['student_p_L_ambiguous'] = round(float((s_pred[ambiguous] != truth[ambiguous]).mean()), 5)
        out['teacher_p_L_ambiguous'] = round(float((t_pred[ambiguous] != truth[ambiguous]).mean()), 5)
    else:
        out['agree_ambiguous'] = ''
        out['student_p_L_ambiguous'] = ''
        out['teacher_p_L_ambiguous'] = ''
    return out


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


def verify_cache_fingerprints(tc, pool_npz):
    """Check the cache's stored fingerprints against the pool and teacher on disk NOW.

    Complements the runtime flips-agreement check rather than replacing it. That check can
    only compare the shots the cache carries; these fingerprints catch the cases it cannot
    see -- a pool regenerated under the same filename, or a teacher checkpoint retrained
    in place. Both would otherwise train a student against soft targets that no longer
    correspond to anything.

    A missing fingerprint is a warning, not an error, so caches written before this check
    existed stay usable.
    """
    from dump_teacher_probs import sha256_file, pool_fingerprint

    if 'pool_flips_sha256' not in tc.files:
        print(f"{LOG} WARNING: cache predates fingerprinting -- identity not verified. "
              "Re-run dump_teacher_probs.py to get the check.", flush=True)
        return

    flips_sha, meas_shape = pool_fingerprint(pool_npz)
    if flips_sha != str(tc['pool_flips_sha256']):
        raise SystemExit(
            f"{LOG} POOL FINGERPRINT MISMATCH -- the pool on disk is not the one the "
            "teacher cache was built from (same path, different content: regenerated "
            "pool?). Re-dump the teacher cache. STOP.")
    if meas_shape != str(tc['pool_measurements_shape']):
        raise SystemExit(
            f"{LOG} pool measurements shape {meas_shape} != cache's "
            f"{str(tc['pool_measurements_shape'])}. STOP.")

    # The teacher checkpoint is only checkable if it is still where the cache recorded it.
    # It usually is locally, but a cache pulled from EAF will point at a path that does not
    # exist here -- that is expected and not an error.
    wpath = str(tc['weights_path']) if 'weights_path' in tc.files else ''
    if 'weights_sha256' in tc.files and wpath and os.path.exists(wpath):
        if sha256_file(wpath) != str(tc['weights_sha256']):
            raise SystemExit(
                f"{LOG} TEACHER CHECKPOINT MISMATCH -- the .weights.h5 at the cached "
                "path has changed since the cache was built (retrained in place?). The "
                "soft targets no longer come from these weights. Re-dump. STOP.")
        print(f"{LOG} fingerprints verified: pool and teacher checkpoint both match "
              "the cache.", flush=True)
    else:
        print(f"{LOG} pool fingerprint verified; teacher checkpoint not present "
              "locally, so its hash was not re-checked.", flush=True)


def load_teacher_cache(path, pool_path, lo, hi, pool_npz=None):
    """Load cached teacher outputs and verify they actually cover [lo, hi) of THIS pool.

    Every failure this checks for is silent otherwise: a cache built from a different
    pool, or covering a shorter range than the requested training prefix, would still
    broadcast into a training run and produce a plausible-looking loss curve fit against
    the wrong shots. Fail loudly instead.
    """
    if not os.path.exists(path):
        raise SystemExit(f"{LOG} MISSING teacher cache {path} -- run dump_teacher_probs.py")
    tc = np.load(path, allow_pickle=False)

    cached_pool = str(tc['pool_path']) if 'pool_path' in tc.files else ''
    if cached_pool and os.path.abspath(cached_pool) != os.path.abspath(pool_path):
        raise SystemExit(
            f"{LOG} teacher cache was built from a DIFFERENT pool:\n"
            f"  cache pool: {cached_pool}\n  train pool: {os.path.abspath(pool_path)}")

    if pool_npz is not None:
        verify_cache_fingerprints(tc, pool_npz)

    idx = tc['shot_idx']
    if idx[0] > lo or idx[-1] < hi - 1:
        raise SystemExit(
            f"{LOG} teacher cache covers shots [{int(idx[0])}, {int(idx[-1]) + 1}) but "
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
                         'Optional; enables the student-vs-teacher agreement columns, '
                         'which are the distillation gap measurement.')
    ap.add_argument('--ambiguous-band', type=float, nargs=2, default=[0.05, 0.95],
                    metavar=('LO', 'HI'),
                    help='teacher-probability band counted as AMBIGUOUS. Agreement is '
                         'reported both raw and restricted to this band. The raw rate is '
                         'dominated by the ~80%% of shots where this teacher is confident '
                         'and reads misleadingly high; the band-restricted rate is the '
                         'one that measures whether the teacher\'s decision function '
                         'actually transferred. Default 0.05-0.95.')
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
    ap.add_argument('--lr', type=float, default=None,
                    help='CONSTANT learning rate, replacing the inherited train_one.py '
                         'schedule. That schedule opens at 1e-2 because it has to kick the '
                         'teacher\'s zero-init state correlator off its slow start; a plain '
                         'MLP/GRU has no such layer, so the opening LR may be too hot and '
                         'could destabilise the alpha=1 and alpha=0 arms asymmetrically. '
                         'Use this to pin an independent student LR before any sweep. '
                         'None => the teacher schedule (not recommended, unverified here).')
    ap.add_argument('--tail-diagnostics', action='store_true',
                    help='evaluate the tail every epoch and log p_L, predicted-positive '
                         'rate vs base rate, a class-collapse flag, and the student logit '
                         'mean/std into the history. Costs one tail inference per epoch.')
    # --- io ---
    ap.add_argument('--data-dir', default=os.path.expanduser('~/rcnn_threshold/pools'))
    ap.add_argument('--pool', default=None, help='explicit training pool npz')
    ap.add_argument('--test-pool', default=None,
                    help='separate npz for a FRESH disjoint tail; strongly preferred')
    ap.add_argument('--out-dir', default=None,
                    help='default: ~/rcnn_threshold/out_{student}, so MLP and GRU results '
                         'never land in the same directory')
    ap.add_argument('--tag', default=None, help='override the auto-generated output tag')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    # Name the architecture in every log line from here on: '[mlp]' or '[gru]'.
    global LOG
    LOG = f'[{args.student}]'
    if args.out_dir is None:
        args.out_dir = os.path.expanduser(f'~/rcnn_threshold/out_{args.student}')

    if args.alpha < 1.0 and not args.teacher_cache:
        raise SystemExit(f"{LOG} --teacher-cache is required unless --alpha 1.0 "
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
    print(f"{LOG} TF {tf.__version__}  GPUs: {tf.config.list_physical_devices('GPU')}",
          flush=True)

    from types_cfg import get_types
    from circuit_partition import split_measurements
    from StudentModels import build_student, assemble_features, student_pred_and_correct

    d, p, r, k, seed = args.d, args.p, args.rounds, args.kernel, args.seed
    ntr, nte = args.n_train, args.n_test
    binary_t, _time_t, idx_t, _packed_t = get_types(d, r, k)

    train_fn = args.pool or os.path.join(args.data_dir, f'data_d{d}_p{p:.3f}_r{r}.npz')
    if not os.path.exists(train_fn):
        raise SystemExit(f"{LOG} MISSING pool {train_fn}")
    ztr = np.load(train_fn)
    N = ztr['measurements'].shape[0]

    # --- training features ---------------------------------------------------------
    if args.test_pool:
        # Fresh disjoint tail: the training prefix may use the whole training pool.
        if ntr > N:
            raise SystemExit(f"{LOG} n_train={ntr} exceeds pool size {N}")
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
        print(f"{LOG} hard-label control arm: teacher column unused (alpha=1).",
              flush=True)
    else:
        z_teach, p_teach, f_cache = load_teacher_cache(args.teacher_cache, train_fn, 0, ntr,
                                                       pool_npz=ztr)
        # The cache carries its own copy of `flips`; if it disagrees with the pool's, the
        # two files are not describing the same shots and every soft target is misaligned.
        if not np.array_equal(f_cache.reshape(-1), f_tr.reshape(-1)):
            raise SystemExit(f"{LOG} teacher cache `flips` disagree with the pool's -- "
                             "the cache is misaligned with these shots. STOP.")
        print(f"{LOG} teacher cache ok: {ntr:,} shots, mean p_teacher={p_teach.mean():.5f}",
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
    optimizer = (tf.keras.optimizers.Adam(learning_rate=args.lr) if args.lr is not None
                 else 'adam')
    model.compile(optimizer=optimizer,
                  loss=make_distillation_loss(args.alpha, args.temperature),
                  metrics=[make_hard_accuracy()])
    n_params = int(model.count_params())
    print(f"{LOG} {args.student} inputs={args.inputs} params={n_params:,}", flush=True)

    # With --lr the LR is constant and the teacher's scheduler is deliberately NOT
    # attached -- attaching it would silently overwrite the constant on every epoch and
    # make the LR comparison measure the schedule instead of the LR.
    callbacks = []
    if args.lr is None:
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler))
    if args.tail_diagnostics:
        callbacks.append(TailDiagnostics(x_te, f_te, args.batch_size).build())
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
        print(f"{LOG} --test-pool given: suppressing the stored MWPM ratio (baselines "
              "are tail-specific). Decode MWPM on THIS tail to compare.", flush=True)
    else:
        mwpm = lookup_mwpm(args.data_dir, d, p, r)

    # --- distillation gap: how closely does the student track the teacher? ----------
    # Agreement is measured on DECISIONS over the tail, which is the quantity that
    # actually matters for deployment -- a student can differ in probability everywhere
    # and still decode identically. Only computed when a tail teacher cache is supplied.
    ag = {k: '' for k in ('agree_all', 'agree_ambiguous', 'agree_confident', 'n_ambiguous',
                          'frac_ambiguous', 'student_p_L_ambiguous', 'teacher_p_L_ambiguous',
                          'teacher_tail_p_L')}
    if args.teacher_tail_cache:
        if not os.path.exists(args.teacher_tail_cache):
            raise SystemExit(f"{LOG} MISSING tail cache {args.teacher_tail_cache}")
        tc = np.load(args.teacher_tail_cache)
        if tc['p_teacher'].shape[0] == nte:
            ag = teacher_agreement(pred, tc['p_teacher'], f_te, tuple(args.ambiguous_band))
            print(f"{LOG} agreement vs teacher: all={ag['agree_all']}  "
                  f"ambiguous={ag['agree_ambiguous']} (band {args.ambiguous_band[0]}-"
                  f"{args.ambiguous_band[1]}, n={ag['n_ambiguous']:,} = "
                  f"{ag['frac_ambiguous']:.1%})  confident={ag['agree_confident']}", flush=True)
            print(f"{LOG}   inside the band: student p_L={ag['student_p_L_ambiguous']}  "
                  f"teacher p_L={ag['teacher_p_L_ambiguous']}", flush=True)
        else:
            print(f"{LOG} tail teacher cache has {tc['p_teacher'].shape[0]} shots, "
                  f"tail is {nte} -- skipping agreement.", flush=True)

    # --- write results --------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    wb = 'f32' if args.weight_bits is None else f'w{args.weight_bits}'
    ab = 'f32' if args.act_bits is None else f'a{args.act_bits}'
    tag = args.tag or (f'student_{args.student}_{args.inputs.replace("+", "-")}_'
                       f'{wb}_{ab}_alpha{args.alpha:g}_T{args.temperature:g}_'
                       f'seed{seed}_ntr{ntr}')

    fields = ['architecture', 'student', 'inputs', 'd', 'p', 'rounds', 'seed', 'n_train',
              'n_test', 'alpha', 'temperature', 'lr', 'weight_bits', 'act_bits', 'n_params',
              'epochs', 'epochs_ran', 'batch_size', 'p_L', 'mwpm_p_L', 'ratio_vs_mwpm',
              'base_rate', 'beats_base_rate', 'teacher_tail_p_L',
              # agreement: the raw rate is dominated by the ~80% of shots where this
              # teacher is confident, so agree_ambiguous is the one that carries signal
              'agree_all', 'agree_ambiguous', 'agree_confident', 'n_ambiguous',
              'frac_ambiguous', 'student_p_L_ambiguous', 'teacher_p_L_ambiguous',
              'ambiguous_band', 'best_val_loss', 'train_time_s']
    with open(os.path.join(args.out_dir, tag + '.csv'), 'w', newline='') as cf:
        w = csv.DictWriter(cf, fieldnames=fields)
        w.writeheader()
        w.writerow(dict(
            architecture=model.name, student=args.student, inputs=args.inputs,
            d=d, p=p, rounds=r, seed=seed, n_train=ntr, n_test=nte,
            alpha=args.alpha, temperature=args.temperature,
            lr=('teacher_schedule' if args.lr is None else args.lr),
            weight_bits=('' if args.weight_bits is None else args.weight_bits),
            act_bits=('' if args.act_bits is None else args.act_bits),
            n_params=n_params, epochs=args.epochs, epochs_ran=epochs_ran,
            batch_size=args.batch_size, p_L=round(pL, 6),
            mwpm_p_L=('' if mwpm is None else round(mwpm, 6)),
            ratio_vs_mwpm=('' if mwpm is None else round(pL / mwpm, 4)),
            base_rate=round(base_rate, 5), beats_base_rate=int(beats_base),
            ambiguous_band=f'{args.ambiguous_band[0]}-{args.ambiguous_band[1]}',
            best_val_loss=round(best_val, 5), train_time_s=round(train_time, 1),
            **ag))
    with open(os.path.join(args.out_dir, tag + '.history.json'), 'w') as hf:
        json.dump({k2: [float(x) for x in v] for k2, v in hist.history.items()}, hf)
    wpath = os.path.join(args.out_dir, tag + '.weights.h5')
    model.save_weights(wpath)

    gap = '' if mwpm is None else (f'  MWPM(same-pool tail)={mwpm:.5f}  '
                                   f'ratio={pL / mwpm:.3f}x')
    flag = '' if beats_base else '  <-- FAIL: p_L >= base rate (class collapse)'
    print(f"{LOG} {tag}  p_L={pL:.5f}{gap}  base_rate={base_rate:.3f}"
          f"  agree_ambiguous={ag['agree_ambiguous']}{flag}"
          f"  ({epochs_ran} ep, {train_time:.0f}s)",
          flush=True)
    print(f"{LOG} wrote -> {os.path.join(args.out_dir, tag + '.csv')}  (+ history, weights)",
          flush=True)


if __name__ == '__main__':
    run()
