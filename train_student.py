#!/usr/bin/env python3
# Created: 2026-07-29
# Last modified: 2026-07-31
"""Train one distillation student (MLP or GRU) against the cached FP32 teacher outputs.

A single point is (student, inputs, seed, n_train, alpha, temperature, weight_bits,
act_bits). This is the Phase-2 workhorse: it trains an hls4ml-native student to imitate
the custom RCNN teacher, so the FPGA flow never has to parse the teacher's unsupported
combiner ops (pow/sqrt/log/gather).

Hard-vs-soft is one flag:
  --alpha 1.0  pure hard labels, the control arm -- what the architecture does alone
  --alpha 0.0  pure distillation, learning only from the teacher's soft output
  0 < a < 1    a blend
One code path for all three, so the comparison is not confounded by two trainers.

The hard label is one bit. The teacher's output is a calibrated probability that also
says how ambiguous the syndrome was; a teacher near 0.5 is telling the student the shot
is a coin flip, which the hard label cannot express. That is what --alpha measures.

Temperature acts in logit space: the soft target is sigmoid(z_teacher / T) and the
student's logit is divided by T. The soft term carries a T^2 factor so its gradient stays
comparable to the hard term as T varies, otherwise a temperature scan doubles as a
loss-weight scan. T=1 is the untempered teacher.

Disjointness follows train_one_quantized.py: the tail comes from --test-pool when given,
otherwise from the same pool behind the `ntr <= N - nte` assert. The teacher cache is
also checked to cover the training range and to come from the same pool.

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


def sha16(path):
    """First 16 hex chars of a file's SHA-256, or '' if it is not there."""
    import hashlib
    if not path or not os.path.exists(path):
        return ''
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()[:16]


def run_provenance(args, train_pool, teacher_cache_npz):
    """Identity of every input that determined this run, written into the result CSV so a
    row stays traceable without relying on directory names: training pool and its flips
    fingerprint, tail pool, teacher cache and checkpoint hash, this script's hash, UTC.
    """
    import datetime
    prov = {
        'run_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'train_pool': os.path.basename(train_pool),
        'train_pool_dir': os.path.basename(os.path.dirname(os.path.abspath(train_pool))),
        'test_pool': os.path.basename(args.test_pool) if args.test_pool else 'same-pool-tail',
        'teacher_cache': os.path.basename(args.teacher_cache) if args.teacher_cache else '',
        'teacher_weights_sha': '',
        'train_pool_flips_sha': '',
        'code_sha': sha16(os.path.abspath(__file__)),
    }
    if teacher_cache_npz is not None:
        for key, col in (('weights_sha256', 'teacher_weights_sha'),
                         ('pool_flips_sha256', 'train_pool_flips_sha')):
            if key in teacher_cache_npz.files:
                prov[col] = str(teacher_cache_npz[key])[:16]
    return prov


PROVENANCE_COLS = ['run_utc', 'train_pool', 'train_pool_dir', 'test_pool', 'teacher_cache',
                   'teacher_weights_sha', 'train_pool_flips_sha', 'code_sha']


def make_distillation_loss(alpha, temperature):
    """Blended hard-label and teacher-imitation loss.

    y_true is packed as two columns, [hard_label, teacher_logit], since Keras passes one
    target tensor. y_pred is the student's raw logit. A closure rather than a Loss
    subclass, so reloading weights needs no custom-object registration.
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
    """Per-epoch tail metrics, built as a Keras callback at call time.

    A student drifting toward one class shows a collapsing logit spread before p_L moves,
    so logit_std is the early warning for a too-hot learning rate.

    Per epoch:
      tail_p_L         error rate on the evaluation tail
      pred_pos_rate    fraction predicted positive (logit > 0)
      base_rate        fraction actually positive, logged every epoch so the comparison
                       needs no second file
      collapsed        1 when pred_pos_rate sits within eps of base_rate and the logit
                       spread is degenerate
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
                      f"{'  <-- collapsed' if collapsed else ''}", flush=True)

        return _CB()


def teacher_agreement(student_pred, teacher_prob, truth, band):
    """Student-vs-teacher decision agreement, overall and restricted to ambiguous shots.

    The teacher is confident on ~80% of shots (p<0.05 or p>0.95), where almost any student
    agrees, so the raw rate is dominated by the easy bulk and reads high regardless: a
    student that coin-flips inside the band still scores ~0.90 raw.

    The informative figure is agreement where the teacher is unsure
    (band[0] < p_teacher < band[1]) -- the contested syndromes distillation should transfer.

    Returns the raw rate, the band-restricted rate, the confident-subset rate, the band
    population, and both models' error rates inside the band (agreeing there does not mean
    either is right).
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
    """Accuracy against the hard label only, ignoring the packed teacher column.

    Keras's stock accuracy would compare the logit against a 2-column target and report a
    meaningless number.
    """
    import tensorflow as tf

    def hard_accuracy(y_true, y_pred):
        y_hard = y_true[:, 0:1]
        pred = tf.cast(y_pred > 0.0, y_hard.dtype)  # logit > 0 == prob > 0.5
        return tf.reduce_mean(tf.cast(tf.equal(pred, y_hard), tf.float32))

    return hard_accuracy


def verify_cache_fingerprints(tc, pool_npz):
    """Check the cache's stored fingerprints against the pool and teacher on disk.

    Complements the runtime flips-agreement check, which only sees the shots the cache
    carries. These catch a pool regenerated under the same filename or a checkpoint
    retrained in place. A missing fingerprint warns rather than fails, so older caches
    stay usable.
    """
    from dump_teacher_probs import sha256_file, pool_fingerprint

    if 'pool_flips_sha256' not in tc.files:
        print(f"{LOG} warning: cache predates fingerprinting -- identity not verified. "
              "Re-run dump_teacher_probs.py to get the check.", flush=True)
        return

    flips_sha, meas_shape = pool_fingerprint(pool_npz)
    if flips_sha != str(tc['pool_flips_sha256']):
        raise SystemExit(
            f"{LOG} pool fingerprint mismatch -- the pool on disk is not the one the "
            "teacher cache was built from (same path, different content: regenerated "
            "pool?). Re-dump the teacher cache.")
    if meas_shape != str(tc['pool_measurements_shape']):
        raise SystemExit(
            f"{LOG} pool measurements shape {meas_shape} != cache's "
            f"{str(tc['pool_measurements_shape'])}.")

    # Only checkable if the checkpoint is still at the recorded path; a cache pulled from
    # EAF will point somewhere that does not exist locally, which is expected.
    wpath = str(tc['weights_path']) if 'weights_path' in tc.files else ''
    if 'weights_sha256' in tc.files and wpath and os.path.exists(wpath):
        if sha256_file(wpath) != str(tc['weights_sha256']):
            raise SystemExit(
                f"{LOG} teacher checkpoint mismatch -- the .weights.h5 at the cached "
                "path has changed since the cache was built (retrained in place?). The "
                "soft targets no longer come from these weights. Re-dump.")
        print(f"{LOG} fingerprints verified: pool and teacher checkpoint both match "
              "the cache.", flush=True)
    else:
        print(f"{LOG} pool fingerprint verified; teacher checkpoint not present "
              "locally, so its hash was not re-checked.", flush=True)


def load_teacher_cache(path, pool_path, lo, hi, pool_npz=None):
    """Load cached teacher outputs and verify they cover [lo, hi) of this pool.

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




def _primary_gen_seed(pool_path):
    """Generation seed of a pool, from its fingerprint, or None when absent."""
    fp = pool_path.replace('.npz', '.fingerprint.json')
    if not os.path.exists(fp):
        return None
    try:
        return json.load(open(fp)).get('gen_seed')
    except (ValueError, OSError):
        return None


def cap_gpu_memory(tf, mib):
    """Cap this process's GPU memory so several trainings share one card predictably.

    TensorFlow's allocator grows into whatever is free and does not release it, so an
    unconstrained process on an idle card parks far more than it needs and a later process
    fails its first allocation. A logical-device limit fixes each process's share.
    Must run before any tensor is placed on the device.
    """
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.set_logical_device_configuration(
            g, [tf.config.LogicalDeviceConfiguration(memory_limit=mib)])
    print(f"[gpu] per-process limit {mib} MiB on {len(gpus)} device(s)", flush=True)


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
                    help='teacher-probability band counted as ambiguous. Agreement is '
                         'reported both raw and restricted to this band. The raw rate is '
                         'dominated by the ~80%% of shots where this teacher is confident '
                         'and reads misleadingly high; the band-restricted rate is the '
                         'one that measures whether the teacher\'s decision function '
                         'actually transferred. Default 0.05-0.95.')
    ap.add_argument('--alpha', type=float, default=0.0,
                    help='weight on the hard-label term. 1.0 = hard labels only (control '
                         'arm, needs no teacher cache); 0.0 = pure distillation.')
    ap.add_argument('--temperature', type=float, default=1.0)
    # --- training recipe (mirrors train_one.py) ---
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--n-train', type=int, required=True)
    ap.add_argument('--n-test', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--val-split', type=float, default=0.2,
                    help='fallback only, used when --val-start/--val-n are not given. '
                         'Carves this fraction OUT of the training prefix, so n_train '
                         'overstates the shots that actually produce gradients.')
    # Explicit validation block, mirroring train_one.py's --val-start/--val-n. Without
    # these Keras carves validation_split out of the training prefix, so a student at
    # n_train=N trains on 0.8N while the teacher at the same n_train trains on N -- the
    # two ladders are then not comparable on the x-axis, which is the whole point of
    # plotting them together.
    ap.add_argument('--val-start', type=int, default=None,
                    help='first shot of an explicit validation block. Set with --val-n to '
                         'pass validation_data and disable validation_split, so the full '
                         'training prefix produces gradients.')
    ap.add_argument('--val-n', type=int, default=None)
    ap.add_argument('--val-teacher-cache', default=None,
                    help='teacher cache covering the validation block. Required with '
                         '--val-start when alpha < 1: the soft term of the loss needs '
                         'teacher logits on the validation shots too, and the training '
                         'cache does not cover them (it stops at n_train). Usually the '
                         'same npz passed to --teacher-tail-cache, since validation and '
                         'the scored partition are normally the same shots.')
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--no-early-stopping', action='store_true')
    # --- best-validation checkpointing (added 2026-08-06 for the parameter-reduction
    # study, which selects a checkpoint on one slice and reports on a disjoint one).
    # Both default to None, so every earlier student run reproduces unchanged: with no
    # --ckpt-dir no callback is added and nothing about the run differs.
    ap.add_argument('--ckpt-dir', default=None,
                    help='write the best-val_loss checkpoint and a true final-epoch '
                         'checkpoint here. Without it, only the end-of-fit weights are '
                         'saved, which are the restored best only when early stopping '
                         'fired -- not comparable across runs.')
    ap.add_argument('--run-tag', default=None,
                    help='filename prefix for those checkpoints (defaults to --tag)')
    ap.add_argument('--require-determinism', action='store_true',
                    help='hard-fail unless TF_DETERMINISTIC_OPS=1 and '
                         'TF_CUDNN_DETERMINISTIC=1 were exported by the launcher, before '
                         'this process started.')
    ap.add_argument('--lr', type=float, default=None,
                    help='constant learning rate, replacing the inherited train_one.py '
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
    ap.add_argument('--extra-train-pool', default=None,
                    help='second pool, TRAINING ONLY, appended to the training prefix. Its '
                         'shots come from an independent generation seed, so they cannot '
                         'overlap the validation, evaluation or sealed blocks of --pool. '
                         'Use to train past a pool\'s clean prefix without touching the '
                         'reserved regions.')
    ap.add_argument('--extra-train-n', type=int, default=None,
                    help='how many shots to take from --extra-train-pool (default: all)')
    ap.add_argument('--test-pool', default=None,
                    help='separate npz for a FRESH disjoint tail; strongly preferred')
    ap.add_argument('--eval-start', type=int, default=None,
                    help='first shot to score. Default None = the last --n-test shots. On '
                         'a pool whose final block is the sealed test partition, set this '
                         'to the validation start so the test set is never read.')
    ap.add_argument('--out-dir', default=None,
                    help='default: ~/rcnn_threshold/out_{student}, so MLP and GRU results '
                         'never land in the same directory')
    ap.add_argument('--tag', default=None, help='override the auto-generated output tag')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--gpu-mem-mib', type=int, default=None,
                    help='cap this process to N MiB of GPU memory, so several trainings '
                         'share one card without the first one parking all of it')
    args = ap.parse_args()

    # Name the architecture in every log line from here on: '[mlp]' or '[gru]'.
    global LOG
    LOG = f'[{args.student}]'
    if args.out_dir is None:
        args.out_dir = os.path.expanduser(f'~/rcnn_threshold/out_{args.student}')

    if args.alpha < 1.0 and not args.teacher_cache:
        raise SystemExit(f"{LOG} --teacher-cache is required unless --alpha 1.0 "
                         "(pure hard-label control arm).")

    if args.require_determinism:
        # The parameter-reduction study requires the LAUNCHER to have exported both
        # variables. The setdefault calls below would otherwise mask a launcher that
        # forgot, which is fine for this script alone but hides a broken launcher from
        # every other entry point in the same sweep.
        from slice_guard_r5 import assert_deterministic_env
        assert_deterministic_env()

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
    elif args.gpu_mem_mib:
        cap_gpu_memory(tf, args.gpu_mem_mib)
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
        # Same-pool split: the tail is the last nte shots, so the prefix must stop short.
        assert ntr <= N - nte, f"train/test overlap: ntr={ntr} > N-nte={N - nte}"

    m_tr = ztr['measurements'][0:ntr].astype(binary_t)
    e_tr = ztr['det_evts'][0:ntr].astype(binary_t)
    f_tr = ztr['flips'][0:ntr].astype(binary_t).reshape(-1)
    b_tr, _, _ = split_measurements(m_tr, d, idx_t)
    x_tr = assemble_features(b_tr, e_tr, args.inputs, student=args.student,
                             d=d, rounds=r, p=p)
    del m_tr, b_tr, e_tr

    # --- targets: pack [hard_label, teacher_logit] ----------------------------------
    if args.alpha >= 1.0 and not args.teacher_cache:
        # Hard-label control arm with no teacher available: the teacher column is unused
        # by the loss (weight 1-alpha == 0), so fill it with zeros rather than requiring
        # a cache the arm does not need.
        z_teach = np.zeros(ntr, dtype=np.float32)
        teacher_npz = None
        print(f"{LOG} hard-label control arm: teacher column unused (alpha=1).",
              flush=True)
    else:
        z_teach, p_teach, f_cache = load_teacher_cache(args.teacher_cache, train_fn, 0, ntr,
                                                       pool_npz=ztr)
        teacher_npz = np.load(args.teacher_cache, allow_pickle=False)
        # The cache carries its own copy of `flips`; if it disagrees with the pool's, the
        # two files are not describing the same shots and every soft target is misaligned.
        if not np.array_equal(f_cache.reshape(-1), f_tr.reshape(-1)):
            raise SystemExit(f"{LOG} teacher cache `flips` disagree with the pool's -- "
                             "the cache is misaligned with these shots.")
        print(f"{LOG} teacher cache ok: {ntr:,} shots, mean p_teacher={p_teach.mean():.5f}",
              flush=True)

    # --- optional independent training extension ------------------------------------
    # Appended to the training prefix only. The extension is a separate generation with its
    # own seed, so its shots are independent of every block of the primary pool; the
    # validation, evaluation and sealed regions are read from the primary pool alone and
    # are untouched by this.
    if args.extra_train_pool:
        if not os.path.exists(args.extra_train_pool):
            raise SystemExit(f"{LOG} MISSING extra pool {args.extra_train_pool}")
        zex = np.load(args.extra_train_pool)
        n_ex = zex['measurements'].shape[0]
        take = min(args.extra_train_n or n_ex, n_ex)
        fp = args.extra_train_pool.replace('.npz', '.fingerprint.json')
        if os.path.exists(fp):
            meta_ex = json.load(open(fp))
            if meta_ex.get('gen_seed') == _primary_gen_seed(train_fn):
                raise SystemExit(
                    f"{LOG} extra pool shares gen_seed {meta_ex.get('gen_seed')} with the "
                    f"primary pool. Same seed means the same shot stream -- the extension "
                    f"would duplicate training data and could repeat evaluation shots.")
            print(f"{LOG} extra pool gen_seed={meta_ex.get('gen_seed')} "
                  f"flips_sha={str(meta_ex.get('flips_sha256'))[:16]}", flush=True)
        m_ex = zex['measurements'][0:take].astype(binary_t)
        e_ex = zex['det_evts'][0:take].astype(binary_t)
        f_ex = zex['flips'][0:take].astype(binary_t).reshape(-1)
        b_ex, _, _ = split_measurements(m_ex, d, idx_t)
        x_ex = assemble_features(b_ex, e_ex, args.inputs, student=args.student,
                                 d=d, rounds=r, p=p)
        del m_ex, b_ex, e_ex
        x_tr = np.concatenate([x_tr, x_ex], axis=0)
        f_tr = np.concatenate([f_tr, f_ex], axis=0)
        z_teach = np.concatenate([z_teach, np.zeros(take, dtype=np.float32)], axis=0)
        ntr = ntr + take
        del x_ex
        print(f"{LOG} training set extended by {take:,} shots from "
              f"{os.path.basename(args.extra_train_pool)}; total {ntr:,}", flush=True)

    prov = run_provenance(args, train_fn, teacher_npz)
    print(f"{LOG} provenance: pool={prov['train_pool_dir']}/{prov['train_pool']} "
          f"flips_sha={prov['train_pool_flips_sha']} teacher_sha={prov['teacher_weights_sha']} "
          f"code_sha={prov['code_sha']}", flush=True)

    y_tr = np.stack([f_tr.astype(np.float32), z_teach], axis=1)

    # --- evaluation tail ------------------------------------------------------------
    zte = np.load(args.test_pool) if args.test_pool else ztr
    Nte = zte['measurements'].shape[0]
    if args.eval_start is None:
        te = slice(Nte - nte, Nte)
    else:
        if args.eval_start + nte > Nte:
            raise SystemExit(f"{LOG} --eval-start {args.eval_start} + n-test {nte} "
                             f"exceeds pool size {Nte}")
        te = slice(args.eval_start, args.eval_start + nte)
    print(f"{LOG} scoring shots [{te.start:,}, {te.stop:,})", flush=True)
    m_te = zte['measurements'][te].astype(binary_t)
    e_te = zte['det_evts'][te].astype(binary_t)
    f_te = zte['flips'][te].astype(binary_t).reshape(-1)
    b_te, _, _ = split_measurements(m_te, d, idx_t)
    x_te = assemble_features(b_te, e_te, args.inputs, student=args.student,
                             d=d, rounds=r, p=p)
    del m_te, b_te, e_te

    # --- explicit validation block --------------------------------------------------
    # Mirrors train_one.py: when --val-start/--val-n are given, validation comes from
    # those shots and validation_split is disabled, so every shot in [0, n_train)
    # produces gradients. Disjointness is asserted rather than assumed.
    val_data = None
    if args.val_start is not None:
        if args.val_n is None:
            raise SystemExit(f"{LOG} --val-start requires --val-n")
        v0, v1 = args.val_start, args.val_start + args.val_n
        if v0 < ntr:
            raise SystemExit(f"{LOG} validation [{v0}, {v1}) overlaps the training "
                             f"prefix [0, {ntr}) -- not disjoint")
        if v1 > Nte:
            raise SystemExit(f"{LOG} validation [{v0}, {v1}) exceeds pool size {Nte}")
        m_va = zte['measurements'][v0:v1].astype(binary_t)
        e_va = zte['det_evts'][v0:v1].astype(binary_t)
        f_va = zte['flips'][v0:v1].astype(binary_t).reshape(-1)
        b_va, _, _ = split_measurements(m_va, d, idx_t)
        x_va = assemble_features(b_va, e_va, args.inputs, student=args.student,
                                 d=d, rounds=r, p=p)
        del m_va, b_va, e_va
        # The loss expects the packed [hard_label, teacher_logit] target. With alpha=1 the
        # teacher column is unused, so zeros are honest filler. With alpha<1 they are not:
        # zeros are a teacher saying p=0.5 on every validation shot, which would make the
        # validation loss measure the wrong thing and, with early stopping, select on it.
        # So the logits must come from a cache covering the validation block.
        if args.alpha < 1.0:
            if not args.val_teacher_cache:
                raise SystemExit(
                    f"{LOG} --val-start with alpha={args.alpha} < 1 requires "
                    f"--val-teacher-cache: the soft term needs teacher logits on the "
                    f"validation shots, and the training cache stops at n_train. Dump a "
                    f"cache over [{v0}, {v1}) and pass it (usually the same npz as "
                    f"--teacher-tail-cache).")
            z_va, _p_va, f_va_cache = load_teacher_cache(
                args.val_teacher_cache, train_fn, v0, v1, pool_npz=None)
            # The cache carries its own copy of the labels. If they disagree with the
            # pool's, the cache describes different shots than the ones just loaded --
            # which is exactly the silent-misalignment failure the fingerprints exist to
            # catch, so check it here too rather than trusting the range arithmetic.
            if not np.array_equal(f_va_cache.reshape(-1).astype(np.int8),
                                  f_va.reshape(-1).astype(np.int8)):
                raise SystemExit(
                    f"{LOG} --val-teacher-cache flips disagree with the pool over "
                    f"[{v0}, {v1}) -- the cache does not describe these shots.")
            z_va = z_va.reshape(-1).astype(np.float32)
        else:
            z_va = np.zeros(len(f_va), np.float32)
        y_va = np.stack([f_va.astype(np.float32), z_va], axis=1)
        val_data = (x_va, y_va)
        steps = int(np.ceil(ntr / args.batch_size))
        print(f"{LOG} partitions: train [0, {ntr:,})  validation [{v0:,}, {v1:,})  "
              f"(validation_split disabled)", flush=True)
        print(f"{LOG} {steps} steps/epoch x {args.epochs} epochs = "
              f"{steps * args.epochs:,} optimizer updates", flush=True)
    else:
        eff = int(ntr * (1 - args.val_split))
        steps = int(np.ceil(eff / args.batch_size))
        print(f"{LOG} WARNING: no --val-start; validation_split={args.val_split} carves "
              f"{ntr - eff:,} shots OUT of the training prefix, so only {eff:,} of "
              f"{ntr:,} produce gradients.", flush=True)
        print(f"{LOG} {steps} steps/epoch x {args.epochs} epochs = "
              f"{steps * args.epochs:,} optimizer updates", flush=True)

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

    # With --lr the rate is constant and the teacher's scheduler is not attached; it would
    # overwrite the constant each epoch and turn an LR comparison into a schedule test.
    callbacks = []
    if args.lr is None:
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler))
    if args.tail_diagnostics:
        callbacks.append(TailDiagnostics(x_te, f_te, args.batch_size).build())
    if not args.no_early_stopping:
        callbacks.insert(0, tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.patience, restore_best_weights=True))
    if args.ckpt_dir:
        # Same construction train_one.py uses. The final-epoch saver is placed BEFORE
        # EarlyStopping, because EarlyStopping(restore_best_weights=True) swaps the best
        # weights back in during its own on_epoch_end on the stopping epoch, and anything
        # that must observe the true final epoch has to run first.
        os.makedirs(args.ckpt_dir, exist_ok=True)
        _ck = os.path.join(args.ckpt_dir, (args.run_tag or args.tag or 'run'))

        class _SaveLastEpoch(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                self.model.save_weights(_ck + '.lastepoch.weights.h5')

        callbacks.insert(0, _SaveLastEpoch())
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            _ck + '.best.weights.h5', monitor='val_loss', mode='min',
            save_best_only=True, save_weights_only=True, verbose=0))
        print(f"{LOG} checkpoints -> {_ck}.best.weights.h5 (min val_loss) and "
              f"{_ck}.lastepoch.weights.h5", flush=True)
    # Printed, not asserted: EarlyStopping(restore_best_weights=True) swaps weights during
    # its own on_epoch_end, so anything that must observe the true final epoch has to
    # appear before it. Logging the constructed order makes that checkable by inspection.
    print(f"{LOG} callback order: "
          f"{[type(c).__name__ for c in callbacks]}", flush=True)

    fit_kw = (dict(validation_data=val_data) if val_data is not None
              else dict(validation_split=args.val_split))
    t0 = time.time()
    hist = model.fit(x=x_tr, y=y_tr, batch_size=args.batch_size, epochs=args.epochs,
                     shuffle=True, verbose=2, callbacks=callbacks, **fit_kw)
    train_time = time.time() - t0
    epochs_ran = len(hist.history['loss'])
    best_val = float(min(hist.history['val_loss']))

    # --- score on the tail ----------------------------------------------------------
    logits = model.predict(x_te, batch_size=args.batch_size, verbose=0)
    pred, correct = student_pred_and_correct(logits, f_te)
    pL = float((~correct).mean())
    base_rate = float(f_te.mean())
    beats_base = pL < base_rate

    # MWPM baselines are tail-specific. lookup_mwpm returns the one computed on the
    # --data-dir pool's own tail, so under an explicit --test-pool it describes different
    # shots and no ratio from it is meaningful. Decode MWPM on the actual tail
    # (eval_on_tail.py --mcnemar) instead.
    if args.test_pool or args.eval_start is not None:
        mwpm = None
        print(f"{LOG} --test-pool given: suppressing the stored MWPM ratio (baselines "
              "are tail-specific). Decode MWPM on THIS tail to compare.", flush=True)
    else:
        mwpm = lookup_mwpm(args.data_dir, d, p, r)

    # --- distillation gap: how closely does the student track the teacher? ----------
    # Measured on decisions, not probabilities: a student can differ in probability
    # everywhere and still decode identically. Needs a tail teacher cache.
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
    # The pool directory is part of the tag: the same (student, alpha, seed, ntr) trained
    # on a different pool is a different experiment, and must not land on the same
    # filenames as an earlier one.
    pool_tag = prov['train_pool_dir']
    tag = args.tag or (f'student_{args.student}_{args.inputs.replace("+", "-")}_'
                       f'{wb}_{ab}_alpha{args.alpha:g}_T{args.temperature:g}_'
                       f'seed{seed}_ntr{ntr}_{pool_tag}')

    # Never silently replace an existing result. Collisions are archived with the run's
    # timestamp rather than overwritten, so an accidental re-run cannot destroy a number
    # that has not been written up yet.
    csv_path = os.path.join(args.out_dir, tag + '.csv')
    if os.path.exists(csv_path):
        stamp = prov['run_utc'].replace(':', '').replace('-', '')
        for ext in ('.csv', '.history.json', '.weights.h5'):
            old = os.path.join(args.out_dir, tag + ext)
            if os.path.exists(old):
                os.rename(old, os.path.join(args.out_dir, f'{tag}.superseded_{stamp}{ext}'))
        print(f"{LOG} existing results for this tag archived as "
              f"{tag}.superseded_{stamp}.*", flush=True)

    fields = ['architecture', 'student', 'inputs', 'd', 'p', 'rounds', 'seed', 'n_train',
              'n_test', 'alpha', 'temperature', 'lr', 'weight_bits', 'act_bits',
              # architecture size, so a row is interpretable without parsing its tag
              'hidden', 'units', 'n_params',
              'epochs', 'epochs_ran', 'batch_size', 'p_L', 'mwpm_p_L', 'ratio_vs_mwpm',
              'base_rate', 'beats_base_rate', 'teacher_tail_p_L',
              # agreement: the raw rate is dominated by the ~80% of shots where this
              # teacher is confident, so agree_ambiguous is the one that carries signal
              'agree_all', 'agree_ambiguous', 'agree_confident', 'n_ambiguous',
              'frac_ambiguous', 'student_p_L_ambiguous', 'teacher_p_L_ambiguous',
              'ambiguous_band', 'best_val_loss', 'train_time_s'] + PROVENANCE_COLS
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
            hidden='-'.join(str(h) for h in args.hidden) if args.hidden else '',
            units=(args.units if args.student == 'gru' else ''),
            n_params=n_params, epochs=args.epochs, epochs_ran=epochs_ran,
            batch_size=args.batch_size, p_L=round(pL, 6),
            mwpm_p_L=('' if mwpm is None else round(mwpm, 6)),
            ratio_vs_mwpm=('' if mwpm is None else round(pL / mwpm, 4)),
            base_rate=round(base_rate, 5), beats_base_rate=int(beats_base),
            ambiguous_band=f'{args.ambiguous_band[0]}-{args.ambiguous_band[1]}',
            best_val_loss=round(best_val, 5), train_time_s=round(train_time, 1),
            **ag, **prov))
    with open(os.path.join(args.out_dir, tag + '.history.json'), 'w') as hf:
        json.dump({k2: [float(x) for x in v] for k2, v in hist.history.items()}, hf)
    wpath = os.path.join(args.out_dir, tag + '.weights.h5')
    model.save_weights(wpath)

    gap = '' if mwpm is None else (f'  MWPM(same-pool tail)={mwpm:.5f}  '
                                   f'ratio={pL / mwpm:.3f}x')
    flag = '' if beats_base else '  <-- fails: p_L >= base rate (class collapse)'
    print(f"{LOG} {tag}  p_L={pL:.5f}{gap}  base_rate={base_rate:.3f}"
          f"  agree_ambiguous={ag['agree_ambiguous']}{flag}"
          f"  ({epochs_ran} ep, {train_time:.0f}s)",
          flush=True)
    print(f"{LOG} wrote -> {os.path.join(args.out_dir, tag + '.csv')}  (+ history, weights)",
          flush=True)


if __name__ == '__main__':
    run()
