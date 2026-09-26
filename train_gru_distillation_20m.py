#!/usr/bin/env python3
# Created: 2026-09-22
# Last modified: 2026-09-22
"""Train one reduced GRU student from the 20M-shot u140 GRU teacher, d=5 r=5 p=0.004.

Purpose
-------
The u140 GRU trained on 20M shots (15M primary + 5M training-only extension, Run 11 of
Experiment 16) beats MWPM at d=5, p=0.004. This script trains one smaller GRU against it,
so the experiment can measure whether knowledge distillation holds that performance while
the parameter count falls to something more attractive for hls4ml / FPGA synthesis.
Everything stays FP32 here. Quantization is a later, separate experiment.

One run = (units, alpha, temperature, seed). The launcher eaf_run_gru_distill_20m.sh
drives the 24-run grid.

  alpha = 1.0   hard Stim labels only (the control arm)
  alpha = 0.0   pure distillation: match the frozen teacher's logit, softened by T

Why a separate script from train_student.py
-------------------------------------------
train_student.py appends --extra-train-pool shots with a teacher logit of exactly 0 for
every extension shot (see its "optional independent training extension" block). At
alpha=1 the teacher column is multiplied by zero and the fill is harmless. At alpha<1 it
tells the student that a quarter of the 20M training set is a p=0.5 coin flip, which
would corrupt the distilled arm. Rather than edit the generic trainer that every earlier
study used, this file handles the two-pool case explicitly and leaves train_student.py
untouched, so every earlier result stays reproducible from unchanged code.

What is reused, so nothing drifts from the earlier studies
----------------------------------------------------------
  StudentModels.build_gru_student      the GRU (reset_after=False, linear logit head)
  StudentModels.assemble_features      det_evts -> [6 timesteps, 24 positions] layout
  StudentModels.student_pred_and_correct   decision rule: logit > 0
  train_student.make_distillation_loss    the alpha / temperature loss
  train_student.make_hard_accuracy     accuracy against the hard label column only
  train_student.teacher_agreement      student-vs-teacher agreement, raw and ambiguous-band
  train_student.cap_gpu_memory         per-process GPU cap
  train_one.set_seeds                  seed order identical to every earlier student run
  dump_teacher_probs.pool_fingerprint / sha256_file   pool and checkpoint identity
  eval_on_tail.build_circuit / mcnemar_from_correct   MWPM circuit and the paired test

Data flow for the 20M training set
----------------------------------
  primary pool   [0, main_n_train)        15M shots, teacher cache --teacher-main-cache
  extension pool [0, extra_n)              5M shots, teacher cache --teacher-extra-cache
      concatenated in that order -> x_train [20M, 6, 24], y_train [20M, 2]
      y_train[:, 0] = Stim logical flip (hard label)
      y_train[:, 1] = teacher logit for the SAME shot, same position in the array
  validation     [val_start, +val_n)       primary pool, teacher cache --teacher-val-cache
  evaluation     [eval_start, +eval_n)     primary pool, read only AFTER training finishes
  sealed test    [sealed_start, end)       never read

Checks that stop the run (SystemExit) rather than warn
------------------------------------------------------
  * pool fingerprint JSON present, geometry (d, r, p) matches, flips SHA-256 on disk
    matches the fingerprint
  * extension pool generated with a different seed than the primary pool
  * training / validation / evaluation / sealed blocks do not overlap
  * every teacher cache: GRU teacher of the stated width, built from a pool whose flips
    SHA-256 equals this pool's, covering exactly the requested shot range, carrying flips
    identical to the pool's flips for those shots, finite logits, nonzero spread
  * all teacher caches come from the same checkpoint (same weights SHA-256), and that
    SHA matches --teacher-weights / --teacher-sha256 when given
  * the extension cache holds exactly extra_n genuine outputs (not a constant fill)
  * alpha < 1 cannot start without the main, extension (when used) and validation caches
  * after concatenation, the teacher column is re-compared against both caches and the
    hard-label column against both pools' flips

Outputs, all inside --out-dir (one directory per run, never overwritten)
------------------------------------------------------------------------
  <tag>.csv                   one result row (p_L, MWPM, ratio, teacher, agreement, hashes)
  <tag>.history.json          per-epoch loss / val_loss / hard_accuracy
  <tag>.best.weights.h5       min-val_loss checkpoint, the one that is scored
  <tag>.lastepoch.weights.h5  true final-epoch weights
  <tag>.per_shot_eval.npz     per-shot student logit, MWPM prediction, teacher logit, truth
  <tag>.teacher_alignment.json   per-segment proof that both training pools got teacher logits
  COMPLETE                    written last; the launcher's skip test keys on it
"""
import argparse
import csv
import datetime
import json
import os
import socket
import sys
import time

import numpy as np

LOG = '[gru-kd20m]'

# Key names the dump_teacher_probs.py cache must carry for this script to trust it. Older
# caches without teacher_arch / units / fingerprints are refused; this study has no
# legacy caches to support.
REQUIRED_CACHE_KEYS = ('p_teacher', 'logit_teacher', 'flips', 'shot_idx', 'teacher_arch',
                       'units', 'weights_sha256', 'pool_flips_sha256', 'pool_path')


# ---------------------------------------------------------------------------------------
# Identity and integrity helpers. Import-safe (numpy only), so the preflight script can use
# them without importing TensorFlow.
# ---------------------------------------------------------------------------------------
def sha256_of(path):
    """Full SHA-256 of a file, or '' if the path is empty or missing."""
    from dump_teacher_probs import sha256_file
    if not path or not os.path.exists(path):
        return ''
    return sha256_file(path)


def read_pool(path, label, d, rounds, p):
    """Open a pool, check it against its fingerprint JSON, and return (npz, meta, flips_sha).

    The fingerprint is required. It is the only record of the generation seed, which is
    what proves the extension pool is an independent draw from the primary pool. The
    flips SHA-256 is recomputed from the file on disk, so a pool regenerated under the same
    name with different content is caught here.
    """
    from dump_teacher_probs import pool_fingerprint
    if not os.path.exists(path):
        raise SystemExit(f"{LOG} MISSING {label} pool {path}")
    fp_path = path.replace('.npz', '.fingerprint.json')
    if not os.path.exists(fp_path):
        raise SystemExit(f"{LOG} {label} pool has no fingerprint beside it ({fp_path}); "
                         "the generation seed cannot be verified. STOP.")
    meta = json.load(open(fp_path))
    if (int(meta['d']), int(meta['rounds']), round(float(meta['p']), 6)) != \
            (d, rounds, round(p, 6)):
        raise SystemExit(f"{LOG} {label} pool fingerprint says d={meta['d']} "
                         f"r={meta['rounds']} p={meta['p']}; this run is d={d} r={rounds} "
                         f"p={p}. STOP.")
    z = np.load(path)
    flips_sha, meas_shape = pool_fingerprint(z)
    if flips_sha != meta['flips_sha256']:
        raise SystemExit(f"{LOG} {label} pool flips SHA-256 on disk {flips_sha[:16]} != "
                         f"fingerprint {meta['flips_sha256'][:16]}. File changed since it "
                         "was generated. STOP.")
    n = int(z['flips'].shape[0])
    print(f"{LOG} {label} pool {path}", flush=True)
    print(f"{LOG}   shots={n:,}  gen_seed={meta['gen_seed']}  flips_sha256={flips_sha}  "
          f"measurements{meas_shape}  purpose={meta.get('purpose', 'primary')}", flush=True)
    return z, meta, flips_sha


def load_teacher_block(cache_path, label, pool_flips_sha, pool_flips, lo, hi,
                       teacher_units):
    """Load one teacher cache and prove it describes shots [lo, hi) of the given pool.

    pool_flips is the pool's own flips array for exactly [lo, hi). Returns a dict with the
    teacher logits and probabilities for those shots plus the cache's identity fields.
    Every check here fails the run: a misaligned cache still produces a smooth loss curve,
    so nothing downstream would reveal it.
    """
    if not cache_path:
        raise SystemExit(f"{LOG} no teacher cache given for the {label} block. STOP.")
    if not os.path.exists(cache_path):
        raise SystemExit(f"{LOG} MISSING teacher cache for the {label} block: {cache_path}")
    tc = np.load(cache_path, allow_pickle=False)
    missing = [k for k in REQUIRED_CACHE_KEYS if k not in tc.files]
    if missing:
        raise SystemExit(f"{LOG} {label} cache {cache_path} lacks {missing}; re-dump it with "
                         "the current dump_teacher_probs.py. STOP.")

    # Which teacher produced it.
    if str(tc['teacher_arch']) != 'gru':
        raise SystemExit(f"{LOG} {label} cache teacher_arch={tc['teacher_arch']}, expected "
                         "'gru'. STOP.")
    if int(tc['units']) != teacher_units:
        raise SystemExit(f"{LOG} {label} cache was dumped from a units={int(tc['units'])} "
                         f"GRU, expected the u{teacher_units} teacher. STOP.")

    # Which pool it describes. The flips SHA-256 is the identity; the stored path is
    # printed only, because a pool may legitimately be read from a different mount.
    if str(tc['pool_flips_sha256']) != pool_flips_sha:
        raise SystemExit(f"{LOG} {label} cache was built from a pool with flips SHA "
                         f"{str(tc['pool_flips_sha256'])[:16]}, this pool is "
                         f"{pool_flips_sha[:16]}. Different shots. STOP.")

    # Which shots. The cache must cover [lo, hi) contiguously.
    idx = tc['shot_idx']
    n_cache = int(idx.shape[0])
    if n_cache == 0 or int(idx[-1]) - int(idx[0]) + 1 != n_cache:
        raise SystemExit(f"{LOG} {label} cache shot_idx is empty or not contiguous. STOP.")
    if int(idx[0]) > lo or int(idx[-1]) < hi - 1:
        raise SystemExit(f"{LOG} {label} cache covers [{int(idx[0]):,}, {int(idx[-1]) + 1:,}) "
                         f"but this run needs [{lo:,}, {hi:,}). STOP.")
    off, n = lo - int(idx[0]), hi - lo
    logit = tc['logit_teacher'][off:off + n].astype(np.float32).reshape(-1)
    prob = tc['p_teacher'][off:off + n].astype(np.float32).reshape(-1)
    flips_cache = tc['flips'][off:off + n].astype(np.int8).reshape(-1)
    if logit.shape[0] != n or flips_cache.shape[0] != n:
        raise SystemExit(f"{LOG} {label} cache returned {logit.shape[0]:,} shots, "
                         f"expected {n:,}. STOP.")

    # Shot-by-shot alignment: the cache's own copy of the labels must equal the pool's.
    if not np.array_equal(flips_cache, np.asarray(pool_flips).astype(np.int8).reshape(-1)):
        n_bad = int((flips_cache != np.asarray(pool_flips).astype(np.int8).reshape(-1)).sum())
        raise SystemExit(f"{LOG} {label} cache flips disagree with the pool on {n_bad:,} of "
                         f"{n:,} shots. Cache is misaligned. STOP.")

    # Genuine model output: finite, not constant, and not a zero fill. A trained GRU's
    # float32 logit lands on exactly 0.0 essentially never, so more than 0.1% exact zeros
    # means a placeholder, not a model.
    if not np.all(np.isfinite(logit)):
        raise SystemExit(f"{LOG} {label} cache has non-finite teacher logits. STOP.")
    frac_zero = float((logit == 0.0).mean())
    logit_std = float(logit.std())
    if frac_zero > 1e-3 or logit_std < 1e-3:
        raise SystemExit(f"{LOG} {label} cache looks like a placeholder, not teacher output: "
                         f"exact-zero fraction {frac_zero:.4f}, logit std {logit_std:.5f}. STOP.")

    t_pred = (logit > 0.0).astype(np.int8)
    stats = dict(
        block=label, cache=os.path.abspath(cache_path),
        cache_sha256=sha256_of(cache_path),
        cache_pool_path=str(tc['pool_path']),
        range=[int(lo), int(hi)], n_shots=int(n),
        teacher_weights_sha256=str(tc['weights_sha256']),
        teacher_units=int(tc['units']),
        logit_mean=round(float(logit.mean()), 5), logit_std=round(logit_std, 5),
        logit_min=round(float(logit.min()), 4), logit_max=round(float(logit.max()), 4),
        frac_exact_zero=frac_zero,
        mean_p_teacher=round(float(prob.mean()), 6),
        teacher_error_rate=round(float((t_pred != flips_cache).mean()), 6),
        flips_match_pool=True)
    print(f"{LOG}   {label:10s} cache ok: shots [{lo:,}, {hi:,})  n={n:,}  "
          f"logit mean={stats['logit_mean']:+.3f} std={stats['logit_std']:.3f}  "
          f"exact zeros={frac_zero:.2e}  teacher err on these shots="
          f"{stats['teacher_error_rate']:.6f}", flush=True)
    return dict(logit=logit, prob=prob, stats=stats)


def check_partitions(n_main_pool, main_n_train, val_start, val_n, eval_start, eval_n,
                     sealed_start):
    """Assert the four primary-pool blocks are in order and disjoint. Integers only."""
    v1, e1 = val_start + val_n, eval_start + eval_n
    problems = []
    if main_n_train > val_start:
        problems.append(f"training [0, {main_n_train:,}) overlaps validation at {val_start:,}")
    if v1 > eval_start:
        problems.append(f"validation [{val_start:,}, {v1:,}) overlaps evaluation at "
                        f"{eval_start:,}")
    if e1 > sealed_start:
        problems.append(f"evaluation [{eval_start:,}, {e1:,}) runs into the sealed test at "
                        f"{sealed_start:,}")
    if sealed_start > n_main_pool:
        problems.append(f"sealed start {sealed_start:,} beyond pool size {n_main_pool:,}")
    if problems:
        raise SystemExit(f"{LOG} partition error: " + "; ".join(problems) + ". STOP.")
    print(f"{LOG} partitions (primary pool): train [0, {main_n_train:,})  "
          f"val [{val_start:,}, {v1:,})  eval [{eval_start:,}, {e1:,})  "
          f"sealed [{sealed_start:,}, {n_main_pool:,}) not read -- disjoint", flush=True)


def fill_sequences(dst, dst_offset, det_evts, d, rounds, p, chunk=1_000_000):
    """Write the GRU input layout for det_evts into dst[dst_offset:], one chunk at a time.

    Same assemble_features call as every earlier student run, applied per chunk into a
    preallocated array. The values are identical to converting all at once; the chunking
    only avoids holding a second full-size float32 copy of the 20M-shot training set
    (about 11.5 GB) during concatenation.
    """
    from StudentModels import assemble_features
    n = det_evts.shape[0]
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        dst[dst_offset + a:dst_offset + b] = assemble_features(
            None, det_evts[a:b], 'evts', student='gru', d=d, rounds=rounds, p=p)


def run():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    # --- geometry (fixed by the teacher) ---
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--p', type=float, default=0.004)
    # --- data ---
    ap.add_argument('--main-pool', required=True,
                    help='primary pool npz; supplies training prefix, validation and '
                         'evaluation blocks')
    ap.add_argument('--main-n-train', type=int, default=15_000_000,
                    help='training prefix [0, N) of the primary pool')
    ap.add_argument('--extra-train-pool', default=None,
                    help='training-only extension pool npz (different generation seed)')
    ap.add_argument('--extra-n', type=int, default=5_000_000,
                    help='shots [0, N) taken from the extension pool')
    ap.add_argument('--val-start', type=int, default=15_000_000)
    ap.add_argument('--val-n', type=int, default=200_000)
    ap.add_argument('--eval-start', type=int, default=15_200_000)
    ap.add_argument('--eval-n', type=int, default=1_800_000)
    ap.add_argument('--sealed-start', type=int, default=17_000_000,
                    help='first shot of the sealed test block; only used to assert that '
                         'nothing reaches it')
    # --- teacher ---
    ap.add_argument('--teacher-main-cache', default=None,
                    help='dump_teacher_probs.py output over [0, main_n_train) of --main-pool')
    ap.add_argument('--teacher-extra-cache', default=None,
                    help='dump_teacher_probs.py output over [0, extra_n) of '
                         '--extra-train-pool')
    ap.add_argument('--teacher-val-cache', default=None,
                    help='dump over the validation block. Needed when alpha < 1, because '
                         'the validation loss (and so checkpoint selection) uses the same '
                         'distillation loss as training.')
    ap.add_argument('--teacher-tail-cache', default=None,
                    help='dump over the evaluation block. Read only after training, for '
                         'teacher p_L and student-teacher agreement.')
    ap.add_argument('--teacher-units', type=int, default=140)
    ap.add_argument('--teacher-weights', default=None,
                    help='teacher .weights.h5; when given, its SHA-256 must equal the one '
                         'recorded in every cache')
    ap.add_argument('--teacher-sha256', default=None,
                    help='expected full SHA-256 of the teacher checkpoint; when given, every '
                         'cache must carry it')
    # --- student and loss ---
    ap.add_argument('--units', type=int, required=True, help='student GRU width')
    ap.add_argument('--alpha', type=float, required=True,
                    help='1.0 = hard labels only; 0.0 = pure distillation')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--ambiguous-band', type=float, nargs=2, default=[0.05, 0.95],
                    metavar=('LO', 'HI'))
    # --- recipe (defaults = Run 11, the teacher's own recipe) ---
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--batch-size', type=int, default=10_000)
    ap.add_argument('--lr', type=float, default=0.003, help='constant Adam learning rate')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=0,
                    help='early-stopping patience on val_loss; 0 = off (Run 11 recipe: the '
                         'full epoch budget runs and the min-val_loss checkpoint is scored)')
    # --- io ---
    ap.add_argument('--out-dir', required=True, help='this run\'s own directory')
    ap.add_argument('--tag', required=True, help='filename prefix for every output')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--gpu-mem-mib', type=int, default=None)
    args = ap.parse_args()

    t_start = time.time()
    d, r, p = args.d, args.rounds, args.p
    use_extra = bool(args.extra_train_pool)
    distilling = args.alpha < 1.0

    # --- refuse to overwrite ---------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    for name in ('COMPLETE', f'{args.tag}.csv', f'{args.tag}.best.weights.h5'):
        if os.path.exists(os.path.join(args.out_dir, name)):
            raise SystemExit(f"{LOG} {os.path.join(args.out_dir, name)} already exists. "
                             "Results are append-only; the launcher moves an incomplete "
                             "attempt aside before retrying. STOP.")

    # --- alpha < 1 needs every training-side teacher cache --------------------------
    if distilling:
        need = {'--teacher-main-cache': args.teacher_main_cache,
                '--teacher-val-cache': args.teacher_val_cache}
        if use_extra:
            need['--teacher-extra-cache'] = args.teacher_extra_cache
        absent = [k for k, v in need.items() if not v]
        if absent:
            raise SystemExit(f"{LOG} alpha={args.alpha} < 1 requires {absent}. A missing "
                             "cache would leave the distilled arm without real teacher "
                             "targets. STOP.")
    if use_extra and args.teacher_main_cache and not args.teacher_extra_cache:
        raise SystemExit(f"{LOG} a main teacher cache was given without an extension "
                         "cache; the extension shots would have no teacher logits. STOP.")

    # --- determinism, before TensorFlow is imported ----------------------------------
    # The launcher exports both variables; this refuses to run if it did not, because
    # setting them after `import tensorflow` has no effect.
    from slice_guard_r5 import assert_deterministic_env
    assert_deterministic_env()
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    assert tf.__version__.startswith('2.15'), (
        f'need TF 2.15.x (Keras 2); got TF {tf.__version__}. Activate the pinned env.')
    from train_student import (cap_gpu_memory, make_distillation_loss, make_hard_accuracy,
                               teacher_agreement)
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')
    elif args.gpu_mem_mib:
        cap_gpu_memory(tf, args.gpu_mem_mib)
    print(f"{LOG} TF {tf.__version__}  GPUs: {tf.config.list_physical_devices('GPU')}  "
          f"host {socket.gethostname()}", flush=True)

    from train_one import set_seeds
    from StudentModels import build_gru_student, detector_sequence_layout, \
        student_pred_and_correct

    # --- pools -----------------------------------------------------------------------
    zm, meta_m, sha_m = read_pool(args.main_pool, 'primary', d, r, p)
    n_main_pool = int(zm['flips'].shape[0])
    check_partitions(n_main_pool, args.main_n_train, args.val_start, args.val_n,
                     args.eval_start, args.eval_n, args.sealed_start)

    n_extra = 0
    if use_extra:
        zx, meta_x, sha_x = read_pool(args.extra_train_pool, 'extension', d, r, p)
        if meta_x['gen_seed'] == meta_m['gen_seed']:
            raise SystemExit(f"{LOG} extension gen_seed {meta_x['gen_seed']} equals the "
                             "primary pool's. Same seed = same shot stream, so the "
                             "extension would repeat primary shots, including evaluation "
                             "shots. STOP.")
        if sha_x == sha_m:
            raise SystemExit(f"{LOG} extension and primary pools have identical flips. STOP.")
        if meta_x.get('purpose') != 'training_only':
            print(f"{LOG}   note: extension fingerprint purpose="
                  f"{meta_x.get('purpose')!r}, expected 'training_only'", flush=True)
        n_extra = int(args.extra_n)
        if n_extra > int(zx['flips'].shape[0]):
            raise SystemExit(f"{LOG} --extra-n {n_extra:,} exceeds extension pool size "
                             f"{int(zx['flips'].shape[0]):,}. STOP.")
    else:
        zx, meta_x, sha_x = None, {}, ''

    n_main = args.main_n_train
    n_train = n_main + n_extra
    n_t, n_pos, _ = detector_sequence_layout(d, r, p)

    # --- training set: preallocate, then fill [main | extension] ---------------------
    flips_main_all = zm['flips'].astype(np.int8).reshape(-1)   # 19M int8, small
    f_main = flips_main_all[0:n_main]
    x_tr = np.empty((n_train, n_t, n_pos), dtype=np.float32)
    det_main = zm['det_evts']                                   # full array, int8
    fill_sequences(x_tr, 0, det_main[0:n_main], d, r, p)
    if use_extra:
        f_extra = zx['flips'][0:n_extra].astype(np.int8).reshape(-1)
        fill_sequences(x_tr, n_main, zx['det_evts'][0:n_extra], d, r, p)
    else:
        f_extra = np.zeros(0, np.int8)
    f_tr = np.concatenate([f_main, f_extra])
    print(f"{LOG} training inputs: {n_main:,} primary + {n_extra:,} extension = "
          f"{n_train:,} shots, x{x_tr.shape[1:]} float32 ({x_tr.nbytes / 1024 ** 3:.1f} GB)",
          flush=True)

    # --- teacher logits for the training set, same order ------------------------------
    have_train_caches = bool(args.teacher_main_cache)
    alignment = {'tag': args.tag, 'alpha': args.alpha, 'segments': []}
    teacher_shas = set()
    if have_train_caches:
        print(f"{LOG} teacher caches:", flush=True)
        t_main = load_teacher_block(args.teacher_main_cache, 'main-train', sha_m, f_main,
                                    0, n_main, args.teacher_units)
        z_parts = [t_main['logit']]
        alignment['segments'].append(dict(t_main['stats'], train_rows=[0, n_main]))
        teacher_shas.add(t_main['stats']['teacher_weights_sha256'])
        if use_extra:
            t_extra = load_teacher_block(args.teacher_extra_cache, 'extra-train', sha_x,
                                         f_extra, 0, n_extra, args.teacher_units)
            if t_extra['logit'].shape[0] != n_extra:
                raise SystemExit(f"{LOG} extension cache holds {t_extra['logit'].shape[0]:,} "
                                 f"outputs, expected {n_extra:,}. STOP.")
            z_parts.append(t_extra['logit'])
            alignment['segments'].append(dict(t_extra['stats'],
                                              train_rows=[n_main, n_train]))
            teacher_shas.add(t_extra['stats']['teacher_weights_sha256'])
        z_tr = np.concatenate(z_parts).astype(np.float32)
        teacher_column = 'teacher_logits'
    else:
        # Only reachable with alpha == 1 (checked above). The column is multiplied by
        # (1 - alpha) = 0 in the loss, so it cannot influence training.
        z_tr = np.zeros(n_train, np.float32)
        teacher_column = 'unused_zeros_alpha1'
        print(f"{LOG} alpha=1 without teacher caches: teacher column unused.", flush=True)

    y_tr = np.stack([f_tr.astype(np.float32), z_tr], axis=1)

    # Post-concatenation proof: re-read each segment out of the packed target and compare
    # it with its source. Catches any off-by-one in the offsets above.
    assert y_tr.shape == (n_train, 2)
    assert x_tr.shape[0] == n_train
    assert np.array_equal(y_tr[:n_main, 0].astype(np.int8), f_main)
    assert np.array_equal(y_tr[n_main:, 0].astype(np.int8), f_extra)
    if have_train_caches:
        assert np.array_equal(y_tr[:n_main, 1], t_main['logit'])
        if use_extra:
            assert np.array_equal(y_tr[n_main:, 1], t_extra['logit'])
            ext_col = y_tr[n_main:, 1]
            print(f"{LOG} packed target check: rows [{n_main:,}, {n_train:,}) carry the "
                  f"extension teacher logits (mean {ext_col.mean():+.4f}, std "
                  f"{ext_col.std():.4f}, exact zeros {int((ext_col == 0).sum()):,}); rows "
                  f"[0, {n_main:,}) carry the primary teacher logits", flush=True)
    alignment['packed_target_checked'] = True

    # --- validation block ------------------------------------------------------------
    v0, v1 = args.val_start, args.val_start + args.val_n
    f_va = flips_main_all[v0:v1]
    x_va = np.empty((args.val_n, n_t, n_pos), dtype=np.float32)
    fill_sequences(x_va, 0, det_main[v0:v1], d, r, p)
    if args.teacher_val_cache:
        t_val = load_teacher_block(args.teacher_val_cache, 'validation', sha_m, f_va,
                                   v0, v1, args.teacher_units)
        z_va = t_val['logit']
        alignment['segments'].append(t_val['stats'])
        teacher_shas.add(t_val['stats']['teacher_weights_sha256'])
    else:
        z_va = np.zeros(args.val_n, np.float32)  # alpha == 1 only
    y_va = np.stack([f_va.astype(np.float32), z_va], axis=1)
    del det_main

    # --- one teacher behind every cache -----------------------------------------------
    teacher_sha = ''
    if teacher_shas:
        if len(teacher_shas) != 1:
            raise SystemExit(f"{LOG} teacher caches come from different checkpoints: "
                             f"{sorted(s[:16] for s in teacher_shas)}. STOP.")
        teacher_sha = teacher_shas.pop()
    if args.teacher_weights:
        on_disk = sha256_of(args.teacher_weights)
        if not on_disk:
            raise SystemExit(f"{LOG} MISSING teacher weights {args.teacher_weights}")
        if teacher_sha and on_disk != teacher_sha:
            raise SystemExit(f"{LOG} teacher weights on disk {on_disk[:16]} != cache's "
                             f"{teacher_sha[:16]}. STOP.")
        teacher_sha = teacher_sha or on_disk
    if args.teacher_sha256 and teacher_sha and args.teacher_sha256 != teacher_sha:
        raise SystemExit(f"{LOG} expected teacher SHA-256 {args.teacher_sha256[:16]}, caches "
                         f"carry {teacher_sha[:16]}. STOP.")
    if teacher_sha:
        print(f"{LOG} teacher checkpoint sha256={teacher_sha}", flush=True)

    alignment_path = os.path.join(args.out_dir, f'{args.tag}.teacher_alignment.json')
    alignment.update(teacher_weights_sha256=teacher_sha, teacher_column=teacher_column,
                     n_main=n_main, n_extra=n_extra, n_train=n_train)
    with open(alignment_path, 'w') as fh:
        json.dump(alignment, fh, indent=2)

    # --- build, compile, fit ---------------------------------------------------------
    set_seeds(args.seed)
    model = build_gru_student(d=d, rounds=r, inputs='evts', units=args.units, hidden=(),
                              p=p)
    gru = model.get_layer('gru')
    assert gru.reset_after is False, 'student GRU must have reset_after=False'
    assert model.layers[-1].activation.__name__ == 'linear', 'student head must be linear'
    n_params = int(model.count_params())
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss=make_distillation_loss(args.alpha, args.temperature),
                  metrics=[make_hard_accuracy()])
    print(f"{LOG} student GRU units={args.units} params={n_params:,} reset_after=False "
          f"head=linear  alpha={args.alpha} T={args.temperature}  seed={args.seed}", flush=True)
    print(f"{LOG} recipe: Adam lr={args.lr} constant  batch={args.batch_size}  "
          f"epochs={args.epochs}  early_stopping="
          f"{'patience ' + str(args.patience) if args.patience > 0 else 'off'}  "
          f"selection=min val_loss", flush=True)

    ck = os.path.join(args.out_dir, args.tag)
    best_path, last_path = ck + '.best.weights.h5', ck + '.lastepoch.weights.h5'

    class _SaveLastEpoch(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            self.model.save_weights(last_path)

    # Same order as train_student.py: the last-epoch saver runs first, so it records the
    # true final weights even if EarlyStopping later restores the best ones.
    callbacks = [_SaveLastEpoch()]
    if args.patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.patience, restore_best_weights=True))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        best_path, monitor='val_loss', mode='min', save_best_only=True,
        save_weights_only=True, verbose=0))

    steps = int(np.ceil(n_train / args.batch_size))
    print(f"{LOG} {steps} steps/epoch x {args.epochs} epochs = {steps * args.epochs:,} "
          f"updates", flush=True)
    t0 = time.time()
    hist = model.fit(x=x_tr, y=y_tr, batch_size=args.batch_size, epochs=args.epochs,
                     shuffle=True, verbose=2, callbacks=callbacks,
                     validation_data=(x_va, y_va))
    train_time = time.time() - t0
    del x_tr, y_tr, x_va, y_va
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    with open(ck + '.history.json', 'w') as fh:
        json.dump(h, fh)
    epochs_ran = len(h['val_loss'])
    best_epoch = int(np.argmin(h['val_loss'])) + 1
    best_val = float(min(h['val_loss']))
    print(f"{LOG} trained {epochs_ran} epochs in {train_time:.0f}s; best val_loss "
          f"{best_val:.6f} at epoch {best_epoch}", flush=True)

    # --- score the min-val_loss checkpoint once on the evaluation block ---------------
    # First read of the evaluation block in this process.
    if not os.path.exists(best_path):
        raise SystemExit(f"{LOG} best checkpoint {best_path} was not written. STOP.")
    model.load_weights(best_path)
    e0, e1 = args.eval_start, args.eval_start + args.eval_n
    det_te = zm['det_evts'][e0:e1].astype(np.int8)
    truth = flips_main_all[e0:e1]
    x_te = np.empty((args.eval_n, n_t, n_pos), dtype=np.float32)
    fill_sequences(x_te, 0, det_te, d, r, p)
    logits = model.predict(x_te, batch_size=args.batch_size, verbose=0).reshape(-1)
    s_pred, s_correct = student_pred_and_correct(logits, truth)
    pL = float((~s_correct).mean())
    del x_te

    # MWPM decoded on these exact shots, never read from a stored baseline.
    import pymatching
    from eval_on_tail import build_circuit, mcnemar_from_correct
    dem = build_circuit(d, p, r).detector_error_model(decompose_errors=True)
    m_pred = pymatching.Matching.from_detector_error_model(dem).decode_batch(
        det_te, bit_packed_predictions=False, bit_packed_shots=False).astype(np.int8).reshape(-1)
    m_correct = (m_pred == truth)
    mwpm_pL = float((~m_correct).mean())
    mc = mcnemar_from_correct(s_correct, m_correct)

    ag = {k: '' for k in ('agree_all', 'agree_ambiguous', 'agree_confident', 'n_ambiguous',
                          'frac_ambiguous', 'student_p_L_ambiguous', 'teacher_p_L_ambiguous',
                          'teacher_tail_p_L')}
    t_logit_te = None
    if args.teacher_tail_cache:
        t_te = load_teacher_block(args.teacher_tail_cache, 'evaluation', sha_m, truth,
                                  e0, e1, args.teacher_units)
        if teacher_sha and t_te['stats']['teacher_weights_sha256'] != teacher_sha:
            raise SystemExit(f"{LOG} evaluation cache comes from a different teacher "
                             "checkpoint. STOP.")
        t_logit_te = t_te['logit']
        ag = teacher_agreement(s_pred, t_te['prob'], truth, tuple(args.ambiguous_band))

    print(f"{LOG} EVAL [{e0:,}, {e1:,})  student p_L={pL:.6f}  MWPM p_L={mwpm_pL:.6f}  "
          f"ratio={pL / mwpm_pL:.4f}x  teacher p_L={ag['teacher_tail_p_L']}", flush=True)
    print(f"{LOG}   McNemar vs MWPM: student-only={mc['rcnn_only']:,}  MWPM-only="
          f"{mc['mwpm_only']:,}  net={mc['net_rcnn_wins']:+,}  p_exact={mc['p_exact']:.3e}",
          flush=True)
    if ag['agree_all'] != '':
        print(f"{LOG}   agreement with teacher: all={ag['agree_all']}  ambiguous="
              f"{ag['agree_ambiguous']} (n={ag['n_ambiguous']:,})", flush=True)

    # Per-shot record. det_evts is left out (216 MB per run); shot_idx points back into
    # the primary pool, which is where it lives.
    np.savez_compressed(
        ck + '.per_shot_eval.npz',
        shot_idx=np.arange(e0, e1, dtype=np.int64), truth=truth,
        student_logit=logits.astype(np.float32), student_pred=s_pred,
        mwpm_pred=m_pred,
        teacher_logit=(t_logit_te if t_logit_te is not None else np.zeros(0, np.float32)),
        main_pool_flips_sha256=sha_m, teacher_weights_sha256=teacher_sha)

    # --- result row ------------------------------------------------------------------
    here = os.path.dirname(os.path.abspath(__file__))
    row = dict(
        experiment='gru_distill_20m', tag=args.tag,
        d=d, rounds=r, p=p, units=args.units, n_params=n_params, seed=args.seed,
        alpha=args.alpha, temperature=args.temperature,
        mode=('hard' if args.alpha >= 1.0 else 'distill' if args.alpha <= 0.0 else 'blend'),
        n_train_main=n_main, n_train_extra=n_extra, n_train=n_train,
        val_range=f'{v0}-{v1}', eval_range=f'{e0}-{e1}',
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
        patience=args.patience, epochs_ran=epochs_ran, best_epoch=best_epoch,
        best_val_loss=round(best_val, 6),
        p_L=round(pL, 6), mwpm_p_L=round(mwpm_pL, 6), ratio_vs_mwpm=round(pL / mwpm_pL, 4),
        both_right=mc['both_right'], student_only=mc['rcnn_only'],
        mwpm_only=mc['mwpm_only'], both_wrong=mc['both_wrong'],
        net_student_wins=mc['net_rcnn_wins'], mcnemar_p_exact=mc['p_exact'],
        teacher_p_L=ag['teacher_tail_p_L'],
        teacher_ratio_vs_mwpm=(round(float(ag['teacher_tail_p_L']) / mwpm_pL, 4)
                               if ag['teacher_tail_p_L'] != '' else ''),
        agree_all=ag['agree_all'], agree_ambiguous=ag['agree_ambiguous'],
        agree_confident=ag['agree_confident'], n_ambiguous=ag['n_ambiguous'],
        frac_ambiguous=ag['frac_ambiguous'],
        student_p_L_ambiguous=ag['student_p_L_ambiguous'],
        teacher_p_L_ambiguous=ag['teacher_p_L_ambiguous'],
        ambiguous_band=f'{args.ambiguous_band[0]}-{args.ambiguous_band[1]}',
        train_time_s=round(train_time, 1), total_time_s=round(time.time() - t_start, 1),
        teacher_column=teacher_column,
        teacher_weights_sha256=teacher_sha,
        teacher_main_cache=os.path.basename(args.teacher_main_cache or ''),
        teacher_extra_cache=os.path.basename(args.teacher_extra_cache or ''),
        main_pool=os.path.abspath(args.main_pool), main_pool_flips_sha256=sha_m,
        main_pool_gen_seed=meta_m['gen_seed'],
        extra_pool=(os.path.abspath(args.extra_train_pool) if use_extra else ''),
        extra_pool_flips_sha256=sha_x, extra_pool_gen_seed=meta_x.get('gen_seed', ''),
        code_sha_script=sha256_of(os.path.abspath(__file__))[:16],
        code_sha_studentmodels=sha256_of(os.path.join(here, 'StudentModels.py'))[:16],
        code_sha_train_student=sha256_of(os.path.join(here, 'train_student.py'))[:16],
        tf_version=tf.__version__, host=socket.gethostname(),
        run_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'))
    with open(ck + '.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)

    # COMPLETE last: every other artifact exists by now.
    for f in (ck + '.csv', ck + '.history.json', best_path, last_path,
              ck + '.per_shot_eval.npz', alignment_path):
        if not os.path.exists(f):
            raise SystemExit(f"{LOG} expected output {f} missing; not marking COMPLETE.")
    with open(os.path.join(args.out_dir, 'COMPLETE'), 'w') as fh:
        fh.write(args.tag + '\n')
    print(f"{LOG} wrote {ck}.csv (+ history, checkpoints, per-shot, alignment) -> COMPLETE",
          flush=True)


if __name__ == '__main__':
    run()
