#!/usr/bin/env python3
# Created: 2026-08-06
# Last updated: 2026-08-06
"""Score one trained checkpoint on val_report and write its per-shot report.

Covers the scoring half of specification section 7.2. One script handles all three
architectures so the inference convention -- how truth and prediction are flattened, and
where the decision threshold sits -- is defined exactly once and cannot drift between the
RCNN path and the student path.

Inference convention, stated explicitly because the comparison depends on it:
  * the model emits one value per shot; truth and prediction are both reshaped to [N]
    before anything is compared;
  * the RCNN head ends in a sigmoid, so its output is already a probability;
  * the students emit a raw logit, which is converted with a sigmoid here;
  * the predicted class is `probability > 0.5` in both cases;
  * correctness is `predicted_class == truth`, and p_L is the mean of its complement.

The per-shot report keys `shot_idx` are GLOBAL pool indices, never positions inside the
slice. Without that, pairing runs against each other and against the MWPM baseline would
require re-inferencing every run.

The existing eval_on_tail.py and eval_student_on_tail.py are left untouched: they carry
neither the manifest hash nor the measured parameter count in their dumps, and they use
different key names for the same quantities, which this study's collation cannot accept.

Usage:
    export TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1
    python score_val_report_r5.py --arch gru --fraction 0.25 --seed 0 \
        --weights <ckpt>.best.weights.h5 --manifest <dir>/param_manifest_r5.json \
        --pool <pool>.npz --out-dir <run dir>
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

from slice_guard_r5 import (assert_deterministic_env, assert_slice_allowed,
                            assert_tf_version, guard_summary, is_smoke,
                            VAL_REPORT_START, VAL_REPORT_STOP,
                            VAL_SELECT_START, VAL_SELECT_STOP)

LOG = '[score]'
D, P, ROUNDS, KERNEL, NPOL = 5, 0.010, 5, 3, 2


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_sha():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'],
                                        stderr=subprocess.DEVNULL).decode().strip()
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def manifest_entry(manifest, arch, fraction):
    """The one frozen configuration for this architecture and budget rung."""
    hits = [c for c in manifest['configurations']
            if c['architecture'] == arch and abs(c['target_fraction'] - fraction) < 1e-9]
    if len(hits) != 1:
        raise SystemExit(f"{LOG} expected exactly one manifest row for {arch} at "
                         f"{fraction}, found {len(hits)}. STOP.")
    row = hits[0]
    if row['reachable'] != 'YES':
        raise SystemExit(f"{LOG} {arch} at {fraction} is marked unreachable in the "
                         f"manifest: {row['reason']}. STOP.")
    return row


def build_from_manifest(arch, cargs, np_mod):
    """Rebuild the exact model the manifest describes, ready for load_weights."""
    if arch == 'rcnn':
        from CNNModel import FullRCNNModel
        m = FullRCNNModel('ZL', D, cargs['kernel'], ROUNDS,
                          [cargs['hidden']] * cargs['hidden_layers'],
                          npol=cargs['npol'], stop_round=None,
                          has_nonuniform_response=False, do_all_data_qubits=False,
                          return_all_rounds=False)
        det_bits = np_mod.zeros((2, ROUNDS * (D ** 2 - 1)), dtype=np_mod.float32)
        det_evts = np_mod.zeros((2, ROUNDS * (D ** 2 - 1)), dtype=np_mod.float32)
        _ = m([det_bits, det_evts])
        return m
    from StudentModels import build_student
    if arch == 'gru':
        return build_student('gru', d=D, rounds=ROUNDS, inputs='evts',
                             units=cargs['units'], hidden=tuple(cargs['hidden']))
    return build_student('mlp', d=D, rounds=ROUNDS, inputs='evts',
                         hidden=tuple(cargs['hidden']))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arch', required=True, choices=['rcnn', 'gru', 'mlp'])
    ap.add_argument('--fraction', required=True, type=float)
    ap.add_argument('--seed', required=True, type=int)
    ap.add_argument('--n-train', type=int, default=10_000_000,
                    help='recorded in the report; the training prefix this run used')
    ap.add_argument('--weights', required=True,
                    help='the val_select-best checkpoint to reload and score')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--also-score-val-select', action='store_true',
                    help='additionally score val_select, so the pilot can confirm that '
                         'the reloaded checkpoint reproduces the loss it was selected on. '
                         'val_select is a selection slice, so nothing here is reported '
                         'from it.')
    ap.add_argument('--run-tag', default=None,
                    help='filename stem for the outputs. The launcher passes the same tag '
                         'it uses for the checkpoint and for its own completion check, so '
                         'both sides name the identical file. Without it a tag is derived '
                         'here, which is fine for a one-off but not for the sweep.')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    assert_deterministic_env()
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import numpy as np
    import tensorflow as tf
    assert_tf_version(tf)
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = json.load(open(args.manifest))
    mhash = sha256_file(args.manifest)
    if list(manifest['partitions']['val_report']) != [VAL_REPORT_START, VAL_REPORT_STOP]:
        raise SystemExit(
            f"{LOG} manifest val_report={manifest['partitions']['val_report']} but this "
            f"process uses [{VAL_REPORT_START}, {VAL_REPORT_STOP}). One of them is "
            f"smoke-scaled (SLICE_GUARD_SMOKE_DIVISOR). STOP.")
    row = manifest_entry(manifest, args.arch, args.fraction)
    cargs = json.loads(row['constructor_args'])
    expected_params = int(row['actual_params'])
    print(f"{LOG} manifest {os.path.basename(args.manifest)} sha256={mhash}")
    print(f"{LOG} {args.arch} fraction={args.fraction} seed={args.seed} "
          f"config={cargs} expected_params={expected_params:,}")

    model = build_from_manifest(args.arch, cargs, np)
    n_params = int(model.count_params())
    if n_params != expected_params:
        raise SystemExit(f"{LOG} rebuilt model has {n_params:,} parameters but the frozen "
                         f"manifest says {expected_params:,}. The checkpoint and the "
                         f"manifest describe different models. STOP.")
    print(f"{LOG} rebuilt model: {n_params:,} parameters, matches the manifest")
    if not os.path.exists(args.weights):
        raise SystemExit(f"{LOG} MISSING checkpoint {args.weights}. STOP.")
    model.load_weights(args.weights)
    print(f"{LOG} loaded {args.weights} (sha256 {sha256_file(args.weights)[:16]}...)")

    # --- data ---------------------------------------------------------------------------
    z = np.load(args.pool)
    from types_cfg import get_types
    binary_t, _, idx_t, _ = get_types(D, ROUNDS, KERNEL)

    def slice_inputs(lo, hi, purpose):
        lo, hi = assert_slice_allowed(lo, hi, purpose)
        det_evts = z['det_evts'][lo:hi].astype(binary_t)
        truth = z['flips'][lo:hi].reshape(-1).astype(np.int8)
        if args.arch == 'rcnn':
            from circuit_partition import split_measurements
            det_bits, _, _ = split_measurements(
                z['measurements'][lo:hi].astype(binary_t), D, idx_t)
            x = [det_bits, det_evts]
        else:
            from StudentModels import assemble_features
            x = assemble_features(None, det_evts, 'evts', student=args.arch,
                                  d=D, rounds=ROUNDS, p=P)
        return lo, hi, x, truth

    def score(lo, hi, purpose):
        lo, hi, x, truth = slice_inputs(lo, hi, purpose)
        t0 = time.time()
        out = model.predict(x, batch_size=args.batch_size, verbose=0)
        out = np.asarray(out).reshape(-1)
        # One decision convention for all three architectures, defined here only.
        prob = out if args.arch == 'rcnn' else 1.0 / (1.0 + np.exp(-out.astype(np.float64)))
        prob = prob.astype(np.float32)
        if prob.shape[0] != hi - lo:
            raise SystemExit(f"{LOG} model returned {prob.shape[0]:,} predictions for "
                             f"{hi - lo:,} shots. STOP.")
        pred = (prob > 0.5).astype(np.int8)
        correct = (pred == truth)
        p_L = float((~correct).mean())
        print(f"{LOG} {purpose}: p_L={p_L:.6f}  errors={int((~correct).sum()):,}/"
              f"{hi - lo:,}  predicted-positive rate={float(pred.mean()):.4f}  "
              f"prob std={float(prob.std()):.4f}  ({time.time() - t0:.0f}s)")
        return dict(lo=lo, hi=hi, prob=prob, pred=pred, truth=truth, correct=correct,
                    p_L=p_L, predicted_positive_rate=float(pred.mean()),
                    prob_std=float(prob.std()), prob_mean=float(prob.mean()))

    rep = score(VAL_REPORT_START, VAL_REPORT_STOP, 'val_report scoring')

    sel = None
    if args.also_score_val_select:
        # Reported only as a reload-consistency diagnostic in the pilot. No study result
        # is taken from val_select.
        sel = score(VAL_SELECT_START, VAL_SELECT_STOP,
                    'val_select reload check (diagnostic only)')

    # --- per-shot report ----------------------------------------------------------------
    tag = args.run_tag or (f"{args.arch}_frac{args.fraction:g}_p{n_params}_seed{args.seed}"
                           f"_ntr{args.n_train}")
    per_shot = os.path.join(args.out_dir, f'pershot_valreport_{tag}.npz')
    if os.path.exists(per_shot):
        raise SystemExit(f"{LOG} {per_shot} exists. Results are append-only; move the old "
                         f"file aside rather than overwriting it. STOP.")
    np.savez_compressed(
        per_shot,
        shot_idx=np.arange(rep['lo'], rep['hi'], dtype=np.int64),  # GLOBAL pool indices
        truth=rep['truth'],
        predicted_class=rep['pred'],
        predicted_probability=rep['prob'],
        correctness=rep['correct'],
        architecture=args.arch, target_fraction=args.fraction,
        actual_params=n_params, seed=args.seed, n_train=args.n_train,
        checkpoint_path=os.path.abspath(args.weights),
        checkpoint_sha256=sha256_file(args.weights),
        manifest_sha256=mhash, git_sha=git_sha(),
        slice_start=rep['lo'], slice_stop=rep['hi'],
        val_select_slice=[VAL_SELECT_START, VAL_SELECT_STOP],
        pool=os.path.abspath(args.pool),
        smoke=is_smoke(),
        d=D, p=P, rounds=ROUNDS)

    summary = dict(
        architecture=args.arch, target_fraction=args.fraction, actual_params=n_params,
        seed=args.seed, n_train=args.n_train,
        val_report_slice=[rep['lo'], rep['hi']], p_L=rep['p_L'],
        residual_errors=int((~rep['correct']).sum()),
        predicted_positive_rate=rep['predicted_positive_rate'],
        prob_mean=rep['prob_mean'], prob_std=rep['prob_std'],
        val_select_slice=[VAL_SELECT_START, VAL_SELECT_STOP],
        val_select_diagnostic=(None if sel is None else
                               dict(p_L=sel['p_L'], prob_std=sel['prob_std'])),
        checkpoint=os.path.abspath(args.weights),
        checkpoint_sha256=sha256_file(args.weights),
        manifest_sha256=mhash, git_sha=git_sha(),
        per_shot_file=os.path.abspath(per_shot),
        pool=os.path.abspath(args.pool),
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        smoke=is_smoke(),
        slice_guard=guard_summary())
    summary_path = os.path.join(args.out_dir, f'valreport_{tag}.json')
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(guard_summary())
    print(f"{LOG} wrote {per_shot}")
    print(f"{LOG} wrote {summary_path}")
    print(f"{LOG} SCORING COMPLETE {tag}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
