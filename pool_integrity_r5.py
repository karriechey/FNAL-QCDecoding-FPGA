#!/usr/bin/env python3
# Created: 2026-08-06
# Last updated: 2026-08-06
"""Pool integrity check and MWPM baseline recomputation for the d=5, r=5 study.

Covers specification sections 2.2 and 2.4. Trains nothing and reads no sealed shot.

What it checks (section 2.4)
----------------------------
1. The pool exists, opens, and holds at least 20,400,000 shots.
2. The four declared partitions are mutually disjoint as indexed, and every range this
   script itself uses is routed through the shared guard in slice_guard_r5.py.
3. Shapes and dtypes are what the d=5, r=5 circuit implies: 145 measurement bits per
   shot, 120 detectors, one flip label.
4. Whether the shot ordering can be proven to be the generation ordering. The NPZ stores
   three parallel arrays and no global shot index column, so array position is the ONLY
   ordering information present in the file. Position alone does not prove generation
   order -- a shuffled pool would look identical. The check therefore compares the pool
   against its sidecar fingerprint written by gen_pool_r5.py, and if no fingerprint is
   present it reports that generation order cannot be independently proven rather than
   inferring it from position.

What it recomputes (section 2.2)
--------------------------------
MWPM on the exact val_report slice [20,100,000, 20,200,000), decoded from the detector
error model of the same circuit the pool was generated from. The measured value is the
baseline for this study. The value 0.084215 is printed alongside it as a sanity check
only; a numerical difference, including one larger than two binomial standard deviations,
is not on its own evidence of a slice mismatch, because the two numbers describe
different shot counts and possibly different slices. Provenance, configuration, indexing,
prediction-length and reproducibility inconsistencies are hard failures; a bare numerical
gap is reported, not fatal.

Outputs, into --out-dir:
    pool_integrity_r5.json          every check and its result
    mwpm_val_report_r5.json         the recomputed baseline and its provenance
    mwpm_val_report_per_shot.npz    global shot index, truth, MWPM prediction, correctness

Usage (on EAF, where the pool lives):
    export TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1
    python pool_integrity_r5.py \
        --pool ~/rcnn_threshold/pools_r5/data_d5_p0.010_r5_FORMAL.npz \
        --out-dir ~/rcnn_threshold/param_reduction_r5
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

from slice_guard_r5 import (assert_partitions_disjoint, assert_slice_allowed,
                            guard_summary, PARTITIONS, POOL_MIN_SHOTS,
                            VAL_REPORT_START, VAL_REPORT_STOP)

LOG = '[pool-check]'
D, P, ROUNDS = 5, 0.010, 5
# The number quoted in earlier work. Compared against, never used.
LEGACY_MWPM = 0.084215


def git_sha():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'],
                                        stderr=subprocess.DEVNULL).decode().strip()
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pool', required=True, help='the formal 20.4M-shot NPZ')
    ap.add_argument('--out-dir', default='rcnn_threshold/param_reduction_r5')
    ap.add_argument('--skip-array-checksum', action='store_true',
                    help='skip the whole-array flips checksum. That checksum hashes the '
                         'raw bytes of the entire flips array, which includes the sealed '
                         'block. It is a file-integrity check with no per-shot semantics '
                         'and no slice-level indexing, so section 2.3 permits it, but the '
                         'flag exists for anyone who wants the stricter reading.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report = dict(script=os.path.basename(__file__), git_sha=git_sha(),
                  utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  pool=os.path.abspath(args.pool), checks={})

    def check(name, ok, detail):
        report['checks'][name] = dict(pass_=bool(ok), detail=detail)
        print(f"{LOG} {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
        return ok

    # --- 0. partitions ------------------------------------------------------------------
    assert_partitions_disjoint()
    check('partitions_disjoint', True,
          ' '.join(f"{k}=[{v[0]:,},{v[1]:,})" for k, v in PARTITIONS.items()))

    # --- 1. pool present and large enough ----------------------------------------------
    if not os.path.exists(args.pool):
        raise SystemExit(f"{LOG} MISSING {args.pool}. STOP.")
    size_gb = os.path.getsize(args.pool) / 1024 ** 3
    print(f"{LOG} loading {args.pool} ({size_gb:.2f} GB) -- this maps the whole file, "
          f"which section 2.3 does not count as sealed-test access", flush=True)
    z = np.load(args.pool)
    keys = sorted(z.files)
    check('npz_keys', set(keys) >= {'measurements', 'det_evts', 'flips'}, str(keys))

    n_total = int(z['flips'].shape[0])
    check('pool_size', n_total >= POOL_MIN_SHOTS,
          f"{n_total:,} shots (need >= {POOL_MIN_SHOTS:,})")
    if n_total < POOL_MIN_SHOTS:
        raise SystemExit(f"{LOG} pool is too small for the declared partitions. STOP.")

    # --- 2. shapes and dtypes -----------------------------------------------------------
    n_meas_expected = ROUNDS * (D ** 2 - 1) + D ** 2      # 5*24 + 25 = 145
    n_det_expected = ROUNDS * (D ** 2 - 1)                # 120
    m_shape, e_shape, f_shape = (z['measurements'].shape, z['det_evts'].shape,
                                 z['flips'].shape)
    check('measurements_shape', m_shape[1] == n_meas_expected,
          f"{m_shape} (expected [N, {n_meas_expected}])")
    check('det_evts_shape', e_shape[1] == n_det_expected,
          f"{e_shape} (expected [N, {n_det_expected}])")
    check('rows_agree', m_shape[0] == e_shape[0] == f_shape[0],
          f"measurements={m_shape[0]:,} det_evts={e_shape[0]:,} flips={f_shape[0]:,}")

    # --- 3. ordering provenance ---------------------------------------------------------
    # The NPZ carries no global shot index column and no ordering checksum of its own, so
    # nothing inside the file distinguishes generation order from a shuffled view. The
    # only external evidence is the sidecar fingerprint gen_pool_r5.py writes.
    fp_path = args.pool.replace('.npz', '.fingerprint.json')
    has_index_column = any(k in z.files for k in ('shot_idx', 'global_idx', 'order'))
    fingerprint = None
    if os.path.exists(fp_path):
        fingerprint = json.load(open(fp_path))
    ordering_detail = []
    ordering_ok = False
    if has_index_column:
        ordering_detail.append('the NPZ stores an explicit shot index column')
        ordering_ok = True
    else:
        ordering_detail.append(
            'the NPZ stores NO global shot index and NO ordering checksum: generation '
            'order cannot be proven from the NPZ alone, and array position is not '
            'accepted as proof')
    if fingerprint is None:
        ordering_detail.append(f'no sidecar fingerprint at {fp_path}')
    else:
        ordering_detail.append(
            f"sidecar fingerprint present: gen_seed={fingerprint.get('gen_seed')} "
            f"n_total={fingerprint.get('n_total')} "
            f"partition_train={fingerprint.get('partition_train')} "
            f"partition_val={fingerprint.get('partition_val')} "
            f"partition_test_SEALED={fingerprint.get('partition_test_SEALED')}")
        # Byte-level file identity: does this NPZ still hold the flips the fingerprint
        # was written for? This hashes the whole array, sealed rows included, but reads
        # no individual sealed shot and makes no use of any sealed value.
        if not args.skip_array_checksum and fingerprint.get('flips_sha256'):
            t0 = time.time()
            got = hashlib.sha256(np.ascontiguousarray(z['flips']).tobytes()).hexdigest()
            same = (got == fingerprint['flips_sha256'])
            check('flips_sha256_matches_fingerprint', same,
                  f"{got[:16]}... vs recorded {fingerprint['flips_sha256'][:16]}... "
                  f"({time.time() - t0:.0f}s)")
            if not same:
                raise SystemExit(
                    f"{LOG} the pool's flips array does not match its own fingerprint. "
                    f"The file was regenerated or altered under the same name. STOP.")
            ordering_detail.append(
                'the flips array matches the fingerprint written at generation time, so '
                'this IS the generated file, in the order it was written')
            ordering_ok = True
    check('ordering_provable', ordering_ok, '; '.join(ordering_detail))

    # --- 4. MWPM on val_report ----------------------------------------------------------
    lo, hi = assert_slice_allowed(VAL_REPORT_START, VAL_REPORT_STOP,
                                 'MWPM baseline recomputation on val_report')
    n_report = hi - lo
    from eval_on_tail import build_circuit
    import pymatching
    circ = build_circuit(D, P, ROUNDS)
    circuit_cfg = dict(
        generator='circuit_generators.get_builtin_circuit',
        name='surface_code:rotated_memory_z', distance=D, rounds=ROUNDS,
        before_round_data_depolarization=P, after_reset_flip_probability=P,
        after_clifford_depolarization=P, before_measure_flip_probability=P,
        num_detectors=int(circ.num_detectors),
        num_observables=int(circ.num_observables))
    check('circuit_detectors', circ.num_detectors == n_det_expected,
          f"{circ.num_detectors} detectors (pool has {e_shape[1]} columns)")

    det_evts_report = z['det_evts'][lo:hi]
    truth = z['flips'][lo:hi].reshape(-1).astype(np.int8)
    check('report_slice_length', det_evts_report.shape[0] == n_report == truth.shape[0],
          f"det_evts={det_evts_report.shape[0]:,} truth={truth.shape[0]:,} "
          f"expected={n_report:,}")

    dem = circ.detector_error_model(decompose_errors=True)
    pym = pymatching.Matching.from_detector_error_model(dem)
    t0 = time.time()
    pred = pym.decode_batch(det_evts_report, bit_packed_predictions=False,
                            bit_packed_shots=False).astype(np.int8).reshape(-1)
    decode_s = time.time() - t0
    if pred.shape[0] != n_report:
        raise SystemExit(f"{LOG} MWPM returned {pred.shape[0]:,} predictions for "
                         f"{n_report:,} shots. STOP.")

    correct = (pred == truth)
    resid = int((~correct).sum())
    p_L = resid / n_report
    # Binomial standard error on the measured rate, for reporting the comparison honestly.
    se = math.sqrt(max(p_L * (1 - p_L), 1e-12) / n_report)
    base_rate = float(truth.mean())

    # Reproducibility: decoding the same slice a second time must give identical
    # predictions. A mismatch means the decoder is not deterministic and every downstream
    # paired test is unsound, so it is a hard failure.
    pred2 = pym.decode_batch(det_evts_report, bit_packed_predictions=False,
                             bit_packed_shots=False).astype(np.int8).reshape(-1)
    check('mwpm_reproducible', bool(np.array_equal(pred, pred2)),
          'the same slice decoded twice gives identical predictions')
    if not np.array_equal(pred, pred2):
        raise SystemExit(f"{LOG} MWPM is not reproducible on this slice. STOP.")

    delta = p_L - LEGACY_MWPM
    print()
    print(f"{LOG} MWPM on val_report [{lo:,}, {hi:,}):")
    print(f"{LOG}   p_L               = {p_L:.6f}  +/- {se:.6f} (binomial SE)")
    print(f"{LOG}   residual errors   = {resid:,} of {n_report:,}")
    print(f"{LOG}   base flip rate    = {base_rate:.6f}")
    print(f"{LOG}   decode time       = {decode_s / 60:.1f} min")
    print(f"{LOG}   legacy quoted 0.084215: difference {delta:+.6f} = {delta / se:+.1f} SE")
    print(f"{LOG}   That difference is a sanity note only. It is not evidence of a slice "
          f"mismatch on its own; the provenance, indexing, prediction-length and "
          f"reproducibility checks above are what would fail on a real mismatch.")
    print()

    per_shot = os.path.join(args.out_dir, 'mwpm_val_report_per_shot.npz')
    np.savez_compressed(
        per_shot,
        # Global pool indices, not positions inside the slice, so this file pairs directly
        # against every model's per-shot report.
        shot_idx=np.arange(lo, hi, dtype=np.int64),
        truth=truth, mwpm_pred=pred, mwpm_correct=correct,
        slice_start=lo, slice_stop=hi, d=D, p=P, rounds=ROUNDS,
        pool=os.path.abspath(args.pool), git_sha=report['git_sha'])

    mwpm_report = dict(
        slice=[lo, hi], n=n_report, p_L=p_L, binomial_se=se,
        residual_errors=resid, base_flip_rate=base_rate,
        decode_seconds=round(decode_s, 1),
        legacy_quoted_value=LEGACY_MWPM, difference_vs_legacy=delta,
        difference_in_se=delta / se,
        circuit=circuit_cfg, pool=os.path.abspath(args.pool),
        pool_fingerprint=fingerprint, git_sha=report['git_sha'],
        per_shot_file=os.path.abspath(per_shot),
        note='This measured value is the baseline for the parameter-reduction study. '
             'Do not substitute 0.084215.')
    mwpm_path = os.path.join(args.out_dir, 'mwpm_val_report_r5.json')
    with open(mwpm_path, 'w') as fh:
        json.dump(mwpm_report, fh, indent=2)

    report['mwpm_val_report'] = mwpm_report
    report['slice_guard_calls'] = guard_summary()
    all_pass = all(c['pass_'] for c in report['checks'].values())
    report['all_checks_passed'] = all_pass
    out_path = os.path.join(args.out_dir, 'pool_integrity_r5.json')
    with open(out_path, 'w') as fh:
        json.dump(report, fh, indent=2)

    print(guard_summary())
    print(f"{LOG} wrote {out_path}")
    print(f"{LOG} wrote {mwpm_path}")
    print(f"{LOG} wrote {per_shot}")
    print(f"{LOG} {'ALL CHECKS PASSED' if all_pass else 'ONE OR MORE CHECKS FAILED'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
