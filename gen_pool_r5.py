#!/usr/bin/env python3
# Created: 2026-08-03
# Last modified: 2026-08-04
"""Generate the formal d=5, r=5 pool with three disjoint partitions.

Partitions (Option A -- validation is its own block, not carved from training):
  [0, n_train)                       training pool for every nested ladder rung
  [n_train, n_train+n_val)           dedicated validation, for early stopping,
                                     checkpoint selection and seed ranking
  [n_train+n_val, +n_test)           sealed test, read once at reporting time

This is stricter than the exploratory r=3 protocol, which used Keras
validation_split=0.2 from inside the training prefix. The difference is deliberate and
must be recorded, not presented as the same design.

The existing pools/data_d5_p0.010_r5.npz holds 5.01M shots with a 10k tail: too coarse to
resolve parity, and 5M training capacity does not reach the top of the ladder.

MWPM is decoded on the VALIDATION partition only. The sealed test partition is not read
here -- its baseline is decoded at reporting time, so nothing in the build touches it.

Writes, alongside the pool:
  mwpm_baseline.csv row      MWPM on the validation partition
  <pool>.fingerprint.json    partition indices, flips sha256, shapes, generation seed

r=5 costs more than r=3 per shot: 145 measurement bits vs 97, 120 detectors vs 72, so
~1.5x the bytes and the sampling time. At 20.2M shots expect roughly 5 GB.

  python gen_pool_r5.py                # 20.0M train + 200k val + 200k test = 20.4M
  python gen_pool_r5.py --dry-run      # sizes and timing estimate only
"""
import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np

from eval_on_tail import build_circuit
from generate_pools import sample_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--n-train', type=int, default=20000000, help='training partition')
    ap.add_argument('--n-val', type=int, default=200000, help='dedicated validation')
    ap.add_argument('--n-test', type=int, default=200000, help='sealed test')
    ap.add_argument('--gen-seed', type=int, default=42)
    ap.add_argument('--out-dir', default=os.path.expanduser('~/rcnn_threshold/pools_r5'))
    ap.add_argument('--tag', default='FORMAL', help='marks the pool filename')
    ap.add_argument('--dry-run', action='store_true',
                    help='estimate size and time from a 100k probe, write nothing')
    args = ap.parse_args()
    args.n = args.n_train + args.n_val + args.n_test
    VAL0, VAL1 = args.n_train, args.n_train + args.n_val
    TEST0, TEST1 = VAL1, VAL1 + args.n_test

    d, p, r = args.d, args.p, args.rounds
    circ = build_circuit(d, p, r)
    n_det = circ.num_detectors
    n_meas = r * (d ** 2 - 1) + d ** 2
    print(f"[gen-r5] d={d} p={p} rounds={r}: {n_meas} measurements, {n_det} detectors")

    if args.dry_run:
        t0 = time.time()
        probe_n = 100000
        m, e, f = sample_pool(circ, probe_n, args.gen_seed)
        dt = time.time() - t0
        per_shot = (m.nbytes + e.nbytes + f.nbytes) / probe_n
        print(f"[gen-r5] probe {probe_n:,} shots in {dt:.1f}s "
              f"({per_shot:.0f} bytes/shot uncompressed)")
        print(f"[gen-r5] estimate for {args.n:,}: "
              f"{dt * args.n / probe_n / 60:.0f} min sampling, "
              f"{per_shot * args.n / 1024**3:.1f} GB uncompressed")
        print("[gen-r5] dry run, nothing written")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    fn = os.path.join(args.out_dir, f'data_d{d}_p{p:.3f}_r{r}_{args.tag}.npz')
    if os.path.exists(fn):
        raise SystemExit(f"[gen-r5] {fn} exists -- refusing to overwrite. "
                         f"Move it aside if you mean to regenerate.")

    t0 = time.time()
    measurements, det_evts, flips = sample_pool(circ, args.n, args.gen_seed)
    print(f"[gen-r5] sampled {args.n:,} shots in {(time.time() - t0) / 60:.1f} min  "
          f"meas{measurements.shape} det{det_evts.shape}", flush=True)

    # MWPM on the VALIDATION partition. The sealed test block is deliberately not read.
    import pymatching
    te = slice(VAL0, VAL1)
    dem = circ.detector_error_model(decompose_errors=True)
    pym = pymatching.Matching.from_detector_error_model(dem)
    t1 = time.time()
    pred = pym.decode_batch(det_evts[te], bit_packed_predictions=False,
                            bit_packed_shots=False).astype(np.int8).reshape(-1)
    truth = flips[te].reshape(-1)
    resid = int((pred != truth).sum())
    mwpm_pL = resid / args.n_val
    print(f"[gen-r5] MWPM on the {args.n_val:,} validation partition: p_L={mwpm_pL:.6f} "
          f"({resid:,} residual errors, {(time.time() - t1) / 60:.1f} min)", flush=True)
    if resid < 500:
        print(f"[gen-r5] WARNING: only {resid} residual errors -- the p_L "
              f"estimate is count-limited. Consider a larger tail.", flush=True)

    np.savez(fn, measurements=measurements, det_evts=det_evts, flips=flips)
    size_gb = os.path.getsize(fn) / 1024**3

    flips_sha = hashlib.sha256(np.ascontiguousarray(flips).tobytes()).hexdigest()
    meta = dict(path=os.path.abspath(fn), d=d, p=p, rounds=r, n_total=args.n,
                partition_train=[0, args.n_train],
                partition_val=[VAL0, VAL1],
                partition_test_SEALED=[TEST0, TEST1],
                protocol='Option A: dedicated validation block, disjoint from training '
                         'and from the sealed test. Stricter than the r=3 exploratory '
                         'protocol, which used Keras validation_split inside the prefix.',
                n_test=args.n_test, n_val=args.n_val, train_capacity=args.n_train,
                gen_seed=args.gen_seed, flips_sha256=flips_sha,
                measurements_shape=str(measurements.shape),
                det_evts_shape=str(det_evts.shape), mwpm_p_L=mwpm_pL,
                mwpm_resid_errors_test=resid, size_gb=round(size_gb, 2))
    with open(fn.replace('.npz', '.fingerprint.json'), 'w') as jf:
        json.dump(meta, jf, indent=2)

    baseline = os.path.join(args.out_dir, 'mwpm_baseline.csv')
    fields = ["d", "p", "rounds", "n_total", "n_test", "train_capacity", "mwpm_p_L",
              "mwpm_resid_errors_test", "base_rate_test", "base_rate_pool", "gen_seed"]
    new = not os.path.exists(baseline)
    with open(baseline, 'a', newline='') as bf:
        w = csv.DictWriter(bf, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(dict(d=d, p=p, rounds=r, n_total=args.n, n_test=args.n_test,
                        train_capacity=args.n_train, mwpm_p_L=round(mwpm_pL, 6),
                        mwpm_resid_errors_test=resid,
                        base_rate_test=round(float(flips[te].mean()), 5),
                        base_rate_pool=round(float(flips.mean()), 5),
                        gen_seed=args.gen_seed))

    print(f"[gen-r5] wrote {fn}  ({size_gb:.1f} GB)")
    print(f"[gen-r5] flips sha256 {flips_sha[:16]}  -> {fn.replace('.npz', '.fingerprint.json')}")
    print(f"[gen-r5] baseline row -> {baseline}  (validation partition)")
    print(f"[gen-r5] partitions: train [0, {args.n_train:,})  val [{VAL0:,}, {VAL1:,})  "
          f"test SEALED [{TEST0:,}, {TEST1:,})")
    print(f"[gen-r5] total {(time.time() - t0) / 60:.1f} min")


if __name__ == '__main__':
    main()
