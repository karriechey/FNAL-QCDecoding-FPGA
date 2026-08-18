#!/usr/bin/env python3
# Created: 2026-08-18
# Last updated: 2026-08-18
"""Write one auditable JSON describing the exact data geometry a run will use.

Every past attempt to reconstruct, months later, which shots belonged to which evaluation
block has cost more time than generating the pool did. This script produces a single file,
written next to the experiment outputs BEFORE training starts, that answers the question
without needing to re-derive anything from a launcher script or a shell history.

It records, for one pool and one partition layout:

  pool path, byte size, generation seed (read from the generator's fingerprint sidecar
    when one exists, otherwise supplied with --gen-seed)
  a content fingerprint of the pool: sha256 of `flips`, and of `det_evts`, so a pool
    regenerated on a different machine can be compared against this one directly
  the literal half-open index ranges [start, stop) of the training prefix, the monitoring
    validation block, the final shared evaluation block, and the sealed test block
  an explicit disjointness check across all four, which fails the script rather than
    writing a file describing an invalid layout
  MWPM p_L decoded on the EXACT evaluation block named here, not read from any stored
    baseline CSV -- a stored baseline describes whichever block its generator decoded,
    which is repeatedly not the block being asked about
  the logical-flip base rate on that same evaluation block, and on the whole pool

MWPM is decoded on CPU and is the expensive part: roughly 10-20 minutes for 1.8M shots at
d=9. Pass --skip-mwpm to write the geometry alone when the baseline is already known and
recorded, but note that the resulting file then carries no independent check that the pool
is the one it claims to be.

Hashing 19M x 720 int8 (13.7 GB) takes a few minutes. --quick-hash hashes an evenly spaced
subsample instead, which is enough to catch a wrong pool but not enough to prove two pools
are bit-identical; the field name in the output says which was done.

Usage:

  $PY write_run_geometry.py --pool <pool.npz> --d 9 --p 0.004 --rounds 9 \
      --train-start 0 --train-stop 10000000 \
      --val-start 15000000 --val-stop 15200000 \
      --eval-start 15200000 --eval-stop 17000000 \
      --test-start 17000000 --test-stop 19000000 \
      --out results/<run>/run_geometry.json
"""
import argparse
import hashlib
import json
import os
import platform
import socket
import time

import numpy as np


def sha256_array(a, quick=False, sample=2_000_000):
    """Content hash of an array.

    Full mode hashes every byte, in row chunks so no copy of the whole array is made.
    Quick mode hashes an evenly spaced subsample of rows; that detects a different pool
    but does not prove two pools are identical, so the caller labels the field accordingly.
    """
    h = hashlib.sha256()
    n = a.shape[0]
    if quick and n > sample:
        idx = np.linspace(0, n - 1, sample).astype(np.int64)
        h.update(np.ascontiguousarray(a[idx]).tobytes())
        return h.hexdigest(), 'subsampled'
    step = max(1, 1_000_000)
    for i in range(0, n, step):
        h.update(np.ascontiguousarray(a[i:i + step]).tobytes())
    return h.hexdigest(), 'full'


def binom_err(p, n):
    """Standard error of a binomial proportion. The precision on p_L is governed by the
    number of failures actually observed, so this is reported alongside the count."""
    if n == 0:
        return float('nan')
    return (p * (1.0 - p) / n) ** 0.5


def check_disjoint(blocks):
    """blocks: list of (name, start, stop). Raise unless every pair is disjoint and every
    range is non-empty and correctly ordered."""
    for name, s, e in blocks:
        if e <= s:
            raise SystemExit(f"[geom] block {name} is empty or inverted: [{s}, {e}). STOP.")
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            n1, s1, e1 = blocks[i]
            n2, s2, e2 = blocks[j]
            if s1 < e2 and s2 < e1:
                raise SystemExit(
                    f"[geom] blocks overlap: {n1} [{s1:,}, {e1:,}) and "
                    f"{n2} [{s2:,}, {e2:,}). STOP.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--rounds', type=int, required=True)
    ap.add_argument('--train-start', type=int, default=0)
    ap.add_argument('--train-stop', type=int, required=True)
    ap.add_argument('--val-start', type=int, required=True)
    ap.add_argument('--val-stop', type=int, required=True)
    ap.add_argument('--eval-start', type=int, required=True)
    ap.add_argument('--eval-stop', type=int, required=True)
    ap.add_argument('--test-start', type=int, default=None)
    ap.add_argument('--test-stop', type=int, default=None)
    ap.add_argument('--gen-seed', type=int, default=None,
                    help='generation seed, when no .fingerprint.json sidecar is present')
    ap.add_argument('--experiment', default=None,
                    help='free-text label recorded in the file, e.g. the run identity')
    ap.add_argument('--quick-hash', action='store_true',
                    help='hash an evenly spaced subsample instead of every byte')
    ap.add_argument('--skip-mwpm', action='store_true',
                    help='write the geometry without decoding MWPM (much faster, but the '
                         'file then carries no independent check of the pool contents)')
    ap.add_argument('--out', required=True, help='path of the JSON to write')
    args = ap.parse_args()

    if not os.path.exists(args.pool):
        raise SystemExit(f"[geom] MISSING pool {args.pool}. STOP.")

    print(f"[geom] pool {args.pool}")
    z = np.load(args.pool)
    evts, flips = z['det_evts'], z['flips']
    N = int(evts.shape[0])
    print(f"[geom] shots {N:,}   det_evts {evts.shape} {evts.dtype}   flips {flips.shape}")

    blocks = [('train', args.train_start, args.train_stop),
              ('validation', args.val_start, args.val_stop),
              ('evaluation', args.eval_start, args.eval_stop)]
    if args.test_start is not None and args.test_stop is not None:
        blocks.append(('sealed_test', args.test_start, args.test_stop))
    for name, s, e in blocks:
        if e > N:
            raise SystemExit(f"[geom] block {name} [{s:,}, {e:,}) exceeds pool size "
                             f"{N:,}. STOP.")
    check_disjoint(blocks)
    print("[geom] partitions, all disjoint:")
    for name, s, e in blocks:
        print(f"[geom]   {name:<12} [{s:>12,}, {e:>12,})   n={e - s:,}")

    # Generation provenance. The generator writes a sidecar; prefer it over anything typed
    # on the command line, so the recorded seed is the one actually used.
    side = args.pool.replace('.npz', '.fingerprint.json')
    fingerprint_sidecar = None
    gen_seed = args.gen_seed
    if os.path.exists(side):
        with open(side) as fh:
            fingerprint_sidecar = json.load(fh)
        gen_seed = fingerprint_sidecar.get('gen_seed', gen_seed)
        print(f"[geom] generator sidecar {side}  gen_seed={gen_seed}")
    else:
        print(f"[geom] no generator sidecar at {side}; gen_seed from --gen-seed: {gen_seed}")

    t0 = time.time()
    flips_sha, flips_mode = sha256_array(flips, quick=args.quick_hash)
    evts_sha, evts_mode = sha256_array(evts, quick=args.quick_hash)
    print(f"[geom] flips sha256    ({flips_mode}) {flips_sha}")
    print(f"[geom] det_evts sha256 ({evts_mode}) {evts_sha}   ({time.time() - t0:.0f}s)")

    truth_all = np.asarray(flips).reshape(-1)
    base_pool = float(truth_all.mean())
    ev_truth = truth_all[args.eval_start:args.eval_stop]
    n_eval = int(args.eval_stop - args.eval_start)
    base_eval = float(ev_truth.mean())
    print(f"[geom] base rate  whole pool {base_pool:.5f}   evaluation block "
          f"{base_eval:.5f} +/- {binom_err(base_eval, n_eval):.5f}")

    mwpm = None
    if not args.skip_mwpm:
        import pymatching
        from eval_on_tail import build_circuit
        circ = build_circuit(args.d, args.p, args.rounds)
        dem = circ.detector_error_model(decompose_errors=True)
        pym = pymatching.Matching.from_detector_error_model(dem)
        t1 = time.time()
        pred = pym.decode_batch(evts[args.eval_start:args.eval_stop],
                                bit_packed_predictions=False,
                                bit_packed_shots=False).astype(np.int8).reshape(-1)
        dt = time.time() - t1
        fails = int((pred != ev_truth).sum())
        p_L = fails / n_eval
        err = binom_err(p_L, n_eval)
        mwpm = dict(p_L=p_L, stderr=err, failures=fails, n=n_eval,
                    block=[args.eval_start, args.eval_stop],
                    decode_seconds=round(dt, 1),
                    source='decoded_here_on_the_named_evaluation_block')
        print(f"[geom] MWPM on [{args.eval_start:,}, {args.eval_stop:,}): "
              f"p_L={p_L:.6f} +/- {err:.6f}  ({fails:,} failures, {dt:.0f}s)")
        if fails < 500:
            print(f"[geom] WARNING: only {fails} MWPM failures on the evaluation block -- "
                  f"the p_L estimate is count-limited.")

    meta = dict(
        written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        experiment=args.experiment,
        host=socket.gethostname(),
        platform=platform.platform(),
        machine=platform.machine(),
        pool=dict(path=os.path.abspath(args.pool),
                  size_bytes=os.path.getsize(args.pool),
                  n_total=N,
                  det_evts_shape=list(evts.shape),
                  flips_shape=list(flips.shape),
                  gen_seed=gen_seed,
                  flips_sha256=flips_sha, flips_hash_mode=flips_mode,
                  det_evts_sha256=evts_sha, det_evts_hash_mode=evts_mode,
                  generator_sidecar=fingerprint_sidecar),
        partitions={name: dict(start=s, stop=e, n=e - s) for name, s, e in blocks},
        partitions_disjoint=True,
        base_rate=dict(whole_pool=base_pool, evaluation_block=base_eval,
                       evaluation_block_stderr=binom_err(base_eval, n_eval)),
        mwpm=mwpm,
        d=args.d, p=args.p, rounds=args.rounds,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f"[geom] wrote {args.out}")


if __name__ == '__main__':
    main()
