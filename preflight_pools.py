#!/usr/bin/env python3
# Created: 2026-08-20
# Last updated: 2026-08-20
"""Assert a pool and its MWPM baseline are what a run assumes, before any GPU time.

  python preflight_pools.py --pool <pool.npz> --d 5 --rounds 5 --p 0.004 \
      --baseline <mwpm_eval_block.json>

Checks, all fatal:
  fingerprint exists beside the pool, and its d / rounds / p match the run
  n_total >= the layout's end (default 19M)
  detector and measurement widths equal r(d^2-1) and r(d^2-1)+d^2
  train / validation / evaluation / sealed indices fit inside the pool and are disjoint
  the baseline JSON covers exactly [eval_start, eval_stop), on this pool's flips SHA
  the baseline's residual count and n agree with its own p_L

Exit status is 0 only when every check passes, so a caller can gate a lane on it.
"""
import argparse
import json
import os
import sys

import numpy as np

FAIL = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--rounds', type=int, required=True)
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--baseline', default=None, help='mwpm_on_block.py JSON for the eval block')
    ap.add_argument('--n-train-max', type=int, default=15000000)
    ap.add_argument('--val-start', type=int, default=15000000)
    ap.add_argument('--val-n', type=int, default=200000)
    ap.add_argument('--eval-start', type=int, default=15200000)
    ap.add_argument('--eval-n', type=int, default=1800000)
    ap.add_argument('--sealed-start', type=int, default=17000000)
    ap.add_argument('--pool-min', type=int, default=19000000)
    args = ap.parse_args()

    print(f"[preflight] {args.pool}")
    if not check('pool exists', os.path.exists(args.pool)):
        sys.exit(1)

    fp = args.pool.replace('.npz', '.fingerprint.json')
    if not check('fingerprint exists', os.path.exists(fp), fp):
        sys.exit(1)
    m = json.load(open(fp))
    check('fingerprint geometry', (m['d'], m['rounds'], float(m['p'])) ==
          (args.d, args.rounds, args.p),
          f"file says d={m['d']} r={m['rounds']} p={m['p']}")
    check('gen_seed recorded', m.get('gen_seed') is not None, f"gen_seed={m.get('gen_seed')}")

    z = np.load(args.pool, mmap_mode='r')
    n_total = int(z['det_evts'].shape[0])
    n_det, n_meas = int(z['det_evts'].shape[1]), int(z['measurements'].shape[1])
    want_det = args.rounds * (args.d ** 2 - 1)
    want_meas = want_det + args.d ** 2

    check('n_total', n_total >= args.pool_min, f"{n_total:,} shots")
    check('detector width', n_det == want_det, f"{n_det} (expected {want_det})")
    check('measurement width', n_meas == want_meas, f"{n_meas} (expected {want_meas})")
    check('fingerprint n_total agrees', int(m['n_total']) == n_total,
          f"fingerprint {m['n_total']:,} vs array {n_total:,}")

    v1, e1 = args.val_start + args.val_n, args.eval_start + args.eval_n
    check('train prefix clears validation', args.n_train_max <= args.val_start,
          f"[0, {args.n_train_max:,}) then validation at {args.val_start:,}")
    check('validation clears evaluation', v1 <= args.eval_start)
    check('evaluation clears sealed', e1 <= args.sealed_start)
    check('sealed inside pool', args.sealed_start < n_total,
          f"sealed [{args.sealed_start:,}, {n_total:,})")

    if args.baseline:
        if check('baseline exists', os.path.exists(args.baseline), args.baseline):
            b = json.load(open(args.baseline))
            check('baseline block start', int(b['block_start']) == args.eval_start,
                  f"{b['block_start']:,}")
            check('baseline block stop', int(b['block_stop']) == e1, f"{b['block_stop']:,}")
            check('baseline n', int(b['n']) == args.eval_n, f"{b['n']:,} shots")
            check('baseline geometry', (b['d'], b['rounds'], float(b['p'])) ==
                  (args.d, args.rounds, args.p))
            sha_b, sha_p = str(b.get('flips_sha256', '')), str(m.get('flips_sha256', ''))
            check('baseline pool identity', bool(sha_b) and sha_b == sha_p,
                  f"baseline {sha_b[:16]} vs pool {sha_p[:16]}")
            resid, n, p_L = int(b['residual_errors']), int(b['n']), float(b['mwpm_p_L'])
            check('baseline p_L consistent', abs(resid / n - p_L) < 1e-9,
                  f"{resid:,}/{n:,} = {resid / n:.6f}, recorded {p_L:.6f}")
            se = (p_L * (1 - p_L) / n) ** 0.5
            print(f"  [baseline] MWPM p_L={p_L:.6f} +/- {se:.6f}  "
                  f"({resid:,} failures of {n:,})")

    if FAIL:
        print(f"[preflight] {len(FAIL)} FAILED: {', '.join(FAIL)}")
        sys.exit(1)
    print("[preflight] all checks passed")


if __name__ == '__main__':
    main()
