#!/usr/bin/env python3
# Created: 2026-08-13
# Last updated: 2026-08-13
"""Report the statistics that decide whether a pool can support stable training.

Written for the below-threshold design decision: p=0.004 has to be far enough below
threshold that distance suppresses errors, while still producing enough logical failures
that p_L can be measured to a useful precision.

Four numbers decide that, and this script reports all four:

  base rate        fraction of shots whose logical observable flips before any decoding.
                   This is the label the network learns, so it sets how many positive
                   examples a training block contains. A base rate near 0 leaves almost
                   nothing to learn from.
  MWPM p_L         the residual error rate after matching, on a named block. This is the
                   target the decoder is chasing and the denominator of xMWPM.
  observed errors  MWPM failures actually seen in the block. p_L is a binomial proportion,
                   so its precision is governed by this count rather than by block size.
  projected errors what that p_L implies for a 200k block and a 2M block, which is the
                   practical question when sizing a validation or test partition.

The relative uncertainty on a binomial proportion is approximately 1/sqrt(k) for k
observed failures, so k=100 gives about 10 percent, k=1000 about 3 percent, and k=10000
about 1 percent. A comparison between two decoders needs their difference to exceed the
combined uncertainty, which is why the observed count matters more than p_L alone.

MWPM is decoded here rather than read from mwpm_baseline.csv, so the number describes the
exact shots named on the command line. A stored baseline describes whichever block the
generator decoded, which is not always the block being asked about.

Run on the pod, pinned interpreter:

  $PY pool_diagnostics.py --pool <pool.npz> --d 9 --p 0.004 --rounds 9
  $PY pool_diagnostics.py --pool <pool.npz> --d 5 --p 0.004 --rounds 5 --block-n 70000
"""
import argparse
import time

import numpy as np


def binom_err(p, n):
    """Standard error of a binomial proportion."""
    if n == 0:
        return float('nan')
    return (p * (1.0 - p) / n) ** 0.5


def report_projection(p_L, sizes=(200000, 2000000)):
    """Expected MWPM failures, and the resulting precision, for candidate block sizes.

    Reported because a block that resolves p_L poorly cannot support a decoder
    comparison, however large the pool behind it is.
    """
    print(f"  {'block size':>12} {'expected failures':>18} {'rel. uncertainty':>18}")
    for n in sizes:
        k = p_L * n
        rel = (1.0 / k ** 0.5) if k > 0 else float('nan')
        print(f"  {n:>12,} {k:>18,.0f} {rel * 100:>17.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--rounds', type=int, required=True)
    ap.add_argument('--block-start', type=int, default=None,
                    help='first shot of the block to decode. Defaults to the start of the '
                         'block that follows the training capacity, when that can be '
                         'inferred; otherwise 0.')
    ap.add_argument('--block-n', type=int, default=200000,
                    help='number of shots to decode. Larger blocks cost matching time but '
                         'buy precision on p_L.')
    args = ap.parse_args()

    print(f"[pool] {args.pool}")
    z = np.load(args.pool)
    evts, flips = z['det_evts'], z['flips']
    N = evts.shape[0]
    print(f"[pool] d={args.d} p={args.p} rounds={args.rounds}")
    print(f"[pool] shots {N:,}   det_evts {evts.shape} {evts.dtype}   flips {flips.shape}")

    # Base rate over the whole pool, then over the decoded block. The two should agree to
    # within sampling noise; a disagreement means the partitions are not drawn from the
    # same distribution and nothing downstream can be trusted.
    truth_all = np.asarray(flips).reshape(-1)
    base_all = float(truth_all.mean())

    start = args.block_start
    if start is None:
        start = max(0, N - args.block_n)
    stop = min(start + args.block_n, N)
    n_block = stop - start
    print(f"\n[block] decoding [{start:,}, {stop:,})  n={n_block:,}")

    truth = truth_all[start:stop]
    base_block = float(truth.mean())

    print(f"\n  logical-flip base rate")
    print(f"    whole pool : {base_all:.5f}  ({int(truth_all.sum()):,} of {N:,} shots)")
    print(f"    this block : {base_block:.5f} +/- {binom_err(base_block, n_block):.5f}"
          f"  ({int(truth.sum()):,} of {n_block:,})")
    print(f"    positives in a 10M training prefix: {base_all * 10_000_000:,.0f}")

    import pymatching
    from eval_on_tail import build_circuit
    circ = build_circuit(args.d, args.p, args.rounds)
    dem = circ.detector_error_model(decompose_errors=True)
    pym = pymatching.Matching.from_detector_error_model(dem)

    t0 = time.time()
    pred = pym.decode_batch(evts[start:stop], bit_packed_predictions=False,
                            bit_packed_shots=False).astype(np.int8).reshape(-1)
    dt = time.time() - t0
    fails = int((pred != truth).sum())
    p_L = fails / n_block
    err = binom_err(p_L, n_block)

    print(f"\n  MWPM on this block  ({dt:.0f}s)")
    print(f"    p_L            : {p_L:.6f} +/- {err:.6f}")
    print(f"    failures seen  : {fails:,} of {n_block:,}")
    rel = (1.0 / fails ** 0.5 * 100) if fails > 0 else float('nan')
    print(f"    rel. uncertainty from {fails:,} failures: {rel:.1f}%")

    print(f"\n  projection to candidate block sizes")
    report_projection(p_L)

    print(f"\n  summary line for the ledger:")
    print(f"    d={args.d} p={args.p} r={args.rounds}  base_rate={base_all:.5f}  "
          f"mwpm_p_L={p_L:.6f}+/-{err:.6f}  fails={fails}/{n_block}")


if __name__ == '__main__':
    main()
