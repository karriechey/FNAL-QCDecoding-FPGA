#!/usr/bin/env python3
# Created: 2026-07-13
# Last modified: 2026-08-03
"""Generate a FRESH, disjoint 200k test tail for the QAT Pareto (Step 2 gate 1).

Disjoint by construction: samples with a NEW gen-seed (default 43) != the training
pool's gen-seed (42, per mwpm_baseline.csv), so these shots are an independent draw
from the same (d,p,rounds) distribution -- never in any training prefix. Same circuit
as generate_pools / eval_on_tail (4-channel rotated_memory_z), rounds=3 explicitly.

Writes {out} with keys measurements/det_evts/flips (+ gen_seed), matching the pool
schema train_one_quantized/eval_on_tail expect. MWPM is decoded separately by
eval_on_tail.py --pool {out} --mcnemar (that gives the fresh-tail MWPM anchor).

Also usable as a general pool generator: --n and --gen-seed are free, and
--allow-training-seed lifts the seed-42 refusal for the case where reproducing an
existing training pool is the point. Note the sample stream depends on n as well as the
seed -- two pools drawn at the same seed with different --n are independent, not nested,
so reproducing a pool requires matching both.

  python make_fresh_tail.py                 # 200k, seed 43 -> pools/data_d5_p0.010_r3_TAIL200k.npz
  python make_fresh_tail.py --n 20200000 --gen-seed 42 --allow-training-seed --out ...
"""
import argparse, os
import numpy as np
from eval_on_tail import build_circuit
from generate_pools import sample_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--n', type=int, default=200000)
    ap.add_argument('--gen-seed', type=int, default=43,
                    help='must differ from the training pool gen-seed (42) for a test tail')
    ap.add_argument('--allow-training-seed', action='store_true',
                    help='permit --gen-seed 42. Only for regenerating a training pool, where '
                         'matching the original draw is the point; never for a test tail.')
    ap.add_argument('--expect-flips-sha', default=None,
                    help='verify the generated flips array against this sha256 prefix and '
                         'fail if it differs, so a reproduced pool is confirmed bit-identical')
    ap.add_argument('--out', default=os.path.expanduser('~/rcnn_threshold/pools/data_d5_p0.010_r3_TAIL200k.npz'))
    args = ap.parse_args()
    if args.gen_seed == 42 and not args.allow_training_seed:
        raise SystemExit('[fresh-tail] gen-seed 42 == training pool seed -> not disjoint. '
                         'Pick another, or pass --allow-training-seed to regenerate a '
                         'training pool deliberately.')

    circ = build_circuit(args.d, args.p, args.rounds)
    meas, det_evts, flips = sample_pool(circ, args.n, args.gen_seed)

    if args.expect_flips_sha:
        import hashlib
        got = hashlib.sha256(np.ascontiguousarray(flips).tobytes()).hexdigest()
        if not got.startswith(args.expect_flips_sha):
            raise SystemExit(f'[fresh-tail] flips sha256 {got[:16]} != expected '
                             f'{args.expect_flips_sha} -- this is not the same draw. '
                             f'Check --n and --gen-seed against the original.')
        print(f'[fresh-tail] flips sha256 {got[:16]} matches -- bit-identical draw.', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, measurements=meas, det_evts=det_evts, flips=flips,
                        gen_seed=args.gen_seed)
    print(f"[fresh-tail] wrote {args.out}  meas{meas.shape} det{det_evts.shape} "
          f"flips{flips.shape}  gen_seed={args.gen_seed}", flush=True)
    print(f"[fresh-tail] next: eval_on_tail.py --pool {args.out} --n-test {args.n} "
          f"--mcnemar  (per FP32 seed) -> fresh MWPM + clean FP32 anchor", flush=True)


if __name__ == '__main__':
    main()
