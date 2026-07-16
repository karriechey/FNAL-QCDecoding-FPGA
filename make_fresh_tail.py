#!/usr/bin/env python3
# Created: 2026-07-13
# Last modified: 2026-07-16
"""Generate a FRESH, disjoint 200k test tail for the QAT Pareto (Step 2 gate 1).

Disjoint by construction: samples with a NEW gen-seed (default 43) != the training
pool's gen-seed (42, per mwpm_baseline.csv), so these shots are an independent draw
from the same (d,p,rounds) distribution -- never in any training prefix. Same circuit
as generate_pools / eval_on_tail (4-channel rotated_memory_z), rounds=3 explicitly.

Writes {out} with keys measurements/det_evts/flips (+ gen_seed), matching the pool
schema train_one_quantized/eval_on_tail expect. MWPM is decoded separately by
eval_on_tail.py --pool {out} --mcnemar (that gives the fresh-tail MWPM anchor).

  python make_fresh_tail.py                 # 200k, seed 43 -> pools/data_d5_p0.010_r3_TAIL200k.npz
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
    ap.add_argument('--gen-seed', type=int, default=43, help='MUST differ from the training pool gen-seed (42)')
    ap.add_argument('--out', default=os.path.expanduser('~/rcnn_threshold/pools/data_d5_p0.010_r3_TAIL200k.npz'))
    args = ap.parse_args()
    if args.gen_seed == 42:
        raise SystemExit('[fresh-tail] gen-seed 42 == training pool seed -> NOT disjoint. Pick another.')

    circ = build_circuit(args.d, args.p, args.rounds)
    meas, det_evts, flips = sample_pool(circ, args.n, args.gen_seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, measurements=meas, det_evts=det_evts, flips=flips,
                        gen_seed=args.gen_seed)
    print(f"[fresh-tail] wrote {args.out}  meas{meas.shape} det{det_evts.shape} "
          f"flips{flips.shape}  gen_seed={args.gen_seed} (!= 42)", flush=True)
    print(f"[fresh-tail] next: eval_on_tail.py --pool {args.out} --n-test {args.n} "
          f"--mcnemar  (per FP32 seed) -> fresh MWPM + clean FP32 anchor", flush=True)


if __name__ == '__main__':
    main()
