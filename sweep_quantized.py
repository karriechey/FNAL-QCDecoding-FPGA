#!/usr/bin/env python3
"""Step 2 driver: weight-bit-width Pareto for FullRCNNModel at d5/r3/p0.010.

Params are written PLAINLY here (not buried in CLI defaults) so it's clear what ran:
    WEIGHT_BITS = [8, 6, 4, 3, 2]      # QAT points (trained fresh)
    SEEDS       = [0, 1, 2]
    N_TRAIN     = 10_000_000           # matches the FP32 anchor
    N_TEST      = 200_000              # FRESH disjoint tail (TEST_POOL)
FP32 (32-bit) is the top-of-Pareto anchor: NOT retrained -- the existing 10M weights
in ANCHOR_DIR are evaluated on the same fresh tail via eval_on_tail.py.

Resume-safe: each (bits,seed) writes one per-run CSV; a run is SKIPPED if its CSV
exists. One subprocess per run (WeightQuant is process-global -> one config per proc).

*** BEFORE LAUNCH: TEST_POOL must be a FRESH 200k pool disjoint from the first
    N_TRAIN training shots. The 10.01M training pool's own last-200k overlaps a
    10M-train set by 190k shots (in-sample). Generate it with generate_pools.py
    (new gen-seed) and MWPM-decode it, then point TEST_POOL at it. ***

WHY DISJOINT IS ENFORCED (measured, not asserted). The 10M FP32 weights were evaluated
in-sample vs held-out on the existing pool's last-200k tail (positional split confirmed
in train_one.py: tr=slice(0,ntr)). Optimism is real but small:
    seed  in-sample p_L(190k)  held-out p_L(10k)   delta    held-out SE
      0        0.04590              0.04700        +0.00110    +/-0.00212
      1        0.04582              0.04870        +0.00288    +/-0.00215
      2        0.04654              0.04720        +0.00066    +/-0.00212
Mean delta ~ +0.0016 p_L (0.5-1.3 sigma per seed). It softens the 10M "slight beat"
toward statistical parity; it does NOT invalidate the decoder comparison. The fresh
tail is for rigor + +/-0.0005 precision (every sweep point is 10M, same regime), not
damage control.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# ---- plainly-stated sweep grid -------------------------------------------
WEIGHT_BITS = [8, 6, 4, 3, 2]
SEEDS       = [0, 1, 2]
D, P, ROUNDS, KERNEL = 5, 0.010, 3, 3
N_TRAIN     = 10_000_000
N_TEST      = 200_000
HIDDEN, HIDDEN_LAYERS, NPOL = 100, 2, 2
EPOCHS, BATCH = 50, 10_000

TRAIN_POOL_DIR = os.path.expanduser('~/rcnn_threshold/pools')          # data_d5_p0.010_r3.npz
TEST_POOL      = os.path.expanduser('~/rcnn_threshold/pools/data_d5_p0.010_r3_TAIL200k.npz')
ANCHOR_DIR     = os.path.expanduser('~/rcnn_threshold/out_t200k_w')    # FP32 10M weights
OUT_DIR        = os.path.expanduser('~/rcnn_threshold/out_q')
# ---------------------------------------------------------------------------


def csv_tag(bits, seed):
    return f'rcnn_d{D}_p{P:.3f}_r{ROUNDS}_w{bits}_seed{seed}_ntr{N_TRAIN}'


def done(bits, seed):
    return os.path.exists(os.path.join(OUT_DIR, csv_tag(bits, seed) + '.csv'))


def run_qat(bits, seed):
    cmd = [PY, os.path.join(HERE, 'train_one_quantized.py'),
           '--d', str(D), '--p', str(P), '--rounds', str(ROUNDS), '--kernel', str(KERNEL),
           '--seed', str(seed), '--n-train', str(N_TRAIN), '--n-test', str(N_TEST),
           '--weight-bits', str(bits), '--epochs', str(EPOCHS), '--batch-size', str(BATCH),
           '--hidden', str(HIDDEN), '--hidden-layers', str(HIDDEN_LAYERS), '--npol', str(NPOL),
           '--data-dir', TRAIN_POOL_DIR, '--test-pool', TEST_POOL, '--out-dir', OUT_DIR,
           '--no-early-stopping']
    print('  >', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_fp32_anchor(seed):
    """FP32 point: eval existing 10M weights on the fresh tail (no retrain), and
    decode MWPM on the SAME fresh shots (--mcnemar). Appends one row per seed to
    OUT_DIR/fp32_anchor.csv; its mwpm_p_L column is the fresh-tail MWPM anchor."""
    w = os.path.join(ANCHOR_DIR, f'rcnn_d{D}_p{P:.3f}_r{ROUNDS}_seed{seed}_ntr{N_TRAIN}.weights.h5')
    if not os.path.exists(w):
        print(f"  [skip fp32 seed{seed}] missing {w}", flush=True); return
    out_csv = os.path.join(OUT_DIR, 'fp32_anchor.csv')
    cmd = [PY, os.path.join(HERE, 'eval_on_tail.py'),
           '--weights', w, '--d', str(D), '--p', str(P), '--rounds', str(ROUNDS),
           '--n-test', str(N_TEST), '--kernel', str(KERNEL), '--hidden', str(HIDDEN),
           '--hidden-layers', str(HIDDEN_LAYERS), '--npol', str(NPOL),
           '--pool', TEST_POOL, '--mcnemar', '--out-csv', out_csv]
    print('  >', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="QAT Pareto sweep. Fan out across Named Servers "
                                 "with --seeds (one seed per server = balanced 5 runs each).")
    ap.add_argument('--bits', default=None, help='comma list, overrides WEIGHT_BITS (e.g. 8,6)')
    ap.add_argument('--seeds', default=None, help='comma list, overrides SEEDS (e.g. 0)')
    ap.add_argument('--skip-fp32', action='store_true',
                    help='skip FP32 anchor re-eval. REQUIRED for parallel servers (avoids '
                         'concurrent append races on fp32_anchor.csv; run it once beforehand).')
    a = ap.parse_args()
    bits_list = [int(b) for b in a.bits.split(',')] if a.bits else WEIGHT_BITS
    seeds_list = [int(s) for s in a.seeds.split(',')] if a.seeds else SEEDS

    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(TEST_POOL):
        raise SystemExit(f"[sweep] FRESH TEST_POOL missing: {TEST_POOL}\n"
                         f"        Generate a disjoint 200k tail first (see module docstring).")
    print(f"=== QAT PARETO SWEEP  bits={bits_list} seeds={seeds_list}  "
          f"n_train={N_TRAIN:,} n_test={N_TEST:,} ===", flush=True)
    if not a.skip_fp32:
        print("--- FP32 anchor (eval existing 10M weights, no retrain) ---", flush=True)
        for s in seeds_list:
            run_fp32_anchor(s)
    for bits in bits_list:
        for s in seeds_list:
            if done(bits, s):
                print(f"[skip] {csv_tag(bits, s)} exists", flush=True); continue
            print(f"--- w{bits} seed{s} ---", flush=True)
            run_qat(bits, s)
    print(f"=== SWEEP DONE. collate: {PY} collate_pareto.py --out-dir {OUT_DIR} ===", flush=True)


if __name__ == '__main__':
    main()
