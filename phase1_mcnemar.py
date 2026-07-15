#!/usr/bin/env python3
# Created: 2026-07-13
# Last modified: 2026-07-15
"""Phase 1: referee-proof knee. Paired McNemar at the two smallest lossless-ish
bit-widths (6, 8) vs MWPM on the SAME fresh disjoint 200k tail as the Pareto.

Why this exists. The Step-2 sweep (sweep_quantized.py) ran --no-save-weights, so
there are no on-disk 6/8-bit models to run a paired test against. This driver
RE-TRAINS just those points WITH --save-weights into a SEPARATE dir (out_q_mcnemar/,
NOT out_q/) so the published Pareto CSVs are never clobbered, then runs
eval_on_tail.py --mcnemar on each to produce the 2x2 contingency + p-value.

The FP32 (32-bit) McNemar is already in out_q/fp32_anchor.csv (the sweep's anchor
step ran --mcnemar). So this only needs bits {6, 8}.

Deliverable: out_q_mcnemar/mcnemar_knee.csv -- one row per (bits, seed) with
both-right / RCNN-only / MWPM-only / both-wrong + p_exact. That table is what
licenses the word "parity" (or "beats") at the knee: it shows the 6-bit gap is
inside test-set noise, shot-by-shot, not asserted.

Resume-safe: a run is SKIPPED if its weights.h5 already exists (train) and its
mcnemar row already exists (eval). Fan out across Named Servers with --bits/--seeds
(one seed per server = 2 runs each).

NOTE the p_L reported here is a FRESH training run per point -- with run-to-run
nondeterminism it will differ slightly from the sweep's Pareto mean. That is fine:
the McNemar test is self-consistent (it pairs THIS model's shots vs MWPM's on the
shared tail). Report the sweep mean as the headline p_L; report McNemar here as the
significance of the knee.
"""
import csv, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

BITS  = [6, 8]
SEEDS = [0, 1, 2]
D, P, ROUNDS, KERNEL = 5, 0.010, 3, 3
N_TRAIN = 10_000_000
N_TEST  = 200_000
HIDDEN, HIDDEN_LAYERS, NPOL = 100, 2, 2
EPOCHS, BATCH = 50, 10_000

TRAIN_POOL_DIR = os.path.expanduser('~/rcnn_threshold/pools')
TEST_POOL      = os.path.expanduser('~/rcnn_threshold/pools/data_d5_p0.010_r3_TAIL200k.npz')
OUT_DIR        = os.path.expanduser('~/rcnn_threshold/out_q_mcnemar')   # separate: no clobber
MCNEMAR_CSV    = os.path.join(OUT_DIR, 'mcnemar_knee.csv')


def tag(bits, seed):
    return f'rcnn_d{D}_p{P:.3f}_r{ROUNDS}_w{bits}_seed{seed}_ntr{N_TRAIN}'


def weights_path(bits, seed):
    return os.path.join(OUT_DIR, tag(bits, seed) + '.weights.h5')


def mcnemar_done(bits, seed):
    """True if this bits/seed already has a row in mcnemar_knee.csv."""
    if not os.path.exists(MCNEMAR_CSV):
        return False
    want = os.path.basename(weights_path(bits, seed))
    for row in csv.DictReader(open(MCNEMAR_CSV)):
        if os.path.basename(str(row.get('weights', ''))) == want:
            return True
    return False


def train(bits, seed):
    if os.path.exists(weights_path(bits, seed)):
        print(f'  [skip train] weights exist: {weights_path(bits, seed)}', flush=True); return
    cmd = [PY, os.path.join(HERE, 'train_one_quantized.py'),
           '--d', str(D), '--p', str(P), '--rounds', str(ROUNDS), '--kernel', str(KERNEL),
           '--seed', str(seed), '--n-train', str(N_TRAIN), '--n-test', str(N_TEST),
           '--weight-bits', str(bits), '--epochs', str(EPOCHS), '--batch-size', str(BATCH),
           '--hidden', str(HIDDEN), '--hidden-layers', str(HIDDEN_LAYERS), '--npol', str(NPOL),
           '--data-dir', TRAIN_POOL_DIR, '--test-pool', TEST_POOL, '--out-dir', OUT_DIR,
           '--save-weights', '--no-early-stopping']
    print('  >', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def mcnemar(bits, seed):
    if mcnemar_done(bits, seed):
        print(f'  [skip mcnemar] row exists for {tag(bits, seed)}', flush=True); return
    cmd = [PY, os.path.join(HERE, 'eval_on_tail.py'),
           '--weights', weights_path(bits, seed), '--d', str(D), '--p', str(P),
           '--rounds', str(ROUNDS), '--n-test', str(N_TEST), '--kernel', str(KERNEL),
           '--hidden', str(HIDDEN), '--hidden-layers', str(HIDDEN_LAYERS), '--npol', str(NPOL),
           '--weight-bits', str(bits),
           '--pool', TEST_POOL, '--mcnemar', '--out-csv', MCNEMAR_CSV]
    print('  >', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Phase 1 knee McNemar. Fan out with --bits/--seeds.')
    ap.add_argument('--bits', default=None, help='comma list, overrides [6,8]')
    ap.add_argument('--seeds', default=None, help='comma list, overrides [0,1,2]')
    a = ap.parse_args()
    bits_list = [int(b) for b in a.bits.split(',')] if a.bits else BITS
    seeds_list = [int(s) for s in a.seeds.split(',')] if a.seeds else SEEDS

    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(TEST_POOL):
        raise SystemExit(f'[phase1] FRESH TEST_POOL missing: {TEST_POOL}')
    print(f'=== PHASE 1 McNEMAR  bits={bits_list} seeds={seeds_list}  '
          f'n_train={N_TRAIN:,} n_test={N_TEST:,}  out={OUT_DIR} ===', flush=True)
    for bits in bits_list:
        for s in seeds_list:
            print(f'--- w{bits} seed{s} ---', flush=True)
            train(bits, s)
            mcnemar(bits, s)
    print(f'=== PHASE 1 DONE. table: {MCNEMAR_CSV} ===', flush=True)


if __name__ == '__main__':
    main()
