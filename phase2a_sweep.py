#!/usr/bin/env python3
# Created: 2026-07-16
# Last modified: 2026-07-16
"""Phase 2a driver: activation-precision sweep by quantization-aware training (QAT).

WHAT THIS MEASURES
------------------
Weights are held fixed at 6 bits (the knee found in the Phase-1 weight sweep). We sweep the
activation word length B over {8, 6, 4} and, for each B, retrain the model with the bounded
activation tensors quantized at B bits (x-like tensors stay float -- see ActQuant / RUN_LOG).
Because the quantizers are live during training (QAT, not post-training quantization), the
network adapts to the reduced precision, so these numbers are a fair operating cost, not the
pessimistic post-training figure.

Each result is compared to the w6 / float-activation anchor (Phase 1): p_L = 0.046675 /
0.047715 / 0.045580 for seeds 0/1/2 on the same fresh 200k tail. The deliverable is the curve
p_L versus activation precision, and the lowest B that stays within noise of the anchor.

HONEST SCOPE
------------
This quantizes only the BOUNDED activation tensors. The x-like (exponential-domain) intermediates
stay in float32, so the resulting model is NOT yet fully fixed-point / synthesizable; that needs
the Phase-4 log-domain (LSE) rewrite. Report this as "all bounded tensors quantized; exponential-
domain intermediates addressed in Phase 4", not "the model is fixed-point".

WARM START
----------
Each run starts (via --init-weights) from the trained 6-bit-weight / float-activation model
(the Phase-1 anchor weights in out_q_mcnemar/), then fine-tunes under activation quantization.
This is cheaper and trains better than starting from scratch.

SMOKE TEST (run this locally FIRST)
-----------------------------------
`--smoke` runs one small point (100k shots, B=6, seed 0, a few epochs) end to end -- train,
evaluate, write CSV -- to confirm the QAT loop runs without dying BEFORE spending EAF GPU time.
Discard its numbers; it only checks that the pipeline is intact. The predicted failure mode at
low B is divergence through the exp path (straight-through-estimator limitation), which is a
finding about how low activation precision can go, not a bug to fix.

Resume-safe: each (act_bits, seed) writes one CSV and is skipped if that CSV already exists.
Fan out across EAF servers with --acts / --seeds (one seed per server = balanced).

  python phase2a_sweep.py --smoke                 # local pipeline check, ~minutes
  python phase2a_sweep.py                          # full sweep {8,6,4} x {0,1,2}
  python phase2a_sweep.py --acts 8 --seeds 0       # one point (e.g. per-server fan-out)
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# ---- sweep grid (stated plainly so it is clear what ran) -----------------------------------
WEIGHT_BITS = 6                       # fixed at the Phase-1 knee
ACT_BITS    = [8, 6, 4]               # activation word lengths to sweep
SEEDS       = [0, 1, 2]
D, P, ROUNDS, KERNEL = 5, 0.010, 3, 3
N_TRAIN     = 10_000_000              # same training volume as the anchor
N_TEST      = 200_000                 # fresh disjoint tail
HIDDEN, HIDDEN_LAYERS, NPOL = 100, 2, 2
EPOCHS, BATCH = 50, 10_000

# Data location: prefer the in-repo ./rcnn_threshold (data pulled from EAF into the repo, so the
# local smoke test works), otherwise ~/rcnn_threshold (the layout on EAF). Same rule as the notebook.
BASE = 'rcnn_threshold' if os.path.isdir('rcnn_threshold') else os.path.expanduser('~/rcnn_threshold')
TRAIN_POOL_DIR = os.path.join(BASE, 'pools')
TEST_POOL      = os.path.join(BASE, 'pools', 'data_d5_p0.010_r3_TAIL200k.npz')
ANCHOR_DIR     = os.path.join(BASE, 'out_q_mcnemar')   # w6 / float-activation anchor weights (Phase 1)
OUT_DIR        = os.path.join(BASE, 'out_q_phase2a')
# --------------------------------------------------------------------------------------------


def anchor_weights(seed):
    """The trained 6-bit-weight / float-activation model to warm-start from (Phase 1)."""
    return os.path.join(ANCHOR_DIR, f'rcnn_d{D}_p{P:.3f}_r{ROUNDS}_w{WEIGHT_BITS}_seed{seed}_ntr{N_TRAIN}.weights.h5')


def csv_tag(act_bits, seed, n_train):
    return f'rcnn_d{D}_p{P:.3f}_r{ROUNDS}_w{WEIGHT_BITS}_a{act_bits}_seed{seed}_ntr{n_train}'


def already_done(act_bits, seed, n_train):
    return os.path.exists(os.path.join(OUT_DIR, csv_tag(act_bits, seed, n_train) + '.csv'))


def run_point(act_bits, seed, n_train, epochs, smoke):
    """Train and evaluate one (act_bits, seed) point via train_one_quantized.py."""
    init = anchor_weights(seed)
    cmd = [PY, os.path.join(HERE, 'train_one_quantized.py'),
           '--d', str(D), '--p', str(P), '--rounds', str(ROUNDS), '--kernel', str(KERNEL),
           '--seed', str(seed), '--n-train', str(n_train), '--n-test', str(N_TEST),
           '--weight-bits', str(WEIGHT_BITS), '--act-bits', str(act_bits),
           '--epochs', str(epochs), '--batch-size', str(BATCH),
           '--hidden', str(HIDDEN), '--hidden-layers', str(HIDDEN_LAYERS), '--npol', str(NPOL),
           '--data-dir', TRAIN_POOL_DIR, '--test-pool', TEST_POOL, '--out-dir', OUT_DIR,
           '--no-early-stopping']
    # Warm-start from the anchor weights when they exist; otherwise train from a seeded init.
    if os.path.exists(init):
        cmd += ['--init-weights', init]
    else:
        print(f'  [warn] anchor weights not found ({init}); training from a fresh init', flush=True)
    if smoke:
        cmd += ['--save-weights']   # keep the smoke model so the pipeline check is fully exercised
    print('  >', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Phase 2a activation-precision sweep (weights fixed at 6 bits).')
    ap.add_argument('--acts', default=None, help='comma list, overrides {8,6,4}')
    ap.add_argument('--seeds', default=None, help='comma list, overrides {0,1,2}')
    ap.add_argument('--smoke', action='store_true',
                    help='local pipeline check: 100k shots, B=6, seed 0, few epochs; discard numbers')
    ap.add_argument('--smoke-epochs', type=int, default=3)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(TEST_POOL):
        raise SystemExit(f'[phase2a] fresh test tail missing: {TEST_POOL}\n'
                         f'          regenerate it with make_fresh_tail.py, or point TEST_POOL at it.')

    if a.smoke:
        # One small point, end to end, to confirm the QAT loop runs. Numbers are not meaningful.
        print('=== SMOKE TEST: 100k shots, B=6, seed 0 -- pipeline check only, discard numbers ===', flush=True)
        run_point(act_bits=6, seed=0, n_train=100_000, epochs=a.smoke_epochs, smoke=True)
        print('=== smoke test finished: the train->eval->CSV pipeline runs. ===', flush=True)
        return

    acts = [int(x) for x in a.acts.split(',')] if a.acts else ACT_BITS
    seeds = [int(x) for x in a.seeds.split(',')] if a.seeds else SEEDS
    print(f'=== PHASE 2a SWEEP  weights={WEIGHT_BITS}b  act_bits={acts}  seeds={seeds}  '
          f'n_train={N_TRAIN:,} n_test={N_TEST:,} ===', flush=True)
    for act_bits in acts:
        for seed in seeds:
            if already_done(act_bits, seed, N_TRAIN):
                print(f'[skip] {csv_tag(act_bits, seed, N_TRAIN)} exists', flush=True)
                continue
            print(f'--- w{WEIGHT_BITS} a{act_bits} seed{seed} ---', flush=True)
            run_point(act_bits, seed, N_TRAIN, EPOCHS, smoke=False)
    print(f'=== SWEEP DONE. results in {OUT_DIR}. '
          f'Anchor to compare against: w6/float-act p_L 0.046675/0.047715/0.045580. ===', flush=True)


if __name__ == '__main__':
    main()
