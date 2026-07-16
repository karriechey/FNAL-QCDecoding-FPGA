#!/usr/bin/env python3
# Created: 2026-07-16
# Last modified: 2026-07-16
"""Phase 2a driver: activation-precision sweep by quantization-aware training (QAT).

WHAT THIS MEASURES
------------------
Weights are held fixed at 6 bits (the knee found in the Phase-1 weight sweep). We sweep the
activation word length B over {8, 6, 4} and, for each B, train the model FROM SCRATCH with the
bounded activation tensors quantized at B bits (x-like tensors stay float -- see ActQuant /
RUN_LOG). Because the quantizers are live during training (QAT, not post-training quantization),
the network adapts to the reduced precision, so these numbers are a fair operating cost.

TRAINED FROM SCRATCH, MATCHING THE ANCHOR RECIPE
------------------------------------------------
Each run trains from its seeded initialisation using the EXACT recipe that produced the Phase-1
anchor -- 50 fixed epochs, no early stopping, val_split 0.2, the reference architecture's LR schedule (all inherited
from train_one.py via train_one_quantized.py). We do NOT warm-start from the anchor weights: the
whole design rests on the act-off control (B=None) reproducing the anchor p_L, which requires an
identical, independent training path, not a fine-tune of the anchor.

THE CONTROL (B=32 = activations off)
------------------------------------
ACT_BITS includes 32: any B>=32 disables the activation quantizers (activations stay full
precision), so B=32 is "activation quant off" -- but trained FROM SCRATCH under the identical
recipe as the anchor. This is the internal consistency check: it MUST reconverge to the anchor
p_L (0.046675 / 0.047715 / 0.045580 for seeds 0/1/2 on the fresh 200k tail). If the control does
not return to the anchor, the training config is wrong and every lower-B row would look bad for
that reason, not because of the precision -- verify the control before trusting B=8/6/4.

We use 32 (not None) as the sentinel ON PURPOSE: the trainer writes a distinct '_a32' suffix for
it, so the control's files are ..._w6_a32_seed{s}_... and can NEVER collide with the Phase-1
anchor's un-suffixed stem ..._w6_seed{s}_... (which the McNemar / hls4ml hand-off compares
against). The control is a fresh retrain in out_q_phase2a/, not a lookup of the old anchor.

HONEST SCOPE
------------
This quantizes only the BOUNDED activation tensors. The x-like (exponential-domain) intermediates
stay float32, so the model is NOT yet fully fixed-point / synthesizable; that needs the Phase-4
log-domain (LSE) rewrite. Report as "all bounded tensors quantized; exponential-domain
intermediates addressed in Phase 4", not "the model is fixed-point".

Weights are saved for every run (needed for the knee McNemar, re-evaluation, and the hls4ml
hand-off). Resume-safe: each (act_bits, seed) writes one CSV and is skipped if it exists. Fan out
across EAF servers with --acts / --seeds (one seed per server).

SMOKE / TREND CHECK (run locally FIRST, before EAF)
---------------------------------------------------
`--smoke` runs two small from-scratch points (100k shots, ~12 epochs, seed 0): the control
(B=32, act off) and B=6. The smoke test earlier proved the loop runs; this proves it trains in the RIGHT
DIRECTION -- the control p_L should descend toward the anchor (~0.0467), and B=6 should track
close to it if 6-bit activation is cheap. Numbers are not converged (small data/epochs); we only
read the trend. If the control does not head toward the anchor, STOP -- the recipe is wrong and
EAF would only burn hours confirming it. (A low-B divergence through the exp path is the
straight-through-estimator limit -- a finding about how low B can go, not a bug.)

  python phase2a_sweep.py --smoke                 # local trend check (control + B=6), ~minutes
  python phase2a_sweep.py                          # full sweep {32,8,6,4} x {0,1,2}
  python phase2a_sweep.py --acts 8 --seeds 0       # one point (per-server fan-out)
  python phase2a_sweep.py --acts 32 --seeds 0      # just the control (activations off) for one seed
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# ---- sweep grid (stated plainly so it is clear what ran) -----------------------------------
WEIGHT_BITS = 6                       # fixed at the Phase-1 knee
# Activation word lengths. 32 = the reconvergence CONTROL (>=32 disables the quantizers, so
# activations are full precision; trained from scratch under the anchor recipe -> must return to
# the anchor p_L). It gets a distinct '_a32' filename so it cannot overwrite the anchor. 8/6/4 are
# the real activation-precision sweep.
ACT_BITS    = [32, 8, 6, 4]
SEEDS       = [0, 1, 2]
D, P, ROUNDS, KERNEL = 5, 0.010, 3, 3
N_TRAIN     = 10_000_000              # same training volume as the anchor
N_TEST      = 200_000                 # fresh disjoint tail
HIDDEN, HIDDEN_LAYERS, NPOL = 100, 2, 2
EPOCHS, BATCH = 50, 10_000            # 50 fixed epochs, no early stopping -- matches the anchor

# Data location: prefer the in-repo ./rcnn_threshold (data pulled from EAF into the repo, so the
# local smoke test works), otherwise ~/rcnn_threshold (the layout on EAF). Same rule as the notebook.
BASE = 'rcnn_threshold' if os.path.isdir('rcnn_threshold') else os.path.expanduser('~/rcnn_threshold')
TRAIN_POOL_DIR = os.path.join(BASE, 'pools')
TEST_POOL      = os.path.join(BASE, 'pools', 'data_d5_p0.010_r3_TAIL200k.npz')
OUT_DIR        = os.path.join(BASE, 'out_q_phase2a')
# --------------------------------------------------------------------------------------------


def csv_tag(act_bits, seed, n_train):
    """Output filename stem. Every point (control 32 and the real 8/6/4) carries an _a{bits}
    suffix, matching what the trainer writes when --act-bits is passed. The control's _a32 keeps
    it distinct from the anchor's un-suffixed ..._w6_seed{s}_... stem."""
    return f'rcnn_d{D}_p{P:.3f}_r{ROUNDS}_w{WEIGHT_BITS}_a{act_bits}_seed{seed}_ntr{n_train}'


def already_done(act_bits, seed, n_train):
    return os.path.exists(os.path.join(OUT_DIR, csv_tag(act_bits, seed, n_train) + '.csv'))


def run_point(act_bits, seed, n_train, epochs):
    """Train (from scratch) and evaluate one (act_bits, seed) point via train_one_quantized.py.

    No warm start: from the seeded init, exactly as the Phase-1 anchor was trained, so the
    control (act_bits=32, quantizers disabled) reproduces the anchor. --no-early-stopping + 50
    epochs + val_split 0.2 mirror the anchor recipe (phase1_mcnemar.py trained it that way).
    --save-weights keeps the quantized model for the McNemar / re-eval / hls4ml hand-off.
    act_bits is always passed (32 for the control) so the trainer writes the distinct _a{bits}
    filename."""
    cmd = [PY, os.path.join(HERE, 'train_one_quantized.py'),
           '--d', str(D), '--p', str(P), '--rounds', str(ROUNDS), '--kernel', str(KERNEL),
           '--seed', str(seed), '--n-train', str(n_train), '--n-test', str(N_TEST),
           '--weight-bits', str(WEIGHT_BITS), '--act-bits', str(act_bits),
           '--epochs', str(epochs), '--batch-size', str(BATCH),
           '--hidden', str(HIDDEN), '--hidden-layers', str(HIDDEN_LAYERS), '--npol', str(NPOL),
           '--data-dir', TRAIN_POOL_DIR, '--test-pool', TEST_POOL, '--out-dir', OUT_DIR,
           '--no-early-stopping',      # fixed 50-epoch budget, matching how the anchor was trained
           '--save-weights']           # keep the quantized weights for every point
    print('  >', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_acts(s):
    """Parse a --acts list. The tokens 'none'/'fp'/'fp32'/'off' all map to 32, the activation-off
    control (>=32 disables the quantizers)."""
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        out.append(32 if tok.lower() in ('none', 'fp', 'fp32', 'off') else int(tok))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Phase 2a activation-precision sweep (weights fixed at 6 bits).')
    ap.add_argument('--acts', default=None, help="comma list, overrides {None,8,6,4}; use 'none' for the control")
    ap.add_argument('--seeds', default=None, help='comma list, overrides {0,1,2}')
    ap.add_argument('--smoke', action='store_true',
                    help='local trend check: 100k shots, ~12 epochs, from scratch, control + B=6; read the trend only')
    ap.add_argument('--smoke-epochs', type=int, default=12)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(TEST_POOL):
        raise SystemExit(f'[phase2a] fresh test tail missing: {TEST_POOL}\n'
                         f'          regenerate it with make_fresh_tail.py, or point TEST_POOL at it.')

    if a.smoke:
        # Two small from-scratch points to confirm the recipe trains in the right direction.
        print('=== TREND CHECK: 100k, from scratch, '
              f'{a.smoke_epochs} epochs -- control (act off) then B=6 ===', flush=True)
        run_point(act_bits=32, seed=0, n_train=100_000, epochs=a.smoke_epochs)   # control (act off)
        run_point(act_bits=6,  seed=0, n_train=100_000, epochs=a.smoke_epochs)   # B=6
        print('=== read the trend: the control p_L should be DESCENDING toward the anchor (~0.0467); '
              'B=6 should track close to it. (Not converged -- small data/epochs.) ===', flush=True)
        return

    acts = parse_acts(a.acts) if a.acts else ACT_BITS
    seeds = [int(x) for x in a.seeds.split(',')] if a.seeds else SEEDS
    print(f'=== PHASE 2a SWEEP  weights={WEIGHT_BITS}b  act_bits={acts}  seeds={seeds}  '
          f'n_train={N_TRAIN:,} n_test={N_TEST:,}  (from scratch, no warm start) ===', flush=True)
    for act_bits in acts:
        for seed in seeds:
            if already_done(act_bits, seed, N_TRAIN):
                print(f'[skip] {csv_tag(act_bits, seed, N_TRAIN)} exists', flush=True)
                continue
            label = 'control(act off)' if act_bits >= 32 else f'a{act_bits}'
            print(f'--- w{WEIGHT_BITS} {label} seed{seed} ---', flush=True)
            run_point(act_bits, seed, N_TRAIN, EPOCHS)
    print(f'=== SWEEP DONE. results in {OUT_DIR}. Control (act off) must match the anchor '
          f'(0.046675/0.047715/0.045580) before trusting B=8/6/4. ===', flush=True)


if __name__ == '__main__':
    main()
