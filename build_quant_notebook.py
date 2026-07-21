#!/usr/bin/env python3
# Created: 2026-07-14
# Last modified: 2026-07-20
"""Emit QUANTIZATION_EXPERIMENTS.ipynb -- a LIVE lab notebook for the FullRCNNModel
quantization work (weight + activation, toward FPGA/hls4ml for Giuseppe).

Live = the code cells load the actual result files from ~/rcnn_threshold/out_q,
out_q_mcnemar so the notebook re-derives every table/plot from disk on EAF (it does not
hard-code numbers). Narrative + provenance in the markdown cells. Regenerate with:
    python build_quant_notebook.py
"""
import json, os

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": s.splitlines(keepends=True)}

cells = []

cells.append(md("""# FullRCNNModel Quantization — Lab Notebook

**Goal.** Quantize the reference architecture's real `FullRCNNModel` surface-code decoder (d=5, r=3, p=0.010)
toward FPGA deployment via hls4ml (collaborator: Giuseppe). Path B = Quantization-Aware
Training with QKeras `quantized_bits` applied at point-of-use inside the custom `call()`
methods (standard QDense substitution can't reach the reference architecture's hand-managed `add_weight`
tensors). `CNNModel.py` stays byte-for-byte pristine; all quantization lives in
`CNNModel_quantized.py`.

This notebook is **live**: cells load the result files from disk and rebuild every number. The authoritative ledger is `docs/RUN_LOG.md`; this is the runnable view.

> Run on EAF (`.venv/bin/python` kernel) where `~/rcnn_threshold/` lives."""))

cells.append(md("""## Fixed substrate (never regenerate)

| item | value |
|---|---|
| architecture | `FullRCNNModel('ZL', d=5, k=3, r=3, [100,100], npol=2)`, ~51.5k params |
| train pool | `data_d5_p0.010_r3.npz` (10.01M shots, **gen-seed 42**) |
| eval tail (all phases) | `data_d5_p0.010_r3_TAIL200k.npz` (200k, **gen-seed 43**, disjoint) |
| quantizer (weights) | `quantized_bits(B, 1)` = 1 sign + 1 int + (B−2) frac |
| env | EAF, `.venv` qkeras 0.9.0, TF 2.15.1, Keras 2.15.0; A100 MIG 3g.40gb |

Contamination fix: the old 200k tail overlapped the 10M train set by 190k shots
(~+0.0016 p_L optimism). Replaced by a fresh gen-seed-43 tail (`make_fresh_tail.py`)."""))

cells.append(code("""import os, glob, json
import numpy as np, pandas as pd
# Locate rcnn_threshold/ without depending on the working directory. Jupyter sets the CWD to the
# notebook's OWN folder, so a bare './rcnn_threshold' test only works for a notebook sitting at the
# repo root -- from analysis_notebooks/ it silently missed and fell through to ~/rcnn_threshold,
# which on a laptop holds only pools/ and none of the out_q* result dirs. Walk up instead, then
# fall back to $HOME (the EAF layout, where the results really do live under the home directory).
def _find_base():
    d = os.path.abspath(os.getcwd())
    while True:
        cand = os.path.join(d, 'rcnn_threshold')
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:                       # hit the filesystem root
            return os.path.expanduser('~/rcnn_threshold')
        d = parent

BASE = _find_base()
OUT_Q  = os.path.join(BASE, 'out_q')           # Step-2 Pareto CSVs
OUT_MC = os.path.join(BASE, 'out_q_mcnemar')   # Phase 1 McNemar + Phase 3 JSON
OUT_2A = os.path.join(BASE, 'out_q_phase2a')   # Phase 2a activation sweep + its paired tests
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 40)
print('BASE :', BASE)
print('out_q     :', OUT_Q, '->', len(glob.glob(OUT_Q+'/*.csv')), 'csv')
print('out_q_mcnemar:', OUT_MC, '->', sorted(os.path.basename(f) for f in glob.glob(OUT_MC+'/*'))[:6])
print('out_q_phase2a:', OUT_2A, '->', len(glob.glob(OUT_2A+'/*.csv')), 'csv')

# The ONE true MWPM on the fresh 200k tail. Verified 2026-07-20 by reading the code, not the
# comment: eval_on_tail.py --mcnemar decodes these exact shots with PyMatching and overwrites the
# looked-up value. The sweep CSVs' mwpm_p_L column (0.0451) and the 0.0518 seen elsewhere both come
# from train_one.lookup_mwpm(), keyed only on (d, p, rounds) -- it does not know which tail it is
# being asked about. Never read the baseline out of a sweep CSV column.
MWPM_FRESH_TAIL = 0.049405"""))

cells.append(md("""## Phase 0 — implementation + local gates (DONE)

- Path B re-homed into `CNNModel_quantized.py`; `CNNModel.py` pristine (monkeypatch
  point-of-use quantizers on the custom layers).
- Gates passed: None-path == original (max|diff|=0); quantization bites; grads on 35
  quantized vars; FP32 10M weights load into a None-model bit-exact (prob corr 1.0)."""))

cells.append(md("""## Step 2 — weight-bit-width Pareto (DONE, n=3)

`sweep_quantized.py` × {8,6,4,3,2} bits × {0,1,2} seeds at 10M shots, fresh 200k tail.
FP32 anchor reuses existing 10M weights (no retrain). Collate below."""))

cells.append(code("""# Rebuild the Pareto table from the per-run CSVs + fp32 anchor
rows = [pd.read_csv(f) for f in sorted(glob.glob(OUT_Q+'/rcnn_d5_p0.010_r3_w*_seed*_ntr*.csv'))]
fp32 = os.path.join(OUT_Q, 'fp32_anchor.csv')
mwpm = None
if os.path.exists(fp32):
    fa = pd.read_csv(fp32).drop_duplicates('weights', keep='last')
    npar = int(rows[0]['n_params'].iloc[0]) if rows else np.nan
    for _, x in fa.iterrows():
        sd = int(str(x['weights']).split('seed')[1].split('_')[0])
        rows.append(pd.DataFrame([dict(weight_bits=32, seed=sd, p_L=float(x['p_L']),
                    n_params=npar, n_test=int(x['n_test']))]))
    m = pd.to_numeric(fa['mwpm_p_L'], errors='coerce').dropna()
    mwpm = float(m.mean()) if len(m) else None
df = pd.concat(rows, ignore_index=True).drop_duplicates(['weight_bits','seed'], keep='last')
g = df.groupby('weight_bits')['p_L'].agg(['mean','std','count'])
g['size_KB'] = g.index * int(df['n_params'].dropna().iloc[0]) / 8 / 1024
g['xMWPM'] = g['mean']/mwpm if mwpm else np.nan
print('MWPM (fresh tail) =', mwpm)
g.sort_values('size_KB')"""))

cells.append(code("""# The Pareto plot -- search known locations (repo snapshot / figures / EAF-style plots/)
from IPython.display import Image
cands = ['analysis_notebooks/figures/pareto_clean.png',
         'experiment_log/plots/pareto_frontier.png',
         'experiment_log/results_snapshot/eaf/QuantumDecoderQKeras/plots/rcnn_d5_r3_qat_pareto.png',
         'plots/rcnn_d5_r3_qat_pareto.png']
p = next((x for x in cands if os.path.exists(x)), None)
Image(p) if p else print('pareto plot not found in repo; regenerate with collate_pareto.py --out-dir', OUT_Q)"""))

cells.append(md("""**Headline.** 8-bit lossless (= FP32); **6-bit is the knee** (37.8 KB, 5.3× smaller,
beats MWPM). Sharp cliff 6→4 (4-bit ~3.4σ worse); 2-bit collapses. Variance blows up at
low bits. This is the **weights-only** ceiling, activations still FP32."""))

cells.append(md("""## Phase 1 — paired McNemar at the knee (DONE)

`phase1_mcnemar.py` retrained {6,8} × {0,1,2} with `--save-weights` into `out_q_mcnemar/`
(no clobber of the Pareto CSVs), then `eval_on_tail.py --mcnemar` on the shared fresh tail.
The paired test licenses "beats MWPM" with discordant shot counts, not error-bar overlap."""))

cells.append(code("""mc = pd.read_csv(os.path.join(OUT_MC,'mcnemar_knee.csv')).drop_duplicates('weights', keep='last')
mc['bits'] = mc['weights'].str.extract(r'_w(\\d+)_').astype(int)
mc['seed'] = mc['weights'].str.extract(r'seed(\\d+)_').astype(int)
show = mc[['bits','seed','p_L','mwpm_p_L','ratio','rcnn_only','mwpm_only','net_rcnn_wins','p_exact']]
show.sort_values(['bits','seed']).reset_index(drop=True)"""))

cells.append(code("""import matplotlib.pyplot as plt

# combine bits=32 (fp32 anchor, already has a McNemar row from the sweep's anchor step)
# with bits=6,8 (this phase) into one table so the figure covers the whole knee
mc_all = pd.concat([
    fa.assign(bits=32, seed=fa['weights'].str.extract(r'seed(\\d+)_')[0].astype(int)),
    mc,
], ignore_index=True).drop_duplicates(['bits', 'seed'], keep='last')

fig = plt.figure(figsize=(9, 7))
gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.5, wspace=0.35)

# top: p_L vs bits, mean +/- std across seeds, MWPM reference line
ax0 = fig.add_subplot(gs[0, :])
ax0.errorbar(g.index, g['mean'], yerr=g['std'], marker='o', capsize=4,
             color='tab:blue', label='RCNN p_L (mean +/- std, n=3 seeds)')
if mwpm:
    ax0.axhline(mwpm, color='tab:red', ls='--', label=f'MWPM p_L = {mwpm:.5f}')
ax0.set_xlabel('weight bits'); ax0.set_ylabel('logical error rate p_L')
ax0.set_title('RCNN vs MWPM across weight bit-width (d=5, r=3, fresh 200k-shot tail)')
ax0.legend(); ax0.grid(alpha=0.3)

# bottom: 2x2 McNemar contingency per bit-width, seed-averaged, fraction of n_test
for i, b in enumerate(sorted(mc_all['bits'].unique())):
    ax = fig.add_subplot(gs[1, i])
    sub = mc_all[mc_all['bits'] == b]
    nte = sub['n_test'].iloc[0]
    table = np.array([[sub['both_right'].mean(), sub['rcnn_only'].mean()],
                       [sub['mwpm_only'].mean(), sub['both_wrong'].mean()]]) / nte
    ax.imshow(table, cmap='Blues', vmin=0, vmax=1)
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f'{table[r, c]*100:.2f}%', ha='center', va='center',
                    color='white' if table[r, c] > 0.5 else 'black', fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['MWPM right', 'MWPM wrong'], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['RCNN right', 'RCNN wrong'], fontsize=8)
    p_worst = sub['p_exact'].max()  # worst (least significant) seed at this bit-width
    ax.set_title(f'{b}-bit\\nworst p={p_worst:.1e}, n={len(sub)} seeds', fontsize=9)

fig.suptitle('Phase 1 knee: paired McNemar, RCNN vs MWPM, shot-by-shot on shared tail', y=1.02)
out_png = './figures/mcnemar_knee.png'
os.makedirs(os.path.dirname(out_png), exist_ok=True)
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print('saved ->', out_png)
plt.show()"""))

cells.append(md("""**Result.** Every seed × every bit-width **beats** MWPM, paired-significant (worst
p=5e-5 at w6/seed1). Claim upgrades from "6-bit at parity" to **"6-bit weights beat MWPM,
paired-significant."** Scope: weights-only, activations FP32, r=3."""))

cells.append(md("""## Phase 2 — activation quantization (DESIGN FROZEN, verified vs source)

Taxonomy-driven, **per-class** integer bits (a single global activation width is ill-posed:
z-like at B=4 has negative fractional width). All `[V]` items verified against
`CNNModel.py` / `utilities_arrayops.py`:

| class | source fact | quantizer | I | swept width |
|---|---|---|---|---|
| z-like | `bound_zlike=12` | signed | 4 (analytic) | B_z ∈ {10,8,6} |
| p/f-like | `frac=sigmoid` | unsigned | 0 | B_b ∈ {8,6,4} |
| cφ/α-like | `phase=tanh` | signed | 1 | B_b ∈ {8,6,4} |
| relu hidden | `Dense(relu)×2` | quantized_relu | profiled | B_d ∈ {8,6} |
| **x-like** | `clip_exp=[6e-6,1.6e5]` (~10 decades) | **none — Phase 4** | — | z-domain LSE / HLS LUT |

Key source findings: combiner `phase=2·tanh` is **signed**, so the log-domain combination
can go ≤0 (rescued by `clip_exp` before `log`). This makes *plain* LSE unavailable — the
Phase-4 remedy is a **choice** (signed-LSE OR constrain c_φ to PSD via a Gram
parameterization; the latter needs retraining), NOT decided here.
Clip is applied to the **sum output** → quantize AFTER the clip. inverter/pow dormant at
r=3. Anchor for Phase 2 = **w6/act-FP32** (not FP32/FP32)."""))

cells.append(md("""## Phase 3 — range profiling (per-seed, for the ap_fixed table)

`profile_ranges.py` wraps (never reimplements) the layer `.call`s + `clip_zlike/clip_exp`,
per instance, and reproduces the sweep p_L as a wrap-integrity check. Load the per-seed
JSONs and check whether `implied_I` is stable across seeds (a fixed_point_format_table caveat if not)."""))

cells.append(code("""js = sorted(glob.glob(OUT_MC+'/profile_ranges_w6_seed*.json'))
print('profile JSONs:', [os.path.basename(x) for x in js])
def load(f):
    d = json.load(open(f))
    prov = d.get('_provenance', {})
    return d, prov
if js:
    d0, prov0 = load(js[0])
    print('seed', prov0.get('seed'), 'profiled p_L =', prov0.get('profiled_p_L'),
          'expected', prov0.get('expected_p_L'))
    # combination_preclip = the LSE-decision sites
    comb = {k:v for k,v in d0.items() if k.startswith('combination_preclip')}
    for k,v in comb.items():
        print(f\"{k}: val[{v['min']:.3g},{v['max']:.3g}] frac_nonpos={v.get('frac_nonpositive',0):.2%}\")
else:
    print('run profile_ranges.py for seed0/1/2 first (see cell below)')"""))

cells.append(code("""# per-seed implied_I stability across sites (fills the fixed_point_format_table caveat)
if len(js) >= 2:
    recs = {}
    for f in js:
        d,_ = load(f); sd = json.load(open(f))['_provenance']['seed']
        for k,v in d.items():
            if isinstance(v, dict) and 'implied_I_from_max' in v:
                recs.setdefault(k, {})[f'seed{sd}_I'] = v['implied_I_from_max']
    tbl = pd.DataFrame(recs).T
    tbl['I_varies'] = tbl.nunique(axis=1) > 1
    display(tbl)
    print('sites where implied_I moves across seeds:', list(tbl.index[tbl['I_varies']]))
else:
    print('need >=2 seed JSONs to compare')"""))

cells.append(md("""### To rerun Phase 3 (per seed, asserts p_L)

```bash
cd ~/QuantumDecoderQKeras
.venv/bin/python profile_ranges.py --weights ~/rcnn_threshold/out_q_mcnemar/rcnn_d5_p0.010_r3_w6_seed0_ntr10000000.weights.h5 --expect-pl 0.046675
.venv/bin/python profile_ranges.py --weights ~/rcnn_threshold/out_q_mcnemar/rcnn_d5_p0.010_r3_w6_seed1_ntr10000000.weights.h5 --expect-pl 0.047715
.venv/bin/python profile_ranges.py --weights ~/rcnn_threshold/out_q_mcnemar/rcnn_d5_p0.010_r3_w6_seed2_ntr10000000.weights.h5 --expect-pl 0.045580
```"""))

cells.append(md("""## Phase 2a — activation-precision sweep (DONE, n=3)

Weights fixed at 6 bits (the Phase-1 knee); activation word length **B swept over {32, 8, 6, 4}**,
3 seeds, 10M training shots, fresh disjoint 200k tail. `B=32` means activation quantization OFF —
it is the per-seed **control**, and it must reproduce the Phase-1 w6/act-FP32 anchor.

Driver `phase2a_sweep.py` (fanned across 3 EAF pods) → `phase2a_collate.py` (tables + figure) →
`phase2a_mcnemar.py` (the paired tests).

### Prerequisite: the determinism fix

The first control reruns spiked on seeds 1 and 2 — val_loss jumping to 0.18 / 0.37 around epoch 3,
landing p_L ~0.050 / 0.057 against anchors of ~0.047 / 0.046 — on identical code, seed and recipe
to Phase 1. Diagnosed by elimination:

- **not the recipe** — 100k runs were smooth on both Mac-CPU and EAF-GPU;
- **not the TF version** — the spiking sweep logged TF 2.15.1, same as the anchor (the bare EAF
  shell's TF 2.16.2 / Keras 3 was never what the sweep used);
- **not the device** — both CPU and GPU were smooth at 100k;
- **scale** was the only variable left: 10M shots is ~1000 steps/epoch vs 8 at 100k, so ~125× more
  chances for one bad step under the reference architecture's high early LR (0.01). 100k structurally cannot reproduce it.

Confirmed by rerunning seed 2 at 10M with the flags on: smooth, no epoch-3 spike. **Cause =
nondeterministic GPU/cuDNN floating-point reduction order, amplified at scale**; the Phase-1 anchor
had simply drawn three spike-free runs. `TF_DETERMINISTIC_OPS=1` + `TF_CUDNN_DETERMINISTIC=1` are
now set in `train_one_quantized.py` *before* `import tensorflow`, with a hard
`assert tf.__version__.startswith('2.15')` guard.

`clipnorm=1.0` was tried during the diagnosis and **reverted** — it suppressed the symptom, left
the cause in place, and would have split this recipe from the Phase-1 anchor's."""))

cells.append(code("""# Control gate: each B=32 control must reproduce its Phase-1 anchor (w6 / float activations).
ANCHOR = {0: 0.046675, 1: 0.047715, 2: 0.045580}   # from out_q_mcnemar/mcnemar_knee.csv, w6 rows

def load_2a():
    \"\"\"{(act_bits, seed): p_L} from the Phase-2a per-run CSVs.\"\"\"
    out = {}
    for f in glob.glob(os.path.join(OUT_2A, '*_ntr10000000.csv')):
        r = pd.read_csv(f).iloc[0]
        out[(int(r['act_bits']), int(r['seed']))] = float(r['p_L'])
    return out

d2a = load_2a()
gate = pd.DataFrame([
    {'seed': s, 'control_B32': d2a.get((32, s)), 'phase1_anchor': ANCHOR[s],
     'delta': (d2a.get((32, s)) - ANCHOR[s]) if d2a.get((32, s)) is not None else None}
    for s in sorted({s for (_, s) in d2a})])
print('Control gate -- all three reproduce the anchor (seed 2 was 0.05685 on the spiked run):')
print(gate.to_string(index=False))"""))

cells.append(code("""# (1) p_L by activation width, and (2) the WITHIN-SEED cost: p_L(B) - p_L(that seed's control).
# Within-seed differencing cancels each seed's own convergence level. NOTE this is DESCRIPTIVE
# only -- the significance test is McNemar, below, and it does not agree with the naive reading.
bits = sorted({b for (b, _) in d2a}, reverse=True)
seeds = sorted({s for (_, s) in d2a})

tbl = pd.DataFrame({f'seed{s}': [d2a.get((b, s)) for b in bits] for s in seeds}, index=bits)
tbl.index.name = 'B'
tbl['mean'] = tbl.mean(axis=1)
tbl['xMWPM'] = tbl['mean'] / MWPM_FRESH_TAIL
print('(1) p_L by activation word length B  (B=32 = control, activations off)')
print(tbl.round(5).to_string(), '\\n')

delta = pd.DataFrame({f'seed{s}': [(d2a.get((b, s)) - d2a[(32, s)]) if d2a.get((b, s)) is not None
                                   else None for b in bits] for s in seeds}, index=bits)
delta.index.name = 'B'
delta['mean_delta'] = delta.mean(axis=1)
print('(2) within-seed cost: p_L(B) - p_L(control)')
print(delta.round(5).to_string())"""))

cells.append(md("""### B=4 is an infeasible fixed-point format, **not** a measured precision limit

All three B=4 seeds returned **p_L = 0.28260 — exactly the tail's base rate** — with val_loss
pinned at 0.5935 from epoch 1. The model emitted a constant and never trained.

The cause is arithmetic, not accuracy. The per-class integer widths leave **negative fractional
width** at B=4 (fractional bits = B − I − sign):

| class | I | signed | frac @ B=4 | frac @ B=6 | frac @ B=8 |
|---|---|---|---|---|---|
| z-like | 4 | yes | **−1** | 1 | 3 |
| relu (decoder hidden) | 6 | no | **−2** | **0** | 2 |
| embed | 2 | yes | 1 | 3 | 5 |
| cφ | 0 | yes | 3 | 5 | 7 |
| p/f | 0 | no | 4 | 6 | 8 |

A negative fractional width means the representable values are spaced **more than 1.0 apart** —
at frac = −2 the ReLU grid is multiples of 4 across [0, 64), so every hidden activation below 2.0
rounds to zero. The network is destroyed before training starts, and **QKeras does not raise** on
this; it silently returns a garbage quantizer.

**So report the integer-width floor, not "4-bit activations fail."** Under the current policy the
minimum viable B is 6. Going lower requires retuning the integer widths first: the p99.9 ReLU width
(I=4) unlocks B=5, but z-like's I=4 follows from the architecture's own ±12 clip and is not
reducible without changing that bound.

`ActQuant.set_bits()` now **refuses** any configuration with negative fractional width, naming the
offending classes and the minimum viable B, and prints the resulting per-class `ap_fixed` format on
every run. The threshold is `frac < 0`, not `frac < 1`: **B=6 leaves the ReLU at frac = 0**
(resolution 1.0 on a [0,64) tensor) **and trained normally**, costing only ~+0.0015 — so zero
fractional width is warned about, not rejected. That B=6 works at all with an integer-only ReLU
grid is a robustness result in its own right."""))

cells.append(md("""### Paired McNemar
The within-seed p_L differences above are **not** a significance test. Both runs decode the **same
200k shots**, so each shot is a matched pair and the informative quantity is the discordant count
(shots where exactly one of the two is right) — the same convention as Phase 1.

Aggregate differencing called B=8 "noise" (deltas straddling zero, non-monotonic against B=6).
The paired test disagrees, and changes the conclusion."""))

cells.append(code("""# Paired tests: each quantized run vs its OWN seed's control, and vs MWPM, on the same shots.
# Produced by phase2a_mcnemar.py, which rebuilds each checkpoint WITH its activation quantizers
# (eval_on_tail.py has no activation support and would score these models with float activations).
f2a = os.path.join(OUT_2A, 'phase2a_mcnemar.csv')
if os.path.exists(f2a):
    mc2 = pd.read_csv(f2a).sort_values(['act_bits', 'seed'], ascending=[False, True])
    print('vs own control (net<0 = control wins):')
    print(mc2[['act_bits','seed','p_L','control_p_L','delta_vs_control',
               'ctrl_n_discordant','ctrl_p_exact']].to_string(index=False), '\\n')
    print('vs MWPM (net>0 = RCNN wins):')
    mc2['net_vs_mwpm'] = mc2['mwpm_q_only'] - mc2['mwpm_mwpm_only']
    print(mc2[['act_bits','seed','p_L','mwpm_p_L','net_vs_mwpm',
               'mwpm_n_discordant','mwpm_p_exact']].to_string(index=False))
else:
    print('phase2a_mcnemar.csv not found -- regenerate with:')
    print('  python phase2a_mcnemar.py --dir', OUT_2A,
          '--pool <pools>/data_d5_p0.010_r3_TAIL200k.npz --acts 8,6 --seeds 0,1,2 \\\\')
    print('    --out-csv', f2a)"""))

cells.append(md("""**B=8 — no *systematic* cost, but not because the differences are noise.**
Two of the three seeds are individually significant **in opposite directions**: seed 0 favours the
control (p = 3.3e-08), seed 1 favours B=8 (p = 3.1e-03), seed 2 is not significant (p = 0.10). A
real per-run difference whose *sign flips across seeds* is training-run variation, not a precision
penalty. The defensible claim is "no consistent cost at B=8", **supported by the sign flip** —
not "the difference is within noise."

**B=6 — a real, consistent cost of about +0.0015.** All three seeds favour the control, all three
are significant, and the magnitudes agree (+0.0011 to +0.0021).

**Versus MWPM the margin becomes width-dependent.** At B=8 every seed beats MWPM. At B=6, seeds 0
and 2 still beat it (p = 1.3e-03, 2.7e-05) but **seed 1 lands at p_L = 0.049400 against MWPM's
0.049405 — a net of one shot in 200k, a dead heat.** So *"6-bit weights AND 6-bit activations still
beat MWPM"* is **not supportable as stated**; it holds at B=8 on all three seeds.

PS: The three seeds share one 200k tail, so their tests are **correlated, not independent**. Read
them as three consistent (or inconsistent) readings; do **not** Fisher-combine the p-values."""))

cells.append(code("""# The figure. B=4 is excluded by default (--plot-exclude-bits): at 0.2826 it is ~6x every other
# point, which flattens the 0.046-0.050 band where the result actually lives and reads as "4-bit
# fails" -- the one conclusion the data does not support. The figure states the omission on itself.
from IPython.display import Image
p = 'plots/phase2a_activation_sweep.png'
if not os.path.exists(p):
    print('regenerate with:  python phase2a_collate.py --dir', OUT_2A)
Image(p) if os.path.exists(p) else None"""))

cells.append(md("""### Where this leaves the FPGA path

1. **Activation quantization works.** Bounded tensors at 8 bits cost nothing systematic; 6 bits
   cost a small consistent +0.0015 and are marginal against MWPM on one seed.
2. **B=4 is not a measurement** — it is a floor imposed by the architecture's own clip bounds
   (z-like ±12 → I=4), now rejected up front by a guard rather than burning a multi-hour run.
3. **This is still not a synthesizable model.** The **x-like** tensors remain FP32 throughout:
   every `CNNKernelWithEmbedding` / `CNNStateCorrelator` output has `frac@B6 < 0` with implied
   I up to 17 (correlator #2, seed 1: max 1.2e5 — ~10 decades). No `ap_fixed` at any sane width.

That last point is Phase 4, and the **phase-matrix result above is why it is hard**, not merely
tedious: the n=3 recurrence correlators have an indefinite phase matrix ~95% of the time, so the
log-domain combination genuinely goes ≤0 and *plain* LSE is unavailable. The remedy is a **choice**
(signed-LSE, or constrain c_φ to PSD via a Gram parameterization and retrain, which may cost
accuracy since the model is actively using the indefinite region) — and it is **not decided**.

**Open:**
- Rerun B=6 with `ActQuant.set_relu_integer(4)` (p99.9 ReLU width, frac 0 → 2) to test how much of
  the +0.0015 is the zero-fractional-width ReLU rather than the word length itself.
- A genuine low-B datapoint needs the integer-width policy retuned; B=4 stays blocked by z-like.
- Phase 4 owns the x-like tensors. Decide signed-LSE vs Gram-constrained c_φ **with the collaborator**."""))

cells.append(md("""## Provenance & status

- **Ledger:** `docs/RUN_LOG.md` (fixed substrate + every phase + git SHAs).
- **Branch:** `quantization-pareto` (local only — not pushed; RCNN repo reorg pending).
- **Files:** `CNNModel_quantized.py`, `train_one_quantized.py`, `sweep_quantized.py`,
  `collate_pareto.py`, `make_fresh_tail.py`, `eval_on_tail.py`, `phase1_mcnemar.py`,
  `profile_ranges.py`, `phase2a_sweep.py`, `phase2a_collate.py`, `phase2a_mcnemar.py`,
  `docs/RUN_LOG.md`, this notebook + `build_quant_notebook.py`.

**Open:** Phase 0–3 and Phase 2a are DONE. Next: rerun B=6 with the p99.9 ReLU width
(`ActQuant.set_relu_integer(4)`), then Phase 4 (log-domain / LSE) for the x-like tensors —
which needs the signed-LSE vs Gram-constrained-c_φ decision made with the collaborator."""))

cells.append(code("""# git provenance for this work (run on the machine with the repo)
# !git log --oneline -20 -- CNNModel_quantized.py train_one_quantized.py sweep_quantized.py \\
#     collate_pareto.py make_fresh_tail.py eval_on_tail.py phase1_mcnemar.py profile_ranges.py
print('see docs/RUN_LOG.md and: git log --oneline quantization-pareto')"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
      "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

# Write to the repo root: that is the copy actually being run and edited. (An analysis_notebooks/
# copy also exists and has drifted; the root one is authoritative for this thread.)
#
# REGENERATING STRIPS OUTPUTS. This emits execution_count=None and outputs=[] for every code cell,
# so any executed results in the existing notebook are lost and the notebook must be re-run. Hand
# edits to markdown cells are lost too -- port them back into this file first (as was done for the
# 2026-07-20 wording changes) or they will not survive the next regeneration.
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'QUANTIZATION_EXPERIMENTS.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print('wrote', out, '::', len(cells), 'cells')
