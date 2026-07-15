#!/usr/bin/env python3
# Created: 2026-07-14
# Last modified: 2026-07-15
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

This notebook is **live**: cells load the result files from disk and rebuild every number,
so it stays honest. The authoritative ledger is `RUN_LOG.md`; this is the runnable view.

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
# Auto-detect: prefer the in-repo ./rcnn_threshold (data pulled from EAF into the repo),
# fall back to ~/rcnn_threshold when running ON EAF. Works in both places.
BASE = 'rcnn_threshold' if os.path.isdir('rcnn_threshold') else os.path.expanduser('~/rcnn_threshold')
OUT_Q  = os.path.join(BASE, 'out_q')           # Step-2 Pareto CSVs
OUT_MC = os.path.join(BASE, 'out_q_mcnemar')   # Phase 1 McNemar + Phase 3 JSON
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 40)
print('BASE :', BASE)
print('out_q     :', OUT_Q, '->', len(glob.glob(OUT_Q+'/*.csv')), 'csv')
print('out_q_mcnemar:', OUT_MC, '->', sorted(os.path.basename(f) for f in glob.glob(OUT_MC+'/*'))[:6])"""))

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
low bits. This is the **weights-only** ceiling — activations still FP32."""))

cells.append(md("""## Phase 1 — paired McNemar at the knee (DONE)

`phase1_mcnemar.py` retrained {6,8} × {0,1,2} with `--save-weights` into `out_q_mcnemar/`
(no clobber of the Pareto CSVs), then `eval_on_tail.py --mcnemar` on the shared fresh tail.
The paired test licenses "beats MWPM" with discordant shot counts, not error-bar overlap."""))

cells.append(code("""mc = pd.read_csv(os.path.join(OUT_MC,'mcnemar_knee.csv')).drop_duplicates('weights', keep='last')
mc['bits'] = mc['weights'].str.extract(r'_w(\\d+)_').astype(int)
mc['seed'] = mc['weights'].str.extract(r'seed(\\d+)_').astype(int)
show = mc[['bits','seed','p_L','mwpm_p_L','ratio','rcnn_only','mwpm_only','net_rcnn_wins','p_exact']]
show.sort_values(['bits','seed']).reset_index(drop=True)"""))

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

cells.append(md("""### How to (re)run Phase 3 (per seed, asserts p_L)

```bash
cd ~/QuantumDecoderQKeras
.venv/bin/python profile_ranges.py --weights ~/rcnn_threshold/out_q_mcnemar/rcnn_d5_p0.010_r3_w6_seed0_ntr10000000.weights.h5 --expect-pl 0.046675
.venv/bin/python profile_ranges.py --weights ~/rcnn_threshold/out_q_mcnemar/rcnn_d5_p0.010_r3_w6_seed1_ntr10000000.weights.h5 --expect-pl 0.047715
.venv/bin/python profile_ranges.py --weights ~/rcnn_threshold/out_q_mcnemar/rcnn_d5_p0.010_r3_w6_seed2_ntr10000000.weights.h5 --expect-pl 0.045580
```"""))

cells.append(md("""## Provenance & status

- **Ledger:** `RUN_LOG.md` (fixed substrate + every phase + git SHAs).
- **Branch:** `quantization-pareto` (local only — not pushed; RCNN repo reorg pending).
- **Files:** `CNNModel_quantized.py`, `train_one_quantized.py`, `sweep_quantized.py`,
  `collate_pareto.py`, `make_fresh_tail.py`, `eval_on_tail.py`, `phase1_mcnemar.py`,
  `profile_ranges.py`, `RUN_LOG.md`, this notebook + `build_quant_notebook.py`.

**Open:** run Phase 3 on 3 seeds → read design → implement `ActQuant` + Phase 2 sweep."""))

cells.append(code("""# git provenance for this work (run on the machine with the repo)
# !git log --oneline -20 -- CNNModel_quantized.py train_one_quantized.py sweep_quantized.py \\
#     collate_pareto.py make_fresh_tail.py eval_on_tail.py phase1_mcnemar.py profile_ranges.py
print('see RUN_LOG.md and: git log --oneline quantization-pareto')"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
      "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'QUANTIZATION_EXPERIMENTS.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print('wrote', out, '::', len(cells), 'cells')
