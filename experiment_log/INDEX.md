# Book-keeping — paper source material

Everything needed to write the paper, in one place. Generated 2026-07-13.
Organized around what the collaborator asked for in the 2026-07-09 meeting (points 3–7).

**Rule of this folder:** it is a *reading* archive, not a working directory.
Scripts still read/write the live `results/`, `logs/`, `plots/` at repo root.
`docs/` was **moved** here (nothing imports it). `plots/`, `logs/`, `results_snapshot/`
are **copies** frozen at 2026-07-13 — re-copy if you rerun anything.

---

## Doc status — which of these can I trust? (audited 2026-07-13)

Dates are **git commit dates** where tracked, file mtime otherwise. Every doc now carries a
status banner in its own header too.

| Doc | Date | Status |
|---|---|---|
| `docs/RUN_LOG.md` | 2026-07-13 | ✅ **CURRENT — authoritative.** Where docs disagree, this wins. |
| `docs/QUANTIZATION_SCHEME.md` | 2026-07-13 | ✅ **CURRENT — rewritten.** Was stale (see below). Part 1 = the paper's method. |
| `docs/RCNN_d5_r3_ladder_REFERENCE.md` | 2026-06-30 | ✅ **CURRENT.** The paper's data-volume result (EAF/GPU, 200k tail). |
| `docs/DEMO_CRUMBLE.md` | 2026-06-19 | ➖ Valid, but a demo guide — no paper numbers. |
| `docs/WorkLog-2026-06-22-2122.md` | 2026-06-22 | 📜 Historical. Plans, not results. Threshold study is **unfinished**. |
| `docs/SWEEP_SUMMARY.md` | 2026-06-23 | ⚠️ **SUPERSEDED.** Mac-CPU sweep, 10k tail. Cite only for the ~31 h CPU wall-clock. |
| `docs/EXPERIMENTS.md` | 2026-06-05 | ❌ **STALE — do not cite.** Empty notes, dead file refs, documents the retired QDense toy. |
| `notebooks/EXPERIMENTS_LOG.ipynb` | 2026-07-07 | ✅ Recent. |
| `notebooks/WORKLOG_pL_vs_N_sweep.ipynb` | 2026-06-23 | ⚠️ Pairs with the superseded CPU sweep. |

### The `QUANTIZATION_SCHEME.md` problem (found 2026-07-13, now fixed)

The original doc described **the wrong experiment**. It sourced from `QATfinal_notebook.ipynb`
(`build_qdense_cnn` / `build_qdense_rcnn`) and documented a 2-D `(w_bits, a_bits)` sweep with
`quantized_relu` activations. **That is the QDense-toy lineage** — a stand-in architecture of
plain QDense layers, retired; `build_qdense_*` appears in no current `.py`.

Verified against the code, the experiment the paper actually reports is:

- **Codebase:** `CNNModel_quantized.py` — point-of-use `quantized_bits(B, 1)` patched onto
  the reference architecture's real custom layers (`QDense` only inside `StateDecoder`).
- **Weights only.** No `quantized_relu`, no `QActivation` anywhere. Activations stay FP32.
- **1-D sweep:** `WEIGHT_BITS = [8, 6, 4, 3, 2]` + FP32 anchor. There is **no `a_bits` axis**.
- **QAT with a fake-quant forward pass:** FP32 master weights, quantized at point of use before
  every matmul, straight-through gradients. *Not* post-training quantization. ← the collaborator's question 1.

Consequence for the paper: the size numbers (201 KB → 12.6 KB) are **weight-storage only**, and
activation quantization remains un-done (Phase 2). Say so explicitly.

---

## Meeting point 3 — "start writing the paper now"

Draft sections and the material that feeds each:

| Paper section | Source in this folder |
|---|---|
| Architecture | `docs/RCNN_d5_r3_ladder_REFERENCE.md` §2 (FullRCNNModel, 51,547 params) |
| Dataset / noise model | `docs/SWEEP_SUMMARY.md` (config table), `docs/RCNN_d5_r3_ladder_REFERENCE.md` |
| Training details | `docs/RUN_LOG.md` "Fixed substrate" + `docs/SWEEP_SUMMARY.md` |
| Quantization method | `docs/QUANTIZATION_SCHEME.md` (**the section the collaborator kept circling back to**) |
| Performance results | `docs/RUN_LOG.md` Pareto table, `results_snapshot/`, `plots/` |
| Hardware outlook | `docs/RUN_LOG.md` Phase 2 / Phase 3 plans |

## Meeting point 4 — full-precision result (the headline)

From `docs/RUN_LOG.md` Phase 0 + `docs/RCNN_d5_r3_ladder_REFERENCE.md`:

- RCNN FP32, 10M shots, 3 seeds: **p_L = 0.0462 ± 0.0003**
- MWPM (PyMatching), same 200k tail: **p_L = 0.0494 ± 0.0005**
- Ratio 0.934×. Paired **McNemar** confirms the beat (net +579..+775, p ~ 1e-21..1e-12).
- Eval tail: 200k shots, gen-seed 43, **disjoint** from the training pool (gen-seed 42).
  An earlier 200k tail overlapped training by 190k shots (+0.0016 p_L optimism) — fixed.

## Meeting point 5 — training-set size / saturation

`docs/RCNN_d5_r3_ladder_REFERENCE.md` (the EAF ladder) and `docs/SWEEP_SUMMARY.md`
(the earlier 31-hour Mac-CPU sweep). Sizes 100k → 10M → 20M × 3–5 seeds.
Result: parity with MWPM around 5M, ~5% below MWPM at 10M, **no further gain at 20M**
(variance blows up ~5×) → structural floor by 10M, not data-limited.
Figures: `plots/rcnn_d5_pl_vs_n_v2.png`, `plots/rcnn_d5_pl_vs_mwpm.png`.

## Meeting point 6 — training details the collaborator wants recorded

Pinned in `docs/RUN_LOG.md`; restated here because these are the exact numbers the paper needs:

| Item | Value |
|---|---|
| Hardware (current) | Fermilab EAF, **A100 80GB in MIG `3g.40gb` slice** (3/7 of card) |
| Hardware (earlier sweep) | Apple Silicon, **CPU-only** — see `docs/SWEEP_SUMMARY.md` |
| Wall-clock | **~195 s/epoch, ~2.8 h/run** (10M shots, EAF); the CPU sweep was ~31 h for 25 runs |
| Training pool | 10.01M shots, gen-seed 42, `data_d5_p0.010_r3.npz` |
| Holdout | **200k shots, gen-seed 43**, disjoint — never used for training or validation |
| Epochs / batch | 50 epochs, batch 10,000 (`train_one_quantized.py` defaults) |
| Seeds | n = 3 per point (sweep); n = 5 in the earlier CPU sweep |
| Physics config | rotated_memory_z, d = 5, p = 0.010, rounds = 3 (**above threshold**, ~0.0065) |
| Stack | TF 2.15.1 / Keras 2.15.0 / QKeras 0.9.0 / Stim / PyMatching |

## Meeting point 7 — parameter count / compactness

**51,547 parameters.** `FullRCNNModel('ZL', d=5, k=3, r=3, [100,100], npol=2)`.
Model size vs precision (weights-only QAT, from `docs/RUN_LOG.md`):

| bits | size | mean p_L | vs MWPM |
|---|---|---|---|
| 32 (FP32 anchor) | 201.4 KB | 0.04614 | 0.934× |
| 8 | 50.3 KB | 0.04594 | 0.930× — **lossless** |
| 6 | 37.8 KB | 0.04718 | 0.955× — **the knee**, 5.3× smaller, still beats MWPM |
| 4 | 25.2 KB | 0.05464 | 1.106× |
| 3 | 18.9 KB | 0.06116 | 1.238× |
| 2 | 12.6 KB | 0.12796 | 2.590× — collapsed |

Quantizer: `quantized_bits(B, 1)` weights-only = 1 sign + 1 integer + (B−2) fractional,
range ≈ [−2, 2). Trainable variables stay FP32; the forward pass uses quantized weights
→ this is **QAT (fake-quant forward)**, not post-training quantization. That is the answer
to the collaborator's question 1. Full detail + the sign-bit subtlety: `docs/QUANTIZATION_SCHEME.md`.

---

## EAF snapshot — pulled 2026-07-13

`results_snapshot/eaf/` holds the run outputs that live outside the repo on EAF
(`~/rcnn_threshold/`) and were never in git. 116 files:

- `rcnn_threshold/out_q/` — the QAT Pareto per-run CSVs + `.history.json` (5 bits × 3 seeds)
  and `fp32_anchor.csv`. **These are the raw rows behind the bits/size/p_L table.**
- `rcnn_threshold/out/` — the data-volume ladder per-run CSVs (100k → 10M × 3 seeds).
  ⚠️ Contains one `r5_seed0_ntr200000` run — that's the old smoke test, **not ladder data**.
- `rcnn_threshold/out_q_mcnemar/` — the w6 paired-McNemar results + weights.
- `rcnn_threshold/pools/data_d5_p0.010_r3_TAIL200k.npz` — the 200k eval tail (gen-seed 43),
  the substrate every published number is scored on.
- `QuantumDecoderQKeras/plots/` — EAF-generated figures, incl. `rcnn_d5_r3_qat_pareto.png`
  and `rcnn_d5_p0.010_r3_ladder.png` (not present in the local `plots/`).

**This is a snapshot, not live state.** Re-pull after any run finishes.

## What is NOT here yet (gaps to close before writing)

1. **w8 McNemar is still running** (seed2 pod, as of 2026-07-13). Until it lands,
   *"8-bit is lossless"* is mean-vs-mean, not a paired test. Confirm w8 seed0/seed1
   were launched too. Re-pull `out_q_mcnemar/mcnemar_knee.csv` when done.
   (w6 knee **is** done and confirmed: 3/3 seeds beat MWPM — see `docs/RUN_LOG.md`.)
2. **⚠️ MWPM baseline discrepancy.** Sweep CSVs say `mwpm_p_L = 0.0451`; anchor and
   McNemar CSVs say `0.049405` — same tail, same config. `RUN_LOG.md`'s table used the
   correct 0.04940, so published ratios are fine, but never compute xMWPM from the sweep
   CSVs' own column. Find the code path that wrote 0.0451. See `docs/RUN_LOG.md`.
3. **Activation quantization (Phase 2) not run.** Current numbers are weights-only.
4. **Per-layer fixed-point range profiling (Phase 3) not run** — needed for Giuseppe's
   `ap_fixed<B,I>` handoff.
5. `docs/EXPERIMENTS.md` has empty **Notes** fields and points at
   `results/results_scaling.csv`, which does not exist. Stale — trust `RUN_LOG.md` instead.

## Folder map

- `docs/` — the written record (moved from repo root)
  - `RUN_LOG.md` — the reproducibility ledger; **the authoritative one**
  - `QUANTIZATION_SCHEME.md` — quantization methods note (paper section + Giuseppe)
  - `RCNN_d5_r3_ladder_REFERENCE.md` — the EAF data-volume ladder study
  - `SWEEP_SUMMARY.md` — the earlier 31-hour CPU p_L-vs-N sweep
  - `EXPERIMENTS.md` — early tracking log (**stale**, see gap 5)
  - `DEMO_CRUMBLE.md` — Crumble spacetime demo guide
  - `WorkLog-2026-06-22-2122.md` — threshold-overlay worklog
- `notebooks/` — narrative record notebooks (moved)
  - `EXPERIMENTS_LOG.ipynb`, `WORKLOG_pL_vs_N_sweep.ipynb`
- `plots/` — figure copies, frozen 2026-07-13
- `logs/` — stdout run logs, frozen 2026-07-13
- `results_snapshot/` — CSV/JSON result copies, frozen 2026-07-13
