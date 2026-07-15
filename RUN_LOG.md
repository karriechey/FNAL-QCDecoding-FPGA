# QAT weight/activation quantization — RUN LOG

*Created: 2026-07-13 | Last modified: 2026-07-15*

Reproducibility ledger for the FullRCNNModel quantization Pareto (FPGA / hls4ml handoff,
collaborator Giuseppe). One row per run: date, git SHA, command, host, result line.
Rule: no result is "real" without git SHA + pool gen-seed + saved weights all pinned here.

## Fixed substrate (never regenerate)

- **Architecture:** `FullRCNNModel('ZL', d=5, k=3, r=3, [100,100], npol=2, ...)`, ~51,547 params.
  Surface code rotated_memory_z, d=5, p=0.010, rounds=3.
- **Train pool:** `~/rcnn_threshold/pools/data_d5_p0.010_r3.npz` (10.01M shots, gen-seed **42**).
  Positional split: `tr=slice(0,ntr)`, `te=slice(N-nte,N)`.
- **Eval tail (all phases):** `~/rcnn_threshold/pools/data_d5_p0.010_r3_TAIL200k.npz`
  200k shots, gen-seed **43** (disjoint from training's 42). Built by `make_fresh_tail.py`.
  THE eval substrate for every phase — do not regenerate, do not swap.
- **FP32 anchor weights:** `~/rcnn_threshold/out_t200k_w/rcnn_d5_p0.010_r3_seed{0,1,2}_ntr10000000.weights.h5`
  (existing 10M FP32 models; reused, never retrained).
- **Quantizer:** `quantized_bits(B, 1)` weights-only = 1 sign + 1 integer + (B-2) frac,
  range ~[-2,2). See `QUANTIZATION_SCHEME.md`.
- **Env:** EAF Jupyter, `.venv/bin/python` (qkeras 0.9.0, TF 2.15.1, Keras 2.15.0).
  GPU = A100 MIG slice (`3g.40gb` = 3/7 of 80GB A100). ~195s/epoch, ~2.8hr/run.
  `kill -9` on a CUDA job leaks GPU memory — restart the Named Server (not pkill) to free it;
  verify `nvidia-smi` shows ~114MiB before relaunch.

---

## Phase 0 — implementation + gates (2026-07, DONE)

- Path B QAT re-homed into `CNNModel_quantized.py`; `CNNModel.py` byte-for-byte pristine.
- Gates: None-path == original (max|diff|=0.0); quantization bites; grads on 35 quantized
  vars; FP32 10M weights load into None-model bit-exact (prob corr 1.0, max|dprob|=0).
- Tail contamination found + fixed: old 200k tail overlapped 10M-train by 190k shots
  (in-sample optimism ~+0.0016 p_L). Fixed via fresh gen-seed-43 tail.
- Clean FP32 anchor (fresh tail, 3 seeds): RCNN p_L 0.0462±0.0003 vs MWPM 0.0494;
  McNemar net +579..+775 wins, p~1e-21..1e-12 — significant beat.

## Phase 1 — harden Exp 1 (referee-proof baseline)  [DONE 2026-07-13]

Objective: same-tail confirmation for every row + paired McNemar 2x2 at the knee (6, 8 bit)
vs MWPM. Licenses "parity"/"beats" with discordant counts, not assertion.

- Driver: `phase1_mcnemar.py` — retrained bits {6,8} × seeds {0,1,2} WITH --save-weights into
  `~/rcnn_threshold/out_q_mcnemar/` (separate dir, does NOT clobber Pareto CSVs in out_q/),
  then `eval_on_tail.py --mcnemar --pool <fresh tail>` → `out_q_mcnemar/mcnemar_knee.csv`.
- FP32 (32-bit) McNemar already in `out_q/fp32_anchor.csv` (sweep anchor ran --mcnemar).

| date | git SHA | run | host | result |
|------|---------|-----|------|--------|
| 2026-07-13 | 46079f5 | w6/w8 × seed0,1,2 McNemar, fresh 200k tail | EAF (3 pods) | ALL 6 beat MWPM, paired-significant. |

**Phase 1 McNemar result (`out_q_mcnemar/mcnemar_knee.csv`, MWPM p_L=0.049405 fresh tail):**

| bits | seed | p_L | ratio | RCNN-only | MWPM-only | net wins | p_exact |
|---|---|---|---|---|---|---|---|
| 6 | 0 | 0.046675 | 0.945 | 3619 | 3073 | +546 | 2.6e-11 |
| 6 | 1 | 0.047715 | 0.966 | 3625 | 3287 | +338 | 5.0e-5 |
| 6 | 2 | 0.045580 | 0.923 | 3696 | 2931 | +765 | 5.7e-21 |
| 8 | 0 | 0.046060 | 0.932 | 3658 | 2989 | +669 | 2.4e-16 |
| 8 | 1 | 0.045645 | 0.924 | 3673 | 2921 | +752 | 2.1e-20 |
| 8 | 2 | 0.046595 | 0.943 | 3683 | 3121 | +562 | 1.0e-11 |

Every seed × every bit-width: RCNN wins MORE discordant shots than MWPM, paired McNemar
significant (worst p=5e-5 at w6/seed1 ≪ 0.05). Refutes "6-bit is inside test-set noise" —
shot-by-shot, 6-bit weights BEAT MWPM, all 3 seeds. Claim upgrades from "6-bit at parity" to
**"6-bit weights beat MWPM, paired-significant."** Scope: weights-only, activations FP32, r=3.
NOTE: `w6_seed0` row is duplicated in the CSV (relaunch after a GPU-OOM race) — identical
values, cosmetic; dedup by (bits,seed). Weights saved: `out_q_mcnemar/*.weights.h5`.

## Phase 2 — activation quantization sweep  [DESIGN FROZEN, verified against source]

Site inventory + all `[V]` VERIFIED against CNNModel.py / utilities_arrayops.py:
- `bound_zlike=12` → **z-like I=4** analytic (no profiling).
- `clip_exp=[1/e^12, e^12]=[6e-6, 1.6e5]` → **x-like ~10 decades, un-quantizable in x-domain → Phase 4** (z-domain LSE rewrite or HLS LUT).
- `phase_activation=tanh`, ×2 → **(−2,2) signed**; added to combiner sum → sum can go ≤0, rescued by `clip_exp` before `log`. Signed quadratic sᵀCs → not clean LSE.
- clip applied to the **sum output** (`res=clip_zlike(res)`) → **quantize AFTER the clip**.
- `frac=sigmoid`→p/f I=0; `cphi/alpha=tanh`→I=1; inverter/pow **dormant at r=3**; **inputs binary** (no aux real input).
- StateDecoder = `Dense(100,relu)×2` + `Dense(1,sigmoid)` → **only dec_h1_out/dec_h2_out need profiling**.

Quantizer assignment (integer bits fixed by taxonomy; only relu I profiled):
| class | quantizer | I | swept width |
|---|---|---|---|
| z-like | signed qb | 4 | B_z ∈ {10,8,6} (B_z≥6 or frac<1) |
| p/f-like | unsigned qb | 0 | B_b ∈ {8,6,4} |
| cφ/α-like | signed qb | 1 | B_b ∈ {8,6,4} |
| relu hidden | quantized_relu | profiled | B_d ∈ {8,6} |
| x-like | NONE (Phase 4) | — | — |

Sweep = per-class budget (NOT a single global `a`), L-slice. Anchor = **w6/act-FP32** (not FP32/FP32).
Run tag encodes w + every activation B. Smoke test asserts: no NaN through log/sqrt post-quant.

## Phase 3 — range profiling  [DONE 2026-07-14, 3 seeds, per-instance]

`profile_ranges.py` on the w6 model, all custom-layer instances wrapped (pure-wrap: p_L
EXACT vs sweep all 3 seeds), clip sites keyed per instance (0 #agg of 91). Outputs
`profile_ranges_w6_seed{0,1,2}.json`; collate `collate_profile.py`.

**STRUCTURAL RESULT (measured, 3 seeds) — phase matrix C definiteness:**
Measured n (state-count) + per-shot lambda_min(C) (C: diag 1, off-diag = actual c_phi):
```
  correlator   n   lambda_min(C)<0        combination<=0 (symptom)
  #0-#4        2   0.00% all seeds        0.00%     <- provably PSD; also proves C-assembly correct
  #5-#9        3   90-98% all seeds       2-19%
```
The RCNN phase matrix is UNCONSTRAINED -> PSD for the n=2 (lead-in) correlators, indefinite
~95% for the n=3 (recurrence) correlators. This is the CAUSE of the log-domain combination
going <=0. Remedy is a Phase-4 CHOICE, NOT decided: (a) signed-LSE, or (b) constrain c_phi to
PSD (Gram parameterization) -> plain LSE works but needs retrain, may cost accuracy (model uses
the indefinite region). n=2 correctness assert (lambda_min=1-|c|>0) PASSED. Necessity bound
frac(lam<0)>=frac(nonpos) holds (n=2 fp-dust needs 1e-3 tol).

**x-like un-quantizable (per instance, 3 seeds):** every CNNKernelWithEmbedding / CNNStateCorrelator
output has frac@B6<0, I(max) up to 17 (correlator #2 seed1: max 1.2e5). No ap_fixed at sane B.
-> log-domain (Phase 4). x-like I moves wildly across seeds (irrelevant, un-quantizable).

**Phase-2 bounded quantizer config (seed-stable to +-1, take max across seeds):**
z-like I=4 (post-clip), decoder relu I=5-6, DetectorEvent embedder I=2-3, DetectorBit I=1,
Triplet/sigmoid I=0. relRMSE@B6 mostly 2-8% (I from p99.9 halves the worst).

**Accumulator finding (Phase-4 HLS, NOT Phase-2):** zlike_preclip absmax = 43.5 / 41.1 / 72
across seeds -- the z_e+z_m and correlator pre-clip accumulators EXCEED the assumed +-24;
need I=7 (2^7=128). Post-clip z-like (the QAT quantizer target) stays ⊆+-12 -> I=4 valid.

Deliverable: per-layer ap_fixed table = the numerical contract for Giuseppe's HLS handoff.

---

## Completed sweep — Step 2 Pareto (2026-07-10, DONE)

Driver `sweep_quantized.py`, n=3, fresh 200k tail, 10M shots. Fanned across 3 Named Servers.
Collate `collate_pareto.py` → `plots/rcnn_d5_r3_qat_pareto.png`, `out_q/`.

```
 bits  size_KB  mean_pL   comb_err   xMWPM   n
   32   201.4   0.04614   0.00056   0.934    3   (FP32 anchor, reused)
    8    50.3   0.04594   0.00053   0.930    3   lossless = FP32
    6    37.8   0.04718   0.00056   0.955    3   KNEE — beats MWPM, paired-significant (Phase 1)
    4    25.2   0.05464   0.00148   1.106    3   ~3.4σ above MWPM
    3    18.9   0.06116   0.00170   1.238    3
    2    12.6   0.12796   0.00709   2.590    3   collapsed
 MWPM = 0.04940 ± 0.00048  (fresh tail, n_test=200000)
```

Headline: 8-bit lossless, **6-bit is the knee** (37.8 KB, 5.3× smaller than FP32, beats MWPM).
Sharp cliff 6→4. Variance blows up at low bits. Weights-only ceiling; Phase 2 adds activations.
