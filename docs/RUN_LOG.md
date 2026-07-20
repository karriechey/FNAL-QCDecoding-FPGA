# QAT weight/activation quantization — RUN LOG

*Created: 2026-07-13 | Last modified: 2026-07-20*
*Last verified against code: 072e010, 2026-07-20*

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

**Phase-2 bounded quantizer config (fixed_point_format_table, collate_profile.py section 4;
seed-stable under max-across-seeds):** Detector{Bit,Event} embedders type 'embedding', PROFILED
I=1/2 (P, not cphi/alpha -- see correction below); combiners z-like I=4 (A); Triplet embedder +
decoder sigmoid p/f I=0 (A); dec_in z-like I=4 (A); decoder relu dec_layer0 I=4 / dec_layer1 I=5
(P, profiled). relRMSE@B6 mostly 2-8%.

**Taxonomy correction (found by reading source + profiling):** the Detector{Bit,Event}StateEmbedder
OUTPUTS are NOT cphi/alpha. embed_pol_state (CNNModel.py:653/824) returns (-1,1) diagonal sub-entries
+ an UNBOUNDED non-diagonal polynomial Sum({x^2,x,1}*embedding_params) (params are exp/sigmoid of
clip_zlike). So these outputs have no analytic bound -> PROFILED I (observed DetectorBit I=1,
DetectorEvent I=2; the difference is learned-param magnitude, NOT a type difference or a "bound
violation"). An earlier draft mislabeled them cphi/alpha I=1 and reported a spurious "violation" --
retracted. The one REAL analytic-bound finding is the +-24 z-sum accumulator below.

**Accumulator finding (Phase-4 HLS, NOT Phase-2):** zlike_preclip absmax = 43.5 / 41.1 / 72
across seeds -- the z_e+z_m and correlator pre-clip accumulators EXCEED the assumed +-24;
need I=7 (2^7=128). Post-clip z-like (the QAT quantizer target) stays ⊆+-12 -> I=4 valid.

Deliverable: per-layer ap_fixed table = the numerical contract for Giuseppe's HLS handoff.

## Phase 2a — activation-QAT build notes (pre-build decisions, NOT yet run)

**Decoder ReLU seeding decision (make it NOW, don't discover it):** seed ActQuant._int_bits for the
two decoder ReLU sites (dec_layer0/1) at the ABS-MAX I=6 (safe, nothing clips) for the FIRST sweep,
NOT the p99.9 I=4/5 in the format table. Reason: start at p99.9 and if Phase 2a trains badly you can't
tell clip-loss from quant-loss. Start safe (I=6), tighten to p99.9 as an optimization ONCE the model
is confirmed to train. One-line change to the ActQuant seed.

**Honest scope of Phase 2a:** it quantizes weights (6-bit) + the ~37 BOUNDED activation sites from the
fixed_point_format_table; x-like intermediates stay FP32. So the output is a PARTIALLY quantized model
-- reportable (Fig 2: accuracy vs activation precision) but NOT synthesizable. The synthesizable model
needs Phase 4 (x-domain LSE rewrite) -- that is the Giuseppe deliverable. Frame as "all bounded tensors
quantized; exponential-domain intermediates addressed in Phase 4", NOT "model is now fixed-point".

**Predicted failure mode (mechanism, not a bug to hunt):** the straight-through estimator is shaky
upstream of an exp (Δz -> e^Δ multiplicative). If the activation-bit sweep DIVERGES at low bits, that is
the STE-through-exp mechanism, expected -- not a code bug.

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

---

## Phase 2a — activation-precision sweep (2026-07-19/20, DONE)

Weights fixed at 6 bits (the Phase-1 knee); activation word length B swept over {32, 8, 6, 4},
3 seeds, 10M training shots, fresh disjoint 200k tail. B=32 means activation quantization OFF and
is the per-seed control. Driver `phase2a_sweep.py`, fanned across 3 EAF pods; collation
`phase2a_collate.py`; paired tests `phase2a_mcnemar.py`.

### Determinism fix (prerequisite, 2026-07-19)

The first control reruns spiked on seeds 1 and 2 — val_loss jumping to 0.18 / 0.37 around epoch 3
and never recovering, landing p_L ~0.050 / 0.057 against anchors of ~0.047 / 0.046 — on identical
code, seed and recipe to Phase 1. Diagnosed by elimination: not the recipe (100k runs were smooth
on both Mac-CPU and EAF-GPU), not the TF version (the spiking sweep logged TF 2.15.1, same as the
anchor — the bare EAF shell's TF 2.16.2 / Keras 3 was never what the sweep used), not the device.
The only remaining variable was scale: 10M shots is ~1000 steps/epoch versus 8 at 100k, so ~125×
more chances for one bad step under the reference architecture's high early LR (0.01). Confirmed by rerunning seed 2
at 10M with `TF_DETERMINISTIC_OPS=1` and `TF_CUDNN_DETERMINISTIC=1` — smooth, no epoch-3 spike.

Cause: nondeterministic GPU/cuDNN floating-point reduction order, amplified at scale. The Phase-1
anchor had simply drawn three spike-free runs. Both flags are now set in `train_one_quantized.py`
before `import tensorflow`, alongside a hard `assert tf.__version__.startswith('2.15')` guard.
`clipnorm=1.0` was tried during the diagnosis and **reverted**: it suppressed the symptom, left the
cause in place, and would have split this recipe from the Phase-1 anchor's.

Control gate — all three reproduce their Phase-1 anchors:

```
        control    anchor     delta
seed 0  0.04659    0.046675   -0.0001
seed 1  0.04731    0.047715   -0.0004
seed 2  0.046565   0.045580   +0.0010     (was 0.05685 on the spiked run)
```

### Result: B=4 is infeasible by construction, not a measured accuracy limit

All three B=4 seeds returned p_L = 0.28260 — exactly the tail's base rate — with val_loss pinned at
0.5935 from epoch 1. The model emitted a constant and never trained. This is a fixed-point FORMAT
failure, not a statement about 4-bit activations. The per-class integer widths leave negative
fractional width at B=4: z-like is signed with I=4, so frac = 4−4−1 = −1, and the decoder ReLU is
unsigned with I=6, so frac = 4−6 = −2 (representable values spaced 4 apart over [0,64), so every
hidden activation below 2.0 rounds to zero). QKeras accepts this silently.

**Report it as the integer-width floor, not as "4-bit activations fail."** Under the current width
policy the minimum viable B is 6. Going lower requires retuning the integer widths first: the p99.9
ReLU width (I=4) unlocks B=5, but z-like's I=4 follows from the architecture's ±12 clip and is not
reducible without changing that bound.

`ActQuant.set_bits()` now refuses any configuration with negative fractional width, naming the
offending classes and the minimum viable B, and prints the resulting per-class ap_fixed format on
every run. The threshold is frac < 0, not frac < 1: B=6 leaves the ReLU at frac = 0 (resolution
1.0) and trained normally, so zero fractional width is warned about, not rejected.

### Result: paired McNemar on B=8 and B=6

`phase2a_collate.py`'s within-seed p_L differences are descriptive only. The two runs decode the
SAME 200k shots, so the correct test is paired McNemar on the discordant shots, as in Phase 1.
Aggregate differencing called B=8 "noise"; the paired test disagrees.

```
B vs its own seed's control          B vs MWPM (0.049405, decoded on this tail)
       delta      net    p_exact            net    p_exact
B=8 s0 +0.00198   -396   3.3e-08            +167   0.051
B=8 s1 -0.00102   +204   3.1e-03            +623   7.4e-14
B=8 s2 +0.00058   -115   0.103              +453   5.7e-08
B=6 s0 +0.00146   -291   4.7e-05            +272   1.3e-03
B=6 s1 +0.00209   -418   1.3e-08            +  1   1.00
B=6 s2 +0.00109   -217   1.3e-03            +351   2.7e-05
```

**B=8: no systematic cost, but not because the differences are noise.** Two of the three seeds are
individually significant — and in OPPOSITE directions (seed 0 favours the control, seed 1 favours
B=8). A real per-run difference whose sign flips across seeds is training-run variation, not a
precision penalty. The honest claim is "no consistent cost at B=8", supported by the sign flip,
rather than "the difference is within noise."

**B=6: a real, consistent cost of about +0.0015.** All three seeds favour the control, all three
are significant, and the magnitudes agree (+0.0011 to +0.0021). This is the one place the sweep
shows activation precision actually costing accuracy.

**Versus MWPM at B=6 the margin becomes seed-dependent:** seeds 0 and 2 still beat MWPM
(p = 1.3e-03, 2.7e-05), but seed 1 lands at p_L = 0.049400 against MWPM's 0.049405 — a net of one
shot in 200k, a dead heat. So "6-bit weights AND 6-bit activations still beat MWPM" is not
supportable as stated; at B=8 it holds on all three seeds.

Note B=6 leaves the decoder ReLU with zero fractional bits (resolution 1.0 on a [0,64) tensor) and
still costs only ~+0.0015 — a robustness result in its own right, and a concrete argument for
rerunning B=6 with the p99.9 ReLU width (I=4, giving frac=2), which should recover part of that cost.

The three seeds share one tail, so their tests are correlated. Read them as three consistent or
inconsistent readings; do NOT Fisher-combine the p-values.

### MWPM baseline provenance (verified 2026-07-20)

The sweep CSVs' `mwpm_p_L` column reads 0.0451 and must be ignored: it comes from
`train_one.lookup_mwpm()`, which reads `pools/mwpm_baseline.csv` keyed only on (d, p, rounds) and
has no idea which tail is being evaluated. The 0.0518 seen elsewhere came from the same lookup.

0.049405 is a decode, not a lookup — confirmed by reading the code rather than trusting the
comment: `eval_on_tail.py --mcnemar` builds the 4-channel rotated_memory_z circuit, takes its
detector error model, decodes these exact shots via `pymatching.Matching.decode_batch`, and
overwrites the looked-up value with that result. Every row of `out_q_mcnemar/mcnemar_knee.csv` and
`out_q/fp32_anchor.csv` carries 0.049405 from that path, and `phase2a_mcnemar.py` re-decodes it
independently to the same value. This is the denominator of every "beats MWPM" claim in the paper.

### Artifacts

```
out_q_phase2a/rcnn_d5_p0.010_r3_w6_a{32,8,6,4}_seed{0,1,2}_ntr10000000.{csv,history.json,weights.h5}
out_q_phase2a/phase2a_mcnemar.csv     paired-test rows, both comparisons
```

### Open

- Rerun B=6 with `ActQuant.set_relu_integer(4)` (p99.9 ReLU width) to test whether the +0.0015
  cost is partly the zero-fractional-width ReLU rather than the word length itself.
- A genuine low-B datapoint needs the integer-width policy retuned first; B=4 remains blocked by
  z-like's ±12 clip.
- Phase 4 (log-domain / LSE rewrite) still owns the x-like tensors, which are left FP32 throughout.
