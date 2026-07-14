# RCNN Surface-Code Decoder — d=5, p=0.01, r=3 Convergence-Ladder Study
### Reference guide for the work update (generated 2026-06-30)

> **Doc status: CURRENT (as of 2026-07-13).** This is the paper's data-volume result —
> EAF/GPU, 200k tail. Its predecessor, the Mac-CPU `SWEEP_SUMMARY.md`, is superseded.
> Raw per-run CSVs: `../results_snapshot/eaf/rcnn_threshold/out/`.

> Renders as rich text in any Markdown viewer (VS Code, GitHub, Jupyter). Also
> plain-text readable. Keep this next to you when presenting.

---

## 1. One-paragraph summary (the elevator version)

We reproduced the reference architecture's RCNN surface-code decoder learning curve — **logical error
rate `p_L` vs training-set size** — entirely on the Fermilab EAF GPU platform, and
extended it with a new **10-million-shot** point. Across 6 training sizes (100k →
10M) × 3 random seeds, the RCNN's error rate falls monotonically: it **reaches parity
with MWPM at ~5M shots**, and at 10M sits **~5% below MWPM** (0.04628 vs 0.04875, ratio
0.949×, all 3 seeds below) on a high-precision **200k-shot** test tail — and this **10M beat
is CONFIRMED** by the paired **McNemar** test (all 3 retrained models beat MWPM on the shared
shots, net +497/+626, p ≈ 10⁻⁹–10⁻¹⁴). Pushing to **20M gives NO further gain** and run-to-run
variance blows up ~5× (2 of 3 retrainings beat MWPM, 1 loses) → a **structural floor reached by
~10M**, not data-limited descent. (A first pass on a small 10k tail couldn't resolve any of this
— see §7.) This is a decoder-vs-decoder comparison at a single **above-threshold** operating
point — not a scalability or below-threshold claim.

---

## 2. What & why (project context)

- **Model:** `FullRCNNModel` — the reference architecture's actual recurrent-CNN decoder architecture
  (TF 2.15 / Keras 2 / QKeras 0.9), 51,547 parameters.
- **Task:** decode surface-code syndrome data (detector events) → predict the logical
  bit flip. Lower `p_L` = better decoder.
- **Benchmark:** MWPM (minimum-weight perfect matching, via PyMatching) — the standard
  classical decoder. The RCNN is trying to *match or beat* it.
- **Bigger arc:** toward a paper (mentor **Gabriel Perdue**, SQMS) and eventually an
  FPGA deployment of the real architecture via hls4ml (Giuseppe collaboration).
- **Platform:** Fermilab EAF, interactive pod, **NVIDIA A100 80GB in MIG mode** (we
  get a ~10 GB / 14-SM slice, ~1/7 of the card).

---

## 3. Experimental design (the part that makes it trustworthy)

**Nested train prefixes, fixed test tail — one pool.** All rungs slice from a single
pool of 10,010,000 shots:

```
Pool:  [ 0 ─────────── training region (10M) ───────────┃─ test 10k ─]
        train(100k) = shots [0 : 100k]        ┐ nested, grow from the FRONT
        train(300k) = shots [0 : 300k]        │ (each is a superset of the last;
          ...                                 │  all disjoint from the tail)
        train(10M)  = shots [0 : 10,000,000]  ┘
        TEST tail   = shots [10,000,000 : 10,010,000]  ← SAME 10k for all 18 runs
```

Why this matters:
- **Fixed tail** → the `p_L`-vs-N curve reflects *training size only*, not test-set
  variation. Every point is scored on identical shots.
- **Nested front prefixes** → provably disjoint from the tail; the model never sees
  its own test data.
- **One pool** → the points are directly comparable (same noise realization).

---

## 4. Configuration & parameters (exact)

| Parameter | Value | Notes |
|---|---|---|
| code distance `d` | 5 | surface code |
| physical error rate `p` | 0.010 | **above** threshold (~0.0065) |
| rounds `r` | 3 | matches the reference architecture's `surface_code_d5_r3_RCNN.ipynb` |
| noise model | 4-channel circuit-level | all four Stim channels set to `p` |
| circuit | `surface_code:rotated_memory_z` | observable `ZL` |
| kernel | 3 | RCNN conv kernel |
| hidden / layers | 100 / 2 | |
| npol | 2 | |
| epochs | 50 | **no early stopping** (fixed budget) |
| batch size | 10,000 | |
| val split | 0.2 | carved from the front of the train prefix |
| LR schedule | the reference architecture's | warm-up ramp epochs 0–9, then stepped decay |
| seeds | 0, 1, 2 | 3 replicates per rung |
| n_params | 51,547 | |
| pool sampling seed | 42 | deterministic per platform |
| test tail `n_test` | 10,000 | **the precision bottleneck — see §7** |

**4-channel noise** = `before_round_data_depolarization`, `after_reset_flip_probability`,
`after_clifford_depolarization`, `before_measure_flip_probability`, all = `p`. The
identical circuit is used to *generate* the data and to build the MWPM decoder's error
model, so nothing is silently mis-baselined.

---

## 5. Results (final, all 18 runs on EAF)

MWPM anchor = **0.04510** (measured on the same fixed 10k tail; ~451 residual errors).

| n_train | mean `p_L` | SEM (seeds) | binomial σ (tail) | combined err | ratio to MWPM |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 0.22517 | 0.01049 | 0.00418 | 0.01129 | 4.99× |
| 300,000 | 0.13560 | 0.00546 | 0.00342 | 0.00644 | 3.01× |
| 800,000 | 0.08063 | 0.00192 | 0.00272 | 0.00333 | 1.79× |
| 2,000,000 | 0.05577 | 0.00387 | 0.00229 | 0.00450 | 1.24× |
| 5,000,000 | 0.04887 | 0.00096 | 0.00216 | 0.00236 | 1.08× |
| **10,000,000** | **0.04520** | **0.00061** | **0.00208** | **0.00216** | **1.002×** |

Per-seed 10M values: 0.04510, 0.04420, 0.04630 (mean 0.04520).

**Read of the result:**
- Clean **monotonic descent** to parity — the reproduction succeeded.
- At 10M the RCNN **sits at MWPM parity (1.00×)**; the 0.0001 gap is ~20× smaller than
  the ±0.00216 error bar → **not resolvable, not a "beat."**
- Honest-uncertainty note: from 5M onward the **binomial (test-set) σ dominates** the
  seed SEM (0.00208 vs 0.00061 at 10M). The seed spread and the tail noise are two
  *different* noise sources (see §8 glossary).

### 5b. DEFINITIVE result — 200k-tail re-measurement (`out_t200k`, MWPM = 0.04875 ± 0.00048)

Full 100k→10M ladder, 3 seeds each, scored on a **200,000-shot** tail (±0.0005 precision,
~4.5× tighter). Presentation figure: `plots/rcnn_d5_pl_vs_n_t200k_full.png`.

| n_train | mean p_L | ±combined | ratio | verdict |
|---:|---:|---:|---:|---|
| 100,000 | 0.19814 | ±0.00272 | 4.06× | |
| 300,000 | 0.14341 | ±0.00729 | 2.94× | |
| 800,000 | 0.07485 | ±0.00319 | 1.54× | |
| 2,000,000 | 0.05331 | ±0.00060 | 1.09× | |
| 5,000,000 | 0.04867 | ±0.00068 | 0.998× | **parity** |
| 10,000,000 | **0.04628** | ±0.00058 | **0.949×** | **~5% below — suggestive** |

**Status of the 10M dip (be precise):** all 3 seeds below MWPM; gap ~4.2× the combined
error → **suggestive, borderline at n=3** (df=2 t-crit ≈4.30). One-sample t of the 3 seeds
vs the MWPM constant gives t≈7 (p≈0.02); folding in MWPM's own uncertainty pulls the margin
to ~4.2. **The paired McNemar test (per-shot, ~200k shots) is the resolver — running (Exp 4)**;
optionally 5 seeds at 5M/10M for retraining robustness. Don't state a clean beat until McNemar.

Note vs the 10k figure: the whole curve sits closer to parity (2M 1.24×→1.09×, 800k
1.79×→1.54×) — NOT the model changing, but the *true* MWPM (0.04875) replacing the
low-draw 0.0451 in the denominator. Low rungs' error bars are seed-dominated (little data
= noisy training); high rungs' are test-set-dominated.

**⚠️ Caption-bug note (fixed 2026-07-02):** `plot_pl_vs_n_gpu_v2.py` used to HARDCODE a
"parity, gap < noise" caption — correct on the 10k tail, but **false on 200k** (it asserted
"no measurable sub-MWPM behavior" while the gap was ~4× the error). Now data-driven. If you
have an old `..._t200k*.png`, **regenerate it** with the fixed script.

- **10M beats MWPM by ~5%, ~3σ.** All 3 seeds below (0.04643, 0.04562, 0.04678),
  `sem_seed` 0.00034; gap −0.00247 vs combined RCNN+MWPM σ ~0.0007.
- **This beat is invisible on the 10k tail** — there 10M read 1.002× within ±0.002 noise.
  The result is real but only *resolvable* with the bigger tail. (The old 10k MWPM 0.0451
  was an anomalously low draw; the true, precise MWPM is 0.04875 — consistent with the
  original CPU run's ~0.049.)
- **Definitive headline:** *at d=5/p=0.01/r=3 (above threshold), the RCNN reaches MWPM
  parity at ~5M shots and beats MWPM by ~5% at 10M on a 200k-shot tail.* One operating
  point; not a scaling/below-threshold claim.

---

## 6. Key findings (state these; do NOT inflate)

1. **Reproduction succeeded.** The RCNN `p_L`-vs-N curve matches the original CPU run
   (5M ≈ 0.0489 on both). The model learns the same thing on CPU and GPU.
2. **"Parity at 5M" (original) vs "parity at 10M" (this run) is a *baseline* artifact,
   not a model change.** The RCNN barely moved (0.0487 → 0.0489 at 5M); what moved was
   the MWPM anchor (~0.0492 → 0.0451). Divide the same RCNN curve by a lower baseline
   and the parity crossing slides right. The RCNN reproduced; the yardstick shifted.
3. **The 10k tail cannot resolve parity.** Adding the 3rd seed flipped the auto-verdict
   from "0.990× BELOW" to "1.002× ABOVE" — one seed crossed the line. Ratio precision
   is ±4.7%; the MWPM anchor itself swings ±0.005 by which 10k shots are sampled
   (0.0451 on the EAF pool vs 0.0518 on the Mac pool — same config).
4. **Does more data (20M) beat MWPM? Unresolved.** 5M→10M was a real ~6% drop (still
   descending), and a power-law+floor fit hints at a floor ~0.68× MWPM — BUT the fit
   under-predicts the actual 10M and its floor is ±24% uncertain. Not a basis to claim
   20M wins.

---

## 7. Why a 10k tail at first, and why we're moving to 200k

**Why 10,000 initially:**
- A tail must hold enough logical errors that the estimate isn't dominated by tiny-count
  noise. The pool was sized for ≥~500 residual errors in the tail; at `p_L≈0.045`, 10k
  shots give ~450 errors. That's the standard "enough to be meaningful" threshold.
- 10k is a *tiny* fraction of a 10M pool, so it barely reduces training data.
- It is perfectly adequate for the **big** effect — the 5× descent from 100k to 10M is
  far larger than the ±0.002 noise, so the curve shape is solid.

**Why it's not enough for the parity question, and the fix:**
- The parity gap (~0.0001–0.001) is *smaller* than the 10k tail's ±0.002 noise → can't
  be resolved. The MWPM anchor also wobbles ±0.005 pool-to-pool.
- Evaluation noise = `√(p(1−p)/n_test)`, shrinking as `1/√n_test`:

  | tail size | σ on `p_L≈0.045` | ~errors |
  |---:|---:|---:|
  | 10k (used) | ±0.00207 | ~450 |
  | 50k | ±0.00093 | ~2,250 |
  | **200k (planned)** | **±0.00046** | ~9,000 |
  | 1M | ±0.00021 | ~45,000 |

- A **200k tail** makes the MWPM anchor and each RCNN point ~4.5× sharper (±0.0005) and
  reproducible pool-to-pool. Demonstrated locally: a 200k tail gave MWPM 0.0494 ± 0.0005
  — pinning the "true" value that the noisy 10k draws (0.0451, 0.0518) were scattering
  around.
- **What the bigger tail does NOT do:** it doesn't reduce seed-to-seed training jitter
  (that needs more seeds), and it won't manufacture a beat that isn't there — if the RCNN
  truly sits at parity, 200k *confirms* parity tightly rather than revealing a win.

---

## 8. Glossary (technical terms)

- **`p_L` (logical error rate):** fraction of test shots where the decoder predicts the
  wrong logical bit. The core metric. Lower = better.
- **Test tail / hold-out set:** the shots reserved *only* for evaluation, never trained
  on. Physically the *last* `n_test` shots of the pool (`te = slice(N−n_test, N)`).
- **`n_test` (tail size):** how many shots are in the test tail (10,000 here). Sets the
  measurement precision via `√(p(1−p)/n_test)`.
- **MWPM (minimum-weight perfect matching):** the standard classical decoder (via
  PyMatching). A *fixed algorithm* — **not trained**, so its `p_L` does **not** improve
  with more shots; more shots only measure it more precisely. It's the flat dashed line.
- **Parity:** RCNN `p_L` = MWPM `p_L`, i.e. ratio = 1.0. "Beating" MWPM = ratio < 1.
- **Ratio to MWPM:** `RCNN p_L ÷ MWPM p_L`. 4.99× means 5× worse; 1.00× means equal.
- **SEM (standard error of the mean):** spread of `p_L` *across seeds* ÷ √(n_seeds).
  Captures **training noise** only (all seeds share the same tail).
- **Binomial / test-set noise:** `√(p(1−p)/n_test)` — sampling error from a *finite*
  test set. Captures **evaluation noise**. Common to all seeds (they share the tail).
  *These two noise sources are separate knobs: more seeds shrink SEM; a bigger tail
  shrinks binomial noise.*
- **Combined error bar:** `√(SEM² + binomial²)` — the honest total used in the v2 figure.
- **Nested train prefixes:** train(100k) ⊂ train(300k) ⊂ … ⊂ train(10M), all from the
  front of one pool.
- **4-channel circuit-level noise:** the four Stim noise channels (data depolarization,
  reset flip, Clifford depolarization, measurement flip), all set to `p`.
- **`d` / `p` / `r`:** code distance / physical error rate / syndrome rounds.
- **Threshold (~0.0065):** the physical error rate below which increasing `d` *reduces*
  `p_L`. We run at **p=0.01 > threshold** (above threshold) — so this is not a
  sub-threshold or scaling study.
- **Detector events / observable flip:** the syndrome inputs / the ground-truth label.
- **DEM (detector error model):** the graph MWPM decodes; built from the same circuit.
- **Base rate / class collapse:** the all-zero predictor's error rate (~0.29 here);
  a decoder must beat it or it has "collapsed." All rungs pass.
- **XLA compile:** TensorFlow's one-time kernel compilation on epoch 1 (~130 s), then
  fast steady-state — don't mistake the slow first epoch for a problem.
- **MIG (Multi-Instance GPU):** the A100 is partitioned; the pod gets a ~10 GB / 14-SM
  slice (~1/7 of the card). Explains why the GPU is only ~Mac-CPU speed for this model.
- **Stim / PyMatching:** the circuit sampler (data generation) / the MWPM decoder.

---

## 9. File locations

**EAF pod (`kchey@jupyter-kchey`, home = `/home/kchey`):**
- Repo: `~/QuantumDecoderQKeras/`
- Pool (10k tail): `~/rcnn_threshold/pools/data_d5_p0.010_r3.npz` (1.6 GB, 10.01M shots)
- MWPM baseline: `~/rcnn_threshold/pools/mwpm_baseline.csv`
- Per-run results: `~/rcnn_threshold/out/rcnn_d5_p0.010_r3_seed{S}_ntr{N}.csv`
- Run logs: `~/rcnn_threshold/out/ladder_*.log`, `ladder_driver.log`
- Collated table: `~/QuantumDecoderQKeras/results/ladder_d5_p0.010_r3_collated.csv`
- Figures: `~/QuantumDecoderQKeras/plots/rcnn_d5_p0.010_r3_ladder.png`,
  `plots/rcnn_d5_pl_vs_n_gpu_v2.png` (the honest-uncertainty v2)
- Bigger-tail rerun (planned): `~/rcnn_threshold/pools_t200k/`, `~/rcnn_threshold/out_t200k/`

**Scripts (in `~/QuantumDecoderQKeras/`, canonical copies also on the Mac at
`~/Documents/QuantumDecoderQKeras/`):**
- `generate_datasets.py` — CLI pool generator (`--rounds 3` is the key flag)
- `mwpm_on_pool.py` — computes the MWPM baseline on a pool's fixed tail
- `train_one.py` — one training run (patched: `lookup_mwpm` now matches on rounds)
- `run_ladder.sh` — the 18-run driver (resume-safe, cheap→expensive)
- `collate_ladder.py` — aggregates per-run CSVs → mean±SEM + basic figure
- `plot_pl_vs_n_gpu_v2.py` — the v2 honest-uncertainty figure
- `this file` — `RCNN_d5_r3_ladder_REFERENCE.md`

**Environment:** run everything on EAF with `uv run python …` (the project venv has
Stim/PyMatching/TF 2.15). Kerberos ticket via `kinit kchey@FNAL.GOV` if needed.

---

## 10. Anticipated questions — with answers

**Q: Is p=0.01 above or below threshold?**
A: Above (~0.0065). This is a decoder-vs-decoder comparison at one above-threshold point,
not a sub-threshold or `d`-scaling claim.

**Q: Why r=3, when the threshold study uses rounds = d?**
A: r=3 reproduces the reference architecture's reference config (`surface_code_d5_r3_RCNN.ipynb`) so we have
a validated number to compare against. The rounds=d convention is a *different* study.

**Q: Does the RCNN beat MWPM?**
A: No — it *reaches parity* at ~10M. The 10M gap (~0.0001) is far below the ±0.002
measurement noise, so no sub-MWPM behavior is resolvable at this tail size.

**Q: Then why did the CPU run "reach parity at 5M"?**
A: Baseline artifact. The RCNN reproduced (5M ≈ 0.0489 both runs); the MWPM anchor
differed (~0.0492 vs 0.0451) because it was estimated on a small, differently-sampled
10k tail. We're fixing this with a 200k tail.

**Q: How confident is the MWPM baseline?**
A: On a 10k tail, only ±0.002, and it swings ±0.005 pool-to-pool. That's why the v2
figure draws MWPM as a *band*, and why the next run uses a 200k tail (±0.0005).

**Q: Is the comparison fair / apples-to-apples?**
A: Yes. Same fixed 10k tail for both decoders; the 4-channel circuit used to generate
the data is the same one used to build MWPM's error model; nested prefixes are disjoint
from the tail.

**Q: Is the result reproducible across platforms?**
A: The RCNN performance is (CPU ≈ GPU). Note Stim sampling is *not* bit-identical across
CPU architectures (same seed → different shots on arm64 vs x86), but the pools are
statistically equivalent. The absolute MWPM number is tail-dependent (hence 200k).

**Q: Why 3 seeds?**
A: To put error bars on training noise (weight init, shuffle order, float
non-associativity → ~0.012 per-run jitter). Averaging over seeds tightens the mean.

**Q: Does this scale to larger d, or below threshold?**
A: Not tested here — single d=5, above-threshold point. `d`-scaling and sub-threshold
behavior are the higher-value follow-up experiments.

**Q: Would 20M training shots beat MWPM?**
A: Unresolved. The curve is still descending at 10M, but a fit's floor estimate is too
uncertain and the 10k tail can't measure a sub-parity result anyway. The bigger tail,
not 20M, is the right next step.

**Q: What did it cost?**
A: ~22 GPU-hr on the A100 MIG slice, run sequentially on the interactive pod (batch was
blocked on a vault-authorization issue — separate escalation to the collaborator/Burt).

---

## 11. Next step (in progress): the 200k-tail rerun

Regenerate the pool with a **200k** tail (train cap 9.8M–10M) and **retrain 5M + 10M ×
3 seeds**, evaluating on the big tail — to pin the MWPM anchor and each point to ±0.0005
and finally answer *at / below / above* parity with tight error bars. (Retrain needed
because `--no-save-weights` was on; the trained models weren't kept.) Estimated ~17
GPU-hr (or ~11 for 10M-only). Outputs land in `~/rcnn_threshold/{pools_t200k,out_t200k}/`
and reuse `plot_pl_vs_n_gpu_v2.py --out-dir … --data-dir …`.

---

## 12. Work-update presentation outline (3 slides + backup)

**Slide 1 — What & why (context)**
- Continuing the reference architecture's RCNN surface-code decoder toward a paper (mentor: the collaborator Perdue).
- Question: can a learned decoder (RCNN) match the standard classical decoder (MWPM)?
- This piece: reproduce the **learning curve** — logical error rate `p_L` vs how much
  training data — on the Fermilab EAF GPU, and extend it to 10M shots.
- *Speaker note:* d=5, p=0.01, r=3, above-threshold, 4-channel circuit noise.

**Slide 2 — The result (lead with the figure)**
- Show `plots/rcnn_d5_pl_vs_n_t200k.png` (the precise 200k-tail figure).
- One line: **"The RCNN reaches MWPM parity at ~5M shots and beats it by ~5% at 10M
  (0.949×, ~3σ) — on a high-precision 200k-shot test tail."**
- Reproduction succeeded: matches the earlier CPU run (5M ≈ 0.049 on both).
- *Speaker note:* 3 seeds/point, **all 3 below MWPM at 10M**; error bars combine seed
  spread + test-set sampling.

**Slide 3 — Why precision mattered + next steps**
- The beat is only *resolvable* on the 200k tail: a first pass on a 10k tail read 1.002×
  within ±0.002 noise — an unresolvable coin-flip. The bigger tail (±0.0005) turned it
  into a ~3σ result.
- The earlier "parity at 5M vs 10M" confusion was a **baseline artifact** — MWPM estimated
  on a small, differently-sampled tail (the 10k 0.0451 was a low draw); the precise MWPM
  0.04875 matches the original CPU run.
- **Scope honestly:** one above-threshold point (d=5, p=0.01, r=3) — a ~5% beat *here*,
  not a scaling or below-threshold claim.
- **Next:** the higher-value experiments — sub-threshold & d-scaling (d=3,5,7) — run with
  a big tail + `--save-weights` from the start so precision is cheap.

**Backup slides (only if asked):** the config table (§4), the sources-of-variability
explanation (§6/§8), the p_L-vs-N fit / 20M discussion (§6.4), infra (A100 MIG, batch
auth blocked).

---

## 13. Command cheatsheet

> On EAF, **everything runs with `uv run python …`** (the project venv has Stim /
> PyMatching / TF 2.15). Plain `python` will fail with `ModuleNotFoundError`.
> If auth is needed: `kinit kchey@FNAL.GOV` (Fermilab Services password).

**A. Watch a running job**
```bash
# is it alive? (look for a `train_one.py` process)
ps -ef | grep train_one | grep -v grep

# how many runs finished (10k-tail dir)
ls ~/rcnn_threshold/out/rcnn_*.csv 2>/dev/null | wc -l

# what the driver is doing right now
tail -3 ~/rcnn_threshold/out/driver.log        # (or ladder_driver.log for the first run)

# live epoch-by-epoch of the current rung
tail -f $(ls -t ~/rcnn_threshold/out/*.log | head -1)     # Ctrl-C to stop watching

# auto-refreshing dashboard
watch -n 30 'ls ~/rcnn_threshold/out/rcnn_*.csv 2>/dev/null | wc -l; tail -2 $(ls -t ~/rcnn_threshold/out/*.log | head -1)'
```
*(For the 200k rerun, swap `out` → `out_t200k`.)*

**B. Pull / view results**
```bash
# the p_L for every finished run
grep -h '\[train\]' ~/rcnn_threshold/out/*.log

# the collated mean±SEM table + basic figure
uv run python collate_ladder.py

# the honest-uncertainty v2 figure + the printed table
uv run python plot_pl_vs_n_gpu_v2.py

# same, for the 200k-tail rerun
uv run python plot_pl_vs_n_gpu_v2.py --out-dir ~/rcnn_threshold/out_t200k --data-dir ~/rcnn_threshold/pools_t200k --plot-out ./plots/rcnn_d5_pl_vs_n_t200k.png
```
View a PNG: open `plots/…png` in the Jupyter file browser (or download it).

**C. Full pipeline from scratch (what each step does)**
```bash
# 1) generate the data pool (Stim sampling; CPU; ~20s, 1.6 GB)
uv run python generate_datasets.py --distances 5 --probs 0.010 --rounds 3 --n-samples 10010000 --seed 42 --out-dir ~/rcnn_threshold/pools

# 2) compute the MWPM baseline on the fixed tail
uv run python mwpm_on_pool.py --d 5 --p 0.010 --rounds 3 --n-test 10000 --data-dir ~/rcnn_threshold/pools --gen-seed 42

# 3) train ONE point (one size, one seed)
uv run python train_one.py --d 5 --p 0.010 --rounds 3 --seed 0 --n-train 800000 --n-test 10000 --epochs 50 --no-early-stopping

# 4) run the whole 18-point ladder in the background
nohup bash run_ladder.sh > ~/rcnn_threshold/out/ladder_driver.log 2>&1 &
```

**D. Resume after a pod cull** (the driver skips finished CSVs):
```bash
cd ~/QuantumDecoderQKeras && nohup bash run_ladder.sh > ~/rcnn_threshold/out/ladder_driver.log 2>&1 &
```

**E. If a big rung dies with CUDA out-of-memory:** add `--batch-size 5000` to that run.

**F. Save weights so you never retrain just to re-measure** (the collaborator's advice; the fix for
the 17-hr-retrain tax). Add `--save-weights` to any `train_one.py` command → it writes
`{tag}.weights.h5`. Then re-score that saved model on ANY tail with no training:
```bash
# add --save-weights when training (writes rcnn_..._ntr10000000.weights.h5)
uv run python train_one.py --d 5 --p 0.010 --rounds 3 --seed 0 --n-train 10000000 --n-test 200000 --epochs 50 --no-early-stopping --save-weights

# later: re-measure that saved model on a different tail — seconds, no retrain
uv run python eval_on_tail.py --weights ~/rcnn_threshold/out/rcnn_d5_p0.010_r3_seed0_ntr10000000.weights.h5 --d 5 --p 0.010 --rounds 3 --n-test 200000 --data-dir ~/rcnn_threshold/pools_t200k
```
*Best practice for comparison studies: generate the pool with a big tail up front AND
train with `--save-weights` — then any precision question is a cheap re-eval, not a
retrain. (Keep `--kernel/--hidden/--hidden-layers/--npol` identical when re-loading.)*

---

## 14. How to READ the outputs (annotated)

### The `[step1] df.head()` block
```
 d    p  rounds  seed  n_train  n_test    p_L  mwpm_p_L
 5 0.01       3     0   100000   10000 0.2056    0.0451
 5 0.01       3     0 10000000   10000 0.0451    0.0451
 ...
```
- Each **row = one training run** (one size, one seed). Columns: the config (`d,p,rounds`),
  the `seed`, the training size `n_train`, the test-tail size `n_test`, the RCNN's result
  `p_L`, and `mwpm_p_L` (the MWPM baseline it's compared to).
- **Why the rows look out of order** (100k, then 10M, then 2M…): `df.head()` just shows the
  first few files in *filename* order (`ntr100000` < `ntr10000000` < `ntr2000000`…
  alphabetical), *not* sorted by size. Harmless — the script groups by `n_train` internally.
  It's only a sanity peek that the columns loaded correctly.
- **Why `mwpm_p_L` is 0.0451 on every row:** there is ONE fixed MWPM baseline; every run is
  compared against the same number. (Not a bug — it *should* be constant.)

### The `[step1] (a)/(b)/(c)` facts
- **(a) seeds at each n_train** — confirms you have 3 replicates at every size (incl. 10M).
  Fewer than 3 = a run is missing/still going.
- **(b) n_test** — the test-tail size. `[10000]` = all runs used a 10k tail. If it printed
  two numbers, runs were mixed (a problem).
- **(c) MWPM anchor + source + its n_test** — the baseline value, which file it came from,
  and how many shots it was scored on.
- **`*** FLAG:`** — a built-in warning that the anchor sits on a small (10k) tail, so it's
  only good to ±0.002. This is *expected* for the 10k run and *should disappear* on the
  200k rerun (bigger tail = no flag).

### The `[table]` (the numbers behind the figure)
```
 n_train      mean_pL   sem_seed  binom_sig  comb_err   xMWPM  n
 10,000,000   0.04520   0.00061    0.00208    0.00216   1.002x  3
```
Column by column:
- **`mean_pL`** — average logical error rate over the seeds. *Lower = better.*
- **`sem_seed`** — how much the seeds disagree (÷√n). This is **training noise**.
- **`binom_sig`** — `√(p(1−p)/n_test)` = **test-set sampling noise**. Shrinks with a bigger
  tail. *At 5M–10M this is the DOMINANT error* (0.00208 ≫ 0.00061).
- **`comb_err`** — `√(sem_seed² + binom_sig²)` = the honest ± you put on the point.
- **`xMWPM`** — `mean_pL ÷ MWPM`. **>1 = worse than MWPM, 1.0 = parity, <1 = beats it.**
- **`n`** — number of seeds.

**How to say a row out loud (10M example):** *"At 10 million shots the RCNN's error rate is
0.0452 ± 0.0022, which is 1.00× MWPM — i.e. parity. The gap to MWPM is 0.0001, ~20× smaller
than the ± error, so it's parity within noise, not a beat."*

**What "good" looks like:** `mean_pL` falling as `n_train` grows; `xMWPM` marching toward
1.0; `comb_err` small relative to the differences between rows (so the descent is real, not
noise). At the top rungs, `binom_sig` dominating `sem_seed` is the signal that **the test
tail — not the number of seeds — is now your precision limit** (→ that's why we go to 200k).

### The figure (`plots/…png`)
- **Blue line + dots** = RCNN mean with combined error bars; **faint blue dots** = individual
  seeds. **Green dashed line + shaded band** = MWPM and its ±uncertainty.
- The blue line **touching the green band at 10M** = parity. **`×` labels** = the ratio at
  each point. The bottom caption scopes the claim (above-threshold, decoder-vs-decoder).
