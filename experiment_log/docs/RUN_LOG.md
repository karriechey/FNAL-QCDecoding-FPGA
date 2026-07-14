# QAT weights-only quantization — RUN LOG

> **Doc status: CURRENT — the authoritative ledger. Last updated 2026-07-13.**
> Where this and any other doc disagree, **this file wins.** (Title said "weight/activation";
> corrected to **weights-only** — activation quantization is Phase 2 and has not been run.)

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
  This is THE eval substrate for every phase — do not regenerate, do not swap.
- **FP32 anchor weights:** `~/rcnn_threshold/out_t200k_w/rcnn_d5_p0.010_r3_seed{0,1,2}_ntr10000000.weights.h5`
  (existing 10M FP32 models; reused, never retrained).
- **Quantizer:** `quantized_bits(B, 1)` weights-only = 1 sign + 1 integer + (B-2) frac,
  range ~[-2,2). See `QUANTIZATION_SCHEME.md`.
- **Env:** EAF Jupyter, `.venv/bin/python` (qkeras 0.9.0, TF 2.15.1, Keras 2.15.0).
  GPU = A100 MIG slice (`3g.40gb` = 3/7 of 80GB A100). ~195s/epoch, ~2.8hr/run.
  `kill -9` on a CUDA job leaks GPU memory — use plain kill / pkill, verify with nvidia-smi.

---

## Phase 0 — implementation + gates (2026-07, DONE)

- Path B QAT re-homed into `CNNModel_quantized.py`; `CNNModel.py` byte-for-byte pristine.
- Gates: None-path == original (max|diff|=0.0); quantization bites; grads on 35 quantized
  vars; FP32 10M weights load into None-model bit-exact (prob corr 1.0, max|dprob|=0).
- Tail contamination found + fixed: old 200k tail overlapped 10M-train by 190k shots
  (in-sample optimism ~+0.0016 p_L). Fixed via fresh gen-seed-43 tail.
- Clean FP32 anchor (fresh tail, 3 seeds): RCNN p_L 0.0462±0.0003 vs MWPM 0.0494;
  McNemar net +579..+775 wins, p~1e-21..1e-12 — significant beat.

## Phase 1 — harden Exp 1 (referee-proof baseline)  [w6 DONE / w8 IN FLIGHT]

Objective: same-tail confirmation for every row + paired McNemar 2x2 at the knee (6, 8 bit)
vs MWPM. Licenses "parity"/"beats" with discordant counts, not assertion.

- Driver: `phase1_mcnemar.py` — retrains bits {6,8} × seeds {0,1,2} WITH --save-weights into
  `~/rcnn_threshold/out_q_mcnemar/` (separate dir, does NOT clobber Pareto CSVs in out_q/),
  then `eval_on_tail.py --mcnemar --pool <fresh tail>` → `out_q_mcnemar/mcnemar_knee.csv`.
  Resume-safe: skips a (bits,seed) whose weights + McNemar row already exist.
- FP32 (32-bit) McNemar already in `out_q/fp32_anchor.csv` (sweep anchor ran --mcnemar).

### w6 knee — DONE. 3/3 seeds beat MWPM on the shared 200k tail (gen-seed 43).

Source: `out_q_mcnemar/mcnemar_knee.csv`. MWPM = 0.049405, n_test = 200000, base_rate 0.2826.

| seed | p_L (w6) | ratio | rcnn_only | mwpm_only | net wins | chi2_cc | p_exact |
|------|----------|-------|-----------|-----------|----------|---------|---------|
| 0 | 0.046675 | 0.945 | 3619 | 3073 | +546 | 44.39 | 2.6e-11 |
| 1 | 0.047715 | 0.966 | 3625 | 3287 | +338 | 16.43 | 5.0e-05 |
| 2 | 0.045580 | 0.923 | 3696 | 2931 | +765 | 88.08 | 5.7e-21 |

**All 3 significant.** The 6-bit knee's beat over MWPM is now a paired result, not mean-vs-mean.

### w8 — IN FLIGHT as of 2026-07-13

seed2 training on pod `jupyter-kchey-seed2` (`phase1_s2.log`), 50 epochs, ~320 s/epoch.
Until it lands, **"8-bit is lossless" remains mean-vs-mean only** — no paired test yet.
`mcnemar_knee.csv` grows one row per completed (bits,seed); re-pull it when the runs finish.

### ⚠️ MWPM baseline discrepancy — resolve before publishing

The per-run sweep CSVs (`out_q/rcnn_*_w*_seed*.csv`) all carry `mwpm_p_L = 0.0451`.
`fp32_anchor.csv` and `mcnemar_knee.csv` carry `mwpm_p_L = 0.049405`. **Same 200k tail,
same config, two different MWPM numbers.** The Pareto table below used 0.04940 (correct —
every mean_pL in it reproduces exactly from the per-run rows), so the published ratios are
fine. But do NOT compute xMWPM from the sweep CSVs' own mwpm column — it inflates the ratio.
Track down which code path wrote 0.0451 (`train_one_quantized.py` vs `eval_on_tail.py`).

| date | git SHA | run | host | result |
|------|---------|-----|------|--------|
| 2026-07 | ≤46079f5 | w6 × seed0,1,2 McNemar | EAF | 3/3 beat MWPM, p 5e-5 … 6e-21 |
| 2026-07-13 | | w8 × seed2 McNemar | EAF seed2 pod | running (epoch 25/50) |
| _not started_ | | w8 × seed0, seed1 | EAF | confirm whether launched |

## Phase 2 — activation quantization sweep  [PLANNED]

Fix w=6 (knee) + w=8 (anchor); sweep activation bits a ∈ {8,6,4} via point-of-use
`quantized_bits`/`quantized_relu` on intermediate tensors. **Combiner nonlinearities**
(`tf.math.pow/sqrt/log`, `VariableBounds.clip_exp/clip_zlike`) are log-domain, wide range —
use signed `quantized_bits(a, I)` NOT `quantized_relu` there; expect more integer bits.
L-slice (fix w6 sweep a; confirm w6/w8 at best a). Deliverable: Table 2 / Fig 2, joint
operating point (e.g. w6/a8 — empirical).

## Phase 3 — fixed-point range profiling  [PLANNED]

Per-tensor min/max + 99.9/0.1 percentiles over the fresh tail → per-layer `ap_fixed<B,I>`
table. **Do the profiling READ before finalizing Phase 2** so integer bits are set from
measured range, not assumed (avoids a Phase-2 redo). Deliverable: Table 3, the numerical
contract for Giuseppe's HLS handoff.

---

## Completed sweep — Step 2 Pareto (2026-07-10, DONE)

Driver `sweep_quantized.py`, n=3, fresh 200k tail, 10M shots. Fanned across 3 Named Servers.
Collate `collate_pareto.py` → `plots/rcnn_d5_r3_qat_pareto.png`, `out_q/`.

```
 bits  size_KB  mean_pL   comb_err   xMWPM   n
   32   201.4   0.04614   0.00056   0.934    3   (FP32 anchor, reused)
    8    50.3   0.04594   0.00053   0.930    3   lossless = FP32
    6    37.8   0.04718   0.00056   0.955    3   KNEE — ~3σ below MWPM, still beats
    4    25.2   0.05464   0.00148   1.106    3   ~3.4σ above MWPM
    3    18.9   0.06116   0.00170   1.238    3
    2    12.6   0.12796   0.00709   2.590    3   collapsed
 MWPM = 0.04940 ± 0.00048  (fresh tail, n_test=200000)
```

Headline: 8-bit lossless, **6-bit is the knee** (37.8 KB, 5.3× smaller than FP32, still
beats MWPM). Sharp cliff 6→4. Variance blows up at low bits (comb_err 0.0006@6/8 →
0.0015@4 → 0.007@2). This is the weights-only ceiling; Phase 2 adds activations.
