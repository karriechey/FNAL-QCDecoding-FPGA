# p_L-vs-N Sweep — Summary (the ~30-hour run)

> **Doc date: 2026-06-23. Superseded for the paper's numbers.** This is the *Mac-CPU* sweep
> (25 runs, 5 sizes × 5 seeds, 10k test tail). The paper's data-volume result is the **EAF/GPU
> ladder** in `RCNN_d5_r3_ladder_REFERENCE.md`, scored on the 200k tail. Cite this one only for
> the CPU wall-clock figure (~31 h) and as the study's origin — **do not mix its p_L numbers
> with the EAF ladder's** (different platform, different test tail).

**What it was:** the no-shortcut data-volume scaling study for the reference architecture's
`FullRCNNModel` surface-code decoder vs an MWPM (pymatching) baseline, at a
single physics config (d=5, p=0.01, r=3, 4-channel noise). Its job: answer the
question the loss curve only hinted at — *is the gap to MWPM structural (the
model class can't do better) or just data-limited (feed it more shots and it
closes)?*

- **Started:** 2026-06-21 10:37:31
- **Finished:** 2026-06-22 17:37:49
- **Wall clock:** ~31 h  |  **Pure training time:** 30.9 h
- **Hardware:** Apple Silicon, **CPU-only (no GPU on this machine)**, TF 2.15.1
- **25 training runs:** 5 data sizes × 5 seeds (0–4)

---

## Configs / params (identical across all 25 runs except n_train & seed)

| Param | Value |
|---|---|
| Architecture | `FullRCNNModel` (the reference architecture's real arch, CNNModel.py), 51,547 params |
| Distance d | 5 |
| Physical error p | 0.01 |
| Rounds r | 3 |
| Kernel | 3 |
| Hidden | 100 × 2 layers, npol=2 |
| Noise | 4-channel (before_round_data_depol, after_reset_flip, after_clifford_depol, before_measure_flip — all = p) |
| Epochs | 50, fixed, **no early stopping** (final weights) |
| Batch size | 10,000 |
| Val split | 0.2 |
| LR schedule | the reference architecture's stepped decay (0.01→ collapses to ~7e-9 by ep50); `LR_FLOOR=0.0` = his recipe bit-for-bit |
| Data pool | `datasets_sweep/data_d5_p0.010_r3.npz` (5.2M shots, 4-channel) |
| Split | fixed 200k **test tail** (same eval shots for every N); train = nested prefixes [0:N] |
| n_train sweep | 100k, 300k, 800k, 2M, 5M |
| Seeds | 0, 1, 2, 3, 4 |

---

## Results

| n_train | mean p_L | SEM | σ (run-to-run) | per-N train time |
|---|---|---|---|---|
| 100k | 0.2033 | 0.0078 | 0.0174 | 0.4 h |
| 300k | 0.1317 | 0.0046 | 0.0103 | 1.1 h |
| 800k | 0.0839 | 0.0089 | 0.0198 | 2.8 h |
| 2M   | 0.0543 | 0.0006 | 0.0013 | 8.0 h |
| **5M** | **0.0488** | **0.0003** | **0.0006** | **18.6 h** |
| **MWPM** | **0.0492** | — | — | — |

**Headline:** RCNN reaches **MWPM parity at 5M** (0.0488 vs 0.0492 — a hair below,
statistically indistinguishable). Monotonic descent the whole way; error bars at
2M and 5M don't overlap, so it was *still descending*, not plateaued early.

**Three conclusions:**
1. **The gap was data-limited, not structural.** At 800k the model looked
   converged (flat loss) yet sat at 1.7× MWPM — that tempted a "structural gap"
   read. 6× more data closed it entirely. The 800k loss-saturation was
   misleading.
2. **Convergence instability was a small-data artifact.** σ collapses ~30×
   (0.0174 → 0.0006) as N grows. The whole run-to-run nondeterminism saga was a
   low-data phenomenon; at the data volume that carries the result, training is
   rock-stable.
3. **Cross-check passed.** Sweep-800k (from 5.2M pool) 0.0839±0.0198 vs the
   earlier anchor-800k (from 1M pool) 0.0744±0.0118 agree within ~0.9σ → the two
   datasets are interchangeable.

---

## Where everything lives

| Item | Path |
|---|---|
| **Benchmark script** | `benchmark_rcnn.py` |
| **Sweep orchestrator** | `run_sweep.sh` (loops the 5 N-values, builds figure, runs cross-check) |
| **Results CSV (raw)** | `results/sweep.csv` (25 rows; p_L col 13, mwpm_p_L col 14, train_time_s col 16) |
| **Anchor CSV (cross-check)** | `results/anchor.csv` (5× 800k from the 1M pool) |
| **Per-epoch histories** | `results/sweep/N<N>/rcnn_d5_p0.010_r3_k3_seed<s>.json` |
| **Run log** | `logs/sweep.log` |
| **THE figure** | `plots/rcnn_d5_pl_vs_n.png` |
| Figure script | `plot_pl_vs_n.py` |
| Diagnostic (convergence) figure | `plots/rcnn_d5_diagnostic.png` (`plot_diagnostic.py`) |
| p_L-vs-MWPM bar figure | `plots/rcnn_d5_pl_vs_mwpm.png` (`plot_pl_vs_mwpm.py`) |

Rebuild the figure without retraining:
`.venv/bin/python plot_pl_vs_n.py --csv results/sweep.csv --out plots/rcnn_d5_pl_vs_n.png`

---

## How to read `rcnn_d5_pl_vs_n.png`

- **X axis (log):** n_train, the number of training shots.
- **Y axis:** logical error rate p_L (lower = better decoder).
- **Faint dots:** the 5 individual seeds at each N (shows real spread).
- **Solid line + caps:** RCNN mean ± SEM across the 5 seeds.
- **Dashed horizontal line:** MWPM baseline (~0.0492) — same for every N because
  it's evaluated on the identical 200k test tail.
- **The story to tell:** the RCNN curve starts ~4× above MWPM at 100k and slides
  monotonically down, the error bars shrink as N grows, and at 5M the curve
  *touches the MWPM dashed line*. So: **"the reference architecture's RCNN matches MWPM at d=5/p=0.01
  once given ~5M training shots; the earlier gap was a data shortage, not a
  ceiling of the architecture."**

---

## "Why we plateau to using the GPU"

Two separate things worth being precise about:

**1. The p_L curve "plateaus" at MWPM.** It doesn't plateau early — it descends
all the way to the MWPM line and flattens there because MWPM is a near-optimal
decoder at this small distance. Matching the optimal baseline *is* the ceiling
worth hitting; you can't expect to beat MWPM by a lot at d=5. (To test for a
*sub-MWPM* win you'd need a point beyond 5M.)

**2. Why this pushes us toward the GPU.** This machine has **no GPU** — all 25
runs were CPU-only, which is exactly why a single 5M run took ~18 h and the whole
sweep took ~31 h. The cost scales with n_train, and the result we just proved is
that **good p_L *requires* large n_train** (5M to reach parity). So the natural
next steps are all data-hungry and CPU-prohibitive:
   - more shots (10M+) to test sub-MWPM,
   - larger code distances (d=7, d=9) where shot counts and model size both grow,
   - the eventual quantization / hls4ml FPGA path.

A 31-hour CPU sweep for one physics config is the practical wall. Moving training
to a GPU venue (EAF) is what makes the d-scaling and >5M experiments tractable —
that's the "plateau → GPU" handoff. (Inference/deployment still targets FPGA per
the north star; GPU is for the *training* scale-up.)
