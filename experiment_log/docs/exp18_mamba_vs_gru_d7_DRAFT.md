# Experiment 18 — parameter-matched GRU vs Mamba at d=7, r=7, p=0.004 · 2026-09-22

*Created: 2026-09-21 | Last updated: 2026-09-22*
*DRAFT for `analysis_notebooks/EXPERIMENTS_LOG.ipynb`. Not appended. Latest entry in the
notebook is Experiment 17 (cell 51); this becomes Experiment 18. Every TBD is a value
that exists only after the GPU runs.*

> **Status: code written and validated locally on synthetic tensors and a 60k-shot
> synthetic d=7 pool; no run of record has started.** Recorded before any result exists.

**Question.** Holding the data, the dense `[8 timesteps × 48 positions]` detector
sequence, the hard logical-flip labels, the training recipe and the parameter budget
fixed, does a selective state-space recurrence (Mamba) decode better than the gated RNN
recurrence (GRU) that Experiment 16 measured at d=7? This isolates the recurrence and
nothing else. Not in scope here: distillation, d=9, sparse defect tokens, cumulative-XOR
features, larger Mamba teachers.

**Held fixed.** d=7, r=7, p=0.004; the Experiment 16 pool and partitions; `det_evts`
only, scattered by `StudentModels.to_sequence()`; hard labels (alpha=1); Adam, constant
LR; batch 10,000; 200 epochs, no early stopping; minimum-`val_loss` checkpoint; seeds
0/1/2; `TF_DETERMINISTIC_OPS=1`, `TF_CUDNN_DETERMINISTIC=1`, `--require-determinism`.

**Manipulated variable.** The recurrent layer, and the learning rate for the Mamba arm
only (a 3-value sweep on the validation block, because the GRU's 3e-3 was chosen for a
different layer type and would otherwise be a confound in either direction).

### Pool of record — d=7, r=7, p=0.004 (Experiment 16)

| | |
|---|---|
| path | `~/pools_d7_p004_19M/data_d7_p0.004_r7_FORMAL.npz` on **grace1** — NEEDS_EAF_CHECK: not known to exist on EAF |
| shots | 19,000,000 |
| arrays | `det_evts` (19M, 336), `measurements` (19M, 385, not read here), `flips` (19M, 1) |
| generation seed | 42 (aarch64 stim stream) |
| `flips` SHA-256 | `b49fbe40786da9e6…` |
| MWPM on `[15.2M, 17.0M)` | 0.004291 ± 0.000049 (7,724 / 1.8M) |
| MWPM on `[17.0M, 19.0M)` | TBD — `mwpm_on_block.py`, run in `MODE=eval` |

| block | range | use |
|---|---|---|
| training | `[0, 10M)` | gradients |
| validation | `[15.0M, 15.2M)` | LR selection, checkpoint selection |
| evaluation | `[15.2M, 17.0M)` | scored once per seed, comparable with Experiment 16 rows |
| test | `[17.0M, 19.0M)` | sealed; scored once, both models and MWPM, at the end |

### Models

| | GRU (Experiment 16) | Mamba (this experiment) |
|---|---|---|
| file | `StudentModels.build_gru_student` | `MambaModel.build_mamba_decoder` |
| input | `[8, 48]` float32 from `to_sequence()` | identical |
| recurrence | `GRU(140, reset_after=False)` | 1 × `MambaBlock(d_model=100, d_state=16, expand=2, d_conv=4, dt_rank=7)` + pre-norm residual |
| readout | final state → `Dense(1)` | final timestep → RMSNorm → `Dense(1)` |
| output | logit, decision `logit > 0` | identical |
| parameters | **79,521** | **79,001** (−0.65%) |
| framework | TF 2.15 / Keras 2 | TF 2.15 / Keras 2; scan unrolled over T=8 in plain ops |

Parameter match measured with `count_params()` on built models
(`check_mamba_local.py`), the same way Experiment 16 matched the MLP. Runner-up
configs: d_model=104/d_state=8 (79,665, +0.18%), d_model=56/3 layers (80,641, +1.41%).
One layer chosen so both models have exactly one recurrent layer; d_state=16 is the
Mamba default.

### Protocol

| stage | command | reads | writes |
|---|---|---|---|
| A smoke | `MODE=smoke` | train 500k, val 50k | s/epoch → sizes B–D |
| B LR sweep | `MODE=lrsweep`, LR ∈ {1e-3, 3e-3, 1e-2}, seed 0, 10M, `LR_EPOCHS`=40 (TBD from A) | train + validation only | best `val_loss` per LR |
| C seed 0 | `MODE=full SEEDS=0 LR=<from B>` | train + val; eval block once | `.best` ckpt, per-shot npz |
| D seeds 1, 2 | `MODE=full SEEDS="1 2"` | same | same |
| E sealed | `MODE=eval` | `[17M, 19M)` once: Mamba ×3, GRU ×3 (existing `.best` ckpts), MWPM | per-shot npz ×7 |

The test block is not read in A–D. Learning-rate selection uses `best_val_loss` only.

### Output paths

`$RT/results/mamba_d7_p004_<mode>_<stamp>/` with `driver.log`, `MANIFEST.txt`,
`pool_provenance.json`, `runs/<tag>.{csv,history.json,config.json}`,
`ckpt/<tag>.{best,lastepoch}.weights.h5`, `per_shot_<tag>_<block>.npz`,
`eval_<block>.csv`. Tarball in `$RT/transfer/` (checkpoints excluded).

Tag format: `mamba_d7_p004_dm100_hard_seed<S>_ntr10000000_b10000_lr<LR>`.

### Metrics to record

| quantity | source | value |
|---|---|---|
| LR sweep: best_val_loss per LR, chosen LR | `runs/mamba_d7_lrsweep_*.csv` | TBD |
| s/epoch, wall per seed | `runs/*.csv` `sec_per_epoch`, `train_time_s` | TBD |
| eval block `[15.2M,17M)`: p_L per seed, mean ± sd, ×MWPM (0.004291) | `eval_eval.csv` | TBD |
| sealed `[17M,19M)`: MWPM p_L, 95% CI | `mwpm_sealed_test_d7.json` | TBD |
| sealed: Mamba p_L per seed, 95% CI, ×MWPM; mean ± sd | `analyze_mamba_vs_gru.py` | TBD |
| sealed: GRU p_L per seed, 95% CI, ×MWPM; mean ± sd | same | TBD |
| McNemar Mamba vs GRU per seed: b, c, n_discordant, p_exact, Δp_L | same | TBD |
| McNemar Mamba vs MWPM, GRU vs MWPM per seed | same | TBD |

p-values reported per seed; not Fisher-combined.

Reference for the GRU on the eval block (Experiment 16, run 2): 0.012324 ± 0.000447,
2.872× MWPM; per seed 0.012806 / 0.011922 / 0.012245.

### Caveats

- Platform: Experiment 16's d=7 corpus, pool and GRU checkpoints are on grace1
  (GH200, aarch64). Running the Mamba arm on EAF requires copying the pool
  (12.8 GB) and the three GRU `.best` checkpoints; an EAF-generated d=7 pool would be a
  different set of shots and would break the paired test. NEEDS_EAF_CHECK.
- The Mamba LR is swept; the GRU's was not (3e-3 inherited from Experiment 14). If the
  sweep picks a different LR, the comparison is "each at its own selected LR," with
  selection on the same validation block.
- Batch 10,000 × 10M shots → 14.3 GiB training array on device (Experiment 16 measured
  17,016 MiB per d=7 process). On a 40 GB MIG slice one seed at a time; on 80 GB, two.
- Runtime unknown until `MODE=smoke` prints `sec_per_epoch`. Mamba's unrolled scan does
  more elementwise work per step than cuDNN GRU; expect slower per epoch. No estimate
  is given without that measurement.
- `dt_proj` bias init draws from `np.random`, seeded by `set_seeds(seed)`; initial
  weights depend on `--seed` only (verified in `check_mamba_local.py`).

### Next step

After E: run `analyze_mamba_vs_gru.py` on the seven sealed-block dumps, fill the TBDs,
append this entry to the notebook as Experiment 18. Decision after that, not before:
Mamba teacher scaling, distillation, or the sparse-token input.
