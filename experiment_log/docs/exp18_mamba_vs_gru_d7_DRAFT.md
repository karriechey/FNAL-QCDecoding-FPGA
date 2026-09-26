# Experiment 18 — parameter-matched GRU vs Mamba at d=7, r=7, p=0.004 · 2026-09-22

*Created: 2026-09-21 | Last updated: 2026-09-22*
*DRAFT for `analysis_notebooks/EXPERIMENTS_LOG.ipynb`. Not appended yet — the run of
record is still training (epoch 180 of 200 at the time of writing) and the evaluation
block has not been scored. Latest entry in the notebook is Experiment 17 (cell 51); this
becomes Experiment 18. Fields still marked TBD need the finished run.*

> **Status: seed 0 at LR 3e-3 trained to epoch 180 of 200 on EAF. The training-side
> result is already unambiguous — the parameter-matched Mamba sits ~3× the GRU's
> validation loss and is flat. Evaluation-block p_L, the other two learning rates, and
> seeds 1–2 are not measured.**

**Question.** Holding the data, the dense `[8 timesteps × 48 positions]` detector
sequence, the hard logical-flip labels, the training recipe and the parameter budget
fixed, does a selective state-space recurrence (Mamba) decode better than the gated RNN
recurrence (GRU) that Experiment 16 measured at d=7? Out of scope here: distillation,
d=9, sparse defect tokens, cumulative-XOR features, larger Mamba teachers.

**Held fixed.** d=7, r=7, p=0.004; the Experiment 16 pool and partitions; `det_evts`
only, scattered by `StudentModels.to_sequence()`; hard labels (alpha=1); Adam, constant
LR 3e-3; batch 10,000; 200 epochs, no early stopping; minimum-`val_loss` checkpoint;
`TF_DETERMINISTIC_OPS=1`, `TF_CUDNN_DETERMINISTIC=1`, `--require-determinism`.

**Manipulated variable.** The recurrent layer. (A 3-value LR sweep for the Mamba arm is
planned as a follow-up control — see *Caveats*.)

### Platform — EAF, not grace1

Experiment 16's d=7 corpus was produced on grace1. This experiment runs on the **EAF
Jupyter pod**, so the pool and the GRU checkpoints were copied there rather than
regenerated: a pool regenerated on x86-64 would be a different set of shots and would
break the paired comparison.

| | |
|---|---|
| pod | `jupyter-kchey`, one **17.8 GiB** GPU slice (MIG), 45 GiB cgroup memory limit |
| pool | `/scratch/fast/7DayLifetime/kchey/pools_d7_p004_19M/data_d7_p0.004_r7_FORMAL.npz` |
| pool identity | 19,000,000 shots, `det_evts` (19M, 336), gen_seed 42, `flips` SHA `b49fbe40786da9e6…` — SHA-256 of the whole file verified equal to grace1's after transfer |
| GRU checkpoints | `/scratch/fast/7DayLifetime/kchey/gh200_gru_d7_e200_20260819T192524Z/seed{0,1,2}/ckpt/` |
| scratch expiry | **7-day lifetime — pool and results are deleted around 2026-09-29** |

Partitions are the Experiment 13 / 16 layout: training `[0, 10M)`, validation
`[15.0M, 15.2M)`, evaluation `[15.2M, 17.0M)`, sealed test `[17.0M, 19.0M)`.
MWPM on the evaluation block is 0.004291 ± 0.000049 (7,724 / 1.8M), measured in
Experiment 16 on these same shots.

### Models

| | GRU (Experiment 16) | Mamba (this experiment) |
|---|---|---|
| file | `StudentModels.build_gru_student` | `MambaModel.build_mamba_decoder` |
| input | `det_evts` → `[8, 48]` | identical (fed as int8, cast on device) |
| recurrence | `GRU(140, reset_after=False)` | 1 × `MambaBlock(d_model=100, d_state=16, expand=2, d_conv=4, dt_rank=7)`, pre-norm residual |
| readout | final state → `Dense(1)` | final timestep → RMSNorm → `Dense(1)` |
| output | logit, decision `logit > 0` | identical |
| parameters | **79,521** | **79,001 (−0.65%)** |
| framework | TF 2.15 / Keras 2 | TF 2.15 / Keras 2, scan unrolled over T=8 in plain ops |

Parameter match measured with `count_params()` on built models, as Experiment 16 matched
the MLP. Runner-up configurations from the search: `d_model=104, d_state=8` (79,665,
+0.18%) and `d_model=56, 3 layers` (80,641, +1.41%). One layer was chosen so both models
have exactly one recurrent layer; `d_state=16` is the Mamba default.

### Result — training side, seed 0, LR 3e-3

`figures/exp18_mamba_vs_gru_d7.png` (rebuild: `.venv/bin/python plot_mamba_vs_gru_d7.py`).

| | Mamba | GRU seed 0 | GRU seed 1 | GRU seed 2 |
|---|---|---|---|---|
| parameters | 79,001 | 79,521 | 79,521 | 79,521 |
| best `val_loss` | **0.1031** (epoch 176 of 180 so far) | 0.0350 (ep 102) | 0.0335 (ep 192) | 0.0331 (ep 179) |
| final-epoch `val_loss` | 0.1037 | 0.0756 | 0.0341 | 0.0357 |
| s/epoch | **218** | 11 | 11 | 11 |
| eval-block p_L | TBD | 0.012806 | 0.011922 | 0.012245 |
| ×MWPM | TBD | 2.984 | 2.778 | 2.854 |

Matched-epoch comparison, Mamba against the mean of the three GRU seeds:

| epoch | Mamba | GRU mean | ratio |
|---:|---|---|---:|
| 10 | 0.1455 | 0.0638 | 2.28× |
| 30 | 0.1248 | 0.0456 | 2.74× |
| 60 | 0.1148 | 0.0389 | 2.95× |
| 100 | 0.1085 | 0.0374 | 2.90× |
| 150 | 0.1059 | 0.0369 | 2.87× |
| 180 | 0.1037 | 0.0411 | 2.52× |

The ratio narrows at epoch 180 only because GRU seed 0's curve rises late, not because
Mamba gained.

**Three readings, in order of how well they are supported.**

*The gap is large and closed early.* Mamba is worse than the GRU from epoch 10 onward and
the separation is ~3× by epoch 60. Improvement has decelerated to about 0.0006 per 10
epochs over the last 40; the remaining 20 epochs cannot close a factor of three. At this
configuration a selective SSM does not match a gated RNN on this decoding problem.

*It underfits rather than overfits.* Training loss 0.1031 against validation 0.1037 — the
two curves lie on top of each other for the whole run. With 10M training shots and 79k
parameters the model is not memorising; it is failing to represent the decision function.
That points the follow-ups at capacity and optimisation, not at regularisation or data
volume.

*Its optimisation is far more stable.* Mean absolute epoch-to-epoch change in `val_loss`
over the second half: **Mamba 0.00047, GRU 0.00168** — 3.6× steadier. The GRU curves
oscillate and seed 0 degrades sharply after ~epoch 140 (0.0350 at its best, 0.0756 at
epoch 200), which is what makes the minimum-`val_loss` checkpoint rule necessary there.
The Mamba curve is monotone and its final epoch is its best. Worth noting alongside
Experiment 17's instability finding, though the two are different architectures.

**Cost.** 218 s/epoch against the GRU's 11 s/epoch at matched parameters — ~20× per epoch,
about 12 hours for one 200-epoch seed. The unrolled 8-step scan is many small elementwise
kernels, so the run is dispatch-bound rather than compute-bound; `jit_compile=True` made
it slightly *slower* (272 vs 250 ms/step on the smoke test) and was left off. Some of the
gap against the 11 s/epoch reference is platform: the GRU number is a full GH200, this is
a 17.8 GiB A100 MIG slice.

**FPGA relevance.** The selective scan needs `exp`, `softplus`, and a per-timestep state
update — the same class of operations that put `FullRCNNModel` outside hls4ml's supported
set and motivated the student-model approach. Even a winning Mamba would be a teacher or
an accuracy-ceiling probe, not a deployment candidate, unless the scan were restructured.

### Protocol and run inventory

| # | run | seeds | status |
|---:|---|---|---|
| 1 | smoke, 500k shots, 3 epochs | 0 | done 2026-09-22, pipeline verified end to end |
| 2 | **LR 3e-3, 10M, 200 epochs** | **0** | **running, epoch 180/200, ETA ~11:35 CDT** |
| 3 | eval-block scoring of run 2's `.best` checkpoint | 0 | pending run 2 |
| 4 | LR 1e-3, 10M, 200 epochs | 0 | not started |
| 5 | LR 1e-2, 10M, 200 epochs | 0 | not started |
| 6 | committed seeds at the selected LR | 1, 2 | not started |
| 7 | sealed-block `[17M, 19M)`: Mamba ×3, GRU ×3, MWPM | 0,1,2 | not started |

Output: `$RT/results/mamba_d7_p004_lr0.003_seed0_20260922T072448Z/` with `driver.log`,
`MANIFEST.txt`, `pool_provenance.json`, `runs/<tag>.{csv,history.json,config.json}`,
`ckpt/<tag>.{best,lastepoch}.weights.h5`.

### Code

| file | role |
|---|---|
| `MambaModel.py` | the selective-SSM decoder and the parameter-match search |
| `train_mamba.py` | trains one run; reads training + validation only, scores nothing |
| `eval_mamba_on_tail.py` | scores one block, decodes MWPM on the same shots, dumps per-shot outcomes keyed by `shot_idx` |
| `analyze_mamba_vs_gru.py` | p_L, Clopper-Pearson 95% CI, ×MWPM, seed mean/sd, McNemar (Mamba vs GRU, each vs MWPM), strict index alignment |
| `check_mamba_local.py` | 16 structural checks — parameter counts, gradients on every weight, causality, bit-exact checkpoint round trip, seed determinism |
| `eaf_run_mamba_d7.sh` | driver: `MODE=smoke\|lrsweep\|full\|eval` |
| `plot_mamba_vs_gru_d7.py` | the figure above |

### Caveats

- **One learning rate, one seed.** 3e-3 was inherited from the GRU and has not been shown
  appropriate for an SSM. The LR sweep is the first control to run; until it does, the
  honest claim is "at the GRU's recipe", not "Mamba is worse".
- **One configuration.** `d_model=100, d_state=16, 1 layer` was picked to match parameters,
  not tuned. A 3-layer/narrower or wider-state variant at the same budget is untested.
- **No p_L yet.** Everything above is validation loss. The decoder-level number and the
  paired McNemar come from the evaluation block.
- **Sequence length.** T=8. A selective SSM's machinery targets long-range dependence; at
  8 timesteps it may be spending parameters on capability the problem does not need. A
  hypothesis, not a measurement.
- **Platform split.** GRU s/epoch is grace1 GH200; Mamba's is an EAF MIG slice. Timing is
  indicative, not a matched benchmark. The *accuracy* comparison is unaffected — same
  pool, same shots, same partitions.
- **Scratch expiry 2026-09-29.** Pull results and checkpoints before then.

### Next step

Finish run 2, score the evaluation block, then the LR sweep. Decide after the sweep
whether Mamba stays in the study as an accuracy probe or is recorded as a negative
control. Distillation, larger teachers and sparse-token inputs remain out of scope until
then.
