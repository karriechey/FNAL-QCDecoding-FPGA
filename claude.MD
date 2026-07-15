# CLAUDE.md

## What this repo is
Quantization-aware training and FPGA-deployment study for the compact RCNN
surface-code decoder, plus the earlier data-volume/scaling study that
established the full-precision baseline. Extracted from `QuantumDecoderQKeras`
via `git filter-repo` on 2026-07-14 — history here is real (dates/authorship
preserved), but this repo only contains files relevant to this paper's
research, not the full Fermilab working tree.

**Architecture:** the reference architecture Sarıca's RCNN (github.com/usarica/FNAL-QCDecodingTests)
**Advisor:** Gabriel Perdue (github.com/gnperdue/FNAL-QCDecodingTests)
**This work:** Karrie Chey, Fermilab OQI Fellowship

## Research threads (both go into the paper)
1. **Data-volume scaling study** — the earlier study establishing the
   full-precision baseline that the quantization sweep starts from.
   **Numbers below are UNVERIFIED — do not repeat them as fact until
   checked against `docs/RUN_LOG.md`, `docs/RCNN_d5_r3_ladder_REFERENCE.md`,
   and `docs/SWEEP_SUMMARY.md`:** training-set size sweep, RCNN vs MWPM at
   fixed d=5, r=3. If asked to describe this study, verify against source
   first — trace every number to a specific file before stating it.
2. **Quantization study** — weights-only QAT bit-width sweep (8/6/4/3/2 +
   FP32 anchor) at a fixed data-volume point from thread 1. McNemar
   validation vs MWPM. This is the paper's primary new contribution.
3. **Range profiling (in progress)** — per-tensor ap_fixed<B,I> assignment
   for hls4ml, motivated by Giuseppe's FPGA deployment questions.

## Stack
TF 2.15.1 / Keras 2 / QKeras 0.9 / Stim / PyMatching / hls4ml.
TF pinned at 2.15 — Keras 3 breaks the custom Layer/add_weight subclasses in
CNNModel.py. Do not suggest upgrading TF/Keras without checking this first.

## Ground truth hierarchy
1. `docs/RUN_LOG.md` — authoritative reproducibility ledger, covers both
   research threads
2. `docs/QUANTIZATION_SCHEME_VERIFIED.md` — if present, supersedes any other
   quantization doc. If absent, treat quantization method claims as
   unverified until traced into source.
3. Code itself, always, over any .md file's claims

## Documentation discipline
Every file under `docs/` must carry a header:
`Last verified against code: <commit sha>, <date>`
When asked to update a doc, re-verify against current source first — do not
edit a doc from memory of what it used to say. If a doc's claims can't be
confirmed from the code, say so explicitly rather than preserving existing
(possibly stale) text.

## Script/file creation discipline
**Every script, notebook, or doc you generate must include a header comment
with creation date, and update a "last modified" line on every edit:**
```python
# Created: YYYY-MM-DD
# Last modified: YYYY-MM-DD
```
This applies retroactively too — if you're editing a file that lacks this
header, add it (using `git log` for the true creation date if available,
not today's date).

## Notebook discipline
When asked to log an experiment result, append to the existing notebook for
that research thread under `analysis_notebooks/` — one notebook per thread
(data-volume ladder, quantization weights, range profiling, etc.), not a
new notebook per session. Only create a new notebook if the thread
genuinely doesn't exist yet, and ask first before doing so.

## EAF results — remote, not local
Compute-heavy runs happen on Fermilab EAF (A100 GPU). Result plots, CSVs,
and `.history.json` files generated there do NOT automatically sync to this
repo or to the local Mac — they sit on EAF (e.g. `~/rcnn_threshold/`) until
manually pulled via `jobsub_lite`/ifdh to dCache, then copied locally.

**When you reference a plot or result that would logically have been
generated on EAF (e.g. anything from a GPU sweep, `qat_pareto.png`,
`ladder.png`, or similar), check whether it actually exists in this repo's
`results/` or `analysis_notebooks/` first.** If it's missing, tell me
explicitly: "This result appears to exist only on EAF — you'll need to pull
it before I can [reference/analyze/include] it." Do not assume a result
exists locally just because `RUN_LOG.md` describes the run that produced it.

## Git discipline
Stage files individually (`git add <file>`), never `git add -A` or
`git add .` — this repo has untracked scratch files that should stay
untracked. Multi-line commit messages via heredoc, not `-m` for anything
non-trivial.

## Do not
- Modify `CNNModel.py` (the reference architecture's original) — quantization logic lives only
  in `CNNModel_quantized.py`
- Mix the 200k eval tail with the 10k tail — they have different MWPM
  baselines (0.0451 vs 0.04940); this caused a real bug previously
- Trust `docs/EXPERIMENTS.md` if it's still present — flagged stale, has
  empty Notes fields and points at a nonexistent results file
- State any number from the data-volume scaling study without tracing it
  to a specific file in this session or a prior verified doc