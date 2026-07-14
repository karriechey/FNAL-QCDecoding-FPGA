# Quantization scheme

> **Doc status — rewritten 2026-07-13.** Original version (committed 2026-06-19) documented
> **only** the QDense-toy sweep and described it as "the QAT sweep." That was wrong for the
> paper: the published Pareto comes from a *different codebase* running a *different sweep*.
> Both experiments are now documented below, in order. **Part 1 is the paper's result.**
> Verified against `CNNModel_quantized.py`, `train_one_quantized.py`, `sweep_quantized.py`
> at commit ≤`46079f5`.

Methods note for the paper and for Giuseppe (hls4ml). The integer/fractional split is a real
FPGA design decision, not just a bit count — so it is stated explicitly.

---

# Part 1 — CURRENT: weights-only QAT on the real architecture ← **the paper's result**

**Source of truth:** `CNNModel_quantized.py` (built 2026-07), driven by `train_one_quantized.py`
and `sweep_quantized.py`. Produces `out_q/` and the Pareto in `RUN_LOG.md`.

## Method: QAT, weights-only, fake-quant forward pass

This answers the collaborator's meeting question 1 directly (QAT vs post-training vs hybrid):

- Trainable variables are stored **full-precision (FP32)**.
- The **forward pass quantizes each weight tensor at its point of use** — `WeightQuant.q(w)`
  is applied immediately before every matmul (`CNNModel_quantized.py:83-87`).
- Gradients flow to the FP32 masters through QKeras's straight-through estimator.
- So: **quantization-aware training with a fake-quant forward pass.** Not post-training
  quantization. Quantization is applied identically at training and at inference.

Why point-of-use patching rather than QDense layer substitution: the reference architecture's custom `Layer`
subclasses hand-manage their weights via `add_weight` inside `call()`, so the standard
`Dense → QDense` swap cannot reach them. This module patches the layer classes instead,
mirroring each original method and adding **only** the quantizer at the weight site.

## What is passed

```python
WeightQuant._quantizer = quantized_bits(bits, 1)   # total bits, 1 integer bit
# weights only — NO activation quantizer anywhere in this path
```

## What is quantized (r=3 default forward graph)

| Layer | Quantized tensors |
|---|---|
| `CNNKernelWithEmbedding` | `kernel_weights_det_bits`, `kernel_weights_det_evts`, `kernel_bias` |
| `CNNStateCorrelator` | `params_state_evolutions`, `params_b` |
| `RCNNKernelCombiner` | `TranslationFrac_*`, `TranslationPhase_*`, `Inverter_*`, `NonUniformResponseAdj` |
| `StateDecoder` | stock `Dense` → `QDense` (`kernel_quantizer` + `bias_quantizer`) |

**Left unquantized, deliberately:** all **activations** (full precision — there is no
`quantized_relu` / `QActivation` in this path); `RCNNRecurrenceBaseKernel.kernel_weights_triplet_states`
(dormant at r=3); the detector-state embedders. Biases **are** quantized, at the same width.

## Sweep grid — 1-D, weight bits only

```python
WEIGHT_BITS = [8, 6, 4, 3, 2]      # + 32-bit FP32 anchor (reused, never retrained)
SEEDS       = [0, 1, 2]
```

`weight_bits` None or ≥32 → quantizer is a **no-op**, byte-identical FP32 (the in-sweep baseline).
There is **no `a_bits` axis.** Fixed: d=5, p=0.010, r=3, k=3, 10M train / 200k test, 50 epochs,
batch 10k, hidden [100,100], npol=2.

## Caveat for the paper (weights-only)

The reported Pareto is a **weights-only** result — activations remain FP32. Model-size numbers
(201 KB → 12.6 KB) are therefore *weight-storage* numbers, and a real FPGA deployment still has
to quantize activations. That is Phase 2 in `RUN_LOG.md`, not yet run. **Say this explicitly**
rather than letting a referee find it.

Also: the integer split is **pinned at 1 integer bit for every width** — only total width B is
swept. Weights span ≈[−2, +2) at every width; only resolution changes; any trained weight beyond
±2 saturates. A real deployment should pick integer bits from *measured* ranges (hls4ml profiling)
rather than assuming 1. That's Phase 3, and naming it pre-empts the critique.

---

# Part 2 — SUPERSEDED: the QDense-toy `(w_bits, a_bits)` sweep

> **Do not cite this as the paper's method.** Kept because its results file
> (`results/results_quantization_sweep_qdense.csv`, 16 configs) still exists, and because the
> sign-bit analysis below is correct and carries over to Part 1 unchanged.

**Source:** `QATfinal_notebook.ipynb` (`build_qdense_cnn` / `build_qdense_rcnn`), also
`train_quantized_qdense.ipynb`. Both notebooks, last touched 2026-06-20. `build_qdense_*`
appears in **no** current `.py` file — this lineage is retired.

**How it differs from Part 1:** a **stand-in** architecture of plain QDense layers, not
the reference architecture's real custom layers; and a **2-D** sweep `w ∈ {2,4,8,32} × a ∈ {2,4,8,32}` with
quantized activations.

```python
w_q = quantized_bits(w_bits, 1)              # kernel_quantizer and bias_quantizer
a_q = quantized_relu(a_bits)                 # QActivation between layers
```

---

# The sign-bit subtlety — **applies to BOTH parts**

QKeras `quantized_bits(bits, integer, keep_negative=True)` defaults `keep_negative=True`, and
`integer` counts integer bits **above the binary point, excluding the sign**. Real layout:

```
total bits = 1 (sign) + integer + fractional
fractional = bits - integer - 1
```

`quantized_bits(4, 1)` is **1 sign + 1 integer + 2 fractional**, NOT 1 integer + 3 fractional.
(Naive readings that ignore `keep_negative` are off by one bit.) Verified empirically from the
quantized grid's step size.

## Weight quantizer — measured grid (signed). Current sweep uses B ∈ {2,3,4,6,8}.

| call | sign | integer | fractional | step | range |
|---|---|---|---|---|---|
| `quantized_bits(2, 1)` | 1 | 1 | 0 | 1.0   | [-2, +1]      |
| `quantized_bits(4, 1)` | 1 | 1 | 2 | 0.25  | [-2, +1.75]   |
| `quantized_bits(8, 1)` | 1 | 1 | 6 | 2^-6  | [-2, +1.984]  |
| `w=32`                 | — | — | — | float32 (unquantized) | — |

At 2-bit, weights have **zero fractional bits** — literally integers {−2,−1,0,1}. (Consistent
with the observed collapse to p_L ≈ 0.128 at 2 bits.)

## Activation quantizer — measured grid (unsigned, ReLU ≥ 0)

**Part 2 only.** Part 1 leaves activations in FP32.

| call | integer | fractional | step | range |
|---|---|---|---|---|
| `quantized_relu(2)` | 0 | 2 | 0.25   | [0, 0.75]   |
| `quantized_relu(4)` | 0 | 4 | 0.0625 | [0, 0.9375] |
| `quantized_relu(8)` | 0 | 8 | 2^-8   | [0, 0.996]  |

## hls4ml / ap_fixed equivalent (Giuseppe's language)

ap_fixed's integer field **includes** the sign bit, so it is `integer + 1`:

- Weights (Part 1 + 2): `quantized_bits(B, 1, keep_negative=1)` ↔ `ap_fixed<B, 2>` (range ≈ [−2, +2))
- Activations (Part 2 only): `quantized_relu(A)` ↔ `ap_ufixed<A, 0>` (range [0, 1))

For the current weights-only result, the handoff to Giuseppe is: **`ap_fixed<6, 2>` weights at
the knee**, activations still FP32 pending Phase 2.
