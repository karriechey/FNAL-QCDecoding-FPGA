# Experiment Tracking Log

> ### ⚠️ STALE — committed 2026-06-05, never updated. **Do not cite.**
> Superseded by `RUN_LOG.md` (the authoritative ledger). Problems: Notes fields are all
> empty; it points at `results/results_scaling.csv`, which **does not exist**; and its
> "Experiment 2: Quantization Sweep (16 configs)" is the retired **QDense-toy** 2-D
> (w,a) sweep, **not** the weights-only QAT sweep the paper reports.
> Kept only as a record of what was attempted in May–June 2026.

## Overview
Track all quantization experiments: architecture, quantization sweep, distance scaling, threshold analysis.

## Run Log

### Experiment 1: Architecture Comparison (Full Precision)
- **Date**: May 2026
- **Config**: d=5, p=0.01, full precision (w=32, a=32)
- **Models**: CNN, RCNN, MWPM baseline
- **Results File**: `results/results_mwpm.csv, results/results_fullprecision.csv`
- **Notes**: 

### Experiment 2: Quantization Sweep (16 configs)
- **Date**: May 2026
- **Config**: d=5, p=0.01, w ∈ {2,4,8,32}, a ∈ {2,4,8,32}
- **Results File**: `results/results_quantization_sweep_qdense.csv`
- **Notes**: 

### Experiment 3: Code Distance Scaling
- **Date**: June 2026
- **Config**: p=0.01, d ∈ {3, 5, 7}, 4 quantization levels
- **Results File**: `results/results_scaling.csv`
- **Notes**: 

### Experiment 4: Threshold Analysis
- **Date**: June 2026
- **Config**: d=5, p ∈ {0.001, 0.005, 0.01, 0.05}, multiple quantization levels
- **Results File**: `results/results_threshold.csv`
- **Notes**: 

## Results CSV Template

Each experiment should produce a CSV with columns:
```
experiment, date, code_distance, error_rate, weight_bits, activation_bits, logical_error_rate, model_size_kb, mac_count, inference_time_ms, seed, notes
```

## Hyperparameters (Fixed Across All Runs)
- Dataset size: 1,000,000 samples per (d,p) config
- Train/val/test split: 80/10/10 (800k/100k/100k)
- Random seed: 42
- Batch size: 256
- Max epochs: 100 (early stopping with patience=10)
- Optimizer: Adam (lr=0.001), legacy Adam on Apple Silicon

## Key Metrics
- **Primary**: Logical error rate p_L (count wrong predictions / total predictions)
- **Secondary**: Model size (KB), MAC count, inference latency
- **Analysis**: Pareto frontier, threshold p*, scaling with distance
