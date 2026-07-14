# Results Directory Structure

## Subdirectories

### exp1_architecture/
Full-precision architecture comparison: CNN vs RCNN vs MWPM baseline
- `cnn_d5_p001.csv` — CNN results on d=5, p=0.01
- `rcnn_d5_p001.csv` — RCNN results on d=5, p=0.01
- `mwpm_d5_p001.csv` — MWPM baseline on d=5, p=0.01
- `learning_curves_cnn.png` — Training progress plot
- `learning_curves_rcnn.png` — Training progress plot

### exp2_quantization/
16-config (w,a) sweep on d=5, p=0.01
- `sweep_results.csv` — Main results table with all 16 configs
- `pareto_frontier.png` — Model size vs logical error rate

### exp3_distance/
Distance scaling: d ∈ {3, 5, 7}, p=0.01, 4 quantization levels
- `distance_scaling_results.csv` — Results by distance
- `distance_scaling_plot.png` — p_L vs d for each quantization level

### exp4_threshold/
Threshold analysis: p ∈ {0.001, 0.005, 0.01, 0.05}, multiple quantization levels
- `threshold_results.csv` — p_L vs p for each config
- `threshold_plot.png` — Threshold crossing visualization
- `pseudo_thresholds.csv` — Estimated p* per decoder/quantization level

## Results CSV Format

Standard columns for all experiments:
- `experiment` — name of experiment (exp1, exp2, etc)
- `date` — ISO format (YYYY-MM-DD)
- `code_distance` — d ∈ {3, 5, 7}
- `physical_error_rate` — p ∈ {0.001, 0.005, 0.01, 0.05}
- `weight_bits` — w ∈ {2, 4, 8, 32}
- `activation_bits` — a ∈ {2, 4, 8, 32}
- `logical_error_rate` — primary metric: p_L = (# wrong predictions) / (# total)
- `model_size_kb` — weight size in KB
- `mac_count` — multiply-accumulate operations
- `inference_time_ms` — latency per sample
- `seed` — random seed for reproducibility
- `notes` — any special conditions or observations

## Key Metrics

**Primary**: `logical_error_rate` (lower is better)
**Secondary**: `model_size_kb` (for Pareto frontier)
**Derivative**: `pseudo_threshold` = p* (error rate at crossover point)

## Plotting

Use `plotting_utils.py` functions:
- `plot_pareto_frontier()` — exp2 results
- `plot_learning_curves()` — exp1 training progress
- `plot_distance_scaling()` — exp3 results
- `plot_threshold_analysis()` — exp4 results
