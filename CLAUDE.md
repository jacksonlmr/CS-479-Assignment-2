# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
# Activate the virtual environment
source .venv/Scripts/activate   # bash/zsh on Windows

# Install dependencies
pip install -r requirements.txt
```

## Running Scripts

```bash
python experiment_1.py   # Run the main classification experiment
python gpu_check.py      # Verify CuPy/GPU availability
```

There are no tests or linters configured for this project.

## Architecture

This is a CS 479 (Pattern Recognition) assignment implementing Bayesian classifiers on 2D Gaussian data.

**Source files:**
- `experiment_1.py` — main script; generates data, runs all three classification experiments, and prints results
- `classifier.py` — `bayesian_case_1` (linear) and `bayesian_case_3` (quadratic) discriminant functions
- `estimation.py` — `ml_estimation(data)` returns `(mu_hat, sigma_hat)` via maximum likelihood
- `display_helpers.py` — all Rich table rendering logic; exports `console`, `build_param_table`, `build_rate_table`, `param_legend`, `rate_legend`, `with_legend`

**Data flow in `experiment_1.py`:**
1. Generate synthetic 2D Gaussian datasets: w1 (60k samples, μ=[1,1], Σ=I) and w2 (140k samples, μ=[4,4], Σ=I).
2. Sample fractions `[0.0001, 0.001, 0.01, 0.1, 1]` (0.01% → 100%) from each class.
3. Run `ml_estimation` on each fraction to get estimated parameters, stored in `samples_set1_est_params[frac]` and `samples_set2_est_params[frac]`.
4. Classify the full combined dataset three ways using `bayesian_case_3`:
   - **Real params**: true μ/Σ values
   - **Estimated params**: ML estimates from each sample fraction
   - **Zeroed diagonal**: same estimates but with off-diagonal σ elements forced to 0 (applied in-place — this mutates `samples_set{1,2}_est_params`)
5. Compute per-class and total misclassification rates; compare estimated/zeroed rates against the real-params baseline.

**Critical ordering constraint:** The parameter display tables must be printed before the zeroed-diagonal classification loop runs, because that loop mutates the stored sigma arrays in-place (`sigma_est[0,1] = sigma_est[1,0] = 0`). After the mutation, the estimated sigmas are no longer recoverable.

**GPU/CPU portability:** All three computation files (`experiment_1.py`, `classifier.py`, `estimation.py`) use the same `try: import cupy as xp / except: import numpy as xp` pattern. Arrays flow through as either CuPy or NumPy arrays. `display_helpers.py:to_cpu()` converts them to NumPy before formatting.

**Display:** `display_helpers.py` sets `sys.stdout.reconfigure(encoding='utf-8')` and `Console(legacy_windows=False)` to enable Unicode block characters (`█████`) for color swatches in Rich table legends on Windows/PowerShell. The console width is fixed at 220 to keep side-by-side table+legend layouts from wrapping.

**Image data** in `P2_Data/Data_Prog2/` contains `.ppm` training and reference images for classes 1, 3, and 6 (separate part of the assignment, not yet implemented).
