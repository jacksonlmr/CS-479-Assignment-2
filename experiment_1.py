import argparse
import os
import warnings

# Parse args BEFORE importing project modules so FORCE_CPU is visible
# to classifier.py and estimation.py at their import-time GPU detection.
parser = argparse.ArgumentParser(description="Bayesian classifier experiment")
group = parser.add_mutually_exclusive_group()
group.add_argument("--cpu", action="store_true", help="Force NumPy (CPU) even if GPU is available")
group.add_argument("--gpu", action="store_true", help="Require CuPy (GPU); exit if unavailable")
args = parser.parse_args()

if args.cpu:
    os.environ["FORCE_CPU"] = "1"

# Project imports — classifier.py / estimation.py see FORCE_CPU here
import numpy as np_cpu
from classifier import bayesian_case_3, bayesian_case_1
from estimation import ml_estimation
from display_helpers import (
    console, build_param_table, build_rate_table,
    param_legend, rate_legend, with_legend, Group, Rule,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="cupy")

if args.cpu:
    import numpy as xp
    print("Using NumPy (CPU) — forced via --cpu")
elif args.gpu:
    try:
        import cupy as xp
        xp.linalg.cholesky(xp.eye(2, dtype=float))  # validates cublas
        print("Using CuPy (GPU) — forced via --gpu")
    except Exception as e:
        print(f"GPU requested but CuPy is unavailable: {e}")
        print("Check your CUDA version:  nvcc --version")
        print("Install matching CuPy:    pip install cupy-cuda11x  (CUDA 11.x)")
        print("                       or pip install cupy-cuda12x  (CUDA 12.x)")
        print("Or run on CPU:            python experiment_1.py --cpu")
        raise SystemExit(1)
else:
    try:
        import cupy as xp
        xp.linalg.cholesky(xp.eye(2, dtype=float))  # validates cublas
        print("GPU detected — using CuPy")
    except Exception:
        import numpy as xp
        print("No GPU detected — using NumPy")

# set seed for consistent results
xp.random.seed(42)

mu1 = xp.array([1, 1])
sigma1 = xp.array([[1, 0], [0, 1]])

mu2 = xp.array([4, 4])
sigma2 = xp.array([[1, 0], [0, 1]])

# mu1 = xp.array([1, 1])
# sigma1 = xp.array([[1, 0], [0, 1]])

# mu2 = xp.array([4, 4])
# sigma2 = xp.array([[4, 0], [0, 8]])

set1 = xp.random.multivariate_normal(mu1, sigma1, 60000)
set1_size = set1.shape[0]
set1_labels = xp.ones(set1_size)

set2 = xp.random.multivariate_normal(mu2, sigma2, 140000)
set2_size = set2.shape[0]
set2_labels = xp.full(set2_size, 2)

combined_set = xp.vstack((set1, set2))
combined_set_labels = xp.concatenate((set1_labels, set2_labels))
combined_set_size = combined_set.shape[0]

# 0.01%, 0.1%, 1%, 10%, 100%
fractions = [0.0001, 0.001, 0.01, 0.1, 1]

# stores samples for each desired percentage
# {fraction: ndarray of samples}
samples_set1 = {}
samples_set2 = {}
samples_combined = {}

for frac in fractions:
    # Add entire dataset to frac == 1
    if (frac == 1):
        samples_set1[frac] = set1
        samples_set2[frac] = set2
        samples_combined[frac] = combined_set
        break

    # Calculate the integer sizes for the percent to sample
    size1 = int(set1.shape[0] * frac)
    size2 = int(set2.shape[0] * frac)

    # Sample by index
    idx1 = xp.random.choice(set1.shape[0], size=size1, replace=False)
    idx2 = xp.random.choice(set2.shape[0], size=size2, replace=False)

    # add samples to dict
    samples_set1[frac] = set1[idx1]
    samples_set2[frac] = set2[idx2]
    samples_combined[frac] = xp.vstack((samples_set1[frac], samples_set2[frac]))

# estimate parameters
# {fraction: (mu, sigma)}
samples_set1_est_params = {}
samples_set2_est_params = {}

for frac in fractions:
    samples_set1_est_params[frac] = ml_estimation(samples_set1[frac])
    samples_set2_est_params[frac] = ml_estimation(samples_set2[frac])

console.print(with_legend(Group(
    build_param_table("Class 1 - Estimated Parameters vs. True Parameters",
                      "1", mu1, sigma1, samples_set1_est_params, fractions),
    Rule(style="dim"),
    build_param_table("Class 2 - Estimated Parameters vs. True Parameters",
                      "2", mu2, sigma2, samples_set2_est_params, fractions),
), param_legend()))

# calculate classifications — classify all points in one batched GPU call
results_real = bayesian_case_3(mu1, mu2, sigma1, sigma2, set1, set2, combined_set)

real_misses = [
    int(xp.sum((results_real == 2) & (combined_set_labels == 1))),  # w1 misclassified as w2
    int(xp.sum((results_real == 1) & (combined_set_labels == 2))),  # w2 misclassified as w1
]

# {frac: [s1_miss, s2_miss]}
sample_misses = {}

for frac in fractions:
    mu1_est, sigma1_est = samples_set1_est_params[frac]
    mu2_est, sigma2_est = samples_set2_est_params[frac]

    results_est = bayesian_case_3(mu1_est, mu2_est, sigma1_est, sigma2_est, set1, set2, combined_set)

    sample_misses[frac] = [
        int(xp.sum((results_est == 2) & (combined_set_labels == 1))),
        int(xp.sum((results_est == 1) & (combined_set_labels == 2))),
    ]

# Calculate missclassification rates

# sample miss rates
# {frac: (s1_miss_rate, s2_miss_rate, total_miss_rate)}
sample_miss_rates = {}

for frac in fractions:
    if frac not in sample_miss_rates:
        sample_miss_rates[frac] = []

    s1_misses, s2_misses = sample_misses[frac]

    sample_miss_rates[frac].append(s1_misses/set1_size)
    sample_miss_rates[frac].append(s2_misses/set2_size)
    sample_miss_rates[frac].append((s1_misses + s2_misses)/combined_set_size)

# real params miss rates
s1_miss_rate_real_params = real_misses[0]/set1_size
s2_miss_rate_real_params = real_misses[1]/set2_size
total_miss_rate_real_params = (real_misses[0]+real_misses[1])/combined_set_size

# save estimated rates before zeroed section overwrites sample_miss_rates
estimated_miss_rates = dict(sample_miss_rates)

# Classify using zero out diagonals
sample_misses = {}

for frac in fractions:
    mu1_est, sigma1_est = samples_set1_est_params[frac]
    mu2_est, sigma2_est = samples_set2_est_params[frac]

    # zero out non diagonals
    sigma1_est[0, 1] = sigma1_est[1, 0] = 0
    sigma2_est[0, 1] = sigma2_est[1, 0] = 0

    results_est = bayesian_case_3(mu1_est, mu2_est, sigma1_est, sigma2_est, set1, set2, combined_set)

    sample_misses[frac] = [
        int(xp.sum((results_est == 2) & (combined_set_labels == 1))),
        int(xp.sum((results_est == 1) & (combined_set_labels == 2))),
    ]

# Calculate missclassification rates for zero out

# sample miss rates
# {frac: (s1_miss_rate, s2_miss_rate, total_miss_rate)}
sample_miss_rates = {}

for frac in fractions:
    if frac not in sample_miss_rates:
        sample_miss_rates[frac] = []

    s1_misses, s2_misses = sample_misses[frac]

    sample_miss_rates[frac].append(s1_misses/set1_size)
    sample_miss_rates[frac].append(s2_misses/set2_size)
    sample_miss_rates[frac].append((s1_misses + s2_misses)/combined_set_size)


# ── Misclassification Rate Comparison Tables ──────────────────────────────────

console.print(with_legend(Group(
    build_rate_table("W1 Misclassification Rates",    s1_miss_rate_real_params,    0, fractions, estimated_miss_rates, sample_miss_rates),
    Rule(style="dim"),
    build_rate_table("W2 Misclassification Rates",    s2_miss_rate_real_params,    1, fractions, estimated_miss_rates, sample_miss_rates),
    Rule(style="dim"),
    build_rate_table("Total Misclassification Rates", total_miss_rate_real_params, 2, fractions, estimated_miss_rates, sample_miss_rates),
), rate_legend()))