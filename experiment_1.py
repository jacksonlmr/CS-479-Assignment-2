from tabulate import tabulate
import numpy as np_cpu
from classifier import bayesian_case_3, bayesian_case_1
from estimation import ml_estimation

try:
    import cupy as xp
    xp.array([0])  # triggers GPU init; fails if no GPU
    print("GPU detected — using CuPy")
except Exception:
    import numpy as xp
    print("No GPU detected — using NumPy")

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

def to_cpu(arr):
    return arr.get() if hasattr(arr, 'get') else arr

for frac in fractions:
    samples_set1_est_params[frac] = ml_estimation(samples_set1[frac])
    samples_set2_est_params[frac] = ml_estimation(samples_set2[frac])

    table_data = [
        ["mu 1", to_cpu(samples_set1_est_params[frac][0])],
        ["sigma 1", to_cpu(samples_set1_est_params[frac][1])],
        ["mu 2", to_cpu(samples_set2_est_params[frac][0])],
        ["sigma 2", to_cpu(samples_set2_est_params[frac][1])]
    ]
    print(f"Parameters from {frac*100}% of data:")
    print(tabulate(table_data, headers=["Parameter", "Estimate"], tablefmt="grid"))


# calculate classifications — classify all points in one batched GPU call
print("\nClassifying with real parameters...")
results_real = bayesian_case_3(mu1, mu2, sigma1, sigma2, set1, set2, combined_set)

real_misses = [
    int(xp.sum((results_real == 2) & (combined_set_labels == 1))),  # w1 misclassified as w2
    int(xp.sum((results_real == 1) & (combined_set_labels == 2))),  # w2 misclassified as w1
]

# {frac: [s1_miss, s2_miss]}
sample_misses = {}

for frac in fractions:
    print(f"Classifying with {frac*100:.4g}% estimated parameters...")
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

# Print miss rates for real params
print(f"\nReal Parameter Missclassification Rates:")
print(f"    W1: {s1_miss_rate_real_params}")
print(f"    W2: {s2_miss_rate_real_params}")
print(f"    Total: {total_miss_rate_real_params}")

# print miss rates for est params
print("\nEstimated Parameter Missclassification Rates")
for key, value in sample_miss_rates.items():
    print(f"\n{key*100:.4g}% sample:")
    print(f"    W1: {sample_miss_rates[key][0]}")
    print(f"    W2: {sample_miss_rates[key][1]}")
    print(f"    Total: {sample_miss_rates[key][2]}")
    print(f"    W1 percent change: {((s1_miss_rate_real_params - sample_miss_rates[key][0])/s1_miss_rate_real_params)*100}%")
    print(f"    W2 percent change: {((s2_miss_rate_real_params - sample_miss_rates[key][1])/s2_miss_rate_real_params)*100}%")
    print(f"    Total percent change: {((total_miss_rate_real_params - sample_miss_rates[key][2])/total_miss_rate_real_params)*100}%")

# Classify using zero out diagonals
sample_misses = {}

for frac in fractions:
    print(f"Classifying with {frac*100:.4g}% estimated parameters...")
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


# print miss rates for est params with zero non diagonals
print("\nEstimated Parameter Zero'd Non-Diagonals Missclassification Rates")
for key, value in sample_miss_rates.items():
    print(f"\n{key*100:.4g}% sample:")
    print(f"    W1: {sample_miss_rates[key][0]}")
    print(f"    W2: {sample_miss_rates[key][1]}")
    print(f"    Total: {sample_miss_rates[key][2]}")
    print(f"    W1 percent change: {((s1_miss_rate_real_params - sample_miss_rates[key][0])/s1_miss_rate_real_params)*100}%")
    print(f"    W2 percent change: {((s2_miss_rate_real_params - sample_miss_rates[key][1])/s2_miss_rate_real_params)*100}%")
    print(f"    Total percent change: {((total_miss_rate_real_params - sample_miss_rates[key][2])/total_miss_rate_real_params)*100}%")