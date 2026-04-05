import numpy as np
import math
from plot_data import plot_gaussian_dataset
from classifier import bayesian_case_1

mu1 = np.array([1, 1])
sigma1 = np.array([[1, 0], [0, 1]])

mu2 = np.array([4, 4])
sigma2 = np.array([[1, 0], [0, 1]])

set1 = np.random.multivariate_normal(mu1, sigma1, 60000)
set2 = np.random.multivariate_normal(mu2, sigma2, 140000)
combined_set = np.vstack((set1, set2))

# # plot decision
# plot_gaussian_dataset(set1, set2, -1, 4.718, "Decision Boundary Dataset A")

# {coordinate: result}
classifications_real_params = {}
classifications_est_params = {}

# Classify using actual parameters of distribution
for x in combined_set:
    result = bayesian_case_1(mu1, mu2, sigma2, sigma2, set1, set2, np.array(x))

    if (result == 1):
        classifications_real_params[tuple(x)] = result
    elif (result == 2):
        classifications_real_params[tuple(x)] = result

# Classify using estimated parameters from ML
for x in combined_set:
    # result = bayesian_case_1(mu1, mu2, sigma2, sigma2, set1, set2, np.array(x))

    if (result == 1):
        classifications_est_params[tuple(x)] = result
    elif (result == 2):
        classifications_est_params[tuple(x)] = result

# Calculate missclassifications for actual parameters
# model said 1, actual 2
missclassified_s2_real_params = 0
# model said 2, actual 1
missclassified_s1_real_params = 0

for key, value in classifications_real_params.items():
    # model said 1, actual 2
    if (key in set2 and value == 1):
        missclassified_s2_real_params += 1

    if (key in set1 and value == 2):
        missclassified_s1_real_params += 1

s1_miss_rate_real_params = (missclassified_s1_real_params)/len(set1)
s2_miss_rate_real_params = (missclassified_s2_real_params)/len(set2)
total_miss_rate_real_params = (missclassified_s1_real_params + missclassified_s2_real_params)/len(combined_set)

# Calculate missclassifications for estimated parameters

print(f"w1 missclassification rate: {s1_miss_rate_real_params}")
print(f"w2 misclassification rate: {s2_miss_rate_real_params}")
print(f"total missclassification rate: {total_miss_rate_real_params}")