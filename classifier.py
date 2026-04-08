import numpy as np
import math
import os

if os.environ.get("FORCE_CPU"):
    import numpy as xp
else:
    try:
        import cupy as xp
        xp.linalg.cholesky(xp.eye(2, dtype=float))
    except Exception:
        import numpy as xp

def bayesian_case_1(mu1, mu2, sigma1, sigma2, w1_data: np.array, w2_data: np.array, instances):
    """
    Classifies a batch of instances.
    instances: (N, D) array
    Returns: (N,) integer array — 1 if class 1, 2 if class 2
    """

    inv_s1 = xp.linalg.inv(sigma1)
    inv_s2 = xp.linalg.inv(sigma2)

    # calculate w_i = inv(sigma) @ mu_i  (linear weights, shape (D,))
    w1 = inv_s1 @ mu1
    w2 = inv_s2 @ mu2

    # calculate ln(P(w_i))
    total_data_size = w1_data.shape[0] + w2_data.shape[0]
    ln_p_w1 = math.log(w1_data.shape[0] / total_data_size)
    ln_p_w2 = math.log(w2_data.shape[0] / total_data_size)

    # calculate w_i0 (bias scalar)
    w_10 = -(0.5 * (mu1.T @ inv_s1) @ mu1) + ln_p_w1
    w_20 = -(0.5 * (mu2.T @ inv_s2) @ mu2) + ln_p_w2

    # compute discriminants for all instances at once
    # instances @ w_i gives (N,) dot products, then add scalar bias
    g_1 = instances @ w1 + w_10
    g_2 = instances @ w2 + w_20

    return xp.where(g_1 > g_2, 1, 2)

def bayesian_case_3(mu1, mu2, sigma1, sigma2, w1_data: np.array, w2_data: np.array, instances):
    """
    Classifies a batch of instances.
    instances: (N, D) array
    Returns: (N,) integer array — 1 if class 1, 2 if class 2
    """

    inv_s1 = xp.linalg.inv(sigma1)
    inv_s2 = xp.linalg.inv(sigma2)

    # calculate x^t * W_i * x for all instances at once
    # W_i = -0.5 * inv(sigma_i)
    # (instances @ W_i) has shape (N, D); elementwise multiply by instances and sum
    # gives x^T W_i x for each row — shape (N,)
    x_W1_x = (instances @ (-0.5 * inv_s1) * instances).sum(axis=1)
    x_W2_x = (instances @ (-0.5 * inv_s2) * instances).sum(axis=1)

    # calculate w_i = inv(sigma_i) @ mu_i  (shape (D,))
    w1 = inv_s1 @ mu1
    w2 = inv_s2 @ mu2

    # w_i^t * x for all instances — shape (N,)
    w1_t_x = instances @ w1
    w2_t_x = instances @ w2

    # calculate P(w_i)
    total_data_size = w1_data.shape[0] + w2_data.shape[0]
    ln_p_w1 = math.log(w1_data.shape[0] / total_data_size)
    ln_p_w2 = math.log(w2_data.shape[0] / total_data_size)

    # calculate ln|sigma_i|
    sign1, ln_sigma_1 = xp.linalg.slogdet(sigma1)
    ln_sigma_1 = sign1 * ln_sigma_1
    sign2, ln_sigma_2 = xp.linalg.slogdet(sigma2)
    ln_sigma_2 = sign2 * ln_sigma_2

    # calculate w_i0 (bias scalar)
    w_10 = -(0.5 * (mu1.T @ inv_s1) @ mu1) - 0.5 * ln_sigma_1 + ln_p_w1
    w_20 = -(0.5 * (mu2.T @ inv_s2) @ mu2) - 0.5 * ln_sigma_2 + ln_p_w2

    # compute discriminants for all instances at once
    g_1 = x_W1_x + w1_t_x + w_10
    g_2 = x_W2_x + w2_t_x + w_20

    return xp.where(g_1 > g_2, 1, 2)
