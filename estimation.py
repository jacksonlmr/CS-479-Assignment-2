import numpy as np

try:
    import cupy as xp
    xp.array([0])  # triggers GPU init; fails if no GPU
except Exception:
    import numpy as xp

def ml_estimation(data: np.ndarray):
    # calculate sample mean
    mu_hat = xp.mean(data, axis=0)

    # calculate sample covariance matrix
    # (X_centered.T @ X_centered) / N is equivalent to mean of outer products
    X_centered = data - mu_hat
    sigma_hat = (X_centered.T @ X_centered) / data.shape[0]

    return mu_hat, sigma_hat
