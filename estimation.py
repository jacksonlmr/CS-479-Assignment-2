import numpy as np
import math

def ml_estimation(data: np.ndarray):
    # calculate sample mean
    mu_hat = np.mean(data, axis=0)

    # calculate sample covariance matrix
    product_array = []
    for x in data:
        x_centered = x - mu_hat
        product = np.outer(x_centered, x_centered)
        product_array.append(product)

    product_array = np.array(product_array)
    sigma_hat = np.mean(product_array, axis=0)

    return mu_hat, sigma_hat
