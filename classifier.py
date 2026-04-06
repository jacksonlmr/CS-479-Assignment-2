import numpy as np
import math

def bayesian_case_1(mu1, mu2, sigma1, sigma2, w1_data: np.array, w2_data: np.array, instance):
    """
    Returns 1 if instance is likely to belong to class 1, 2 if instance is likely to belong to class 2
    """
    # calculate (w_i)^t 
    w1 = np.linalg.inv(sigma1) @ mu1
    w1_t = np.matrix.transpose(w1)

    w2 = np.linalg.inv(sigma2) @ mu2
    w2_t = np.matrix.transpose(w2)
    
    # multiply (w_i)^t * x
    w1_t_x = w1_t @ np.matrix.transpose(instance)
    w2_t_x = w2_t @ np.matrix.transpose(instance)
    
    # calculate ln(P(w_i))
    total_data_size = w1_data.shape[0] + w2_data.shape[0]
    p_w1 = w1_data.shape[0] / total_data_size
    p_w2 = w2_data.shape[0] / total_data_size

    ln_p_w1 = math.log(p_w1)
    ln_p_w2 = math.log(p_w2)

    # calculate w_i0
    w_10 = (-(0.5 * (np.matrix.transpose(mu1) @ np.linalg.inv(sigma1))) @ mu1) + ln_p_w1
    w_20 = (-(0.5 * (np.matrix.transpose(mu2) @ np.linalg.inv(sigma2))) @ mu2) + ln_p_w2
    
    # calculate g_i
    g_1 = w1_t_x + w_10
    g_2 = w2_t_x + w_20

    if (g_1 > g_2):
        return 1
    return 2

def bayesian_case_3(mu1, mu2, sigma1, sigma2, w1_data: np.array, w2_data: np.array, instance):
    """
    Returns 1 if instance is likely to belong to class 1, 2 if instance is likely to belong to class 2
    """
    # calculate x^t * W_i * x
    x_Wi_x_1 = instance.T @ (-0.5 * np.linalg.inv(sigma1)) @ instance
    x_Wi_x_2 = instance.T @ (-0.5 * np.linalg.inv(sigma2)) @ instance

    # calculate (w_i)^t 
    w1 = np.linalg.inv(sigma1) @ mu1
    w1_t = w1.T

    w2 = np.linalg.inv(sigma2) @ mu2
    w2_t = w2.T

    # multiply (w_i)^t * x
    w1_t_x = w1_t @ instance
    w2_t_x = w2_t @ instance

    # calculate P(w_i)
    total_data_size = w1_data.shape[0] + w2_data.shape[0]
    p_w1 = w1_data.shape[0] / total_data_size
    p_w2 = w2_data.shape[0] / total_data_size

    # calculate ln(P(w_i))
    ln_p_w1 = math.log(p_w1)
    ln_p_w2 = math.log(p_w2)

    # calculate ln|sigma_i|
    sign1, ln_sigma_1 = np.linalg.slogdet(sigma1)
    ln_sigma_1 = sign1*ln_sigma_1
    sign2, ln_sigma_2 = np.linalg.slogdet(sigma2)
    ln_sigma_2 = sign2*ln_sigma_2

    # calculate w_i0
    w_10 = (-(0.5 * (mu1.T @ np.linalg.inv(sigma1))) @ mu1) - 0.5 * ln_sigma_1 + ln_p_w1
    w_20 = (-(0.5 * (mu2.T @ np.linalg.inv(sigma2))) @ mu2) - 0.5 * ln_sigma_2 + ln_p_w2

    # calculate g_i
    g_1 = x_Wi_x_1 + w1_t_x + w_10
    g_2 = x_Wi_x_2 + w2_t_x + w_20

    # print(f"g_1: {g_1}\n")
    if (g_1 > g_2):
        return 1
    return 2