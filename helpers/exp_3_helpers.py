import numpy as np
from theory.classifier import bayesian_img
import math

def extract_faces(img: np.ndarray, img_ref: np.ndarray):
    """
    Returns a smaller array from img where the values correspond to the matching values 
    where img_ref is not [0, 0, 0]

    img and img_ref are (N, 3) shape, where N is the total number of pixels in the image
    """
    # mask for where values are not faces
    face_mask = np.all(img_ref != 0, axis=1)
    output_array = img[face_mask]
    
    return output_array

def make_chromatic(img: np.ndarray):
    """
    Returns a chromatic space image from an rgb space image
    """
    img_sums = np.sum(img, axis=1)
    
    R_values = img[:, 0]
    G_values = img[:, 1]
    
    safe = img_sums > 0
    r_values = np.where(safe, R_values/np.where(safe, img_sums, 1), 0)
    g_values = np.where(safe, G_values/np.where(safe, img_sums, 1), 0)

    img_chromatic = np.vstack((r_values, g_values)).T

    return img_chromatic

def make_chromatic_1(img: np.ndarray):
    """
    Returns a chromatic space image from an rgb space image
    """
    C_b_constants = [-0.169, 0.332, 0.500]
    C_r_constants = [0.500, -0.419, -0.081]

    C_b = (img * C_b_constants).sum(axis=1)
    C_r = (img * C_r_constants).sum(axis=1)

    return np.vstack((C_b, C_r)).T

def reconstruct_img(results: np.ndarray, img_flat: np.ndarray, og_img: np.ndarray):
    """
    Reconstructs an image from the results of face localization and the original image
    """
    og_shape = og_img.shape

    result_flat = np.where(results.reshape(-1, 1), img_flat, [255, 255, 255])

    result = result_flat.reshape(og_shape)
    return result.astype(np.uint8)

def calc_error(img_ref: np.ndarray, results: np.ndarray):
    """
    Calculates the FP and FN rate from the reference image, and the results 
    from classification
    """
    # mask (N,)
    img_ref_mask = np.all(img_ref != 0, axis=1)
    # print(img_ref_mask.shape)
    # model said its skin, its not (FP) (result is 1, img_ref is 0)
    fp = ((results == 1) & (img_ref_mask == False)).sum()
    tn = ((results == 0) & (img_ref_mask == False)).sum()
    
    # FN
    fn = ((results == 0) & (img_ref_mask == True)).sum()
    tp = ((results == 1) & (img_ref_mask == True)).sum()

    set_size = results.shape[0]
    fpr = float(fp/(fp + tn))
    fnr = float(fn/(fn + tp))

    return fpr, fnr

def calc_c(sigma):
    return  1/((2 * math.pi) * (np.linalg.det(sigma) ** 0.5))

def gen_roc(mu, sigma, img_flat, img_ref_flat):
    """
    Calculates values of t, img is flattened img
    """
    # calculate bayesian and error for each 20 values of t

    # generate values of t
    c_value = calc_c(sigma)
    t_values = [0]
    for i in range(1, 5000):
        # print(i)
        t_values.append(t_values[i-1] + float(c_value/5000))
    # calculate bayesian for each 
    # {t_value: (fpr, fnr)}
    bayesian_error_results = np.empty(shape=(len(t_values), 2))
    t_values = np.array(t_values)
    for i, value in enumerate(t_values):
        result = bayesian_img(mu, sigma, img_flat, value)
        bayesian_error_results[i] = calc_error(img_ref_flat, result)

    return t_values, bayesian_error_results
