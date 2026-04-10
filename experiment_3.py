import argparse
import os
import warnings
import cv2

# Parse args BEFORE importing project modules so FORCE_CPU is visible
# to classifier.py and estimation.py at their import-time GPU detection.
parser = argparse.ArgumentParser(description="Bayesian classifier enperiment")
group = parser.add_mutually_exclusive_group()
group.add_argument("--cpu", action="store_true", help="Force NumPy (CPU) even if GPU is available")
group.add_argument("--gpu", action="store_true", help="Require CuPy (GPU); exit if unavailable")
args = parser.parse_args()

if args.cpu:
    os.environ["FORCE_CPU"] = "1"

# Project imports — classifier.py / estimation.py see FORCE_CPU here
from theory.classifier import bayesian_case_3_img
from theory.estimation import ml_estimation
from helpers.display_helpers import (
    console, build_param_table, build_rate_table,
    param_legend, rate_legend, with_legend, Group, Rule,
)
from helpers.startup_helpers import select_array_module

warnings.filterwarnings("ignore", category=FutureWarning, module="cupy")

# np = select_array_module(args)
import numpy as np

# Start of enperiment code 

# Open training images 
base_img_path = os.path.join("P2_Data", "Data_Prog2")
training_1 = cv2.imread(os.path.join(base_img_path, "Training_1.ppm"))
ref_1 = cv2.imread(os.path.join(base_img_path, "ref1.ppm"))

def extract_faces(img, img_ref):
    # flatten the pixels (loses information about pixel location)
    img_reshaped = np.reshape(img, (-1, 3))
    img_ref_reshaped = np.reshape(img_ref, (-1, 3))

    # mask for where values are not faces
    face_mask = np.all(img_ref_reshaped != 0, axis=1)
    output_array = img_reshaped[face_mask]
    
    return output_array

def make_chromatic(img):
    img_sums = np.sum(img, axis=1)
    
    R_values = img[:, 0]
    G_values = img[:, 1]
    
    r_values = R_values/img_sums
    g_values = G_values/img_sums

    img_chromatic = np.vstack((r_values, g_values)).T

    return img_chromatic

face_pixels = extract_faces(training_1, ref_1)

face_pixels_chromatic = make_chromatic(face_pixels)
print(face_pixels_chromatic.shape)

mu, sigma = ml_estimation(face_pixels_chromatic)
print(mu)
print(sigma)

# calculate classifications — classify all points in one batched GPU call
results_t6 = bayesian_case_3_img(mu, sigma, face)

real_misses = [
    int(np.sum((results_t6 == 2) & (combined_set_labels == 1))),  # w1 misclassified as w2
    int(np.sum((results_t6 == 1) & (combined_set_labels == 2))),  # w2 misclassified as w1
]

