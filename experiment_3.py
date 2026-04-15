import argparse
import os
import warnings
import cv2
import math
import time
import matplotlib.pyplot as plt

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
from theory.classifier import bayesian_img
from theory.estimation import ml_estimation
from helpers.display_helpers import plot_roc, plot_roc_2
from helpers.exp_3_helpers import extract_faces, make_chromatic, make_chromatic_1, reconstruct_img, calc_error, gen_roc

warnings.filterwarnings("ignore", category=FutureWarning, module="cupy")

# np = select_array_module(args)
import numpy as np

# Start of experiment code
_start = time.time()

# Open training images
base_img_path = os.path.join("P2_Data", "Data_Prog2")

training_1 = cv2.imread(os.path.join(base_img_path, "Training_1.ppm"))
cv2.imwrite("output/training_1.jpg", training_1)
training_1 = cv2.cvtColor(training_1, cv2.COLOR_BGR2RGB)
ref_1 = cv2.imread(os.path.join(base_img_path, "ref1.ppm"))
cv2.imwrite("output/ref1.jpg", ref_1)
ref_1 = cv2.cvtColor(ref_1, cv2.COLOR_BGR2RGB)

training_3 = cv2.imread(os.path.join(base_img_path, "Training_3.ppm"))
cv2.imwrite("output/training_3.jpg", training_3)
training_3 = cv2.cvtColor(training_3, cv2.COLOR_BGR2RGB)
ref_3 = cv2.imread(os.path.join(base_img_path, "ref3.ppm"))
ref_3 = cv2.cvtColor(ref_3, cv2.COLOR_BGR2RGB)

training_6 = cv2.imread(os.path.join(base_img_path, "Training_6.ppm"))
cv2.imwrite("output/training_6.jpg", training_6)
training_6 = cv2.cvtColor(training_6, cv2.COLOR_BGR2RGB)
ref_6 = cv2.imread(os.path.join(base_img_path, "ref6.ppm"))
ref_6 = cv2.cvtColor(ref_6, cv2.COLOR_BGR2RGB)


# reduce dimensions
training_1_flat = training_1.reshape(-1, 3)
ref_1_flat = ref_1.reshape(-1, 3)

training_3_flat = training_3.reshape(-1, 3)
ref_3_flat = ref_3.reshape(-1, 3)

training_6_flat = training_6.reshape(-1, 3)
ref_6_flat = ref_6.reshape(-1, 3)

# extract face pixels from training data
face_pixels = extract_faces(training_1_flat, ref_1_flat)

face_pixels_chromatic = make_chromatic(face_pixels)

mu, sigma = ml_estimation(face_pixels_chromatic)

# convert testing data to chromatic
training_3_flat_chromatic = make_chromatic(training_3_flat)
training_6_flat_chromatic = make_chromatic(training_6_flat)

# generate ROC curve for part a (combined images)
combined_chromatic_a = np.vstack([training_3_flat_chromatic, training_6_flat_chromatic])
combined_ref_a = np.vstack([ref_3_flat, ref_6_flat])
t_a, ber_a = gen_roc(mu, sigma, combined_chromatic_a, combined_ref_a)
t_intersect_a = plot_roc(t_a, ber_a, "output/part_a")

# calculate classifications based on ROC results
results_t3 = bayesian_img(mu, sigma, training_3_flat_chromatic, t=t_intersect_a)
results_t6 = bayesian_img(mu, sigma, training_6_flat_chromatic, t=t_intersect_a)

# reconstruct images
rec_img_3 = reconstruct_img(results_t3, training_3_flat, training_3)
rec_img_3 = cv2.cvtColor(rec_img_3, cv2.COLOR_RGB2BGR)
cv2.imwrite("output/training_3_a_result.jpg", rec_img_3)

rec_img_6 = reconstruct_img(results_t6, training_6_flat, training_6)
rec_img_6 = cv2.cvtColor(rec_img_6, cv2.COLOR_RGB2BGR)
cv2.imwrite("output/training_6_a_result.jpg", rec_img_6)


# Using YC_rC_b space
training_1_flat = training_1.reshape(-1, 3)
ref_1_flat = ref_1.reshape(-1, 3)

training_3_flat = training_3.reshape(-1, 3)
ref_3_flat = ref_3.reshape(-1, 3)

training_6_flat = training_6.reshape(-1, 3)
ref_6_flat = ref_6.reshape(-1, 3)

# extract face pixels from training data
face_pixels = extract_faces(training_1_flat, ref_1_flat)

face_pixels_chromatic = make_chromatic_1(face_pixels)

mu, sigma = ml_estimation(face_pixels_chromatic)

# convert testing data to chromatic
training_3_flat_chromatic = make_chromatic_1(training_3_flat)
training_6_flat_chromatic = make_chromatic_1(training_6_flat)

# generate ROC curve for part b (combined images)
combined_chromatic_b = np.vstack([training_3_flat_chromatic, training_6_flat_chromatic])
combined_ref_b = np.vstack([ref_3_flat, ref_6_flat])
t_b, ber_b = gen_roc(mu, sigma, combined_chromatic_b, combined_ref_b)
t_intersect_b = plot_roc(t_b, ber_b, "output/part_b")

# calculate classifications based on ROC results
results_t3 = bayesian_img(mu, sigma, training_3_flat_chromatic, t=t_intersect_b)
results_t6 = bayesian_img(mu, sigma, training_6_flat_chromatic, t=t_intersect_b)

# reconstruct images
rec_img_3 = reconstruct_img(results_t3, training_3_flat, training_3)
rec_img_3 = cv2.cvtColor(rec_img_3, cv2.COLOR_RGB2BGR)
cv2.imwrite("output/training_3_b_result.jpg", rec_img_3)

rec_img_6 = reconstruct_img(results_t6, training_6_flat, training_6)
rec_img_6 = cv2.cvtColor(rec_img_6, cv2.COLOR_RGB2BGR)
cv2.imwrite("output/training_6_b_result.jpg", rec_img_6)

plot_roc_2(ber_a, ber_b, "output/a_b_roc_curves.jpg")

print(f"Elapsed time: {time.time() - _start:.2f}s")