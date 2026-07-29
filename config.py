"""
Central place for parameters, so you change one file instead of hunting
through scripts. Matches your proposal's baseline (ORB) / improved (SIFT)
naming.
"""
import os

# --- paths -----------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "templeRing")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
GT_CALIBRATION_PATH = os.path.join(DATASET_DIR, "templeR_par.txt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# --- pipeline parameters -----------------------------------------------------
FEATURE_TYPE = "orb"          # "orb" (baseline) or "sift" (improved classical)
MIN_INLIERS_INIT = 60         # min RANSAC inliers to accept an initial pair
USE_GT_INTRINSICS = True      # TempleRing ships real calibration -- use it
EXPORT_COLMAP = True          # write sparse/ + images/ for Gaussian Splatting later

# fallback intrinsics guess, only used if USE_GT_INTRINSICS is False and no
# calibration is available for a dataset
FOCAL_MM = None
SENSOR_WIDTH_MM = None
