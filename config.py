import os

# paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "templeRing")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
GT_CALIBRATION_PATH = os.path.join(DATASET_DIR, "templeR_par.txt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# pipeline parameters
FEATURE_TYPE = "orb"  # "orb" (baseline) or "sift" (improved classical)
MIN_INLIERS_INIT = 60  # min RANSAC inliers to accept an initial pair
USE_GT_INTRINSICS = True  # TempleRing ships real calibration -- use it
EXPORT_COLMAP = True  # write sparse/ + images/ for Gaussian Splatting later

# TempleRing images are captured in ring order (see templeR_ang.txt) and the
# object has repetitive architecture (columns) -- exhaustive all-pairs
# matching lets non-overlapping, opposite-side views produce false-positive
# matches that corrupt the initial pair and the whole reconstruction.
# "sequential" only matches nearby-in-capture-order images (+ wraparound),
# which matches how the data was actually captured. Use "exhaustive" only for
# genuinely unordered photo sets (e.g. your own casually-captured photos).
MATCHING_STRATEGY = "sequential"
MATCH_WINDOW = 6

# fallback intrinsics guess, only used if USE_GT_INTRINSICS is False and no
# calibration is available for a dataset
FOCAL_MM = None
SENSOR_WIDTH_MM = None

# --- SuperPoint + LightGlue deep pipeline (lightglue_sfm.py) ---------------
# These only affect lightglue_sfm.py -- sfm_baseline.py never reads this
# file, so nothing here touches the classical ORB/SIFT pipeline.
LIGHTGLUE_MAX_KEYPOINTS = 2048  # SuperPoint keypoints per image
LIGHTGLUE_MIN_CONFIDENCE = 0.0  # extra match-confidence bar; 0.0 = off
# (LightGlue already filters internally)
LIGHTGLUE_MIN_PNP_INLIERS = 30  # mirrors the classical pipeline's validated
# incremental-registration reliability bar
LIGHTGLUE_DEVICE = "auto"  # "auto" picks CUDA if available, else CPU
