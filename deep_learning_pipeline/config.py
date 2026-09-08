"""
Central configuration for the Temple Ring SfM pipeline.

Edit DATASET_DIR to point at your local copy of the Temple Ring dataset
(the folder containing templeR0001.png ... templeR0047.png and, optionally,
temple_par.txt / temple_ang.txt with the Middlebury ground-truth calibration).
"""

from pathlib import Path

# ---- Paths ----
DATASET_DIR = Path("./datasets/templeRing/images/")

# Where all pipeline outputs (features, matches, database, sparse model) go.
OUTPUT_DIR = Path("outputs/lightglue")

IMAGE_GLOB = "*.png"  # Temple Ring ships as .png; change to *.jpg if needed

# ---- Feature extraction / matching (hloc configs) ----
# "superpoint_max" runs SuperPoint with no resize cap and a high keypoint
# budget, which suits a small, high-detail dataset like Temple Ring
# (47 images) far better than the resize-limited "superpoint_aachen" config
# that's tuned for large-scale outdoor localization.
FEATURE_CONF = "superpoint_max"

# hloc's built-in LightGlue + SuperPoint matcher config.
MATCHER_CONF = "superpoint+lightglue"

# Temple Ring is small enough (47 images) that exhaustive pairwise matching
# is cheap and gives the mapper the most connectivity to work with.
USE_EXHAUSTIVE_PAIRS = True

# ---- Camera / reconstruction options ---
# All Temple Ring images were captured with the same physical camera, so we
# tell COLMAP to share one set of intrinsics across every image rather than
# estimating a separate camera per image.
CAMERA_MODEL = "SIMPLE_RADIAL"  # good default when intrinsics are unknown

# The pipeline lets COLMAP estimate intrinsics from scratch (robust default,
# no dataset-specific assumptions). If you also have the Middlebury
# temple_par.txt ground truth and want to check accuracy afterwards, pass
# it to `python -m pipeline.evaluate --gt <path> --poses <poses.txt>` —
# see pipeline/evaluate.py.

# ---- Derived paths (do not edit) ----
FEATURES_PATH = OUTPUT_DIR / "features.h5"
MATCHES_PATH = OUTPUT_DIR / "matches.h5"
PAIRS_PATH = OUTPUT_DIR / "pairs-exhaustive.txt"
DATABASE_PATH = OUTPUT_DIR / "database.db"
SFM_DIR = OUTPUT_DIR / "sparse"
