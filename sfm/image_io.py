"""Image discovery and camera-intrinsics helpers."""

import glob
import os

import numpy as np


def load_images(folder):
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    paths = []
    for extension in extensions:
        paths.extend(glob.glob(os.path.join(folder, extension)))
    paths = sorted(
        {os.path.normcase(os.path.abspath(path)): path for path in paths}.values()
    )
    if len(paths) < 2:
        raise RuntimeError(f"Need >=2 images in {folder}, found {len(paths)}")
    return paths


def estimate_intrinsics(shape, focal_mm=None, sensor_width_mm=None):
    h, w = shape[:2]
    if focal_mm is not None and sensor_width_mm is not None:
        fx = fy = (focal_mm / sensor_width_mm) * w
    else:
        fx = fy = max(w, h)
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
