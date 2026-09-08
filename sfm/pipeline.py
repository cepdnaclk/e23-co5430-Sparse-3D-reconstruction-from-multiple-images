"""Top-level orchestration for the classical incremental SfM pipeline."""

import os
import time

import numpy as np

from .bundle_adjustment import bundle_adjust
from .exporters import export_cameras, export_colmap_text, export_ply
from .features import detect_features, make_detector, match_all
from .image_io import estimate_intrinsics, load_images
from .reconstruction import run_incremental_sfm
from .reporting import print_metrics_summary
from .visualization import export_open3d


def run_pipeline(
    images,
    out,
    feature_type="orb",
    focal_mm=None,
    sensor_width_mm=None,
    min_inliers_init=60,
    export_colmap=False,
    K_override=None,
    matching_strategy="exhaustive",
    match_window=6,
):
    """Run detection, matching, reconstruction, BA, reporting, and export."""
    timings = {}
    os.makedirs(out, exist_ok=True)
    paths = load_images(images)
    print(f"[load] {len(paths)} images found")

    start = time.time()
    _, norm_type = make_detector(feature_type)
    frames = detect_features(paths, feature_type=feature_type)
    timings["feature_detection"] = time.time() - start

    if K_override is not None:
        K = np.array(K_override, dtype=np.float64)
        print(f"[intrinsics] using supplied ground-truth K\n{K}")
    else:
        K = estimate_intrinsics(frames[0].shape, focal_mm, sensor_width_mm)
        print(f"[intrinsics] estimated (no calibration supplied)\n{K}")

    start = time.time()
    matches_table = match_all(frames, norm_type, matching_strategy, match_window)
    timings["matching"] = time.time() - start

    start = time.time()
    registered, points3D = run_incremental_sfm(frames, matches_table, K)
    timings["incremental_sfm"] = time.time() - start
    print(
        f"[sfm] registered {len(registered)}/{len(frames)} images, "
        f"{len(points3D)} 3D points before BA"
    )

    start = time.time()
    registered, points3D = bundle_adjust(registered, points3D, K)
    timings["bundle_adjustment"] = time.time() - start

    export_ply(os.path.join(out, "points3D.ply"), points3D)
    export_cameras(os.path.join(out, "cameras.json"), registered, K)
    if export_colmap:
        export_colmap_text(
            os.path.join(out, "sparse", "0"),
            registered,
            points3D,
            K,
            copy_images_to=os.path.join(out, "images"),
        )
    print_metrics_summary(frames, matches_table, points3D, registered, timings, K)
    export_open3d(
        os.path.join(out, "points3D_open3d.ply"), points3D, registered, K
    )
    print(f"[done] wrote outputs to {out}/")
    return {
        "frames": frames,
        "registered": registered,
        "points3D": points3D,
        "K": K,
        "timings": timings,
        "matches_table": matches_table,
    }
