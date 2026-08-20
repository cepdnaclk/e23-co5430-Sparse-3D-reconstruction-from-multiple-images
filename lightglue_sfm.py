"""
Deep-learning Structure-from-Motion pipeline: SuperPoint (detector) +
LightGlue (matcher).

WHY LIGHTGLUE INSTEAD OF SUPERGLUE
-----------------------------------
The project proposal names SuperGlue, but this implementation uses LightGlue,
its direct successor from the same research group (ETH Zurich, Sarlin et al.):
  - 4-10x faster on GPU, comparable-or-better accuracy (ICCV 2023 paper)
  - Apache-2.0 license (SuperGlue's is restrictive/research-only) -- cleaner
    for a repo we're pushing to GitHub
  - Actively maintained, simple pip-installable API
This is a well-justified substitution, not a deviation: LightGlue does the
same job (learned, confidence-filtered feature matching) that SuperGlue was
proposed for. Cite this choice explicitly in the report.

HOW THIS FITS THE REST OF THE PROJECT
--------------------------------------
This file deliberately does NOT reimplement the whole SfM pipeline. Only
feature detection and matching differ from sfm_baseline.py (ORB/SIFT +
brute-force+ratio-test) -- everything downstream is imported and reused
unmodified:
    geometric_verify()      RANSAC + Essential matrix
    initialize_reconstruction()   triangulation-angle-gated initial pair
    run_incremental_sfm()   PnP-based incremental registration
    bundle_adjust()         sparse-Jacobian joint refinement
    export_*()              .ply / cameras.json / COLMAP-format export
    evaluate.py              ground-truth pose scoring
This is intentional: it isolates feature detection/matching as the ONLY
variable, which is exactly what the ORB vs. SIFT vs. SuperPoint+LightGlue
comparison is supposed to measure. sfm_baseline.py itself is untouched.

We also reuse the classical pipeline's candidate-pair selection strategy
(sequential/windowed matching, see config.MATCHING_STRATEGY) for the same
reason -- both pipelines must test the same candidate pairs for the
comparison to be fair.

Usage:
    python lightglue_sfm.py                       # uses config.py defaults
    python lightglue_sfm.py --out outputs/my_run
    python lightglue_sfm.py --max_keypoints 1024   # faster, fewer points
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

import config
import evaluate
from sfm_baseline import (
    Frame,
    RUN_STATS,
    bundle_adjust,
    estimate_intrinsics,
    export_cameras,
    export_colmap_text,
    export_open3d,
    export_ply,
    load_images,
    print_metrics_summary,
    record_match_count,
    run_incremental_sfm,
)


# --------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------


def get_device():
    """Picks CUDA if available, unless overridden by config.LIGHTGLUE_DEVICE
    ("cuda" / "cpu" / "auto"). Runs fine on CPU for a dataset this size --
    just slower than a GPU."""
    preference = getattr(config, "LIGHTGLUE_DEVICE", "auto")
    if preference != "auto":
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Feature detection (SuperPoint)
# --------------------------------------------------------------------------


def load_image_tensor(path):
    """Reads an image with cv2 -- the SAME loader the classical pipeline
    uses -- so keypoint pixel coordinates stay in the exact resolution the
    ground-truth K and evaluate.py expect. (Some LightGlue example scripts
    use their own image loader, which may resize; we deliberately avoid that
    here to guarantee pixel-for-pixel consistency across all three methods.)
    Returns a normalized (3,H,W) float32 tensor in [0,1], plus the raw cv2
    shape (H,W,3) for downstream bookkeeping.
    """
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1)  # (H,W,3) -> (3,H,W), what SuperPoint expects
    return tensor, img_bgr.shape


def detect_features_lightglue(paths, max_keypoints, device):
    """Runs SuperPoint on every image. Mirrors sfm_baseline.detect_features()'s
    role (including RUN_STATS bookkeeping, for the shared metrics printout),
    but builds Frame objects a little differently:

      frame.kp   -- list[cv2.KeyPoint], built from SuperPoint's keypoint
                    coordinates. This is the SAME structure the classical
                    pipeline produces, which is what lets every downstream
                    function (geometric_verify, register_next_image,
                    triangulate_new_points, export_colmap_text, ...) work
                    completely unmodified -- they only ever read kp[i].pt.

      frame.desc -- the RAW SuperPoint feature dict (torch tensors), NOT a
                    numpy descriptor array like the classical pipeline
                    stores. This is safe: match_pair()/make_detector() (the
                    only functions in sfm_baseline.py that touch .desc) are
                    never called on these frames -- LightGlue's matcher needs
                    the full extractor output, not just raw descriptors, so
                    we keep it around here instead.
    """
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)

    frames = []
    for i, path in enumerate(paths):
        tensor, shape = load_image_tensor(path)
        tensor = tensor.to(device)

        with torch.no_grad():
            feats = extractor.extract(tensor)  # batched dict (batch size 1)

        kp_xy = feats["keypoints"][0].cpu().numpy()  # (N, 2) pixel coordinates
        kp = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp_xy]

        frames.append(Frame(i, path, kp, feats, shape))
        RUN_STATS["n_keypoints"][i] = len(kp)
        print(f"[features:lightglue] {os.path.basename(path)}: {len(kp)} keypoints")

    return frames


# --------------------------------------------------------------------------
# Matching (LightGlue)
# --------------------------------------------------------------------------


def match_pair_lightglue(f1, f2, matcher, min_confidence=0.0):
    """Runs LightGlue on one already-extracted SuperPoint feature pair and
    returns a list of cv2.DMatch -- the SAME object type
    sfm_baseline.match_pair() returns, so geometric_verify() and everything
    downstream of it needs zero changes to accept these matches.

    Unlike the classical match_pair(), there is no separate Lowe's ratio
    test here: LightGlue does its own confidence-based filtering internally
    before returning "matches" at all. This matches the project proposal's
    plan ("the learned matcher's own confidence-based filtering for the deep
    learning pipeline"). min_confidence is an OPTIONAL extra bar on top of
    that internal filtering; 0.0 (off) is the honest default -- raise it only
    if you deliberately want to test a stricter deep-pipeline threshold.
    """
    with torch.no_grad():
        raw = matcher({"image0": f1.desc, "image1": f2.desc})
    out = rbd(raw)  # drop the batch dimension

    idx_pairs = out["matches"].cpu().numpy()  # (M, 2) int64: index into kp1, kp2
    scores = out["scores"].cpu().numpy()      # (M,) match confidence, higher = better

    dmatches = []
    for (i1, i2), score in zip(idx_pairs, scores):
        if score < min_confidence:
            continue
        # cv2.DMatch.distance is conventionally "lower = better"; LightGlue's
        # score is "higher = better", so store (1 - score) to keep that
        # convention for anything that later inspects .distance.
        dmatches.append(cv2.DMatch(int(i1), int(i2), float(1.0 - score)))
    return dmatches


def match_all_lightglue(frames, matcher, matching_strategy, match_window, min_confidence):
    """Candidate-pair selection is IDENTICAL to sfm_baseline.match_all()'s
    logic (sequential/windowed vs. exhaustive) -- kept that way on purpose,
    see module docstring. Only the actual pairwise matching call differs."""
    matches_table = {}
    n = len(frames)

    if matching_strategy == "sequential":
        pairs = set()
        for i in range(n):
            for d in range(1, match_window + 1):
                j = (i + d) % n
                pairs.add((min(i, j), max(i, j)))
        print(
            f"[match:lightglue] sequential strategy: {len(pairs)} pairs "
            f"(vs {n * (n - 1) // 2} for exhaustive)"
        )
        pair_iter = sorted(pairs)
    else:
        pair_iter = [(i, j) for i in range(n) for j in range(i + 1, n)]
        print(f"[match:lightglue] exhaustive strategy: {len(pair_iter)} pairs")

    for i, j in pair_iter:
        m = match_pair_lightglue(frames[i], frames[j], matcher, min_confidence)
        record_match_count(i, j, len(m))
        if len(m) >= 20:  # same acceptance bar as the classical pipeline
            matches_table[(i, j)] = m

    return matches_table


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------


def run_pipeline_lightglue(
    images,
    out,
    max_keypoints=2048,
    min_confidence=0.0,
    min_pnp_inliers=30,
    export_colmap=False,
    K_override=None,
    matching_strategy="sequential",
    match_window=6,
    device=None,
):
    """Deep-learning counterpart to sfm_baseline.run_pipeline(). Structured
    identically on purpose, so the two are easy to compare side by side:
    detect features -> get intrinsics -> match -> incremental SfM -> bundle
    adjust -> export -> print metrics. Only the first two stages differ."""
    if device is None:
        device = get_device()
    print(f"[lightglue] using device: {device}")

    timings = {}
    os.makedirs(out, exist_ok=True)
    paths = load_images(images)
    print(f"[load] {len(paths)} images found")

    t0 = time.time()
    frames = detect_features_lightglue(paths, max_keypoints, device)
    timings["feature_detection"] = time.time() - t0

    if K_override is not None:
        K = np.array(K_override, dtype=np.float64)
        print(f"[intrinsics] using supplied ground-truth K\n{K}")
    else:
        K = estimate_intrinsics(frames[0].shape)
        print(f"[intrinsics] estimated (no calibration supplied)\n{K}")

    matcher = LightGlue(features="superpoint").eval().to(device)

    t0 = time.time()
    matches_table = match_all_lightglue(
        frames, matcher, matching_strategy, match_window, min_confidence
    )
    timings["matching"] = time.time() - t0

    n = len(frames)
    t0 = time.time()
    registered, points3D = run_incremental_sfm(
        frames, matches_table, K, min_pnp_inliers=min_pnp_inliers
    )
    timings["incremental_sfm"] = time.time() - t0
    print(
        f"[sfm] registered {len(registered)}/{n} images, "
        f"{len(points3D)} 3D points before BA"
    )

    t0 = time.time()
    registered, points3D = bundle_adjust(registered, points3D, K)
    timings["bundle_adjustment"] = time.time() - t0

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
    print_metrics_summary(frames, matches_table, points3D, registered, timings)
    export_open3d(os.path.join(out, "points3D_open3d.ply"), points3D, registered, K)
    print(f"[done] wrote outputs to {out}/")

    return {
        "frames": frames,
        "registered": registered,
        "points3D": points3D,
        "K": K,
        "timings": timings,
        "matches_table": matches_table,
    }


# --------------------------------------------------------------------------
# CLI entry point -- mirrors run.py's structure (config-driven defaults +
# ground-truth evaluation wiring), so this file is runnable standalone.
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default=config.IMAGES_DIR)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--max_keypoints",
        type=int,
        default=getattr(config, "LIGHTGLUE_MAX_KEYPOINTS", 2048),
    )
    ap.add_argument(
        "--min_confidence",
        type=float,
        default=getattr(config, "LIGHTGLUE_MIN_CONFIDENCE", 0.0),
        help="Extra match-confidence bar on top of LightGlue's own internal "
        "filtering. 0.0 = off (recommended default).",
    )
    ap.add_argument(
        "--min_pnp_inliers",
        type=int,
        default=getattr(config, "LIGHTGLUE_MIN_PNP_INLIERS", 30),
    )
    ap.add_argument("--gt_calibration", default=config.GT_CALIBRATION_PATH)
    ap.add_argument(
        "--no_gt_intrinsics",
        action="store_true",
        help="Ignore ground-truth calibration, use the width/height "
        "intrinsics approximation instead",
    )
    ap.add_argument(
        "--export_colmap",
        action="store_true",
        default=config.EXPORT_COLMAP,
        help="Also export COLMAP-format sparse/ + images/",
    )
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(config.OUTPUT_DIR, "lightglue")

    K_override = None
    gt_poses = None
    if os.path.exists(args.gt_calibration):
        gt_poses = evaluate.load_gt_poses(args.gt_calibration)
        if config.USE_GT_INTRINSICS and not args.no_gt_intrinsics:
            # Same camera for every TempleRing shot -- grab K from any
            # entry. Do NOT use per-image ground-truth R,t: pose is exactly
            # what we're testing.
            any_K, _, _ = next(iter(gt_poses.values()))
            K_override = any_K
    else:
        print(
            f"[lightglue] no ground-truth calibration found at "
            f"{args.gt_calibration} -- falling back to intrinsics approximation"
        )

    result = run_pipeline_lightglue(
        images=args.images,
        out=args.out,
        max_keypoints=args.max_keypoints,
        min_confidence=args.min_confidence,
        min_pnp_inliers=args.min_pnp_inliers,
        export_colmap=args.export_colmap,
        K_override=K_override,
        matching_strategy=config.MATCHING_STRATEGY,
        match_window=config.MATCH_WINDOW,
    )

    if gt_poses is not None:
        evaluate.evaluate_poses(result["registered"], gt_poses)


if __name__ == "__main__":
    main()
