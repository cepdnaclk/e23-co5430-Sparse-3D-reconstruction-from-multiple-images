"""
Main entry point. Run this, not sfm_baseline.py directly, when working with
a dataset that has ground-truth calibration (like TempleRing) -- it wires the
pipeline and the evaluation together and uses config.py for defaults.

Usage:
    python run.py                          # uses config.py defaults (ORB)
    python run.py --feature_type sift      # improved classical variant
    python run.py --out outputs/orb_run1   # custom output folder per run
"""
import argparse
import os

import config
import sfm_baseline as sfm
import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=config.IMAGES_DIR)
    ap.add_argument("--out", default=os.path.join(config.OUTPUT_DIR, config.FEATURE_TYPE))
    ap.add_argument("--feature_type", choices=["sift", "orb"], default=config.FEATURE_TYPE)
    ap.add_argument("--gt_calibration", default=config.GT_CALIBRATION_PATH)
    ap.add_argument("--no_gt_intrinsics", action="store_true",
                     help="Ignore ground-truth calibration, use the width/height "
                          "intrinsics approximation instead")
    args = ap.parse_args()

    K_override = None
    gt_poses = None
    if os.path.exists(args.gt_calibration):
        gt_poses = evaluate.load_gt_poses(args.gt_calibration)
        if config.USE_GT_INTRINSICS and not args.no_gt_intrinsics:
            # TempleRing uses the same camera for every shot -- grab K from
            # whichever entry we have; do NOT use per-image ground-truth R,t,
            # only K, since the pose itself is exactly what we're testing.
            any_K, _, _ = next(iter(gt_poses.values()))
            K_override = any_K
    else:
        print(f"[run] no ground-truth calibration found at {args.gt_calibration} "
              f"-- falling back to intrinsics approximation")

    result = sfm.run_pipeline(
        images=args.images,
        out=args.out,
        feature_type=args.feature_type,
        min_inliers_init=config.MIN_INLIERS_INIT,
        export_colmap=config.EXPORT_COLMAP,
        K_override=K_override,
    )

    if gt_poses is not None:
        evaluate.evaluate_poses(result["registered"], gt_poses)


if __name__ == "__main__":
    main()
