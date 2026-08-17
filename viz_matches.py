"""
Visualize detected keypoints and pairwise feature matches without running the
full SFM pipeline (reuses the exact detection + matching logic from
sfm_baseline.py). Writes PNGs to <out>/keypoints/ and <out>/matches/:

  <out>/keypoints/keypoints_<i>.png    keypoints drawn on each image
  <out>/matches/<i>_<j>.png            ALL raw matches (red) with the RANSAC
                                       inliers overlaid in green
  <out>/matches/inliers_<i>_<j>.png    only the RANSAC-verified inliers

Usage (only --feature_type is required; everything else uses config.py
defaults that mirror run.py):
    python viz_matches.py --feature_type sift
    python viz_matches.py --feature_type orb --pairs "0,1 1,2 2,3"
    python viz_matches.py --feature_type sift --interactive
"""

import argparse
import os

import cv2
import numpy as np

import config
import evaluate
import sfm_baseline as sfm


def load_K(gt_calibration, shape):
    """Ground-truth intrinsics when available (same logic as run.py), else the
    width/height approximation."""
    if os.path.exists(gt_calibration):
        gt_poses = evaluate.load_gt_poses(gt_calibration)
        any_K, _, _ = next(iter(gt_poses.values()))
        return np.array(any_K, dtype=np.float64)
    return sfm.estimate_intrinsics(shape)


def draw_keypoints(frames, out_dir, radius=3):
    """Save one overlay of the detected keypoints per image. Drawn manually
    with a fixed small radius so SIFT's large scales don't produce huge
    overlapping circles."""
    os.makedirs(out_dir, exist_ok=True)
    for f in frames:
        img = cv2.imread(f.path)
        if img is None:
            print(f"[viz] warning: could not read {f.path}, skipping keypoints")
            continue
        out = img.copy()
        for kp in f.kp:
            cv2.circle(
                out,
                (int(round(kp.pt[0])), int(round(kp.pt[1]))),
                radius,
                (0, 0, 255),
                1,
            )
        path = os.path.join(out_dir, f"keypoints_{f.idx}.png")
        cv2.imwrite(path, out)
        print(
            f"[viz] {f.idx} ({os.path.basename(f.path)}): "
            f"{len(f.kp)} keypoints -> {path}"
        )


def small_keypoints(kps, size=5):
    """Return keypoints with a capped size so drawing code draws small circles
    (SIFT keypoint sizes can exceed 100px, which makes overlays a mess)."""
    return [
        cv2.KeyPoint(kp.pt[0], kp.pt[1], size, kp.angle, kp.response, kp.octave)
        for kp in kps
    ]


def draw_matches(frames, matches_table, K, out_dir, pairs=None, interactive=False):
    """Save combined (raw red + inliers green) and inlier-only match images for
    each selected pair. Defaults to every pair in matches_table."""
    os.makedirs(out_dir, exist_ok=True)
    if pairs is None:
        pairs = sorted(matches_table.keys())

    for i, j in pairs:
        key = (i, j) if (i, j) in matches_table else (j, i)
        if key not in matches_table:
            print(f"[viz] warning: pair ({i},{j}) has no matches, skipping")
            continue

        f1, f2 = frames[key[0]], frames[key[1]]
        img1 = cv2.imread(f1.path)
        img2 = cv2.imread(f2.path)
        if img1 is None or img2 is None:
            print(f"[viz] warning: could not read pair ({key[0]},{key[1]}), skipping")
            continue

        raw = matches_table[key]
        inliers = raw
        v = sfm.geometric_verify(f1, f2, raw, K, min_inliers=0)
        if v is not None:
            inliers, E, mask = v

        kp1, kp2 = small_keypoints(f1.kp), small_keypoints(f2.kp)
        combined = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            raw,
            None,
            matchColor=(0, 0, 255),
            singlePointColor=(0, 0, 255),
        )
        if len(inliers):
            # cv2.drawMatches(outImg=...) clears the canvas instead of
            # compositing, so overlay the inlier lines manually.
            w1 = img1.shape[1]
            for m in inliers:
                p1 = f1.kp[m.queryIdx].pt
                p2 = f2.kp[m.trainIdx].pt
                cv2.line(
                    combined,
                    (int(round(p1[0])), int(round(p1[1]))),
                    (int(round(p2[0])) + w1, int(round(p2[1]))),
                    (0, 255, 0),
                    1,
                )
        combined_path = os.path.join(out_dir, f"matches_{key[0]}_{key[1]}.png")
        cv2.imwrite(combined_path, combined)

        inlier_img = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            inliers,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(0, 255, 0),
        )
        inlier_path = os.path.join(out_dir, f"inliers_{key[0]}_{key[1]}.png")
        cv2.imwrite(inlier_path, inlier_img)

        print(
            f"[viz] pair ({key[0]},{key[1]}): "
            f"{len(raw)} raw / {len(inliers)} inliers -> "
            f"{os.path.basename(combined_path)}"
        )

        if interactive:
            cv2.imshow(f"matches {key[0]}-{key[1]}", combined)
            cv2.waitKey(0)

    if interactive:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_type", required=True, choices=["sift", "orb"])
    ap.add_argument("--images", default=config.IMAGES_DIR)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gt_calibration", default=config.GT_CALIBRATION_PATH)
    ap.add_argument(
        "--matching_strategy",
        choices=["sequential", "exhaustive"],
        default=config.MATCHING_STRATEGY,
    )
    ap.add_argument("--match_window", type=int, default=config.MATCH_WINDOW)
    ap.add_argument("--max_features", type=int, default=4000)
    ap.add_argument("--pairs", default=None, help='e.g. "0,1 1,2"')
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="Pop up each match image (cv2.imshow) instead of only saving",
    )
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(config.OUTPUT_DIR, f"{args.feature_type}_viz")

    paths = sfm.load_images(args.images)
    print(f"[viz] {len(paths)} images found")

    _, norm_type = sfm.make_detector(args.feature_type, args.max_features)
    frames = sfm.detect_features(
        paths, feature_type=args.feature_type, max_features=args.max_features
    )

    K = load_K(args.gt_calibration, frames[0].shape)
    print(f"[viz] intrinsics K\n{K}")

    matches_table = sfm.match_all(
        frames, norm_type, args.matching_strategy, args.match_window
    )

    draw_keypoints(frames, os.path.join(args.out, "keypoints"))

    pairs = None
    if args.pairs:
        pairs = [tuple(int(x) for x in p.split(",")) for p in args.pairs.split()]
    draw_matches(
        frames,
        matches_table,
        K,
        os.path.join(args.out, "matches"),
        pairs=pairs,
        interactive=args.interactive,
    )

    print(f"[viz] wrote visualizations to {args.out}/")


if __name__ == "__main__":
    main()
