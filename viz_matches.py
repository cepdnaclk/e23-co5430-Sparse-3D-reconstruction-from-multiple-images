"""
Visualizes keypoints and feature matches from a completed SfM run, using the
exact objects the pipeline produced (frames, matches_table, K).

Usage (called automatically by run.py with --save_matches):
    python run.py --feature_type sift --save_matches
    python run.py --save_matches --pairs "0,1 3,4"

Outputs land in <out>/keypoints/*.png and <out>/matches/*.png:
    keypoints_<i>.png          detected keypoints on image i
    <i>_<j>.png                raw ratio-test matches, color-coded
                               (green = RANSAC inliers, red = outliers)
    inliers_<i>_<j>.png        only the RANSAC-verified inliers
"""

import os

import cv2
import numpy as np

import sfm_baseline as sfm


def draw_keypoints(frames, out_dir):
    kp_dir = os.path.join(out_dir, "keypoints")
    os.makedirs(kp_dir, exist_ok=True)
    for f in frames:
        img = cv2.imread(f.path)
        if img is None:
            img = cv2.imread(f.path, cv2.IMREAD_GRAYSCALE)
        overlay = cv2.drawKeypoints(img, f.kp, None, color=(0, 255, 0))
        dest = os.path.join(kp_dir, f"keypoints_{f.idx}.png")
        cv2.imwrite(dest, overlay)
        # print(f"[viz] {len(f.kp)} keypoints -> {dest}")


def _draw_matches_pair(frames, i, j, raw, K, out_dir):
    img1 = cv2.imread(frames[i].path)
    img2 = cv2.imread(frames[j].path)
    if img1 is None or img2 is None:
        img1 = cv2.imread(frames[i].path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(frames[j].path, cv2.IMREAD_GRAYSCALE)
    w1 = img1.shape[1]

    # RANSAC inliers vs outliers against the essential matrix.
    v = sfm.geometric_verify(frames[i], frames[j], raw, K, min_inliers=0)
    if v is None:
        inliers, outliers = [], list(raw)
    else:
        inlier_matches, _, _ = v
        inlier_set = set(id(m) for m in inlier_matches)
        inliers = [m for m in raw if id(m) in inlier_set]
        outliers = [m for m in raw if id(m) not in inlier_set]

    def line_pairs(matches):
        return [
            (
                (
                    int(frames[i].kp[m.queryIdx].pt[0]),
                    int(frames[i].kp[m.queryIdx].pt[1]),
                ),
                (
                    w1 + int(frames[j].kp[m.trainIdx].pt[0]),
                    int(frames[j].kp[m.trainIdx].pt[1]),
                ),
            )
            for m in matches
        ]

    # combined: outliers red, inliers green overlaid on top. cv2.drawMatches
    # wipes an outImg passed to it, so the colored links are drawn manually.
    combined = cv2.drawMatches(
        img1,
        frames[i].kp,
        img2,
        frames[j].kp,
        [],
        None,
        singlePointColor=(255, 255, 255),
    )
    for p1, p2 in line_pairs(outliers):
        cv2.line(combined, p1, p2, (0, 0, 255), 2)
    for p1, p2 in line_pairs(inliers):
        cv2.line(combined, p1, p2, (0, 255, 0), 2)

    inlier_img = cv2.drawMatches(
        img1,
        frames[i].kp,
        img2,
        frames[j].kp,
        inliers,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 255, 0),
    )

    base = f"{i}_{j}"
    for name, img in (("", combined), ("inliers_", inlier_img)):
        dest = os.path.join(out_dir, "matches", f"{name}{base}.png")
        cv2.imwrite(dest, img)

    # n_in = len(inliers)
    # print(
    #     f"[viz] pair ({i},{j}): {len(raw)} raw matches / {n_in} RANSAC "
    #     f"inliers -> {out_dir}/matches/"
    # )


def draw_matches(frames, matches_table, K, out_dir, pairs=None):
    m_dir = os.path.join(out_dir, "matches")
    os.makedirs(m_dir, exist_ok=True)

    table = sorted(matches_table.items())
    if pairs:
        wanted = [tuple(map(int, p.split(","))) for p in pairs.split()]
        wanted = {(min(a, b), max(a, b)) for a, b in wanted}
        table = [kv for kv in table if kv[0] in wanted]

    if not table:
        print("[viz] no matched pairs to draw")
        return

    for (i, j), matches in table:
        _draw_matches_pair(frames, i, j, matches, K, out_dir)
