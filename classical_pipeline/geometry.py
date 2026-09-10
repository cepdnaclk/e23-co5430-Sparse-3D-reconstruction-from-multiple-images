"""Epipolar geometry and triangulation helpers."""

import cv2
import numpy as np

from .reporting import RUN_STATS


def geometric_verify(f1, f2, matches, K, min_inliers=30):
    if len(matches) < 8:
        return None
    points1 = np.float32([f1.kp[match.queryIdx].pt for match in matches])
    points2 = np.float32([f2.kp[match.trainIdx].pt for match in matches])
    essential, mask = cv2.findEssentialMat(
        points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if essential is None or mask is None:
        return None
    mask = mask.ravel().astype(bool)
    inliers = [match for match, keep in zip(matches, mask) if keep]
    RUN_STATS["n_inliers"][(f1.idx, f2.idx)] = len(inliers)
    if len(inliers) < min_inliers:
        return None
    return inliers, essential, mask


def pair_parallax_score(f1, f2, inliers):
    distances = []
    for match in inliers:
        point1 = np.array(f1.kp[match.queryIdx].pt)
        point2 = np.array(f2.kp[match.trainIdx].pt)
        distances.append(np.linalg.norm(point1 - point2))
    return np.median(distances) if distances else 0.0


def triangulation_angle_deg(pts3d, C1, C2):
    vector1 = pts3d - C1
    vector2 = pts3d - C2
    vector1 /= np.linalg.norm(vector1, axis=1, keepdims=True) + 1e-9
    vector2 /= np.linalg.norm(vector2, axis=1, keepdims=True) + 1e-9
    cosine = np.clip(np.sum(vector1 * vector2, axis=1), -1, 1)
    return np.degrees(np.median(np.arccos(cosine)))


def triangulate_points(K, R1, t1, R2, t2, pts1, pts2):
    projection1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    projection2 = K @ np.hstack([R2, t2.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(
        projection1, projection2, pts1.T, pts2.T
    )
    return (homogeneous[:3] / homogeneous[3]).T


def cheirality_mask(pts3d, R, t):
    points_camera = (R @ pts3d.T + t.reshape(3, 1)).T
    return points_camera[:, 2] > 0
