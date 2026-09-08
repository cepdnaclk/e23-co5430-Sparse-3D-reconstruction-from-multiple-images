"""Two-view initialization and incremental camera registration."""

import os

import cv2
import numpy as np

from .geometry import (
    cheirality_mask,
    geometric_verify,
    triangulate_points,
    triangulation_angle_deg,
)


def initialize_reconstruction(
    frames, K, matches_table, min_inliers=60, min_triangulation_angle_deg=2.0
):
    candidates = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            key = (i, j)
            if key not in matches_table:
                continue
            verified = geometric_verify(
                frames[i], frames[j], matches_table[key], K, min_inliers
            )
            if verified is None:
                continue
            inliers, essential, _ = verified
            points1 = np.float32(
                [frames[i].kp[match.queryIdx].pt for match in inliers]
            )
            points2 = np.float32(
                [frames[j].kp[match.trainIdx].pt for match in inliers]
            )
            _, rotation, translation, _ = cv2.recoverPose(
                essential, points1, points2, K
            )
            translation = translation.ravel()
            points3d = triangulate_points(
                K,
                np.eye(3),
                np.zeros(3),
                rotation,
                translation,
                points1,
                points2,
            )
            good = cheirality_mask(points3d, np.eye(3), np.zeros(3)) & cheirality_mask(
                points3d, rotation, translation
            )
            if good.sum() < min_inliers:
                continue
            center1 = np.zeros(3)
            center2 = -rotation.T @ translation
            angle = triangulation_angle_deg(points3d[good], center1, center2)
            candidates.append(
                (i, j, inliers, essential, rotation, translation, angle)
            )

    if not candidates:
        raise RuntimeError(
            "Could not find a good initial pair. Try more overlap "
            "between images or lower min_inliers."
        )

    qualifying = [
        candidate
        for candidate in candidates
        if candidate[6] >= min_triangulation_angle_deg
    ]
    pool = qualifying if qualifying else candidates
    if not qualifying:
        print(
            f"[init] WARNING: no candidate pair reached "
            f"{min_triangulation_angle_deg}° triangulation angle -- falling back "
            "to the best available. Reconstruction may still be poorly conditioned."
        )
    i, j, inliers, _, rotation, translation, angle = max(
        pool, key=lambda candidate: len(candidate[2])
    )
    print(
        f"[init] chosen initial pair: {os.path.basename(frames[i].path)} / "
        f"{os.path.basename(frames[j].path)}  ({len(inliers)} inliers, "
        f"triangulation angle {angle:.2f}°)"
    )

    rotation1 = np.eye(3)
    translation1 = np.zeros(3)
    frames[i].R, frames[i].t, frames[i].registered = rotation1, translation1, True
    frames[j].R, frames[j].t, frames[j].registered = (
        rotation,
        translation,
        True,
    )
    points1 = np.float32([frames[i].kp[match.queryIdx].pt for match in inliers])
    points2 = np.float32([frames[j].kp[match.trainIdx].pt for match in inliers])
    points3d = triangulate_points(
        K, rotation1, translation1, rotation, translation, points1, points2
    )
    good = cheirality_mask(points3d, rotation1, translation1) & cheirality_mask(
        points3d, rotation, translation
    )

    result = []
    for index, match in enumerate(inliers):
        if not good[index]:
            continue
        point_id = len(result)
        result.append(points3d[index])
        frames[i].point3d_idx[match.queryIdx] = point_id
        frames[j].point3d_idx[match.trainIdx] = point_id
    print(f"[init] triangulated {len(result)} initial points")
    return i, j, np.array(result, dtype=np.float64)


def register_next_image(
    frame,
    registered_frames,
    matches_table,
    K,
    points3D,
    reproj_thresh=8.0,
    min_pnp_inliers=30,
):
    object_points, image_points, keypoint_indices = [], [], []
    seen_point_ids = set()
    for registered in registered_frames:
        i, j = (
            (registered.idx, frame.idx)
            if registered.idx < frame.idx
            else (frame.idx, registered.idx)
        )
        if (i, j) not in matches_table:
            continue
        for match in matches_table[(i, j)]:
            if registered.idx < frame.idx:
                registered_kp, frame_kp = match.queryIdx, match.trainIdx
            else:
                registered_kp, frame_kp = match.trainIdx, match.queryIdx
            point_id = registered.point3d_idx[registered_kp]
            if point_id == -1 or point_id in seen_point_ids:
                continue
            seen_point_ids.add(point_id)
            object_points.append(points3D[point_id])
            image_points.append(frame.kp[frame_kp].pt)
            keypoint_indices.append((frame_kp, point_id))

    if len(object_points) < 6:
        return None
    object_points = np.array(object_points, dtype=np.float64)
    image_points = np.array(image_points, dtype=np.float64)
    ok, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        K,
        None,
        reprojectionError=reproj_thresh,
        confidence=0.999,
        iterationsCount=2000,
    )
    if not ok or inliers is None or len(inliers) < min_pnp_inliers:
        return None
    rotation, _ = cv2.Rodrigues(rotation_vector)
    inlier_set = set(inliers.ravel().tolist())
    for local_index, (keypoint_index, point_id) in enumerate(keypoint_indices):
        if local_index in inlier_set:
            frame.point3d_idx[keypoint_index] = point_id
    return rotation, translation.ravel(), len(inliers)


def triangulate_new_points(
    frame, registered_frames, matches_table, K, points3D_list, min_point_angle_deg=3.0
):
    new_count = 0
    rejected_angle = 0
    for registered in registered_frames:
        if registered.idx == frame.idx:
            continue
        i, j = (
            (registered.idx, frame.idx)
            if registered.idx < frame.idx
            else (frame.idx, registered.idx)
        )
        if (i, j) not in matches_table:
            continue
        points_a, points_b, indices_a, indices_b = [], [], [], []
        for match in matches_table[(i, j)]:
            if registered.idx < frame.idx:
                index_a, index_b = match.queryIdx, match.trainIdx
            else:
                index_a, index_b = match.trainIdx, match.queryIdx
            if (
                registered.point3d_idx[index_a] != -1
                or frame.point3d_idx[index_b] != -1
            ):
                continue
            points_a.append(registered.kp[index_a].pt)
            points_b.append(frame.kp[index_b].pt)
            indices_a.append(index_a)
            indices_b.append(index_b)
        if len(points_a) < 8:
            continue
        points3d = triangulate_points(
            K,
            registered.R,
            registered.t,
            frame.R,
            frame.t,
            np.float32(points_a),
            np.float32(points_b),
        )
        good = cheirality_mask(
            points3d, registered.R, registered.t
        ) & cheirality_mask(points3d, frame.R, frame.t)
        center_a = -registered.R.T @ registered.t
        center_b = -frame.R.T @ frame.t
        for index in range(len(points3d)):
            if not good[index]:
                continue
            angle = triangulation_angle_deg(
                points3d[index : index + 1], center_a, center_b
            )
            if angle < min_point_angle_deg:
                rejected_angle += 1
                continue
            point_id = len(points3D_list)
            points3D_list.append(points3d[index])
            registered.point3d_idx[indices_a[index]] = point_id
            frame.point3d_idx[indices_b[index]] = point_id
            new_count += 1
    if rejected_angle:
        print(
            f"[triangulate]   (rejected {rejected_angle} points below "
            f"{min_point_angle_deg}° triangulation angle)"
        )
    return new_count


def run_incremental_sfm(frames, matches_table, K, min_pnp_inliers=30):
    initial_i, initial_j, points3d = initialize_reconstruction(
        frames, K, matches_table
    )
    points3d = list(points3d)
    registered = [frames[initial_i], frames[initial_j]]
    remaining = [
        frame for frame in frames if frame.idx not in (initial_i, initial_j)
    ]
    progress = True
    while remaining and progress:
        progress = False
        best = None
        for frame in remaining:
            result = register_next_image(
                frame,
                registered,
                matches_table,
                K,
                np.array(points3d),
                min_pnp_inliers=min_pnp_inliers,
            )
            if result is None:
                continue
            rotation, translation, n_inliers = result
            if best is None or n_inliers > best[1]:
                best = (frame, n_inliers, rotation, translation)
        if best is None:
            break
        frame, n_inliers, rotation, translation = best
        frame.R, frame.t, frame.registered = rotation, translation, True
        print(f"[register] {os.path.basename(frame.path)}  ({n_inliers} PnP inliers)")
        new_points = triangulate_new_points(
            frame, registered, matches_table, K, points3d
        )
        print(f"[triangulate] +{new_points} new points (total {len(points3d)})")
        registered.append(frame)
        remaining.remove(frame)
        progress = True
    if remaining:
        print(
            f"[warn] could not register {len(remaining)} image(s): "
            f"{[os.path.basename(frame.path) for frame in remaining]}"
        )
    return registered, np.array(points3d, dtype=np.float64)
