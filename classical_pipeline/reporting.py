"""Runtime statistics and summary reporting."""

import numpy as np


RUN_STATS = {
    "n_keypoints": {},
    "n_matches": {},
    "n_inliers": {},
    "timings": {},
}


def record_match_count(i, j, n):
    RUN_STATS["n_matches"][(i, j)] = n


def print_metrics_summary(frames, matches_table, points3D, registered, timings, K=None):
    """Print the shared M2 metrics output contract."""
    if RUN_STATS["n_keypoints"]:
        kp_min = min(RUN_STATS["n_keypoints"].values())
        kp_max = max(RUN_STATS["n_keypoints"].values())
        kp_mean = float(np.mean(list(RUN_STATS["n_keypoints"].values())))
    elif frames:
        kps = [len(frame.kp) for frame in frames]
        kp_min, kp_max = int(min(kps)), int(max(kps))
        kp_mean = float(np.mean(kps))
    else:
        kp_min = kp_max = 0
        kp_mean = 0.0

    if RUN_STATS["n_matches"]:
        matches_mean = float(np.mean(list(RUN_STATS["n_matches"].values())))
    elif matches_table:
        counts = [len(matches) for matches in matches_table.values()]
        matches_mean = float(np.mean(counts)) if counts else 0.0
    else:
        matches_mean = 0.0

    inliers_mean = (
        float(np.mean(list(RUN_STATS["n_inliers"].values())))
        if RUN_STATS["n_inliers"]
        else 0.0
    )

    n_points = len(points3D) if points3D is not None else 0
    n_obs = 0
    point_obs_counts = {}
    obs_per_image = []
    for frame in registered:
        count = int(np.sum(frame.point3d_idx != -1))
        obs_per_image.append(count)
        n_obs += count
        for point_id in frame.point3d_idx:
            if point_id != -1:
                point_id = int(point_id)
                point_obs_counts[point_id] = point_obs_counts.get(point_id, 0) + 1
    mean_track = (
        float(np.mean(list(point_obs_counts.values()))) if point_obs_counts else 0.0
    )
    mean_obs_per_image = float(np.mean(obs_per_image)) if obs_per_image else 0.0

    mean_reproj = 0.0
    if K is not None and n_obs > 0 and points3D is not None and len(points3D) > 0:
        errors = []
        for frame in registered:
            valid = frame.point3d_idx != -1
            if not np.any(valid):
                continue
            point_ids = frame.point3d_idx[valid]
            mask = point_ids < len(points3D)
            if not np.any(mask):
                continue
            point_ids = point_ids[mask]
            points = points3D[point_ids]
            kp_indices = np.where(valid)[0][mask]
            observations = np.array(
                [frame.kp[index].pt for index in kp_indices], dtype=np.float64
            )
            points_camera = (
                frame.R @ points.T + frame.t.reshape(3, 1)
            ).T
            projected = (K @ points_camera.T).T
            projected = projected[:, :2] / projected[:, 2:3]
            errors.extend(np.linalg.norm(projected - observations, axis=1).tolist())
        if errors:
            mean_reproj = float(np.mean(errors))

    print("\n=== M2 metrics summary ===")
    print(f"Images: {len(frames)}   Registered: {len(registered)}/{len(frames)}")
    print(f"Keypoints per image: min={kp_min} max={kp_max} mean={kp_mean:.1f}")
    print(f"Matched pairs (pre-RANSAC), mean per pair: {matches_mean:.1f}")
    print(f"RANSAC inliers, mean per verified pair: {inliers_mean:.1f}")
    print(f"Reconstructed 3D points: {n_points}")
    print(f"Num observations: {n_obs}")
    print(f"Mean track length: {mean_track:.2f}")
    print(f"Mean observations per image: {mean_obs_per_image:.2f}")
    print(f"Mean reprojection error: {mean_reproj:.3f} px")
    for stage, seconds in timings.items():
        print(f"Time [{stage}]: {seconds:.2f}s")
    print("===========================\n")
