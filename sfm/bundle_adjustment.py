"""Sparse bundle adjustment for cameras and reconstructed points."""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def project(K, R, t, pts3d):
    points_camera = (R @ pts3d.T + t.reshape(3, 1)).T
    projected = (K @ points_camera.T).T
    return projected[:, :2] / projected[:, 2:3]


def pack_params(registered, points3D):
    camera_params = []
    for frame in registered:
        rotation_vector, _ = cv2.Rodrigues(frame.R)
        camera_params.append(
            np.concatenate([rotation_vector.ravel(), frame.t.ravel()])
        )
    return np.concatenate([np.array(camera_params).ravel(), points3D.ravel()])


def unpack_params(x, n_cams, n_pts):
    camera_params = x[: n_cams * 6].reshape(n_cams, 6)
    points3d = x[n_cams * 6 :].reshape(n_pts, 3)
    return camera_params, points3d


def build_observations(registered, points3D):
    camera_indices, point_indices, observations = [], [], []
    for camera_index, frame in enumerate(registered):
        for keypoint_index, point_id in enumerate(frame.point3d_idx):
            if point_id == -1:
                continue
            camera_indices.append(camera_index)
            point_indices.append(point_id)
            observations.append(frame.kp[keypoint_index].pt)
    return (
        np.array(camera_indices),
        np.array(point_indices),
        np.array(observations, dtype=np.float64),
    )


def ba_residuals(x, K, n_cams, n_pts, cam_idx, pt_idx, obs):
    camera_params, points3d = unpack_params(x, n_cams, n_pts)
    residuals = np.zeros((len(obs), 2))
    for camera_index in range(n_cams):
        mask = cam_idx == camera_index
        if not np.any(mask):
            continue
        rotation, _ = cv2.Rodrigues(camera_params[camera_index, :3])
        translation = camera_params[camera_index, 3:]
        projected = project(
            K, rotation, translation, points3d[pt_idx[mask]]
        )
        residuals[mask] = projected - obs[mask]
    return residuals.ravel()


def build_ba_sparsity(n_cams, n_pts, cam_idx, pt_idx):
    n_observations = len(cam_idx)
    matrix = lil_matrix(
        (n_observations * 2, n_cams * 6 + n_pts * 3), dtype=int
    )
    observation_indices = np.arange(n_observations)
    for offset in range(6):
        matrix[2 * observation_indices, cam_idx * 6 + offset] = 1
        matrix[2 * observation_indices + 1, cam_idx * 6 + offset] = 1
    for offset in range(3):
        matrix[
            2 * observation_indices,
            n_cams * 6 + pt_idx * 3 + offset,
        ] = 1
        matrix[
            2 * observation_indices + 1,
            n_cams * 6 + pt_idx * 3 + offset,
        ] = 1
    return matrix


def bundle_adjust(registered, points3D, K, verbose=True):
    n_cameras, n_points = len(registered), len(points3D)
    camera_indices, point_indices, observations = build_observations(
        registered, points3D
    )
    initial = pack_params(registered, points3D)
    residual_args = (
        K,
        n_cameras,
        n_points,
        camera_indices,
        point_indices,
        observations,
    )
    if verbose:
        residuals = ba_residuals(initial, *residual_args)
        error = np.sqrt((residuals.reshape(-1, 2) ** 2).sum(1)).mean()
        print(
            f"[BA] initial mean reprojection error: {error:.3f} px "
            f"over {len(observations)} observations"
        )
    result = least_squares(
        ba_residuals,
        initial,
        jac_sparsity=build_ba_sparsity(
            n_cameras, n_points, camera_indices, point_indices
        ),
        verbose=2 if verbose else 0,
        method="trf",
        args=residual_args,
        max_nfev=100,
        xtol=1e-8,
        ftol=1e-8,
    )
    camera_params, optimized_points = unpack_params(
        result.x, n_cameras, n_points
    )
    for camera_index, frame in enumerate(registered):
        frame.R, _ = cv2.Rodrigues(camera_params[camera_index, :3])
        frame.t = camera_params[camera_index, 3:]
    if verbose:
        residuals = ba_residuals(result.x, *residual_args)
        error = np.sqrt((residuals.reshape(-1, 2) ** 2).sum(1)).mean()
        print(f"[BA] final mean reprojection error:   {error:.3f} px")
    return registered, optimized_points
