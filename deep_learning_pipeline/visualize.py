"""
Quick 3D viewer for the reconstructed sparse point cloud and camera poses.

Usage:
    python -m pipeline.visualize --sfm-dir outputs/templeRing/sparse
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap


def camera_frustum(image, camera, scale=0.05):
    """Build a small wireframe frustum LineSet for one registered image."""
    R_wc = image.cam_from_world().rotation.matrix().T  # cam->world
    C = image.projection_center()  # camera center in world coords

    w, h = camera.width, camera.height
    corners_cam = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float)
    K_inv = np.linalg.inv(camera.calibration_matrix())
    corners_cam = (K_inv @ corners_cam.T).T * scale
    corners_world = (R_wc @ corners_cam.T).T + C

    points = np.vstack([C, corners_world])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    frustum.paint_uniform_color([1.0, 0.2, 0.2])
    return frustum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfm-dir", type=Path, required=True)
    parser.add_argument("--frustum-scale", type=float, default=0.05)
    args = parser.parse_args()

    model = pycolmap.Reconstruction(args.sfm_dir)

    xyz = np.array([p.xyz for p in model.points3D.values()])
    rgb = np.array([p.color for p in model.points3D.values()]) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    geometries = [pcd]
    for image in model.images.values():
        camera = model.cameras[image.camera_id]
        geometries.append(camera_frustum(image, camera, args.frustum_scale))

    o3d.visualization.draw_geometries(
        geometries, window_name="Temple Ring sparse reconstruction"
    )


if __name__ == "__main__":
    main()
