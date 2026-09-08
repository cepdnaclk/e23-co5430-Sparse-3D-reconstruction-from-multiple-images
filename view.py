"""
Interactive viewer for the reconstructed point cloud and recovered cameras.

Usage:
    python view.py outputs/orb
    python view.py outputs/sift
"""

import argparse
import json
import os

import numpy as np
import open3d as o3d


def load_cameras(cameras_json_path):
    with open(cameras_json_path) as f:
        data = json.load(f)

    K = np.array(data["K"])

    cams = [
        (c["image"], np.array(c["R"]), np.array(c["t"]))
        for c in data["cameras"]
    ]

    return K, cams


def camera_frustum(K, R, t, scale=0.05, color=(1, 0.5, 0)):
    """Create a small wireframe pyramid representing a camera."""

    center = -R.T @ t

    corners_cam = (
        np.array(
            [
                [-1, -1, 2],
                [1, -1, 2],
                [1, 1, 2],
                [-1, 1, 2],
            ]
        )
        * scale
    )

    corners_world = (R.T @ corners_cam.T).T + center

    points = np.vstack([center, corners_world])

    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
    ]

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    ls.paint_uniform_color(color)

    return ls


def view_point_cloud(
    ply_path,
    cam_path,
    point_size=3.0,
):
    """Open an interactive Open3D viewer."""

    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)

    print(f"[view] loaded {len(pcd.points)} points from {ply_path}")

    if len(pcd.points) == 0:
        print("[view] ERROR: point cloud is empty.")
        return

    # Load cameras
    K = None
    cams = []

    if os.path.exists(cam_path):
        K, cams = load_cameras(cam_path)
        print(f"[view] loaded {len(cams)} camera poses from {cam_path}")

    # Create visualization geometries
    geometries = [pcd]

    # Camera frustums
    if len(cams):
        bbox = pcd.get_axis_aligned_bounding_box()

        scale = float(np.linalg.norm(bbox.get_extent())) * 0.03

        for idx, (name, R, t) in enumerate(cams):
            frustum = camera_frustum(
                K,
                R,
                t,
                scale=scale,
            )

            geometries.append(frustum)

    print("\n[view] Opening Open3D viewer...")
    print("[view] Close the window to exit.\n")

    # Interactive viewer
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Sparse 3D Reconstruction",
        width=1200,
        height=900,
        point_show_normal=False,
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "results_dir",
        help="e.g. outputs/orb or outputs/sift",
    )

    ap.add_argument(
        "--point_size",
        type=float,
        default=3.0,
    )

    args = ap.parse_args()

    # Find point cloud
    ply_path = os.path.join(
        args.results_dir,
        "points3D_open3d.ply",
    )

    if not os.path.exists(ply_path):
        ply_path = os.path.join(
            args.results_dir,
            "points3D.ply",
        )

    # Camera file
    cam_path = os.path.join(
        args.results_dir,
        "cameras.json",
    )

    # Check point cloud
    if not os.path.exists(ply_path):
        print(f"[view] ERROR: Could not find point cloud:")
        print(f"       {ply_path}")
        return

    view_point_cloud(
        ply_path,
        cam_path,
        args.point_size,
    )


if __name__ == "__main__":
    main()