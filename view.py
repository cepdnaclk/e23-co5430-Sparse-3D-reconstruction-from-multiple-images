"""
Opens an interactive 3D window showing your reconstructed point cloud plus
the recovered camera poses as small frustums (so you can see the ring
structure of the capture). Unlike the offscreen snapshot in the pipeline,
this uses a real window -- rotate/pan/zoom with the mouse, close to exit.

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
    cams = [(c["image"], np.array(c["R"]), np.array(c["t"])) for c in data["cameras"]]
    return K, cams


def camera_frustum(K, R, t, scale=0.05, color=(1, 0.5, 0)):
    """Small wireframe pyramid showing a camera's position + viewing direction."""
    # camera center in world coords, given world->camera: x_cam = R x_world + t
    center = -R.T @ t
    # four corners of a small square in front of the camera, in camera space,
    # then transformed into world space
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
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="e.g. outputs/orb")
    ap.add_argument("--point_size", type=float, default=3.0)
    args = ap.parse_args()

    ply_path = os.path.join(args.results_dir, "points3D_open3d.ply")
    if not os.path.exists(ply_path):
        ply_path = os.path.join(args.results_dir, "points3D.ply")
    cam_path = os.path.join(args.results_dir, "cameras.json")

    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"[view] loaded {len(pcd.points)} points from {ply_path}")

    geoms = [pcd]
    if os.path.exists(cam_path):
        K, cams = load_cameras(cam_path)
        # scale frustums relative to the point cloud's own size, so they're
        # visible but don't dominate the scene
        bbox = pcd.get_axis_aligned_bounding_box()
        scale = float(np.linalg.norm(bbox.get_extent())) * 0.03
        for name, R, t in cams:
            geoms.append(camera_frustum(K, R, t, scale=scale))
        print(f"[view] loaded {len(cams)} camera poses from {cam_path}")

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Reconstruction: {args.results_dir}",
        width=1000,
        height=750,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
