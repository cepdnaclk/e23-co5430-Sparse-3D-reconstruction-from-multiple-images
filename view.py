"""
Renders the reconstructed point cloud plus the recovered camera poses (small
frustums) to PNG files using Open3D's offscreen (EGL) renderer. This works
even on headless machines and on Wayland+NVIDIA setups where Open3D's
interactive windowed viewer fails to create an OpenGL window.

For interactive exploration, open the .ply in MeshLab instead:

    meshlab outputs/sift/points3D.ply

Usage:
    python view.py outputs/orb
    python view.py outputs/sift --angles 8 --point_size 4.0
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
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.paint_uniform_color(color)
    return ls


def render_offscreen(ply_path, cam_path, out_dir, point_size=3.0, n_angles=6):
    """Render the point cloud (+ camera frustums) from n_angles turntable views."""
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"[view] loaded {len(pcd.points)} points from {ply_path}")

    K, cams = None, []
    if os.path.exists(cam_path):
        K, cams = load_cameras(cam_path)
        print(f"[view] loaded {len(cams)} camera poses from {cam_path}")

    renderer = o3d.visualization.rendering.OffscreenRenderer(1200, 900)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    pt_mat = o3d.visualization.rendering.MaterialRecord()
    pt_mat.shader = "defaultUnlit"
    pt_mat.point_size = point_size
    renderer.scene.add_geometry("pcd", pcd, pt_mat)

    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.shader = "unlitLine"

    if len(cams):
        bbox = pcd.get_axis_aligned_bounding_box()
        scale = float(np.linalg.norm(bbox.get_extent())) * 0.03
        for idx, (name, R, t) in enumerate(cams):
            frustum = camera_frustum(K, R, t, scale=scale)
            renderer.scene.add_geometry(f"cam_{idx}", frustum, line_mat)

    center = np.asarray(bbox.get_center(), dtype=np.float32)
    diag = float(np.linalg.norm(bbox.get_extent())) or 1.0
    up = np.array([0, 1, 0], dtype=np.float32)

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for angle_deg in np.linspace(0, 360, n_angles, endpoint=False):
        a = np.radians(angle_deg)
        eye = center + np.array(
            [np.sin(a) * diag * 1.5, 0.4 * diag, np.cos(a) * diag * 1.5],
            dtype=np.float32,
        )
        renderer.setup_camera(60.0, center, eye, up)
        img = renderer.render_to_image()
        path = os.path.join(out_dir, f"points3D_view_{int(round(angle_deg)):03d}.png")
        o3d.io.write_image(path, img)
        paths.append(path)
        print(f"[view] wrote {path}")

    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="e.g. outputs/orb")
    ap.add_argument("--point_size", type=float, default=3.0)
    ap.add_argument(
        "--angles", type=int, default=6, help="number of turntable views to render"
    )
    ap.add_argument(
        "--out", default=None, help="where to write the PNGs (default: results_dir)"
    )
    args = ap.parse_args()

    ply_path = os.path.join(args.results_dir, "points3D_open3d.ply")
    if not os.path.exists(ply_path):
        ply_path = os.path.join(args.results_dir, "points3D.ply")
    cam_path = os.path.join(args.results_dir, "cameras.json")
    out_dir = args.out or args.results_dir

    render_offscreen(ply_path, cam_path, out_dir, args.point_size, args.angles)

    print(
        f"\n[view] This system cannot show an interactive Open3D window "
        f"(Wayland/GLFW), so PNG renders were saved to {out_dir}/."
    )
    print(f"[view] Explore the point cloud interactively in MeshLab:\n       meshlab {ply_path}")


if __name__ == "__main__":
    main()