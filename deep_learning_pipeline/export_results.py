"""
Helpers to export a pycolmap.Reconstruction to plain formats that don't
require COLMAP or pycolmap to open later (PLY point cloud, text poses).
"""

from pathlib import Path

import numpy as np
import pycolmap


def export_point_cloud(model: pycolmap.Reconstruction, ply_path: Path) -> None:
    """Write the sparse 3D points (with color) to a PLY file."""
    points = model.points3D
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p in points.values():
            x, y, z = p.xyz
            r, g, b = (int(c) for c in p.color)
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def export_poses(model: pycolmap.Reconstruction, poses_path: Path) -> None:
    """
    Write one line per registered image with its recovered camera pose:
    image_name qw qx qy qz tx ty tz  (world-from-camera rotation as a
    quaternion, plus the camera center translation), matching the
    convention used by COLMAP's images.txt.
    """
    poses_path.parent.mkdir(parents=True, exist_ok=True)
    with open(poses_path, "w") as f:
        f.write("# image_name qw qx qy qz tx ty tz\n")
        for image in model.images.values():
            q = image.cam_from_world().rotation.quat  # (x, y, z, w) in pycolmap
            qx, qy, qz, qw = q
            t = image.cam_from_world().translation
            f.write(f"{image.name} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]}\n")


def summarize(model: pycolmap.Reconstruction) -> str:
    return (
        f"{len(model.images)} images registered, "
        f"{len(model.points3D)} 3D points, "
        f"mean track length {np.mean([p.track.length() for p in model.points3D.values()]):.2f}"
    )
