"""Open3D point-cloud export and optional snapshot rendering."""

import numpy as np


def _render_worker(ply_path, png_path, cam_lines_json):
    import open3d as o3d

    point_cloud = o3d.io.read_point_cloud(ply_path)
    renderer = o3d.visualization.rendering.OffscreenRenderer(800, 600)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 4.0
    renderer.scene.add_geometry("pcd", point_cloud, material)
    bounds = point_cloud.get_axis_aligned_bounding_box()
    center = np.asarray(bounds.get_center(), dtype=np.float32)
    diagonal = float(np.linalg.norm(bounds.get_extent())) or 1.0
    eye = (center + np.array([1, 1, 1], dtype=np.float32) * diagonal).astype(
        np.float32
    )
    renderer.setup_camera(60.0, center, eye, np.array([0, 1, 0], dtype=np.float32))
    o3d.io.write_image(png_path, renderer.render_to_image())


def export_open3d(path, points3D, registered, K):
    import multiprocessing as mp

    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points3D)
    if len(points3D) > 0:
        point_cloud.paint_uniform_color([0.2, 0.6, 0.9])
    o3d.io.write_point_cloud(path, point_cloud)
    print(f"[open3d] wrote {path}")
    try:
        context = mp.get_context("spawn")
        process = context.Process(
            target=_render_worker,
            args=(path, path.replace(".ply", ".png"), None),
        )
        process.start()
        process.join(timeout=30)
        if process.exitcode == 0:
            print(f"[open3d] wrote rendered snapshot {path.replace('.ply', '.png')}")
        else:
            print(
                "[open3d] snapshot render unavailable in this environment "
                "(no GPU display) — open the .ply locally in Open3D instead"
            )
    except Exception as error:
        print(f"[open3d] snapshot render skipped: {error}")
