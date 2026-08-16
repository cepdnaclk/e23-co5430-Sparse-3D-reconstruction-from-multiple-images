"""
Opens an interactive 3D window showing your reconstructed point cloud plus
the recovered camera poses as small frustums (so you can see the ring
structure of the capture). Unlike the offscreen snapshot in the pipeline,
this uses a real window -- rotate/pan/zoom with the mouse, close to exit.

Usage:
    python view.py outputs/orb
    python view.py outputs/sift
    python view.py outputs/sift --backend matplotlib
    python view.py outputs/sift --backend matplotlib --save snapshot.png
"""

import argparse
import json
import os

# On Wayland sessions Open3D's bundled GLFW can pick the Wayland backend,
# which fails to create an OpenGL context (GLEW init fails, window never
# opens). This best-effort hint pushes GLFW to the X11 backend where an
# XWayland display exists (no-op on X11; some Open3D builds ignore it). The
# matplotlib fallback below is the reliable path either way.
os.environ.setdefault("GLFW_PLATFORM", "x11")

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


def load_ply_points(ply_path):
    """Read xyz from an ASCII or binary PLY without going through Open3D."""
    with open(ply_path, "rb") as f:
        header_bytes = bytearray()
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header not terminated")
            header_bytes += line
            if line.rstrip(b"\r\n") == b"end_header":
                break
        header = header_bytes.decode("ascii")

    fmt = None
    vertex_count = None
    vertex_props = []
    props = None
    in_vertex = False
    for line in header.splitlines()[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
                props = vertex_props
        elif parts[0] == "property" and in_vertex:
            props.append((parts[2], parts[1]))

    if fmt is None or vertex_count is None:
        raise ValueError("PLY header missing format/element vertex")

    ncols = 3  # x y z are the first three numeric columns in our files
    if fmt == "ascii":
        pts = np.loadtxt(
            open(ply_path, encoding="utf-8", errors="replace"),
            skiprows=header.count("\n"),
            max_rows=vertex_count,
            usecols=(0, 1, 2),
        )
        return np.asarray(pts, dtype=np.float64).reshape(-1, 3)

    if fmt not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError(f"Unsupported PLY format: {fmt}")

    dt = np.dtype([(name, _PLY_DTYPES[typ]) for name, typ in vertex_props])
    dt = dt.newbyteorder("<" if fmt == "binary_little_endian" else ">")
    raw = np.fromfile(open(ply_path, "rb"), dtype=dt, count=vertex_count,
                      offset=len(header_bytes))
    xyz = np.column_stack([raw[name] for name in ("x", "y", "z")])
    return np.asarray(xyz, dtype=np.float64)


_PLY_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def plot_matplotlib(points, cameras, results_dir, save_path=None):
    """Render the point cloud + camera frustums with matplotlib, which works
    without a working OpenGL context (Wayland/headless)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Reconstruction: {results_dir}")

    if len(points):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.2, c="skyblue", depthshade=False)

    for name, R, t in cameras:
        center = -R.T @ t
        corners_cam = np.array([[-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2]]) * 0.05
        corners_world = (R.T @ corners_cam.T).T + center
        verts = np.vstack([center, corners_world])
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
        segs = [np.vstack([verts[a], verts[b]]) for a, b in edges]
        lc = Line3DCollection(segs, colors="orange", linewidths=1.2)
        ax.add_collection3d(lc)

    if len(points):
        mn, mx = points.min(axis=0), points.max(axis=0)
        pad = max((mx - mn).max() * 0.05, 1e-6)
        ax.set_xlim(mn[0] - pad, mx[0] + pad)
        ax.set_ylim(mn[1] - pad, mx[1] + pad)
        ax.set_zlim(mn[2] - pad, mx[2] + pad)
    ax.set_box_aspect((1, 1, 1))

    if save_path:
        fig.savefig(save_path, dpi=140)
        print(f"[view] saved matplotlib render -> {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="e.g. outputs/orb")
    ap.add_argument("--point_size", type=float, default=3.0)
    ap.add_argument(
        "--backend",
        choices=["open3d", "matplotlib"],
        default="open3d",
        help="open3d = interactive GL window (default); matplotlib = backend-"
        "agnostic 3D plot that works on Wayland/headless systems",
    )
    ap.add_argument(
        "--save",
        default=None,
        metavar="OUT.png",
        help="With --backend matplotlib: render to this PNG instead of showing "
        "a window (works fully headless)",
    )
    args = ap.parse_args()

    ply_path = os.path.join(args.results_dir, "points3D_open3d.ply")
    if not os.path.exists(ply_path):
        ply_path = os.path.join(args.results_dir, "points3D.ply")
    cam_path = os.path.join(args.results_dir, "cameras.json")

    cams = []
    if os.path.exists(cam_path):
        K, cams = load_cameras(cam_path)
        print(f"[view] loaded {len(cams)} camera poses from {cam_path}")

    if args.backend == "matplotlib":
        points = load_ply_points(ply_path)
        print(f"[view] loaded {len(points)} points from {ply_path}")
        plot_matplotlib(points, cams, args.results_dir, save_path=args.save)
        return

    import open3d as o3d

    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"[view] loaded {len(pcd.points)} points from {ply_path}")

    geoms = [pcd]
    if cams:
        bbox = pcd.get_axis_aligned_bounding_box()
        scale = float(np.linalg.norm(bbox.get_extent())) * 0.03
        for name, R, t in cams:
            geoms.append(camera_frustum(K, R, t, scale=scale))

    probe = o3d.visualization.Visualizer()
    if not probe.create_window(window_name="probe", width=16, height=16, visible=False):
        print(
            "[view] Open3D could not create an OpenGL window (Wayland/headless "
            "environment). Falling back to the matplotlib renderer.\n"
            "  For an interactive plot use: --backend matplotlib\n"
            "  For a headless PNG snapshot use: --backend matplotlib --save out.png"
        )
        points = load_ply_points(ply_path)
        plot_matplotlib(points, cams, args.results_dir, save_path=args.save)
        return
    probe.destroy_window()

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Reconstruction: {args.results_dir}",
        width=1000,
        height=750,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
