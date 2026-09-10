"""PLY, camera JSON, and COLMAP text exporters."""

import json
import os
import shutil

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def export_ply(path, points3D, colors=None):
    with open(path, "w") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(points3D)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for index, (x, y, z) in enumerate(points3D):
            if colors is None:
                file.write(f"{x} {y} {z}\n")
            else:
                red, green, blue = colors[index]
                file.write(
                    f"{x} {y} {z} {int(red)} {int(green)} {int(blue)}\n"
                )


def export_cameras(path, registered, K):
    output = {"K": K.tolist(), "cameras": []}
    for frame in registered:
        output["cameras"].append(
            {
                "image": os.path.basename(frame.path),
                "R": frame.R.tolist(),
                "t": frame.t.tolist(),
            }
        )
    with open(path, "w") as file:
        json.dump(output, file, indent=2)


def export_colmap_text(out_dir, registered, points3D, K, copy_images_to=None):
    os.makedirs(out_dir, exist_ok=True)
    frames_by_index = {frame.idx: frame for frame in registered}
    height, width = registered[0].shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    with open(os.path.join(out_dir, "cameras.txt"), "w") as file:
        file.write(
            "# Camera list, one line per camera: CAMERA_ID MODEL WIDTH HEIGHT PARAMS\n"
        )
        file.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    tracks = {}
    for frame in registered:
        for keypoint_index, point_id in enumerate(frame.point3d_idx):
            if point_id != -1:
                tracks.setdefault(int(point_id), []).append(
                    (frame.idx + 1, keypoint_index)
                )

    with open(os.path.join(out_dir, "images.txt"), "w") as file:
        file.write(
            "# Image list, two lines per image: IMAGE_ID QW QX QY QZ TX TY TZ "
            "CAMERA_ID NAME, then POINTS2D\n"
        )
        for frame in registered:
            qx, qy, qz, qw = Rotation.from_matrix(frame.R).as_quat()
            image_id = frame.idx + 1
            file.write(
                f"{image_id} {qw} {qx} {qy} {qz} {frame.t[0]} {frame.t[1]} "
                f"{frame.t[2]} 1 {os.path.basename(frame.path)}\n"
            )
            parts = []
            for keypoint_index, keypoint in enumerate(frame.kp):
                point_id = frame.point3d_idx[keypoint_index]
                point_id_text = str(int(point_id) + 1) if point_id != -1 else "-1"
                x, y = keypoint.pt
                parts.append(f"{x} {y} {point_id_text}")
            file.write(" ".join(parts) + "\n")

    color_cache = {}

    def sample_color(path, x, y):
        if path not in color_cache:
            color_cache[path] = cv2.imread(path, cv2.IMREAD_COLOR)
        image = color_cache[path]
        if image is None:
            return 128, 128, 128
        x_index = int(np.clip(round(x), 0, image.shape[1] - 1))
        y_index = int(np.clip(round(y), 0, image.shape[0] - 1))
        blue, green, red = image[y_index, x_index]
        return int(red), int(green), int(blue)

    with open(os.path.join(out_dir, "points3D.txt"), "w") as file:
        file.write("# 3D point list: POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for point_id, point in enumerate(points3D):
            observations = tracks.get(point_id, [])
            if observations:
                image_id, keypoint_index = observations[0]
                frame = frames_by_index[image_id - 1]
                x, y = frame.kp[keypoint_index].pt
                red, green, blue = sample_color(frame.path, x, y)
            else:
                red, green, blue = 128, 128, 128
            track = " ".join(
                f"{image_id} {keypoint_index}"
                for image_id, keypoint_index in observations
            )
            file.write(
                f"{point_id + 1} {point[0]} {point[1]} {point[2]} "
                f"{red} {green} {blue} 0.5 {track}\n"
            )
    if copy_images_to:
        os.makedirs(copy_images_to, exist_ok=True)
        for frame in registered:
            shutil.copy(
                frame.path,
                os.path.join(copy_images_to, os.path.basename(frame.path)),
            )
    print(f"[colmap-export] wrote {out_dir}/{{cameras,images,points3D}}.txt")
