"""
Optional: compare recovered camera poses against the Middlebury Temple Ring
ground-truth calibration file (temple_par.txt), since COLMAP's reconstruction
lives in an arbitrary similarity frame (unknown global scale/rotation/
translation) while the dataset's ground truth is metric.

temple_par.txt format (one header line with image count, then per image):
  <name> f 0 0  0 f 0  0 0 1   r11 r12 r13 r21 r22 r23 r31 r32 r33   t1 t2 t3
i.e. 3x3 intrinsics K, 3x3 rotation R, 3x1 translation t (world-to-camera).

We align the estimated camera centers to the ground-truth centers with an
umeyama similarity transform (rotation + isotropic scale + translation),
then report the RMS camera-center error after alignment.

Usage:
    python -m pipeline.evaluate --gt data/templeRing/temple_par.txt \
        --poses outputs/templeRing/poses.txt
"""
import argparse
from pathlib import Path

import numpy as np


def parse_temple_par(path: Path) -> dict[str, np.ndarray]:
    """Return {image_name: camera_center_in_world_coords} from temple_par.txt."""
    centers = {}
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0].strip())
    for line in lines[1 : 1 + n]:
        parts = line.split()
        name = parts[0]
        vals = np.array(parts[1:], dtype=float)
        K = vals[0:9].reshape(3, 3)  # noqa: F841 (unused, kept for clarity)
        R = vals[9:18].reshape(3, 3)
        t = vals[18:21]
        # World-to-camera: x_cam = R x_world + t  =>  camera center = -R^T t
        center = -R.T @ t
        centers[name] = center
    return centers


def parse_estimated_poses(path: Path) -> dict[str, np.ndarray]:
    """Return {image_name: camera_center} from our poses.txt (see export_results.py)."""
    centers = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            name = parts[0]
            qw, qx, qy, qz, tx, ty, tz = (float(x) for x in parts[1:])
            R = quat_to_rotmat(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            center = -R.T @ t
            centers[name] = center
    return centers


def quat_to_rotmat(qw, qx, qy, qz) -> np.ndarray:
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


def umeyama_alignment(src: np.ndarray, dst: np.ndarray):
    """Similarity transform (R, s, t) mapping src -> dst in least squares sense."""
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = dst_c.T @ src_c / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_src = (src_c**2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - s * R @ mu_src
    return R, s, t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="path to temple_par.txt")
    parser.add_argument("--poses", type=Path, required=True, help="path to our poses.txt")
    args = parser.parse_args()

    gt = parse_temple_par(args.gt)
    est = parse_estimated_poses(args.poses)
    common = sorted(set(gt) & set(est))
    if len(common) < 3:
        raise SystemExit(
            f"Only {len(common)} image names matched between GT and estimate; "
            "check that image names in poses.txt match temple_par.txt entries."
        )

    src = np.array([est[n] for n in common])
    dst = np.array([gt[n] for n in common])
    R, s, t = umeyama_alignment(src, dst)
    aligned = (s * (R @ src.T).T) + t
    err = np.linalg.norm(aligned - dst, axis=1)

    print(f"Compared {len(common)} / {len(gt)} images")
    print(f"Mean camera-center error after alignment: {err.mean():.5f}")
    print(f"Median camera-center error:                {np.median(err):.5f}")
    print(f"Max camera-center error:                   {err.max():.5f}")
    print(f"Recovered similarity scale (est -> GT):     {s:.5f}")


if __name__ == "__main__":
    main()
