"""
Loads TempleRing's ground-truth camera calibration (templeR_par.txt) and
scores a reconstruction's estimated poses against it.

Why this matters for your project: your own reconstruction's world
coordinate frame is arbitrary (it's defined by whichever image pair you
happened to initialize from), so raw camera positions can't be compared to
ground truth directly. We first find the best-fit similarity transform
(rotation + translation + uniform scale) that maps your cameras onto the
ground-truth ones -- this is the standard approach (COLMAP, VGGSfM's own
evaluation, etc. all do this) -- then measure the leftover error.
"""

import os
import numpy as np


def load_gt_poses(par_path):
    """Parses a *_par.txt file (TempleRing / Middlebury MVS format):
    first line = image count, then one line per image:
    'name k11 k12 k13 k21 k22 k23 k31 k32 k33 r11..r33 t1 t2 t3'
    Returns dict: image filename -> (K, R, t) as numpy arrays.
    """
    gt = {}
    with open(par_path) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    n = int(lines[0])
    for line in lines[1 : 1 + n]:
        parts = line.split()
        name = parts[0]
        vals = list(map(float, parts[1:]))
        K = np.array(vals[0:9]).reshape(3, 3)
        R = np.array(vals[9:18]).reshape(3, 3)
        t = np.array(vals[18:21])
        gt[name] = (K, R, t)
    print(f"[evaluate] loaded {len(gt)} ground-truth poses from {par_path}")
    return gt


def camera_center(R, t):
    """World-space camera position, given world->camera pose x_cam = R x_world + t."""
    return -R.T @ t


def umeyama_similarity(src, dst):
    """Best-fit similarity transform (scale s, rotation R, translation t) that
    maps points `src` (Nx3) onto `dst` (Nx3) in a least-squares sense, i.e.
    dst ~= s * R @ src + t. Standard Umeyama (1991) closed-form solution."""
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = (dst_c.T @ src_c) / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt

    var_src = (src_c**2).sum() / len(src)
    scale = np.trace(np.diag(S) @ D) / var_src if var_src > 1e-12 else 1.0
    t = dst_mean - scale * R @ src_mean
    return scale, R, t


def evaluate_poses(registered, gt_poses):
    """Aligns the reconstruction's cameras onto ground truth, then reports
    per-camera rotation error (degrees) and position error (in ground-truth
    units, after alignment) -- your project's version of RRE/RTE."""
    names, our_centers, gt_centers, our_Rs, gt_Rs = [], [], [], [], []
    for f in registered:
        name = os.path.basename(f.path)
        if name not in gt_poses:
            continue
        gt_K, gt_R, gt_t = gt_poses[name]
        names.append(name)
        our_centers.append(camera_center(f.R, f.t))
        gt_centers.append(camera_center(gt_R, gt_t))
        our_Rs.append(f.R)
        gt_Rs.append(gt_R)

    if len(names) < 3:
        print(
            "[evaluate] fewer than 3 registered images have matching ground "
            "truth -- can't align, skipping pose evaluation"
        )
        return None

    our_centers = np.array(our_centers)
    gt_centers = np.array(gt_centers)
    scale, R_align, t_align = umeyama_similarity(our_centers, gt_centers)
    aligned_centers = scale * (R_align @ our_centers.T).T + t_align

    pos_errors = np.linalg.norm(aligned_centers - gt_centers, axis=1)

    rot_errors_deg = []
    for our_R, gt_R in zip(our_Rs, gt_Rs):
        # rotation error = angle of (R_align @ our_R) relative to gt_R
        R_aligned_cam = R_align @ our_R
        R_err = R_aligned_cam @ gt_R.T
        cos_angle = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        rot_errors_deg.append(np.degrees(np.arccos(cos_angle)))
    rot_errors_deg = np.array(rot_errors_deg)

    print("\n=== Pose accuracy vs. ground truth (TempleRing calibration) ===")
    print(f"Cameras compared: {len(names)}/{len(registered)}")
    print(
        f"Position error (post-alignment): "
        f"mean={pos_errors.mean():.4f}  median={np.median(pos_errors):.4f}"
    )
    print(
        f"Rotation error (degrees):        "
        f"mean={rot_errors_deg.mean():.3f}  median={np.median(rot_errors_deg):.3f}"
    )
    print("==================================================================\n")

    return {
        "names": names,
        "position_error": pos_errors,
        "rotation_error_deg": rot_errors_deg,
        "scale": scale,
    }
