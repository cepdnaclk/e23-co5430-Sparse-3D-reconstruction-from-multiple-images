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


def average_rotation(rotations):
    """Chordal L2 mean of a list of rotation matrices, via SVD. Used to get a
    robust estimate of the global alignment rotation directly from the known
    per-camera rotations, instead of only from camera centers (positions) --
    centers alone are a weak, sometimes near-degenerate constraint on
    rotation when the registered cameras span only a narrow arc."""
    M = sum(rotations) / len(rotations)
    U, S, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    return U @ D @ Vt


def pairwise_relative_rotation_error(our_Rs, gt_Rs):
    """Median pairwise relative-rotation error in degrees. Cancels out any
    single shared/global misalignment entirely (it never uses an aligned
    frame at all) -- use this as a sanity check whenever the absolute
    (aligned) rotation error looks suspiciously large or uniform across
    cameras, which is the signature of a bad global alignment rather than a
    genuinely bad reconstruction."""
    errs = []
    n = len(our_Rs)
    for a in range(n):
        for b in range(a + 1, n):
            rel_ours = our_Rs[b] @ our_Rs[a].T
            rel_gt = gt_Rs[b] @ gt_Rs[a].T
            err_R = rel_ours @ rel_gt.T
            cos_a = np.clip((np.trace(err_R) - 1) / 2, -1, 1)
            errs.append(np.degrees(np.arccos(cos_a)))
    return np.median(errs), np.mean(errs)


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

    # Robust rotation alignment: average per-camera (gt_R_i @ our_R_i^T)
    # candidates directly, rather than relying solely on the center-fit SVD
    # (which degenerates when the registered cameras span a narrow arc).
    R_align_candidates = [gt_R @ our_R.T for our_R, gt_R in zip(our_Rs, gt_Rs)]
    R_align = average_rotation(R_align_candidates)

    # scale + translation, given the fixed rotation above (standard fixed-R
    # Procrustes: rotate first, then least-squares scale/translation)
    src_mean = our_centers.mean(axis=0)
    dst_mean = gt_centers.mean(axis=0)
    src_c = our_centers - src_mean
    dst_c = gt_centers - dst_mean
    rotated_src_c = (R_align @ src_c.T).T
    denom = (rotated_src_c**2).sum()
    scale = (rotated_src_c * dst_c).sum() / denom if denom > 1e-12 else 1.0
    t_align = dst_mean - scale * (R_align @ src_mean)

    aligned_centers = scale * (R_align @ our_centers.T).T + t_align
    pos_errors = np.linalg.norm(aligned_centers - gt_centers, axis=1)

    rot_errors_deg = []
    for our_R, gt_R in zip(our_Rs, gt_Rs):
        R_aligned_cam = R_align @ our_R
        R_err = R_aligned_cam @ gt_R.T
        cos_angle = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        rot_errors_deg.append(np.degrees(np.arccos(cos_angle)))
    rot_errors_deg = np.array(rot_errors_deg)

    rel_median, rel_mean = pairwise_relative_rotation_error(our_Rs, gt_Rs)

    print("\n=== Pose accuracy vs. ground truth (TempleRing calibration) ===")
    print(f"Cameras compared: {len(names)}/{len(registered)}")
    print(
        f"Position error (post-alignment): "
        f"mean={pos_errors.mean():.4f}  median={np.median(pos_errors):.4f}"
    )
    print(
        f"Rotation error, absolute/aligned (degrees): "
        f"mean={rot_errors_deg.mean():.3f}  median={np.median(rot_errors_deg):.3f}"
    )
    print(
        f"Rotation error, pairwise-RELATIVE (degrees): "
        f"mean={rel_mean:.3f}  median={rel_median:.3f}  "
        f"(cross-check, immune to alignment issues)"
    )
    if rel_median < 5.0 and rot_errors_deg.mean() > 30.0:
        print(
            "[evaluate] NOTE: relative error is low but absolute error is high "
            "-- this pattern means the reconstruction itself is accurate, but "
            "the registered cameras likely span too narrow an arc for a fully "
            "reliable global alignment. Trust the relative number here."
        )
    print("==================================================================\n")

    return {
        "names": names,
        "position_error": pos_errors,
        "rotation_error_deg": rot_errors_deg,
        "relative_rotation_error_median": rel_median,
        "scale": scale,
    }
