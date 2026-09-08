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

This file (classical pipeline) is a SEPARATED implementation from
deep_learning_pipeline/evaluate.py — duplicated logic, identical output
contract (Option A). Both files must print the same M2 / pose sections
with the same headers, formatting, and numeric precision, but they
derive metrics from different reconstruction representations
(Frame list + RUN_STATS vs pycolmap.Reconstruction + h5).
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
        f"mean={pos_errors.mean():.4f}  median={np.median(pos_errors):.4f}  max={pos_errors.max():.4f}"
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
        "relative_rotation_error_mean": rel_mean,
        "scale": scale,
        "position_error_mean": float(pos_errors.mean()),
        "position_error_median": float(np.median(pos_errors)),
        "position_error_max": float(pos_errors.max()),
        "rotation_error_mean": float(rot_errors_deg.mean()),
        "rotation_error_median": float(np.median(rot_errors_deg)),
        "num_compared": len(names),
        "num_registered": len(registered),
    }


# ---------------------------------------------------------------------------
# M2 / structure metrics — duplicated logic (Option A) to match
# deep_learning_pipeline/evaluate.py output contract exactly.
# ---------------------------------------------------------------------------

def _project(K, R, t, pts3d):
    """Project 3D points to 2D using K,R,t (world->camera)."""
    pts_cam = (R @ pts3d.T + t.reshape(3, 1)).T
    # avoid division by zero
    z = pts_cam[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    proj = (K @ pts_cam.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj


def compute_structure_metrics(registered, points3D, K, matches_table=None, frames=None):
    """Compute COLMAP-style structure metrics from classical Frame representation.

    Duplicated implementation (Option A) — same numeric results as
    deep_learning_pipeline/evaluate.py but derived from Frame.point3d_idx
    rather than pycolmap.Reconstruction.

    Returns dict with:
      num_points3D, num_observations, mean_track_length,
      mean_observations_per_image, mean_reprojection_error
    """
    n_points = len(points3D) if points3D is not None else 0
    # observations = total assigned keypoint->3D associations
    n_obs = 0
    track_lengths = []
    obs_per_image = []

    # Build point -> observation count via per-point accumulation
    # For classical we can count via frames' point3d_idx
    point_obs_counts = {}
    for f in registered:
        cnt = int(np.sum(f.point3d_idx != -1))
        obs_per_image.append(cnt)
        n_obs += cnt
        for pid in f.point3d_idx:
            if pid != -1:
                pid = int(pid)
                point_obs_counts[pid] = point_obs_counts.get(pid, 0) + 1

    if point_obs_counts:
        track_lengths = list(point_obs_counts.values())
        mean_track = float(np.mean(track_lengths))
    else:
        mean_track = 0.0

    mean_obs_per_image = float(np.mean(obs_per_image)) if obs_per_image else 0.0

    # mean reprojection error: average over all observations
    mean_reproj = 0.0
    if K is not None and n_obs > 0 and points3D is not None and len(points3D) > 0:
        errors = []
        for f in registered:
            valid = f.point3d_idx != -1
            if not np.any(valid):
                continue
            pids = f.point3d_idx[valid]
            # filter pids that are within points3D range
            mask = pids < len(points3D)
            if not np.any(mask):
                continue
            pids = pids[mask]
            pts3d = points3D[pids]
            # keypoints
            kp_idx = np.where(valid)[0][mask]
            obs = np.array([f.kp[i].pt for i in kp_idx], dtype=np.float64)
            proj = _project(K, f.R, f.t, pts3d)
            err = np.linalg.norm(proj - obs, axis=1)
            errors.extend(err.tolist())
        if errors:
            mean_reproj = float(np.mean(errors))

    return {
        "num_points3D": int(n_points),
        "num_observations": int(n_obs),
        "mean_track_length": float(mean_track),
        "mean_observations_per_image": float(mean_obs_per_image),
        "mean_reprojection_error": float(mean_reproj),
    }


def compute_feature_metrics(frames, matches_table=None, inliers_dict=None):
    """Compute feature/matching stats for M2 summary (classical side)."""
    if frames is None or len(frames) == 0:
        return {
            "keypoints_min": 0,
            "keypoints_max": 0,
            "keypoints_mean": 0.0,
            "matches_mean": 0.0,
            "inliers_mean": 0.0,
            "num_matches_pairs": 0,
            "num_inliers_pairs": 0,
        }
    kps = [len(f.kp) for f in frames]
    kp_min = int(min(kps)) if kps else 0
    kp_max = int(max(kps)) if kps else 0
    kp_mean = float(np.mean(kps)) if kps else 0.0

    matches_mean = 0.0
    num_matches_pairs = 0
    if matches_table is not None and len(matches_table) > 0:
        vals = list(matches_table.values())
        # matches_table stores list of DMatch
        counts = [len(v) for v in vals]
        if counts:
            matches_mean = float(np.mean(counts))
            num_matches_pairs = len(counts)
    elif inliers_dict is not None and len(inliers_dict) > 0:
        # fallback if only inliers available
        pass

    inliers_mean = 0.0
    num_inliers_pairs = 0
    if inliers_dict is not None and len(inliers_dict) > 0:
        counts = list(inliers_dict.values())
        if counts:
            inliers_mean = float(np.mean(counts))
            num_inliers_pairs = len(counts)

    return {
        "keypoints_min": kp_min,
        "keypoints_max": kp_max,
        "keypoints_mean": kp_mean,
        "matches_mean": matches_mean,
        "inliers_mean": inliers_mean,
        "num_matches_pairs": num_matches_pairs,
        "num_inliers_pairs": num_inliers_pairs,
    }


def print_m2_summary(
    n_images,
    n_registered,
    feature_metrics,
    structure_metrics,
    timings=None,
):
    """Print M2 metrics summary with IDENTICAL format to deep pipeline.

    This function is duplicated in deep_learning_pipeline/evaluate.py (Option A).
    Headers, field order, and numeric precision must stay in sync.
    """
    fm = feature_metrics
    sm = structure_metrics
    print("\n=== M2 metrics summary ===")
    print(f"Images: {n_images}   Registered: {n_registered}/{n_images}")
    print(
        f"Keypoints per image: min={fm['keypoints_min']} max={fm['keypoints_max']} mean={fm['keypoints_mean']:.1f}"
    )
    print(f"Matched pairs (pre-RANSAC), mean per pair: {fm['matches_mean']:.1f}")
    print(f"RANSAC inliers, mean per verified pair: {fm['inliers_mean']:.1f}")
    print(f"Reconstructed 3D points: {sm['num_points3D']}")
    print(f"Num observations: {sm['num_observations']}")
    print(f"Mean track length: {sm['mean_track_length']:.2f}")
    print(f"Mean observations per image: {sm['mean_observations_per_image']:.2f}")
    print(f"Mean reprojection error: {sm['mean_reprojection_error']:.3f} px")
    if timings:
        for stage, secs in timings.items():
            print(f"Time [{stage}]: {secs:.2f}s")
    print("===========================\n")


def evaluate_reconstruction(
    frames,
    registered,
    points3D,
    K,
    gt_poses,
    matches_table=None,
    inliers_dict=None,
    timings=None,
):
    """Unified entry point: prints M2 summary + pose accuracy with identical contract.

    Kept separated from deep pipeline's equivalent function (Option A).
    """
    n_images = len(frames) if frames is not None else len(registered)
    n_registered = len(registered)

    feature_metrics = compute_feature_metrics(frames, matches_table, inliers_dict)
    structure_metrics = compute_structure_metrics(registered, points3D, K)

    print_m2_summary(n_images, n_registered, feature_metrics, structure_metrics, timings)

    pose_result = None
    if gt_poses is not None:
        pose_result = evaluate_poses(registered, gt_poses)
    else:
        print("[evaluate] no ground truth provided — skipping pose evaluation\n")

    return {
        "m2": {
            "n_images": n_images,
            "n_registered": n_registered,
            **feature_metrics,
            **structure_metrics,
            "timings": timings or {},
        },
        "pose": pose_result,
    }
