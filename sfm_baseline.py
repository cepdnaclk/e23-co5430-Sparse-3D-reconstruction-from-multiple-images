"""
Classical Incremental Structure-from-Motion baseline.
CO543/CO5430 Computer Vision — Group project M2 (classical baseline).

Pipeline:
  1. SIFT feature detection per image
  2. Pairwise matching (ratio test) + RANSAC-verified essential matrix
  3. Initial pair selection (most inliers + enough parallax)
  4. Initial two-view reconstruction (recoverPose + triangulation)
  5. Incremental registration of remaining images via 3D-3D PnP+RANSAC
  6. New point triangulation after each registration
  7. Bundle adjustment (scipy least_squares) minimizing reprojection error

Usage:
    python sfm_baseline.py --images /path/to/image_folder --out /path/to/output_dir \
        [--focal_mm 26 --sensor_width_mm 6.3]   # optional, for intrinsics estimate

If you don't know your camera's focal length / sensor width (common for phone
cameras), the script falls back to the standard "photographer's" approximation:
    fx = fy = max(width, height)
    cx, cy = image center
This is what most from-scratch SfM baselines and course projects use when no
calibration target (checkerboard) is available. For the "Expected" tier you
should ideally calibrate properly with cv2.calibrateCamera on a checkerboard
video/photos — see calibrate_checkerboard.py stub at the bottom of this file.
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


# Clsses


class Camera:
    def __init__(self, K):
        self.K = K.astype(np.float64)


class Frame:
    def __init__(self, idx, path, kp, desc, image_shape):
        self.idx = idx
        self.path = path
        self.kp = kp  # list[cv2.KeyPoint] keypoints array
        self.desc = desc  # np.ndarray [N, 128] descirptors array
        self.shape = image_shape  # (H, W)
        self.registered = False
        self.R = None  # world->camera rotation
        self.t = None  # world->camera translation

        # point3d_idx[i] = index into global points3D array if keypoint i has
        # a triangulated 3D point, else -1
        self.point3d_idx = -np.ones(len(kp), dtype=np.int64)


# Utility functions


# Get the paths of images
def load_images(folder):
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    paths = []
    for e in extensions:
        paths.extend(glob.glob(os.path.join(folder, e)))
    paths = sorted(paths)
    if len(paths) < 2:
        raise RuntimeError(f"Need >=2 images in {folder}, found {len(paths)}")
    return paths


# create the camera intrinsic matrix for the camera intrinsics, if not avilable, use the default approximation
# fx and fy are the focal lengths of the camera projection expressed in pixels
def estimate_intrinsics(shape, focal_mm=None, sensor_width_mm=None):
    h, w = shape[:2]  # Shape of the image
    # convert focal length from mm to pixels using the sensor width and image width
    if focal_mm is not None and sensor_width_mm is not None:
        fx = fy = (focal_mm / sensor_width_mm) * w
    else:
        # standard fallback approximation (no  calibration available)
        fx = fy = max(w, h)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K


# create the feature detector based on the feature type, either SIFT or ORB
def make_detector(feature_type, max_features=4000):
    feature_type = feature_type.lower()
    if feature_type == "sift":
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
    elif feature_type == "orb":
        # ORB gives binary descriptors -> must be matched with Hamming distance,
        # not L2. WTA_K=2 (default) pairs with NORM_HAMMING.
        return cv2.ORB_create(nfeatures=max_features, fastThreshold=7), cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unknown feature_type '{feature_type}', use 'sift' or 'orb'")


# detect features using opencv and the detector
def detect_features(paths, feature_type="sift", max_features=4000):
    detector, _ = make_detector(feature_type, max_features)
    frames = []
    for i, path in enumerate(paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Could not read image {path}")
        kp, desc = detector.detectAndCompute(img, None)
        frames.append(Frame(i, path, kp, desc, img.shape))
        RUN_STATS["n_keypoints"][i] = len(kp)
        print(
            f"[features:{feature_type}] {os.path.basename(path)}: {len(kp)} keypoints"
        )
    return frames


def match_pair(f1, f2, ratio=0.75, norm_type=cv2.NORM_L2):
    if f1.desc is None or f2.desc is None or len(f1.desc) < 8 or len(f2.desc) < 8:
        return []
    bf = cv2.BFMatcher(norm_type)
    knn = bf.knnMatch(f1.desc, f2.desc, k=2)

    # perform Lowe's ratio test and get the good matches
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def record_match_count(i, j, n):
    RUN_STATS["n_matches"][(i, j)] = n


RUN_STATS = {
    "n_keypoints": {},  # frame idx -> count
    "n_matches": {},  # (i,j) -> count before RANSAC
    "n_inliers": {},  # (i,j) -> count after RANSAC
    "timings": {},  # stage name -> seconds
}


def geometric_verify(f1, f2, matches, K, min_inliers=30):
    """RANSAC-verified essential matrix. Returns (inlier_matches, E, mask) or None."""
    if len(matches) < 8:
        return None
    pts1 = np.float32([f1.kp[m.queryIdx].pt for m in matches])
    pts2 = np.float32([f2.kp[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None or mask is None:
        return None
    mask = mask.ravel().astype(bool)
    inliers = [m for m, ok in zip(matches, mask) if ok]
    RUN_STATS["n_inliers"][(f1.idx, f2.idx)] = len(inliers)
    if len(inliers) < min_inliers:
        return None
    return inliers, E, mask


def pair_parallax_score(f1, f2, inliers):
    """Rough parallax proxy: median pixel displacement of inlier matches.
    Kept for reference -- no longer used to pick the initial pair, since it
    doesn't distinguish 'lots of pixel motion' from 'lots of pixel motion
    from a narrow, poorly-conditioned baseline'. See triangulation_angle_deg()
    below for what replaced it."""
    d = []
    for m in inliers:
        p1 = np.array(f1.kp[m.queryIdx].pt)
        p2 = np.array(f2.kp[m.trainIdx].pt)
        d.append(np.linalg.norm(p1 - p2))
    return np.median(d) if d else 0.0


def triangulation_angle_deg(pts3d, C1, C2):
    """Median angle (degrees), at each 3D point, between the ray back to
    camera 1's center and the ray back to camera 2's center -- the standard
    measure of how well-conditioned a stereo pair is for triangulation."""
    v1 = pts3d - C1
    v2 = pts3d - C2
    v1n = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9)
    v2n = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9)
    cos_ang = np.clip(np.sum(v1n * v2n, axis=1), -1, 1)
    return np.degrees(np.median(np.arccos(cos_ang)))


# --------------------------- Two-view initialization ------------------------ #


def triangulate_points(K, R1, t1, R2, t2, pts1, pts2):
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    return pts3d


def cheirality_mask(pts3d, R, t):
    """Points must be in front of the camera (positive depth)."""
    pts_cam = (R @ pts3d.T + t.reshape(3, 1)).T
    return pts_cam[:, 2] > 0


def initialize_reconstruction(
    frames, K, matches_table, min_inliers=60, min_triangulation_angle_deg=2.0
):
    """Pick the best initial pair and triangulate. A candidate must clear two
    bars: enough RANSAC inliers, AND a real median triangulation angle of at
    least min_triangulation_angle_deg -- this rejects narrow-baseline /
    near-duplicate pairs that can rack up huge inlier counts (because the
    images are nearly identical) while being numerically unstable to
    triangulate from. Among pairs clearing both bars, most inliers wins."""
    candidates = []  # (i, j, inliers, E, R, t, angle_deg)
    n = len(frames)
    for i in range(n):
        for j in range(i + 1, n):
            key = (i, j)
            if key not in matches_table:
                continue
            matches = matches_table[key]
            v = geometric_verify(frames[i], frames[j], matches, K, min_inliers)
            if v is None:
                continue
            inliers, E, mask = v

            pts1 = np.float32([frames[i].kp[m.queryIdx].pt for m in inliers])
            pts2 = np.float32([frames[j].kp[m.trainIdx].pt for m in inliers])
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
            t = t.ravel()

            pts3d = triangulate_points(K, np.eye(3), np.zeros(3), R, t, pts1, pts2)
            good = cheirality_mask(pts3d, np.eye(3), np.zeros(3)) & cheirality_mask(
                pts3d, R, t
            )
            if good.sum() < min_inliers:
                continue

            C1 = np.zeros(3)
            C2 = -R.T @ t
            angle = triangulation_angle_deg(pts3d[good], C1, C2)
            candidates.append((i, j, inliers, E, R, t, angle))

    if not candidates:
        raise RuntimeError(
            "Could not find a good initial pair. Try more overlap "
            "between images or lower min_inliers."
        )

    qualifying = [c for c in candidates if c[6] >= min_triangulation_angle_deg]
    pool = qualifying if qualifying else candidates
    if not qualifying:
        print(
            f"[init] WARNING: no candidate pair reached {min_triangulation_angle_deg}\u00b0 "
            f"triangulation angle -- falling back to the best available. "
            f"Reconstruction may still be poorly conditioned."
        )
    i, j, inliers, E, R, t, angle = max(pool, key=lambda c: len(c[2]))

    print(
        f"[init] chosen initial pair: {os.path.basename(frames[i].path)} / "
        f"{os.path.basename(frames[j].path)}  ({len(inliers)} inliers, "
        f"triangulation angle {angle:.2f}\u00b0)"
    )

    R1 = np.eye(3)
    t1 = np.zeros(3)
    frames[i].R, frames[i].t, frames[i].registered = R1, t1, True
    frames[j].R, frames[j].t, frames[j].registered = R, t, True

    pts1 = np.float32([frames[i].kp[m.queryIdx].pt for m in inliers])
    pts2 = np.float32([frames[j].kp[m.trainIdx].pt for m in inliers])
    pts3d = triangulate_points(K, R1, t1, R, t, pts1, pts2)
    good = cheirality_mask(pts3d, R1, t1) & cheirality_mask(pts3d, R, t)

    points3D = []
    for k, m in enumerate(inliers):
        if not good[k]:
            continue
        pid = len(points3D)
        points3D.append(pts3d[k])
        frames[i].point3d_idx[m.queryIdx] = pid
        frames[j].point3d_idx[m.trainIdx] = pid

    print(f"[init] triangulated {len(points3D)} initial points")
    return i, j, np.array(points3D, dtype=np.float64)


# ------------------------------- Incremental loop ---------------------------- #


def register_next_image(
    frame,
    registered_frames,
    matches_table,
    K,
    points3D,
    reproj_thresh=8.0,
    min_pnp_inliers=30,
):
    """Find 2D-3D correspondences between `frame` and already-registered frames
    (via feature matches), then solve PnP+RANSAC for its pose.
    min_pnp_inliers is a real reliability bar, not the mathematical minimum
    PnP needs (6) -- accepting a registration at 6-8 inliers means trusting a
    pose with essentially no redundancy, and bundle adjustment then treats
    that unreliable pose as equally trustworthy as a 1000-inlier one, which
    can drag the whole joint optimization off course."""
    obj_pts, img_pts, kp_indices = [], [], []
    seen_point_ids = set()

    for rf in registered_frames:
        i, j = (rf.idx, frame.idx) if rf.idx < frame.idx else (frame.idx, rf.idx)
        key = (i, j)
        if key not in matches_table:
            continue
        matches = matches_table[key]
        for m in matches:
            if rf.idx < frame.idx:
                rf_kp_idx, fr_kp_idx = m.queryIdx, m.trainIdx
            else:
                rf_kp_idx, fr_kp_idx = m.trainIdx, m.queryIdx
            pid = rf.point3d_idx[rf_kp_idx]
            if pid == -1 or pid in seen_point_ids:
                continue
            seen_point_ids.add(pid)
            obj_pts.append(points3D[pid])
            img_pts.append(frame.kp[fr_kp_idx].pt)
            kp_indices.append((fr_kp_idx, pid))

    if len(obj_pts) < 6:
        return None  # not enough correspondences yet

    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts,
        img_pts,
        K,
        None,
        reprojectionError=reproj_thresh,
        confidence=0.999,
        iterationsCount=2000,
    )
    if not ok or inliers is None or len(inliers) < min_pnp_inliers:
        return None

    R, _ = cv2.Rodrigues(rvec)
    inlier_set = set(inliers.ravel().tolist())
    for local_i, (fr_kp_idx, pid) in enumerate(kp_indices):
        if local_i in inlier_set:
            frame.point3d_idx[fr_kp_idx] = pid

    return R, tvec.ravel(), len(inliers)


def triangulate_new_points(frame, registered_frames, matches_table, K, points3D_list):
    """After registering `frame`, triangulate new points against each already
    registered neighbor for keypoints that don't yet have a 3D point."""
    new_count = 0
    for rf in registered_frames:
        if rf.idx == frame.idx:
            continue
        i, j = (rf.idx, frame.idx) if rf.idx < frame.idx else (frame.idx, rf.idx)
        key = (i, j)
        if key not in matches_table:
            continue
        matches = matches_table[key]

        pts_a, pts_b, idx_a, idx_b = [], [], [], []
        for m in matches:
            if rf.idx < frame.idx:
                a_idx, b_idx = m.queryIdx, m.trainIdx
            else:
                a_idx, b_idx = m.trainIdx, m.queryIdx
            if rf.point3d_idx[a_idx] != -1 or frame.point3d_idx[b_idx] != -1:
                continue  # already has a 3D point on either side
            pts_a.append(rf.kp[a_idx].pt)
            pts_b.append(frame.kp[b_idx].pt)
            idx_a.append(a_idx)
            idx_b.append(b_idx)

        if len(pts_a) < 8:
            continue
        pts_a = np.float32(pts_a)
        pts_b = np.float32(pts_b)
        pts3d = triangulate_points(K, rf.R, rf.t, frame.R, frame.t, pts_a, pts_b)
        good = cheirality_mask(pts3d, rf.R, rf.t) & cheirality_mask(
            pts3d, frame.R, frame.t
        )

        for k in range(len(pts3d)):
            if not good[k]:
                continue
            pid = len(points3D_list)
            points3D_list.append(pts3d[k])
            rf.point3d_idx[idx_a[k]] = pid
            frame.point3d_idx[idx_b[k]] = pid
            new_count += 1
    return new_count


def run_incremental_sfm(frames, matches_table, K, min_pnp_inliers=30):
    i0, j0, points3D = initialize_reconstruction(frames, K, matches_table)
    points3D = list(points3D)
    registered = [frames[i0], frames[j0]]
    remaining = [f for f in frames if f.idx not in (i0, j0)]

    progress = True
    while remaining and progress:
        progress = False
        # try to register the image with the most 2D-3D correspondences first
        best = None
        for f in remaining:
            res = register_next_image(
                f,
                registered,
                matches_table,
                K,
                np.array(points3D),
                min_pnp_inliers=min_pnp_inliers,
            )
            if res is None:
                continue
            R, t, n_inliers = res
            if best is None or n_inliers > best[1]:
                best = (f, n_inliers, R, t)
        if best is None:
            break
        f, n_inliers, R, t = best
        f.R, f.t, f.registered = R, t, True
        print(f"[register] {os.path.basename(f.path)}  ({n_inliers} PnP inliers)")
        new_pts = triangulate_new_points(f, registered, matches_table, K, points3D)
        print(f"[triangulate] +{new_pts} new points (total {len(points3D)})")
        registered.append(f)
        remaining.remove(f)
        progress = True

    if remaining:
        print(
            f"[warn] could not register {len(remaining)} image(s): "
            f"{[os.path.basename(f.path) for f in remaining]}"
        )

    return registered, np.array(points3D, dtype=np.float64)


# -------------------------------- Bundle adjustment --------------------------- #


def project(K, R, t, pts3d):
    pts_cam = (R @ pts3d.T + t.reshape(3, 1)).T
    proj = (K @ pts_cam.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj


def pack_params(registered, points3D):
    cam_params = []
    for f in registered:
        rvec, _ = cv2.Rodrigues(f.R)
        cam_params.append(np.concatenate([rvec.ravel(), f.t.ravel()]))
    cam_params = np.array(cam_params).ravel()
    return np.concatenate([cam_params, points3D.ravel()])


def unpack_params(x, n_cams, n_pts):
    cam_params = x[: n_cams * 6].reshape(n_cams, 6)
    points3D = x[n_cams * 6 :].reshape(n_pts, 3)
    return cam_params, points3D


def build_observations(registered, points3D):
    """Flatten all (camera_idx, point_idx, observed_2d) triples."""
    cam_idx, pt_idx, obs = [], [], []
    for ci, f in enumerate(registered):
        for kp_i, pid in enumerate(f.point3d_idx):
            if pid == -1:
                continue
            cam_idx.append(ci)
            pt_idx.append(pid)
            obs.append(f.kp[kp_i].pt)
    return np.array(cam_idx), np.array(pt_idx), np.array(obs, dtype=np.float64)


def ba_residuals(x, K, n_cams, n_pts, cam_idx, pt_idx, obs):
    cam_params, points3D = unpack_params(x, n_cams, n_pts)
    residuals = np.zeros((len(obs), 2))
    for ci in range(n_cams):
        mask = cam_idx == ci
        if not np.any(mask):
            continue
        rvec = cam_params[ci, :3]
        t = cam_params[ci, 3:]
        R, _ = cv2.Rodrigues(rvec)
        pids = pt_idx[mask]
        proj = project(K, R, t, points3D[pids])
        residuals[mask] = proj - obs[mask]
    return residuals.ravel()


def build_ba_sparsity(n_cams, n_pts, cam_idx, pt_idx):
    """Tell least_squares which Jacobian entries can be non-zero, so it uses
    sparse linear algebra internally instead of a dense (n_obs x n_params)
    matrix. Without this, BA OOMs on anything beyond a handful of points."""
    n_obs = len(cam_idx)
    m = n_obs * 2
    n = n_cams * 6 + n_pts * 3
    A = lil_matrix((m, n), dtype=int)
    obs_i = np.arange(n_obs)
    for s in range(6):
        A[2 * obs_i, cam_idx * 6 + s] = 1
        A[2 * obs_i + 1, cam_idx * 6 + s] = 1
    for s in range(3):
        A[2 * obs_i, n_cams * 6 + pt_idx * 3 + s] = 1
        A[2 * obs_i + 1, n_cams * 6 + pt_idx * 3 + s] = 1
    return A


def bundle_adjust(registered, points3D, K, verbose=True):
    n_cams, n_pts = len(registered), len(points3D)
    cam_idx, pt_idx, obs = build_observations(registered, points3D)
    x0 = pack_params(registered, points3D)

    if verbose:
        r0 = ba_residuals(x0, K, n_cams, n_pts, cam_idx, pt_idx, obs)
        print(
            f"[BA] initial mean reprojection error: "
            f"{np.sqrt((r0.reshape(-1, 2) ** 2).sum(1)).mean():.3f} px "
            f"over {len(obs)} observations"
        )

    sparsity = build_ba_sparsity(n_cams, n_pts, cam_idx, pt_idx)
    res = least_squares(
        ba_residuals,
        x0,
        jac_sparsity=sparsity,
        verbose=2 if verbose else 0,
        method="trf",
        args=(K, n_cams, n_pts, cam_idx, pt_idx, obs),
        max_nfev=100,
        xtol=1e-8,
        ftol=1e-8,
    )

    cam_params, points3D_opt = unpack_params(res.x, n_cams, n_pts)
    for ci, f in enumerate(registered):
        rvec = cam_params[ci, :3]
        f.R, _ = cv2.Rodrigues(rvec)
        f.t = cam_params[ci, 3:]

    if verbose:
        r1 = ba_residuals(res.x, K, n_cams, n_pts, cam_idx, pt_idx, obs)
        print(
            f"[BA] final mean reprojection error:   "
            f"{np.sqrt((r1.reshape(-1, 2) ** 2).sum(1)).mean():.3f} px"
        )

    return registered, points3D_opt


# ---------------------------------- Export ------------------------------------ #


def export_ply(path, points3D, colors=None):
    n = len(points3D)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points3D[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x} {y} {z}\n")


def export_cameras(path, registered, K):
    out = {"K": K.tolist(), "cameras": []}
    for f in registered:
        out["cameras"].append(
            {
                "image": os.path.basename(f.path),
                "R": f.R.tolist(),
                "t": f.t.tolist(),
            }
        )
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)


# ------------------------------------ Main ------------------------------------- #


def _render_worker(ply_path, png_path, cam_lines_json):
    """Runs in a subprocess so a headless-EGL segfault can't kill the main run."""
    import open3d as o3d
    import json

    pcd = o3d.io.read_point_cloud(ply_path)
    renderer = o3d.visualization.rendering.OffscreenRenderer(800, 600)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 4.0
    renderer.scene.add_geometry("pcd", pcd, mat)
    bounds = pcd.get_axis_aligned_bounding_box()
    center = np.asarray(bounds.get_center(), dtype=np.float32)
    extent = np.asarray(bounds.get_extent(), dtype=np.float32)
    diag = float(np.linalg.norm(extent)) or 1.0
    eye = (center + np.array([1, 1, 1], dtype=np.float32) * diag).astype(np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    renderer.setup_camera(60.0, center, eye, up)
    img = renderer.render_to_image()
    o3d.io.write_image(png_path, img)


def export_open3d(path, points3D, registered, K):
    """Save an Open3D point cloud (.ply) — this is the visualization your
    proposal commits to. Also attempts a rendered .png snapshot for slides,
    isolated in a subprocess since headless/EGL rendering can segfault on
    machines without a GPU display (harmless if it fails; the .ply is what
    matters and opens fine in Open3D locally or MeshLab)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)
    if len(points3D) > 0:
        pcd.paint_uniform_color([0.2, 0.6, 0.9])
    o3d.io.write_point_cloud(path, pcd)
    print(f"[open3d] wrote {path}")

    try:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        p = ctx.Process(
            target=_render_worker, args=(path, path.replace(".ply", ".png"), None)
        )
        p.start()
        p.join(timeout=30)
        if p.exitcode == 0:
            print(f"[open3d] wrote rendered snapshot {path.replace('.ply', '.png')}")
        else:
            print(
                "[open3d] snapshot render unavailable in this environment "
                "(no GPU display) — open the .ply locally in Open3D instead"
            )
    except Exception as e:
        print(f"[open3d] snapshot render skipped: {e}")


def print_metrics_summary(frames, matches_table, points3D, registered, timings):
    """Prints the exact quantitative metrics your proposal commits to."""
    print("\n=== M2 metrics summary ===")
    print(f"Images: {len(frames)}   Registered: {len(registered)}/{len(frames)}")
    print(
        f"Keypoints per image: "
        f"min={min(RUN_STATS['n_keypoints'].values())} "
        f"max={max(RUN_STATS['n_keypoints'].values())} "
        f"mean={np.mean(list(RUN_STATS['n_keypoints'].values())):.1f}"
    )
    if RUN_STATS["n_matches"]:
        print(
            f"Matched pairs (pre-RANSAC), mean per pair: "
            f"{np.mean(list(RUN_STATS['n_matches'].values())):.1f}"
        )
    if RUN_STATS["n_inliers"]:
        print(
            f"RANSAC inliers, mean per verified pair: "
            f"{np.mean(list(RUN_STATS['n_inliers'].values())):.1f}"
        )
    print(f"Reconstructed 3D points: {len(points3D)}")
    for stage, secs in timings.items():
        print(f"Time [{stage}]: {secs:.2f}s")
    print("===========================\n")


def export_colmap_text(out_dir, registered, points3D, K, copy_images_to=None):
    """Write cameras.txt / images.txt / points3D.txt in standard COLMAP text
    format. This is the format Gaussian Splatting trainers (gsplat, the
    original 3DGS repo) expect as input, laid out as:
        <out_dir>/sparse/0/{cameras.txt,images.txt,points3D.txt}
        <out_dir>/images/<original filenames>
    """
    from scipy.spatial.transform import Rotation

    os.makedirs(out_dir, exist_ok=True)
    frame_by_idx = {f.idx: f for f in registered}
    h, w = registered[0].shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    with open(os.path.join(out_dir, "cameras.txt"), "w") as fh:
        fh.write(
            "# Camera list, one line per camera: CAMERA_ID MODEL WIDTH HEIGHT PARAMS\n"
        )
        fh.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    tracks = {}
    for f in registered:
        for kp_i, pid in enumerate(f.point3d_idx):
            if pid == -1:
                continue
            tracks.setdefault(int(pid), []).append((f.idx + 1, kp_i))

    with open(os.path.join(out_dir, "images.txt"), "w") as fh:
        fh.write(
            "# Image list, two lines per image: "
            "IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME, then POINTS2D\n"
        )
        for f in registered:
            qx, qy, qz, qw = Rotation.from_matrix(f.R).as_quat()
            img_id = f.idx + 1
            name = os.path.basename(f.path)
            fh.write(
                f"{img_id} {qw} {qx} {qy} {qz} {f.t[0]} {f.t[1]} {f.t[2]} 1 {name}\n"
            )
            parts = []
            for kp_i, kp in enumerate(f.kp):
                pid = f.point3d_idx[kp_i]
                pid_str = str(int(pid) + 1) if pid != -1 else "-1"
                x, y = kp.pt
                parts.append(f"{x} {y} {pid_str}")
            fh.write(" ".join(parts) + "\n")

    color_cache = {}

    def sample_color(path, x, y):
        if path not in color_cache:
            color_cache[path] = cv2.imread(path, cv2.IMREAD_COLOR)
        img = color_cache[path]
        if img is None:
            return (128, 128, 128)
        xi = int(np.clip(round(x), 0, img.shape[1] - 1))
        yi = int(np.clip(round(y), 0, img.shape[0] - 1))
        b, g, r = img[yi, xi]
        return (int(r), int(g), int(b))

    with open(os.path.join(out_dir, "points3D.txt"), "w") as fh:
        fh.write("# 3D point list: POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for pid, pt in enumerate(points3D):
            obs = tracks.get(pid, [])
            if obs:
                img_id, kp_i = obs[0]
                f = frame_by_idx[img_id - 1]
                x, y = f.kp[kp_i].pt
                r, g, b = sample_color(f.path, x, y)
            else:
                r, g, b = 128, 128, 128
            track_str = " ".join(f"{img_id} {kp_i}" for img_id, kp_i in obs)
            fh.write(f"{pid + 1} {pt[0]} {pt[1]} {pt[2]} {r} {g} {b} 0.5 {track_str}\n")

    if copy_images_to:
        os.makedirs(copy_images_to, exist_ok=True)
        import shutil

        for f in registered:
            shutil.copy(f.path, os.path.join(copy_images_to, os.path.basename(f.path)))

    print(f"[colmap-export] wrote {out_dir}/{{cameras,images,points3D}}.txt")


def run_pipeline(
    images,
    out,
    feature_type="orb",
    focal_mm=None,
    sensor_width_mm=None,
    min_inliers_init=60,
    export_colmap=False,
    K_override=None,
    matching_strategy="exhaustive",
    match_window=6,
):
    """Runs the full pipeline and returns the key objects, so other scripts
    (e.g. run.py, evaluate.py) can use it as a library instead of a CLI.
    K_override: pass a 3x3 intrinsics matrix to skip the width/height
    approximation entirely -- use this whenever the dataset ships ground
    truth calibration (e.g. TempleRing's templeR_par.txt).
    matching_strategy: "exhaustive" tests all O(n^2) pairs -- fine for small
    unordered sets, but on a *ring*-captured dataset with repetitive texture
    (e.g. a temple's repeated columns) it lets non-overlapping, opposite-side
    views produce spuriously "confident" matches that poison the initial
    pair and the whole reconstruction. "sequential" instead only matches each
    image against its `match_window` nearest neighbours in capture order,
    plus a wraparound for ring closure -- matches what you know about how the
    images were actually captured, and is dramatically less prone to this
    failure mode."""
    import time

    timings = {}

    os.makedirs(out, exist_ok=True)
    paths = load_images(images)
    print(f"[load] {len(paths)} images found")

    t0 = time.time()
    _, norm_type = make_detector(feature_type)
    frames = detect_features(paths, feature_type=feature_type)
    timings["feature_detection"] = time.time() - t0

    if K_override is not None:
        K = np.array(K_override, dtype=np.float64)
        print(f"[intrinsics] using supplied ground-truth K\n{K}")
    else:
        sample_shape = frames[0].shape
        K = estimate_intrinsics(sample_shape, focal_mm, sensor_width_mm)
        print(f"[intrinsics] estimated (no calibration supplied)\n{K}")

    t0 = time.time()
    matches_table = {}
    n = len(frames)
    if matching_strategy == "sequential":
        pairs = set()
        for i in range(n):
            for d in range(1, match_window + 1):
                j = (i + d) % n
                pairs.add((min(i, j), max(i, j)))
        print(
            f"[match] sequential strategy: {len(pairs)} pairs "
            f"(vs {n * (n - 1) // 2} for exhaustive)"
        )
        for i, j in sorted(pairs):
            m = match_pair(frames[i], frames[j], norm_type=norm_type)
            record_match_count(i, j, len(m))
            if len(m) >= 20:
                matches_table[(i, j)] = m
    else:
        for i in range(n):
            for j in range(i + 1, n):
                m = match_pair(frames[i], frames[j], norm_type=norm_type)
                record_match_count(i, j, len(m))
                if len(m) >= 20:
                    matches_table[(i, j)] = m
            print(f"[match] frame {i} vs rest done")
    timings["matching"] = time.time() - t0

    t0 = time.time()
    registered, points3D = run_incremental_sfm(frames, matches_table, K)
    timings["incremental_sfm"] = time.time() - t0
    print(
        f"[sfm] registered {len(registered)}/{n} images, "
        f"{len(points3D)} 3D points before BA"
    )

    t0 = time.time()
    registered, points3D = bundle_adjust(registered, points3D, K)
    timings["bundle_adjustment"] = time.time() - t0

    export_ply(os.path.join(out, "points3D.ply"), points3D)
    export_cameras(os.path.join(out, "cameras.json"), registered, K)
    if export_colmap:
        export_colmap_text(
            os.path.join(out, "sparse", "0"),
            registered,
            points3D,
            K,
            copy_images_to=os.path.join(out, "images"),
        )
    print_metrics_summary(frames, matches_table, points3D, registered, timings)
    export_open3d(os.path.join(out, "points3D_open3d.ply"), points3D, registered, K)
    print(f"[done] wrote outputs to {out}/")

    return {
        "frames": frames,
        "registered": registered,
        "points3D": points3D,
        "K": K,
        "timings": timings,
        "matches_table": matches_table,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder of input images")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument(
        "--feature_type",
        choices=["sift", "orb"],
        default="orb",
        help="orb = baseline, sift = improved classical",
    )
    ap.add_argument("--focal_mm", type=float, default=None)
    ap.add_argument("--sensor_width_mm", type=float, default=None)
    ap.add_argument("--min_inliers_init", type=int, default=60)
    ap.add_argument(
        "--export_colmap",
        action="store_true",
        help="Also export COLMAP-format sparse/ + images/, "
        "ready for Gaussian Splatting training",
    )
    args = ap.parse_args()
    run_pipeline(
        images=args.images,
        out=args.out,
        feature_type=args.feature_type,
        focal_mm=args.focal_mm,
        sensor_width_mm=args.sensor_width_mm,
        min_inliers_init=args.min_inliers_init,
        export_colmap=args.export_colmap,
    )


if __name__ == "__main__":
    main()
