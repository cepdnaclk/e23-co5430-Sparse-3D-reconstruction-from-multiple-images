# Structure From Motion

CO543/CO5430 Computer Vision Project, Group 13.
Sparse 3D reconstruction from multiple images.

## Team

| Name | E-Number | Email |
|---|---|---|
| K.L.D.H. Liyanagama | E/23/202 | e23202@eng.pdn.ac.lk |
| W.G.R.P. Gamage | E/23/108 | e23108@eng.pdn.ac.lk |
| C.M.H.K. Chandrasekara | E/23/043 | e23043@eng.pdn.ac.lk |

Team data also lives in `data/index.json` (single source of truth).

## Problem statement

Given a set of overlapping images of a scene, recover the camera pose for
each image and a sparse 3D point cloud of the scene, using classical
feature based Structure from Motion. See `docs/UserGuide.md` for the full
pipeline explanation and `datasets/templeRing/README.txt` for dataset details.

## Pipeline overview

`run.py` orchestrates the classical incremental SfM pipeline:

1. **Feature detection** — ORB (baseline) or SIFT (improved classical)
   keypoints + descriptors per image.
2. **Matching** — brute-force + Lowe's ratio test, with the correct distance
   metric chosen automatically (Hamming for ORB, L2 for SIFT). Pairs are
   matched sequentially with wraparound by default (see `config.py`).
3. **Geometric verification** — RANSAC-verified Essential matrix per pair.
4. **Incremental reconstruction** — pick the best initial pair (most inliers
   + minimum triangulation angle), triangulate, then register each remaining
   image via PnP+RANSAC and triangulate new points.
5. **Bundle adjustment** — jointly refines all camera poses and 3D points
   using a sparse Jacobian.
6. **Evaluation** — compares recovered poses against TempleRing's
   ground-truth calibration and reports position/rotation error.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

This repo does not commit dataset images (see `.gitignore`).
Download TempleRing (Middlebury Multi-View Stereo dataset) yourself and place
the `.png` files in `datasets/templeRing/images/`. The calibration files
(`templeR_par.txt`, `templeR_ang.txt`, `README.txt`) are already included.

## Quick start

```bash
python run.py                          # ORB baseline (default), TempleRing
python run.py --feature_type sift      # SIFT variant
python run.py --out outputs/my_run     # custom output location
```

Each run prints, in order: keypoints per image, matching progress, the chosen
initial pair, per-image PnP registration, bundle adjustment convergence, the
final metrics summary (keypoints/matches/inliers/points/timing), and pose
accuracy against ground truth.

## CLI reference

`run.py` flags (defaults come from `config.py`):

| Flag | Default | Description |
|---|---|---|
| `--feature_type` | `config.FEATURE_TYPE` (`orb`) | `sift` or `orb` |
| `--images` | `config.IMAGES_DIR` | Folder of input images |
| `--out` | `outputs/<feature_type>/` | Output folder (derives from `--feature_type` if not given) |
| `--gt_calibration` | `config.GT_CALIBRATION_PATH` | Ground-truth `*_par.txt` for intrinsics + pose scoring |
| `--no_gt_intrinsics` | off | Ignore GT intrinsics, use the width/height approximation |

## Outputs

Results land in `outputs/<feature_type>/`:

- `points3D.ply`, `points3D_open3d.ply` — sparse point cloud
- `cameras.json` — recovered camera poses + intrinsics
- `sparse/0/{cameras,images,points3D}.txt` — COLMAP format (for Gaussian
  Splatting later)
- `images/` — registered images copied for COLMAP/Gaussian Splatting

## Visualizing results

### 3D reconstruction

Interactive Open3D viewer showing the point cloud and recovered camera
frustums (rotate/pan/zoom with the mouse):

```bash
python view.py outputs/sift
```

MeshLab (if installed) also opens the point cloud directly:

```bash
meshlab outputs/sift/points3D.ply
```

The `.ply` is a sparse, unmeshed point cloud — use Point Splatting in
MeshLab to view it as dots.

### Keypoints & feature matching

Visualize detected keypoints and pairwise feature matches without running the
full pipeline (reuses the exact detection/matching logic). Only
`--feature_type` is required:

```bash
python viz_matches.py --feature_type sift
```

Writes to `outputs/<feature_type>_viz/`:

- `keypoints/keypoints_<i>.png` — keypoints drawn on each image (fixed small
  radius so large SIFT scales stay readable)
- `matches/<i>_<j>.png` — all raw matches in red with RANSAC inliers overlaid
  in green
- `matches/inliers_<i>_<j>.png` — only the RANSAC-verified inliers

Useful flags: `--pairs "0,1 1,2"` (only render specific pairs),
`--interactive` (pop each image up with cv2.imshow), `--matching_strategy`,
`--match_window`, `--max_features`, `--gt_calibration`, `--out`.

## Project structure

```
Root/
├── run.py              # main entry point (wires pipeline + evaluation)
├── sfm_baseline.py      # SfM pipeline (detection -> matching -> RANSAC ->
│                          incremental registration -> BA -> export)
├── evaluate.py          # ground-truth pose loading + accuracy scoring
├── viz_matches.py       # keypoint + feature-match visualization tool
├── view.py              # interactive 3D viewer (point cloud + camera poses)
├── config.py            # parameters
├── requirements.txt
├── data/index.json      # team data
├── datasets/templeRing/ # calibration + image folder (images gitignored)
├── outputs/             # per-run results (gitignored)
└── docs/UserGuide.md    # detailed pipeline + troubleshooting guide
```

## Further reading

- `docs/UserGuide.md` — full pipeline explanation, reading the ground-truth
  evaluation, and troubleshooting
- `datasets/templeRing/README.txt` — dataset details