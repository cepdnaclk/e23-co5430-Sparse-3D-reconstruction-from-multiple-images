# Structure From Motion

CO543/CO5430 Computer Vision Project, Group 13.
Sparse 3D reconstruction from multiple images.

## Team

| Name | E-Number | Email |
|---|---|---|
| K.L.D.H. Liyanagama | E/23/202 | e23202@eng.pdn.ac.lk |
| W.G.R.P. Gamage | E/23/108 | e23108@eng.pdn.ac.lk |
| C.M.H.K. Chandrasekara | E/23/043 | e23043@eng.pdn.ac.lk |

## Problem statement

Given a set of overlapping images of a scene, recover the camera pose for
each image and a sparse 3D point cloud of the scene, using Structure from
Motion. This repo ships two independent pipelines that solve the same task
and can be compared side-by-side:

- **Classical feature-based SfM** (`run.py`, `sfm/`) — ORB (baseline) or
  SIFT (improved classical), brute-force matching, RANSAC geometric
  verification, incremental reconstruction, and sparse bundle adjustment.
- **Deep-learning-based SfM** (`deep_learning_pipeline/`) — SuperPoint
  features, LightGlue matching, and incremental reconstruction + bundle
  adjustment via `pycolmap` (through `hloc`).

Both pipelines output a sparse point cloud and camera poses, and both print
an identical M2/pose evaluation summary so the two methods can be compared
directly. See `docs/UserGuide.md` for the detailed pipeline explanation and
`datasets/templeRing/README.txt` for dataset details.

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

Render the point cloud plus recovered camera frustums to PNGs (a turntable of
views) using Open3D's offscreen renderer. Works on headless machines and
Wayland+NVIDIA setups where the interactive windowed viewer fails:

```bash
python view.py outputs/sift
```

Writes `points3D_view_*.png` into the results folder (add `--angles 12` for
more views, `--out elsewhere` to redirect).

For interactive exploration, open the `.ply` in MeshLab:

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
├── run.py              # main entry point (wires classical pipeline + evaluation)
├── sfm_baseline.py      # Backward-compatible facade for the classical pipeline
├── sfm/                 # Modular classical SfM implementation
│   ├── image_io.py      # Image discovery and camera intrinsics
│   ├── features.py      # Feature detection and matching
│   ├── geometry.py      # Epipolar geometry and triangulation
│   ├── reconstruction.py # Initialization and incremental registration
│   ├── bundle_adjustment.py # Sparse joint pose/point refinement
│   ├── exporters.py     # PLY, camera JSON, and COLMAP output
│   ├── visualization.py # Open3D export and snapshot rendering
│   ├── reporting.py     # Runtime statistics and M2 metrics
│   ├── pipeline.py      # End-to-end orchestration
│   ├── cli.py           # Shared CLI entry point
│   └── __init__.py      # Public API re-export
├── evaluate.py          # ground-truth pose loading + M2/pose scoring
├── viz_matches.py       # keypoint + feature-match visualization tool
├── view.py              # offscreen 3D renderer (point cloud + camera poses)
├── config.py            # parameters
├── requirements.txt    # shared deps for classical + deep pipeline
├── data/index.json      # team data (single source of truth)
├── datasets/templeRing/ # calibration + image folder (images gitignored)
├── outputs/             # per-run results (gitignored)
├── deep_learning_pipeline/ # SuperPoint + LightGlue + pycolmap pipeline
│   ├── run_pipeline.py  # extract → match → incremental mapping → export
│   ├── config.py        # dataset/feature/matcher config
│   ├── export_results.py # pycolmap → PLY + poses.txt
│   ├── visualize.py     # Open3D viewer for the sparse model
│   ├── evaluate.py      # M2 + pose evaluation (duplicated contract)
│   └── requirements.txt # extra deps only needed here
├── Hierarchical-Localization/ # hloc (SuperPoint+LightGlue → COLMAP glue)
│   └── install as editable: `cd Hierarchical-Localization && pip install -e .`
├── gaussian_splatting/  # 3D Gaussian Splatting on COLMAP outputs
│   ├── README.md        # dataset prep + Colab/server training guide
│   ├── prepare_dataset.py # SfM output → gsplat COLMAP layout
│   ├── export_ply.py    # gsplat checkpoint → 3DGS PLY
│   ├── train_server.sh  # idempotent GPU-server training script
│   ├── gsplat_colab.ipynb # Colab walkthrough
│   └── data/            # prepared temple/ dataset (gitignored)
└── docs/UserGuide.md    # detailed pipeline + troubleshooting guide
```

## Further reading

- `docs/UserGuide.md` — full pipeline explanation, reading the ground-truth
  evaluation, and troubleshooting
- `deep_learning_pipeline/README.md` — SuperPoint + LightGlue + pycolmap pipeline
- `gaussian_splatting/README.md` — 3D Gaussian Splatting on the SfM outputs
- `datasets/templeRing/README.txt` — dataset details
