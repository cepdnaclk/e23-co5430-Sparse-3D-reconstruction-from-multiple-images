# Structure From Motion

CO543/CO5430 Computer Vision Project, Group 13.
Sparse 3D reconstruction from multiple images

## Problem statement

Given a set of overlapping images of a scene, recover the camera pose for
each image and a sparse 3D point cloud of the scene, using classical
feature based Structure from Motion. See `docs/UserGuide.md` for the full
pipeline explanation and `datasets/templeRing/README.txt` for dataset details.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

This repo does not commit dataset images (see `.gitignore` ).
Download TempleRing (Middlebury Multi-View Stereo dataset) yourself and place
the `.png` files in `datasets/templeRing/images/`. The calibration files
(`templeR_par.txt`, `templeR_ang.txt`, `README.txt`) are already included.

## Running

```bash
python run.py                          # ORB baseline (default), TempleRing
python run.py --feature_type sift      # SIFT variant
python run.py --out outputs/orb_run1   # custom output location
```

Outputs land in `outputs/<feature_type>/`:
- `points3D.ply`, `points3D_open3d.ply` — sparse point cloud
- `cameras.json` — recovered camera poses + intrinsics
- `sparse/0/{cameras,images,points3D}.txt` — COLMAP format (for Gaussian
  Splatting later)
- Console output includes the metrics your proposal commits to (keypoint
  counts, match counts, RANSAC inlier counts, point count, per-stage timing)
  plus pose accuracy against TempleRing's ground-truth calibration

## Project structure

```
Root/
├── run.py              # main entry point
├── sfm_baseline.py      # SfM pipeline (feature detection -> matching ->
│                          RANSAC -> incremental registration -> BA -> export)
├── evaluate.py          # ground-truth pose loading + accuracy scoring
├── config.py            # parameters
├── requirements.txt
├── datasets/templeRing/ # calibration + image folder (images gitignored)
├── outputs/             # per-run results (gitignored)
└── docs/UserGuide.md    # detailed pipeline + troubleshooting guide
```