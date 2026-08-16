# Structure From Motion

**CO543/CO5430 Computer Vision — Group 13.**

Sparse 3D reconstruction from multiple images using classical, feature-based
Structure-from-Motion (SfM): detect keypoints, match them across views,
verify matches geometrically, and incrementally build a camera pose + sparse
point cloud reconstruction of the scene.

## Team

| Name | E-number | Email |
|---|---|---|
| K.L.D.H. Liyanagama | E/23/202 | e23202@eng.pdn.ac.lk |
| W.G.R.P. Gamage | E/23/108 | e23108@eng.pdn.ac.lk |
| C.M.H.K. Chandrasekara | E/23/043 | e23043@eng.pdn.ac.lk |

## Features

- **Feature detection** — SIFT or ORB keypoints per image
- **Matching** — brute-force matching + Lowe's ratio test, with the correct
  distance metric picked automatically (Hamming for ORB, L2 for SIFT)
- **Geometric verification** — RANSAC-verified Essential matrix per pair
- **Incremental reconstruction** — best initial pair, triangulation, then
  PnP+RANSAC registration of every remaining image with new-point
  triangulation
- **Bundle adjustment** — joint refinement of all poses and 3D points with a
  sparse Jacobian
- **Ground-truth evaluation** — pose accuracy (RRE/RTE) against TempleRing's
  real calibration
- **Exports** — PLY point clouds, camera JSON, and COLMAP-format sparse
  reconstruction (ready for Gaussian Splatting)
- **Visualization** — interactive 3D viewer and per-image/pairwise feature
  match images

## Repository layout

```
Root/
├── run.py              # main entry point (pipeline + evaluation + viz)
├── sfm_baseline.py     # the SfM pipeline itself (CLI or library)
├── evaluate.py         # ground-truth pose loading + accuracy scoring
├── config.py           # parameters (feature type, matching, intrinsics...)
├── viz_matches.py      # keypoint / feature-match visualization
├── view.py             # interactive 3D viewer for a run's output
├── requirements.txt
├── data/index.json     # team information
├── datasets/templeRing/# calibration + image folder (images gitignored)
├── outputs/            # per-run results (gitignored)
└── docs/UserGuide.md   # detailed pipeline + troubleshooting guide
```

## Pipeline overview

1. **Feature detection** — `sfm_baseline.detect_features` extracts ORB or
   SIFT keypoints + descriptors per image.
2. **Matching** — `sfm_baseline.match_pair` brute-force matches descriptors
   and keeps the unambiguous ones via Lowe's ratio test.
3. **Geometric verification** — `sfm_baseline.geometric_verify` fits a RANSAC
   essential matrix per pair and keeps the inlier correspondences.
4. **Incremental reconstruction** — `sfm_baseline.run_incremental_sfm` picks
   the best initial pair (inliers + triangulation angle), triangulates it,
   then registers every remaining image via PnP+RANSAC and triangulates new
   points against already-registered neighbors.
5. **Bundle adjustment** — `sfm_baseline.bundle_adjust` jointly optimizes all
   camera poses and 3D points to minimize reprojection error.
6. **Export + evaluate** — writes PLY/COLMAP outputs, prints the metrics
   summary, and scores pose accuracy against ground truth.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9+ with OpenCV, NumPy, SciPy, and Open3D.

## Dataset (TempleRing)

This repo does not commit dataset images (see `.gitignore`). Download the
TempleRing dataset (Middlebury Multi-View Stereo) yourself and place the
`.png` files in `datasets/templeRing/images/`:

- 47 views sampled on a ring around a plaster temple replica
- `templeR_par.txt` — exact per-image intrinsics + pose (K, R, t)
- `templeR_ang.txt` — latitude/longitude of each view
- `README.txt` — dataset details and coordinate conventions

If your images aren't named to match `templeR_par.txt`, pose evaluation is
skipped (names must match exactly, case-sensitive).

## Usage

### Main entry point

```bash
python run.py                          # ORB baseline (config default), TempleRing
python run.py --feature_type sift      # SIFT (improved classical)
python run.py --out outputs/orb_run1   # custom output folder per run
python run.py --save_matches           # also save keypoint + match visualizations
python run.py --save_matches --pairs "0,1 3,4"   # ...for specific pairs only
```

`run.py` flags:

| Flag | Default | Description |
|---|---|---|
| `--images` | `datasets/templeRing/images` | Input image folder |
| `--out` | `outputs/<feature_type>` | Output folder (derived from the actual feature type) |
| `--feature_type` | `orb` (from config) | `sift` or `orb` |
| `--gt_calibration` | `datasets/templeRing/templeR_par.txt` | Ground-truth calibration file |
| `--no_gt_intrinsics` | off | Ignore GT calibration, use the width/height intrinsics approximation |
| `--save_matches` | off | Write keypoint + match PNGs into `out/` |
| `--pairs` | all matched | Only visualize these pairs, e.g. `"0,1 3,4"` |

### Interactive 3D viewer

```bash
python view.py outputs/sift            # OR: outputs/orb
```

Opens a window with the sparse point cloud and camera frustums (ring
structure visible). Rotate/pan/zoom with the mouse.

If the Open3D window fails to open (common on Wayland sessions, e.g. the
`GLFW/GLEW` errors), `view.py` automatically falls back to a matplotlib 3D
plot. You can also choose the backend explicitly:

```bash
python view.py outputs/sift --backend matplotlib              # interactive plot
python view.py outputs/sift --backend matplotlib --save shot.png  # headless PNG
```

### Running the pipeline standalone

```bash
python sfm_baseline.py --images /path/to/images --out /path/to/out \
    --feature_type sift --export_colmap
```

## Outputs

Everything lands in `outputs/<feature_type>/`:

- `points3D.ply` — sparse point cloud (ASCII PLY)
- `points3D_open3d.ply` — same cloud saved via Open3D, plus
  `points3D_open3d.png` (rendered snapshot; may fail on headless machines —
  the `.ply` is unaffected)
- `cameras.json` — recovered intrinsics + per-image `R`, `t`
- `sparse/0/{cameras,images,points3D}.txt` — COLMAP-format reconstruction
  (for Gaussian Splatting later)
- `images/` — copies of the registered images (COLMAP layout)
- `keypoints/` — `keypoints_<i>.png`, detected keypoints per image
  (with `--save_matches`)
- `matches/` — pairwise feature matches (with `--save_matches`):
  - `<i>_<j>.png` — all ratio-test matches color-coded: **green = RANSAC
    inliers, red = outliers**
  - `inliers_<i>_<j>.png` — only the geometrically verified inliers

Console output includes the metrics summary (keypoints, matches, RANSAC
inliers, 3D points, per-stage timing) and pose accuracy vs ground truth.

## Configuration

All knobs live in `config.py`:

| Parameter | Default | Notes |
|---|---|---|
| `FEATURE_TYPE` | `"orb"` | `"orb"` (baseline) or `"sift"` (improved classical) |
| `MIN_INLIERS_INIT` | 60 | Min RANSAC inliers to accept the initial pair |
| `USE_GT_INTRINSICS` | True | TempleRing ships real calibration — use it |
| `EXPORT_COLMAP` | True | Write `sparse/` + `images/` for Gaussian Splatting |
| `MATCHING_STRATEGY` | `"sequential"` | See below |
| `MATCH_WINDOW` | 6 | Neighbours matched per image (sequential strategy) |

**Matching strategy:** TempleRing was captured in ring order and has
repetitive architecture (columns), so exhaustive all-pairs matching lets
opposite-side views produce false positives that corrupt the initial pair.
`"sequential"` only matches nearby-in-capture-order images (+ wraparound),
matching how the data was actually shot. Use `"exhaustive"` for genuinely
unordered photo sets.

## Evaluation

`evaluate.py` aligns your reconstruction onto TempleRing's ground truth with
a best-fit similarity transform (Umeyama: rotation + translation + scale),
then reports per-camera error:

```
=== Pose accuracy vs. ground truth (TempleRing calibration) ===
Cameras compared: 47/47
Position error (post-alignment): mean=0.0034  median=0.0029
Rotation error, absolute/aligned (degrees): mean=0.412  median=0.351
Rotation error, pairwise-RELATIVE (degrees): mean=0.401  median=0.382
```

- **Position error** — in ground-truth units (the object spans ~0.1 units);
  mean under ~0.01 is good, above ~0.05 suggests drift or a bad initial pair.
- **Rotation error (aligned)** — degrees after global alignment; under a few
  degrees is good.
- **Relative rotation error** — computed without any global alignment
  (cancels shared misalignment). If relative is low but absolute is high, the
  reconstruction itself is accurate but the registered cameras span too
  narrow an arc for reliable alignment — trust the relative number.

Run ORB vs SIFT (and later learned methods) and compare these numbers side
by side — that's your core comparison table.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "Could not find a good initial pair" | Images missing/misnamed in `datasets/templeRing/images/`, or too little overlap |
| Cameras compared << images registered | Filename mismatch with `templeR_par.txt` (must match exactly) |
| High position/rotation error | Try `--feature_type sift`; check how many images registered |
| Open3D snapshot fails with an EGL error | Harmless — `.ply` is still written; open it in Open3D/MeshLab/CloudCompare |
| `view.py` fails to open a window (GLFW/GLEW errors) | Wayland session; `view.py` auto-falls back to matplotlib, or run `python view.py outputs/orb --backend matplotlib --save out.png` |
| Bundle adjustment is slow | Expected with 47 images; it uses a sparse Jacobian so it won't OOM, but can take minutes |

See `docs/UserGuide.md` for the full deep dive and troubleshooting.

## Stretch goal: Gaussian Splatting

`run.py` already writes COLMAP-format `sparse/0/` + `images/` into
`outputs/<feature_type>/` — exactly what Gaussian Splatting trainers expect.
Using [`gsplat`](https://github.com/nerfstudio-project/gsplat):

```bash
pip install gsplat
python -m gsplat.examples.simple_trainer default \
    --data_dir outputs/orb \
    --result_dir outputs/orb_splats \
    --data_factor 1
```

(Check gsplat's docs for the current entry-point name — it has changed across
versions.) Needs an NVIDIA GPU with CUDA. Comparing render quality across SfM
methods tests whether upstream pose accuracy affects downstream rendering.

## Credits

TempleRing dataset (Middlebury Multi-View Stereo), captured by Steve Seitz,
James Diebel, Daniel Scharstein, Brian Curless, and Rick Szeliski. Built on
OpenCV, NumPy, SciPy, and Open3D.