# User Guide

This guide covers both pipelines in this repo. The classical pipeline
(`run.py`) and the deep-learning pipeline (`deep_learning_pipeline/`) solve
the same task — sparse 3D reconstruction + camera poses from the TempleRing
images — and both print an identical M2/pose evaluation summary so their
results can be compared directly.

## Which pipeline should I run first?

- **Start with the classical pipeline** (`run.py`). It has fewer dependencies
  and is the main entry point for the project.
- **Then run the deep-learning pipeline** (`deep_learning_pipeline/`) if you
  want a SuperPoint + LightGlue + pycolmap result to compare against.

## Classical pipeline overview (`run.py`)

`run.py` orchestrates everything using `config.py`'s defaults:

1. **Feature detection** (`sfm.detect_features`) — ORB or SIFT keypoints per image.
2. **Matching** — brute-force + ratio test (`sfm.match_pair`), correct
   distance metric picked automatically (Hamming for ORB, L2 for SIFT).
3. **Geometric verification** — RANSAC-verified Essential matrix per pair
   (`sfm.geometric_verify`).
4. **Incremental reconstruction** — pick the best initial pair, triangulate,
   then register each remaining image via PnP+RANSAC and triangulate new
   points (`sfm.run_pipeline`).
5. **Bundle adjustment** — refines all camera poses and 3D points jointly,
   with a sparse Jacobian so it scales past a handful of points
   (`sfm.bundle_adjust`).
6. **Evaluation against ground truth** — TempleRing ships exact camera
   calibration (`templeR_par.txt`). `evaluate.py` aligns your reconstruction
   onto it with a similarity transform (Umeyama: rotation + translation +
   scale) and reports position/rotation error per camera.

## Deep-learning pipeline overview (`deep_learning_pipeline/`)

```bash
cd deep_learning_pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # pycolmap + hloc + SuperPoint/LightGlue deps
python -m pipeline.run_pipeline
```

Steps:
1. Extract SuperPoint features for every image (`hloc.extract_features`).
2. Build an exhaustive pair list (`hloc.pairs_from_exhaustive`).
3. Match pairs with LightGlue (`hloc.match_features`).
4. Build a COLMAP database and run incremental mapping + bundle adjustment
   (`hloc.reconstruction`, backed by `pycolmap`).
5. Export the sparse point cloud to PLY and camera poses to a text file.

Re-run with `--skip-existing` to reuse cached features/matches from disk.

## Setup (classical + deep pipeline share one requirements file)

```bash
git clone <your-repo>
cd Temple-SfM
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# put TempleRing's .png files here:
#   datasets/templeRing/images/templeR0001.png ... templeR0047.png
```

The deep-learning pipeline depends on `hloc`. The repo ships a local copy in
`Hierarchical-Localization/`; install it as an editable module so
`deep_learning_pipeline` can import it:

```bash
cd Hierarchical-Localization && pip install -e .
```

This brings in `hloc`'s own requirements (including `lightglue` from
`cvg/LightGlue` and a newer `pycolmap>=3.13.0`). If you already installed
the repo's `requirements.txt` and want to avoid duplicate/pycolmap version
conflicts, install `hloc` into the same venv *after* the base deps.

## Running the classical pipeline

```bash
python run.py                     # ORB baseline
python run.py --feature_type sift # SIFT (improved classical)
```

Each run prints, in order: keypoints per image, matching progress, which
initial pair was chosen and why, per-image PnP registration, bundle
adjustment convergence, the final metrics summary (keypoints/matches/inliers/
points/timing), and finally pose accuracy against ground truth.

## Reading the ground truth evaluation

Both pipelines print the same sections, so the numbers are directly comparable:

```
=== M2 metrics summary ===
Images: 47   Registered: 47/47
Keypoints per image: min=... max=... mean=...
Matched pairs (pre-RANSAC), mean per pair: ...
RANSAC inliers, mean per verified pair: ...
Reconstructed 3D points: ...
Num observations: ...
Mean track length: ...
Mean observations per image: ...
Mean reprojection error: ... px
===========================

=== Pose accuracy vs. ground truth (TempleRing calibration) ===
Cameras compared: 47/47
Position error (post-alignment): mean=...  median=...  max=...
Rotation error, absolute/aligned (degrees): mean=...  median=...
Rotation error, pairwise-RELATIVE (degrees): mean=...  median=...  (cross-check, immune to alignment issues)
==================================================================
```

- **Cameras compared** should equal how many images got registered — if it's
  much lower, some images failed to register (see troubleshooting).
- **Position error** is in the same units as the bounding box in
  `datasets/templeRing/README.txt` (the object spans roughly 0.1 units across
  each axis) — so a mean error under ~0.01 is good; above ~0.05 suggests
  drift or a bad initial pair.
- **Rotation error** in degrees — under a few degrees is good for a scene
  this well-textured.
- **Pairwise relative rotation error** is a cross-check that cancels out any
  global misalignment; trust it when the absolute rotation error looks
  suspiciously large.

Run ORB, SIFT, and the SuperPoint+LightGlue pipeline and put these numbers
side by side — that's your core comparison table.

## Troubleshooting (classical pipeline)

| Symptom | Likely cause |
|---|---|
| "Could not find a good initial pair" | Check images are actually in `datasets/templeRing/images/` and named to match `templeR_par.txt` |
| Cameras compared << images registered | Filename mismatch between your images and `templeR_par.txt` entries — names must match exactly (case-sensitive) |
| High position/rotation error | Try `--feature_type sift` (denser, more distinctive matches than ORB); check `[sfm]` log for how many images actually got registered |
| Open3D snapshot render fails, prints an EGL error | Harmless — the `.ply` file is still written correctly; open it locally in Open3D, MeshLab, or CloudCompare. This can happen on headless machines/servers without a GPU display |
| Bundle adjustment is slow | Expected to take longer with more points; it uses a sparse Jacobian so it shouldn't OOM, but 47 images with dense matching can still take a few minutes |

## Troubleshooting (deep-learning pipeline)

| Symptom | Likely cause |
|---|---|
| `No images matching '*.png' found` | `pipeline/config.py` `DATASET_DIR` or `IMAGE_GLOB` doesn't point at the folder with `templeR0001.png` ... `templeR0047.png` |
| Reconstruction failed (hloc/pycolmap could not register enough images) | Not enough well-connected matches; check that features/matches were actually written and that a well-connected pair graph exists. With 47 images, exhaustive pairs are cheap — avoid switching to retrieval-based pairs unless the dataset is much larger |
| Feature/matching extraction is very slow | SuperPoint+LightGlue is much faster on a CUDA GPU; on CPU it still works but expect longer runtimes |
| `pycolmap` install fails | `pycolmap>=0.6.1` ships prebuilt wheels for Linux/macOS/Windows; if a wheel isn't available for your platform you may need to build COLMAP from source first |
| Pose evaluation skipped | `templeR_par.txt` wasn't found where the pipeline looks, or the image names in the model don't match `temple_par.txt` entries exactly (case-sensitive) |

## Stretch goal: Gaussian Splatting

Once you have a reconstruction you're happy with, the classical pipeline
(`run.py`, with `EXPORT_COLMAP=True`) and the deep-learning pipeline both
produce a COLMAP-format model under `outputs/<feature_type>/sparse/` (or
`outputs/lightglue/sparse/`) and copy the registered images alongside it —
the exact layout Gaussian Splatting trainers expect.

The repo includes a dedicated workflow for this in `gaussian_splatting/\
README.md`:

- `gaussian_splatting/prepare_dataset.py` converts an SfM output into the
  flat COLMAP layout `gsplat` expects, copies only the images referenced by
  the model, and prints a summary.
- Training is designed for Google Colab (free T4, 16 GB VRAM) via
  `gaussian_splatting/gsplat_colab.ipynb`, with an alternative
  `gaussian_splatting/train_server.sh` for an SSH-accessible GPU server.
- `gaussian_splatting/export_ply.py` converts a trained checkpoint to a
  standard 3DGS PLY you can open in MeshLab / CloudCompare / SuperSplat.

The natural stretch goal: train once per SfM method (ORB, SIFT,
SuperPoint+LightGlue) and compare downstream render quality — does the
upstream SfM method's accuracy affect the renderable reconstruction?

Needs an NVIDIA GPU with CUDA.