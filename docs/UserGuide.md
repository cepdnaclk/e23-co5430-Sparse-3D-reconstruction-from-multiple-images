# User Guide

## Pipeline overview

`run.py` orchestrates everything using `config.py`'s defaults:

1. **Feature detection** (`sfm_baseline.detect_features`) — ORB or SIFT keypoints per image.
2. **Matching** — brute-force + ratio test (`sfm_baseline.match_pair`), correct
   distance metric picked automatically (Hamming for ORB, L2 for SIFT).
3. **Geometric verification** — RANSAC-verified Essential matrix per pair
   (`sfm_baseline.geometric_verify`).
4. **Incremental reconstruction** — pick the best initial pair, triangulate,
   then register each remaining image via PnP+RANSAC and triangulate new
   points (`sfm_baseline.run_incremental_sfm`).
5. **Bundle adjustment** — refines all camera poses and 3D points jointly,
   with a sparse Jacobian so it scales past a handful of points
   (`sfm_baseline.bundle_adjust`).
6. **Evaluation against ground truth** — TempleRing ships exact camera
   calibration (`templeR_par.txt`). `evaluate.py` aligns your reconstruction
   onto it with a similarity transform (Umeyama: rotation + translation +
   scale) and reports position/rotation error per camera — this is your
   project's version of the pose accuracy metrics used in the VGGSfM-style
   evaluation literature.

## Running locally

```bash
git clone <your-repo>
cd Temple-SfM
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# put TempleRing's .png files here:
#   datasets/templeRing/images/templeR0001.png ... templeR0047.png

python run.py                     # ORB baseline
python run.py --feature_type sift # SIFT (improved classical)
```

Each run prints, in order: keypoints per image, matching progress, which
initial pair was chosen and why, per-image PnP registration, bundle
adjustment convergence, the final metrics summary (keypoints/matches/inliers/
points/timing), and finally pose accuracy against ground truth.

## Reading the ground truth evaluation

```
=== Pose accuracy vs. ground truth (TempleRing calibration) ===
Cameras compared: 47/47
Position error (post-alignment): mean=0.0034  median=0.0029
Rotation error (degrees):        mean=0.412  median=0.351
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

Run all three methods (ORB, SIFT, and later SuperPoint+SuperGlue) and put
these numbers side by side — that's your core M3 comparison table.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "Could not find a good initial pair" | Check images are actually in `datasets/templeRing/images/` and named to match `templeR_par.txt` |
| Cameras compared << images registered | Filename mismatch between your images and `templeR_par.txt` entries — names must match exactly (case-sensitive) |
| High position/rotation error | Try `--feature_type sift` (denser, more distinctive matches than ORB); check `[sfm]` log for how many images actually got registered |
| Open3D snapshot render fails, prints an EGL error | Harmless — the `.ply` file is still written correctly; open it locally in Open3D, MeshLab, or CloudCompare. This can happen on headless machines/servers without a GPU display |
| Bundle adjustment is slow | Expected to take longer with more points; it uses a sparse Jacobian (`build_ba_sparsity`) so it shouldn't OOM, but 47 images with dense matching can still take a few minutes |

## Stretch goal: Gaussian Splatting

Once you have a reconstruction you're happy with, `run.py` already writes
COLMAP-format output to `outputs/<feature_type>/sparse/0/` and copies the
registered images to `outputs/<feature_type>/images/` — exactly the layout
Gaussian Splatting trainers expect.

Using [`gsplat`](https://github.com/nerfstudio-project/gsplat) (lighter
install than the original 3DGS repo):

```bash
pip install gsplat

# gsplat's example trainer expects a COLMAP-style project directory:
#   <data_dir>/images/
#   <data_dir>/sparse/0/{cameras.txt,images.txt,points3D.txt}
# which is exactly outputs/<feature_type>/ already produced above.

python -m gsplat.examples.simple_trainer default \
    --data_dir outputs/orb \
    --result_dir outputs/orb_splats \
    --data_factor 1
```
(Check `gsplat`'s own docs/CLI for the exact current entry point name — the
package has changed its example script layout across versions.)

This trains a set of 3D Gaussians initialized from your sparse point cloud,
and will periodically render held-out views for a PSNR/SSIM comparison.
Running this once per SfM method (ORB, SIFT, SuperGlue) and comparing final
render quality is the natural stretch-goal extension: *does the upstream
SfM method's accuracy affect the downstream renderable reconstruction?*

Needs an NVIDIA GPU with CUDA.