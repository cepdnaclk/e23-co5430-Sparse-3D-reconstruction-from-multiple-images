# Deep-Learning SfM User Guide (SuperPoint + LightGlue + pycolmap)

Learned correspondences (SuperPoint features, LightGlue matching) with
`pycolmap` incremental mapping + bundle adjustment behind the scenes.
`hloc` (local checkout in `Hierarchical-Localization/`) handles the
extract → match → import glue; everything downstream is `pycolmap` directly.

Solves the same task as the classical pipeline so both print the identical
M2/pose evaluation summary for side-by-side comparison.

## 1. What it does

```
images/ → SuperPoint extract (features.h5) → exhaustive pairs
  → LightGlue match (matches.h5) → COLMAP database
  → pycolmap incremental mapping + BA → export PLY + poses.txt → evaluate
```

Code map:

| Step | Code |
|---|---|
| Config | `deep_learning_pipeline/config.py` (`DATASET_DIR`, `OUTPUT_DIR=outputs/lightglue`, `FEATURE_CONF=superpoint_max`, `MATCHER_CONF=superpoint+lightglue`, `CAMERA_MODEL=SIMPLE_RADIAL`) |
| Driver | `deep_learning_pipeline/run_pipeline.py` (`--dataset-dir`, `--output-dir`, `--skip-existing`) |
| Export | `deep_learning_pipeline/export_results.py` (`export_point_cloud()`, `export_poses()`, `summarize()`) |
| Evaluate | `deep_learning_pipeline/evaluate.py` (M2 + Umeyama pose scoring, same contract as `evaluate.py`) |
| Visualize | `deep_learning_pipeline/visualize.py` (`--sfm-dir`, `--frustum-scale`) |

Why hloc: importing LightGlue matches into a COLMAP database (feature import,
match import, two-view verification) is finicky; hloc is the reference
implementation from the LightGlue/SuperGlue group and is used in the LightGlue
paper's own benchmarks.

## 2. Prerequisites

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd Hierarchical-Localization && pip install -e . && cd ..
```

- Install hloc **after** the base requirements (it pulls `lightglue` from
  `cvg/LightGlue` and `pycolmap>=3.13.0`).
- Works on CPU; SuperPoint + LightGlue is much faster on a CUDA GPU (PyTorch
  falls back automatically).
- `Hierarchical-Localization/` is a third-party checkout — only `hloc` is used;
  its `d2-net` / `SuperGluePretrainedNetwork` / etc. subfolders are unused.

Dataset: same 47 TempleRing PNGs, default `DATASET_DIR = datasets/templeRing/images/`
(`IMAGE_GLOB = "*.png"`). Edit `config.py` if your path/extension differs.
Ground truth (`templeR_par.txt` in the parent folder) is only used for scoring,
never for intrinsics — COLMAP estimates `SIMPLE_RADIAL` from scratch
(`CameraMode.SINGLE`, one shared camera).

## 3. Running it

From the repo root:

```bash
python -m deep_learning_pipeline.run_pipeline
python -m deep_learning_pipeline.run_pipeline --skip-existing   # reuse features.h5/matches.h5
python -m deep_learning_pipeline.run_pipeline --dataset-dir /path/to/images --output-dir outputs/lightglue_test
```

Stages print as `[1/5]..[5/5]`: image list → SuperPoint extract →
exhaustive pairs (47 imgs → 1081 pairs, cheap) → LightGlue match →
`hloc.reconstruction.main()` incremental mapping + BA → export → unified
M2 + pose evaluation (auto-finds `templeR_par.txt` next to the dataset).

## 4. Outputs

```
outputs/lightglue/
  features.h5  matches.h5  pairs-exhaustive.txt  database.db
  sparse/            # binary COLMAP model (cameras/images/points3D.bin)
  points3D.ply      # colored sparse cloud (from export_point_cloud)
  poses.txt         # "image_name qw qx qy qz tx ty tz" per registered image
```

![SuperPoint + LightGlue sparse point cloud](assets/deep_pointcloud.png)

## 5. Inspecting the result

```bash
python -m deep_learning_pipeline.visualize --sfm-dir outputs/lightglue/sparse
python -m deep_learning_pipeline.visualize --sfm-dir outputs/lightglue/sparse --frustum-scale 0.08
```

Opens Open3D with the colored cloud + one small red wireframe frustum per
registered camera (`camera_frustum()` uses the true `calibration_matrix()` and
`projection_center()`).

## 6. Evaluating against ground truth

`run_pipeline` already prints M2 + pose if it finds `templeR_par.txt`.
To re-score manually (same contract as the classical `evaluate.py`):

```bash
python -m deep_learning_pipeline.evaluate --gt datasets/templeRing/templeR_par.txt --poses outputs/lightglue/poses.txt
python -m deep_learning_pipeline.evaluate --gt datasets/templeRing/templeR_par.txt --sparse outputs/lightglue/sparse
```

Monocular SfM is up-to-scale, so poses are aligned with a Umeyama similarity
fit before errors are reported. Compare the `M2 metrics summary` +
`Pose accuracy vs. ground truth` blocks directly against
`python run.py --feature_type sift` output.

## 7. Design choices

- `superpoint_max` (no resize cap, high keypoint budget) over
  `superpoint_aachen` (1024px cap): TempleRing is only ~640×480, denser
  keypoints help triangulation.
- Exhaustive pairs: fine at 47 images; for larger sets swap
  `pairs_from_exhaustive` for retrieval pairs (e.g. NetVLAD).
- No masks/silhouettes: background features are kept and left for RANSAC/COLMAP
  to reject. Mask first if your own scenes are cluttered.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `No images matching '*.png' found` | `config.py` `DATASET_DIR`/`IMAGE_GLOB` wrong; check folder + extension |
| `Reconstruction failed ... not enough images` | matches too sparse/disconnected; confirm `features.h5`/`matches.h5` written; keep exhaustive pairs at this scale |
| Very slow extract/match | expected on CPU; use CUDA GPU; `--skip-existing` to avoid recompute |
| `pycolmap` install fails | needs a prebuilt wheel for your platform, else build COLMAP from source |
| Pose evaluation skipped | GT `templeR_par.txt` not found or image basenames differ (case-sensitive) |
