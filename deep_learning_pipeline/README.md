# Deep-learning SfM: SuperPoint + LightGlue + pycolmap

This folder is the **deep-learning pipeline** for the Temple Ring dataset.
It reconstructs camera poses and a sparse 3D point cloud using SuperPoint
features, LightGlue matching, and incremental reconstruction + bundle
adjustment via `pycolmap` (through `hloc`).

It solves the same task as the classical pipeline in the repo root
(`run.py` with ORB or SIFT) so the two can be compared side-by-side using
the identical M2/pose evaluation summary both print.

## Why hloc instead of a from-scratch OpenCV pipeline

We use `pycolmap` as the SfM backend. Getting LightGlue's matches into a
COLMAP database correctly (feature import, match import, two-view geometry
verification) is finicky and easy to get subtly wrong. `hloc` is the
reference implementation for exactly this combination (from the LightGlue /
SuperGlue authors' own group) and is used in the LightGlue paper's own
localization benchmarks, so we build on it rather than reimplementing it.
Everything downstream (mapping, bundle adjustment, model access) is
`pycolmap` directly — `hloc` only handles the extract/match/import step.

## 1. Setup

From the repo root:

```bash
python -m venv venv && source venv/bin/activate   # or use conda
pip install -r requirements.txt
```

The deep-learning pipeline depends on `hloc`, the reference
SuperPoint+LightGlue-to-COLMAP glue code. The repo's copy lives in
`Hierarchical-Localization/` and should be installed as an editable module
so the pipeline can import it:

```bash
# from the repo root, after the venv is active:
cd Hierarchical-Localization && pip install -e .
```

`setup.py` pulls in `hloc`'s own `requirements.txt`, which includes
`lightglue` (installed directly from the `cvg/LightGlue` repo) and a newer
`pycolmap>=3.13.0`. If you already installed the repo's
`requirements.txt` (`pycolmap>=0.6.1`, no `lightglue`), the editable install
will bring the deep pipeline's exact deps into the same venv.

Notes:
- Works on CPU, but SuperPoint+LightGlue extraction/matching will be much
  faster with a CUDA GPU. Nothing in the code assumes a GPU; PyTorch will
  fall back to CPU automatically.
- `pycolmap` ships prebuilt wheels for Linux/macOS/Windows, so you don't
  need a separate COLMAP install.
- The `Hierarchical-Localization` folder is a git submodule-like third-party
  checkout (it has its own `.gitmodules` with `d2-net`,
  `SuperGluePretrainedNetwork`, `deep-image-retrieval`, `r2d2`). Only
  `hloc` is needed for this pipeline; the other third-party folders are
  unused unless you run the repo's own Aachen/InLoc pipelines.

## 2. Get the dataset onto disk

Download the Temple Ring set (47 images, `templeR0001.png` ...
`templeR0047.png`) from the Middlebury Multi-View Stereo page and place the
images in a folder, e.g.:

```
datasets/templeRing/templeR0001.png
datasets/templeRing/templeR0002.png
...
datasets/templeRing/temple_par.txt   # optional, ground-truth calibration
```

The default `deep_learning_pipeline/config.py` already points at
`datasets/templeRing/images/`. Update it if your path or file extension
differs (`DATASET_DIR`, `IMAGE_GLOB`).

## 3. Run the pipeline

From the repo root:

```bash
python -m deep_learning_pipeline.run_pipeline
```

Or from inside the folder:

```bash
cd deep_learning_pipeline
python -m pipeline.run_pipeline
```

This will, in order:
1. Extract SuperPoint keypoints/descriptors for every image
   (`superpoint_max` config — no resize cap, suited to a small, detailed
   dataset).
2. Build an exhaustive pair list (47 images → 1081 pairs is cheap; fine for
   this dataset size).
3. Match every pair with LightGlue.
4. Build a COLMAP database, import the features/matches, run
   `pycolmap.incremental_mapping` (incremental SfM: initial pair, PnP
   registration of remaining images, triangulation, and bundle adjustment
   after each step plus a final global BA).
5. Export `points3D.ply` (sparse cloud) and `poses.txt` (per-image camera
   pose) into `outputs/lightglue/`. The full binary COLMAP model
   (`cameras.bin`, `images.bin`, `points3D.bin`) is left in
   `outputs/lightglue/sparse/` if you want to open it in COLMAP's GUI or
   load it with `pycolmap.Reconstruction(...)` for further analysis.

Re-running with `--skip-existing` reuses cached features/matches from disk
instead of recomputing them (handy if you only want to re-run mapping with
different options).

## 4. Inspect the result

```bash
python -m deep_learning_pipeline.visualize --sfm-dir outputs/lightglue/sparse
```

Opens an Open3D window with the colored sparse point cloud and a small red
wireframe frustum per registered camera.

## 5. (Optional) Evaluate against ground truth

Temple Ring ships known camera calibration in `temple_par.txt`. Since
COLMAP reconstructs up to an arbitrary similarity transform (unknown scale/
rotation/translation — monocular SfM cannot recover absolute scale),
`deep_learning_pipeline/evaluate.py` aligns the recovered camera poses to the
ground truth with a Umeyama similarity fit and reports the residual error:

```bash
python -m deep_learning_pipeline.evaluate \
    --gt datasets/templeRing/templeR_par.txt \
    --poses outputs/lightglue/poses.txt
```

You can also point it at the sparse model directly; it will read poses and
rotation from the `pycolmap.Reconstruction`:

```bash
python -m deep_learning_pipeline.evaluate \
    --gt datasets/templeRing/templeR_par.txt \
    --sparse outputs/lightglue/sparse
```

## Comparing the two pipelines

Both the classical pipeline (`run.py`) and this deep-learning pipeline print
an identical M2 + pose evaluation summary. To compare them:

```bash
# classical
python run.py --feature_type sift

# deep-learning
python -m deep_learning_pipeline.run_pipeline
```

Then put the `M2 metrics summary` and `Pose accuracy vs. ground truth`
sections side by side. The deep-learning pipeline's output lands in
`outputs/lightglue/`; the classical one lands in `outputs/sift/` or
`outputs/orb/` depending on the feature type.

## Assumptions made

- **Camera model**: one shared `SIMPLE_RADIAL` camera for all images
  (`camera_mode="SINGLE"` in `run_pipeline.py`), since Temple Ring was shot
  with a single physical camera. Intrinsics are estimated by COLMAP, not
  read from `temple_par.txt` — this keeps the pipeline dataset-agnostic; use
  `evaluate.py` if you want a metric accuracy check against the known
  calibration.
- **Matching strategy**: exhaustive pairs (all `n*(n-1)/2` pairs), which is
  fine at 47 images. For larger datasets you'd swap
  `pairs_from_exhaustive` for retrieval-based pairs (e.g. NetVLAD via
  `hloc.pairs_from_retrieval`) to keep matching sub-quadratic.
- **Feature config**: `superpoint_max` (no resize limit, high keypoint
  budget) rather than `superpoint_aachen` (which downsizes to 1024px) —
  Temple Ring images are small (~640x480) so there's no need to cap
  resolution, and denser keypoints help triangulation on a mostly-textured
  object.
- No ground-truth silhouettes/masks are used to restrict features to the
  object; background features are included and left for COLMAP/RANSAC to
  filter as outliers. If your images have a lot of background clutter you
  may want to mask them first.

## File overview

| File | Purpose |
|---|---|
| `config.py` | Paths and hyperparameters — edit `DATASET_DIR` here |
| `run_pipeline.py` | End-to-end driver: extract → pair → match → map → export |
| `export_results.py` | Write `pycolmap.Reconstruction` to PLY + text poses |
| `visualize.py` | Open3D viewer for the sparse cloud + camera frustums |
| `evaluate.py` | Align + compare recovered poses to `temple_par.txt` ground truth |
| `requirements.txt` | Extra deps only needed for this pipeline |
