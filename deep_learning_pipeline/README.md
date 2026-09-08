# Temple Ring SfM: SuperPoint + LightGlue + pycolmap

Full structure-from-motion pipeline for the Middlebury **Temple Ring**
dataset: SuperPoint features, LightGlue matching, incremental
reconstruction (poses + sparse point cloud) and bundle adjustment via
COLMAP's own solver, driven through `pycolmap` and Google/ETH's `hloc`
toolbox (which already implements the SuperPoint+LightGlue -> COLMAP
database -> `pycolmap.incremental_mapping` glue code, so we don't have to
hand-roll database import / geometric verification ourselves).

## Why hloc instead of a from-scratch OpenCV pipeline

You asked for pycolmap as the backend. Getting LightGlue's matches into a
COLMAP database correctly (feature import, match import, two-view geometry
verification) is finicky and easy to get subtly wrong. `hloc` is the
reference implementation for exactly this combination (from the LightGlue /
SuperGlue authors' own group) and is used in the LightGlue paper's own
localization benchmarks, so we build on it rather than reimplementing it.
Everything downstream (mapping, bundle adjustment, model access) is
`pycolmap` directly — `hloc` only handles the extract/match/import step.

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate   # or use conda
pip install -r requirements.txt
```

Notes:
- Works on CPU, but SuperPoint+LightGlue extraction/matching will be much
  faster with a CUDA GPU. Nothing in the code assumes a GPU; PyTorch will
  fall back to CPU automatically.
- `pycolmap` ships prebuilt wheels for Linux/macOS/Windows, so you don't
  need a separate COLMAP install.

## 2. Get the dataset onto disk

Download the Temple Ring set (47 images, `templeR0001.png` ...
`templeR0047.png`) from the Middlebury Multi-View Stereo page and place the
images in a folder, e.g.:

```
data/templeRing/templeR0001.png
data/templeRing/templeR0002.png
...
data/templeRing/temple_par.txt   # optional, ground-truth calibration
```

Update `pipeline/config.py` if your path or file extension differs
(`DATASET_DIR`, `IMAGE_GLOB`).

## 3. Run the pipeline

```bash
python -m pipeline.run_pipeline
```

This will, in order:
1. Extract SuperPoint keypoints/descriptors for every image (`superpoint_max` config — no resize cap, suited to a small, detailed dataset).
2. Build an exhaustive pair list (47 images -> 1081 pairs is cheap; fine for this dataset size).
3. Match every pair with LightGlue.
4. Build a COLMAP database, import the features/matches, run
   `pycolmap.incremental_mapping` (incremental SfM: initial pair, PnP
   registration of remaining images, triangulation, and bundle adjustment
   after each step plus a final global BA).
5. Export `points3D.ply` (sparse cloud) and `poses.txt` (per-image camera
   pose) into `outputs/templeRing/`. The full binary COLMAP model
   (`cameras.bin`, `images.bin`, `points3D.bin`) is left in
   `outputs/templeRing/sparse/` if you want to open it in COLMAP's GUI or
   load it with `pycolmap.Reconstruction(...)` for further analysis.

Re-running with `--skip-existing` reuses cached features/matches from disk
instead of recomputing them (handy if you only want to re-run mapping with
different options).

## 4. Inspect the result

```bash
python -m pipeline.visualize --sfm-dir outputs/templeRing/sparse
```

Opens an Open3D window with the colored sparse point cloud and a small red
wireframe frustum per registered camera.

## 5. (Optional) Evaluate against ground truth

Temple Ring ships known camera calibration in `temple_par.txt`. Since
COLMAP reconstructs up to an arbitrary similarity transform (unknown scale/
rotation/translation — monocular SfM cannot recover absolute scale),
`pipeline/evaluate.py` aligns the recovered camera centers to the ground
truth with a Umeyama similarity fit and reports the residual error:

```bash
python -m pipeline.evaluate \
    --gt data/templeRing/temple_par.txt \
    --poses outputs/templeRing/poses.txt
```

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
| `pipeline/config.py` | Paths and hyperparameters — edit `DATASET_DIR` here |
| `pipeline/run_pipeline.py` | End-to-end driver: extract -> pair -> match -> map -> export |
| `pipeline/export_results.py` | Write `pycolmap.Reconstruction` to PLY + text poses |
| `pipeline/visualize.py` | Open3D viewer for the sparse cloud + camera frustums |
| `pipeline/evaluate.py` | Align + compare recovered poses to `temple_par.txt` ground truth |
