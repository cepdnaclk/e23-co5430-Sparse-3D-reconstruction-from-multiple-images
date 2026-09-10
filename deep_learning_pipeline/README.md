# Deep-learning SfM: SuperPoint + LightGlue + pycolmap

> Full user guide: [`docs/DeepLearningGuide.md`](../docs/DeepLearningGuide.md).
> This README is a quick pointer — the guide is the source of truth.

Reconstructs camera poses + sparse cloud with learned correspondences and
`pycolmap` incremental mapping + bundle adjustment (via `hloc` glue).
Prints the same M2/pose summary as `python run.py --feature_type sift` for
side-by-side comparison.

## Setup

```bash
pip install -r requirements.txt
cd Hierarchical-Localization && pip install -e . && cd ..
```

Needs the 47 TempleRing PNGs in `datasets/templeRing/images/`
(`deep_learning_pipeline/config.py: DATASET_DIR`, `IMAGE_GLOB="*.png"`).

## Run

```bash
python -m deep_learning_pipeline.run_pipeline
python -m deep_learning_pipeline.run_pipeline --skip-existing
```

Extract (`superpoint_max`) → exhaustive pairs → LightGlue match
(`superpoint+lightglue`) → `pycolmap` mapping + BA → `outputs/lightglue/`
(`points3D.ply`, `poses.txt`, `sparse/`).

## Inspect / evaluate

```bash
python -m deep_learning_pipeline.visualize --sfm-dir outputs/lightglue/sparse
python -m deep_learning_pipeline.evaluate --gt datasets/templeRing/templeR_par.txt --poses outputs/lightglue/poses.txt
python -m deep_learning_pipeline.evaluate --gt datasets/templeRing/templeR_par.txt --sparse outputs/lightglue/sparse
```

![SuperPoint + LightGlue sparse point cloud](../docs/assets/deep_pointcloud.png)

| File | Purpose |
|---|---|
| `config.py` | Paths + `FEATURE_CONF` / `MATCHER_CONF` / `CAMERA_MODEL=SIMPLE_RADIAL` |
| `run_pipeline.py` | Driver: extract → pair → match → map → export → evaluate |
| `export_results.py` | `pycolmap.Reconstruction` → PLY + text poses |
| `visualize.py` | Open3D colored cloud + red frustums |
| `evaluate.py` | Same M2/pose output contract as root `evaluate.py` |
