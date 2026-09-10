# User Guide (index)

Sparse 3D reconstruction from overlapping images: recover per-image camera
poses + a sparse point cloud, then optionally render it with Gaussian
Splatting. Two independent SfM pipelines solve the same task so they can be
compared with the identical M2/pose evaluation summary.

## Which guide do I need?

| I want to ... | Read |
|---|---|
| Run the hand-written ORB/SIFT pipeline | [ClassicalGuide.md](ClassicalGuide.md) |
| Run SuperPoint + LightGlue + pycolmap | [DeepLearningGuide.md](DeepLearningGuide.md) |
| See keypoints, matches, point clouds | [VisualizationGuide.md](VisualizationGuide.md) |
| Render with Gaussian Splatting | [GaussianSplattingGuide.md](GaussianSplattingGuide.md) |

Start with the **classical pipeline** (`python run.py`) — fewest dependencies,
main project entry point. Then run the **deep pipeline**
(`python -m deep_learning_pipeline.run_pipeline`) for the learned-features
comparison.

## Quick comparison

```bash
python run.py --feature_type sift              # classical → outputs/sift/
python -m deep_learning_pipeline.run_pipeline  # deep      → outputs/lightglue/
```

Both print:

```
=== M2 metrics summary ===
Images: 47   Registered: 47/47
Keypoints per image: min=.. max=.. mean=..
Matched pairs (pre-RANSAC), mean per pair: ..
RANSAC inliers, mean per verified pair: ..
Reconstructed 3D points / Num observations / Mean track length / ...
Mean reprojection error: .. px
===========================

=== Pose accuracy vs. ground truth (TempleRing calibration) ===
Cameras compared: 47/47
Position error (post-alignment): mean=.. median=.. max=..
Rotation error, absolute/aligned (degrees): mean=.. median=..
Rotation error, pairwise-RELATIVE (degrees): mean=.. median=..
==================================================================
```

Put the two blocks side by side — that is the core result table.
Position units follow `datasets/templeRing/README.txt` (object ~0.1 units:
`< ~0.01` good, `> ~0.05` drift). Rotation `< a few °` is good here. If
absolute rotation looks huge but pairwise-relative is `< 5°`, trust the
relative number (narrow-arc global alignment artifact).

## Visual preview

![Classical sparse point cloud](assets/classical_pointcloud.png)
![Deep-learning sparse point cloud](assets/deep_pointcloud.png)
![Detected keypoints](assets/classical_keypoints.png)
![Feature matches](assets/classical_matches.png)
![Gaussian Splatting render](assets/gaussian_render.png)

## Setup (one venv for classical + deep)

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd Hierarchical-Localization && pip install -e . && cd ..   # deep pipeline only (hloc)
# place TempleRing PNGs in datasets/templeRing/images/
```

## Troubleshooting (short version)

| Symptom | Likely cause |
|---|---|
| `Need >=2 images` / `No images matching '*.png'` | wrong image folder (`config.py` / `deep_learning_pipeline/config.py`) |
| `Could not find a good initial pair` | need more overlap; try SIFT; lower `MIN_INLIERS_INIT` |
| `Cameras compared << Registered` | basename mismatch vs `templeR_par.txt` (case-sensitive) |
| `Reconstruction failed (hloc/pycolmap)` | disconnected match graph; keep exhaustive pairs at 47 images |
| Slow BA / extract | expected; BA takes minutes, SuperPoint+LightGlue wants a CUDA GPU |
| `[open3d] snapshot render unavailable` | harmless headless/EGL issue; `.ply` is still valid |

Full tables live in each per-pipeline guide.
