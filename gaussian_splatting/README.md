# 3D Gaussian Splatting on the repo's COLMAP outputs

> Full user guide:
> [`docs/GaussianSplattingGuide.md`](../docs/GaussianSplattingGuide.md).
> This README is a quick pointer.

Stretch-goal rendering: train 3DGS from either SfM pipeline's COLMAP model
with [gsplat](https://docs.gsplat.studio/). Best source is
`outputs/lightglue/sparse` (47/47 registered); classical SIFT
(`outputs/sift/sparse/0/`, `EXPORT_COLMAP=True`) also works.

## Quick path

```bash
python gaussian_splatting/prepare_dataset.py \
    --sparse outputs/lightglue/sparse \
    --out gaussian_splatting/data/temple
cd gaussian_splatting/data && zip -qr temple_colmap.zip temple
```

Then either open `gaussian_splatting/gaussian_splatting.ipynb` in Colab
(T4 GPU) or use `train_server.sh` on an SSH GPU box:

```bash
./train_server.sh --zip temple_colmap.zip
python gaussian_splatting/export_ply.py results/temple/ckpts/ckpt_29999_rank0.pt temple_splats.ply
```

![Novel-view render from the trained model](../docs/assets/gaussian_render.png)

| File | Purpose |
|---|---|
| `prepare_dataset.py` | SfM output → gsplat `images/` + `sparse/0/` layout |
| `gaussian_splatting.ipynb` | Colab walkthrough (T4) |
| `train_server.sh` | Idempotent GPU-server training (Inria `train.py`/`render.py`/`metrics.py`) |
| `export_ply.py` | gsplat `.pt` checkpoint → viewable 3DGS PLY |
