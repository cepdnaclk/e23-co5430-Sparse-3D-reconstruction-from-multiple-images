# Gaussian Splatting User Guide (stretch goal / downstream rendering)

Turn a finished SfM model from either pipeline into a renderable
3D Gaussian Splatting model. No poses/intrinsics are re-estimated here —
the chosen SfM reconstruction is used as-is.

## 1. Pipeline at a glance

```
outputs/lightglue/sparse  (COLMAP model)
        │  gaussian_splatting/prepare_dataset.py  (local)
        ▼
gaussian_splatting/data/temple/  (images/ + sparse/0/, gsplat layout)
        │  zip → upload to Colab / scp to GPU server
        ▼
train (gsplat simple_trainer in Colab, or Inria train.py on server)
        ▼
results/temple/ → checkpoints + PSNR/SSIM + turntable video + PLY
```

Recommended source: **deep pipeline** `outputs/lightglue/sparse` (47/47
registered). Classical SIFT (`outputs/sift/sparse/0`, needs `EXPORT_COLMAP=True`)
also works. Classical ORB (`outputs/orb/`, ~8/47 poses) splats poorly.

## 2. Step 1 — prepare the dataset (local machine)

```bash
python gaussian_splatting/prepare_dataset.py \
    --sparse outputs/lightglue/sparse \
    --out gaussian_splatting/data/temple
cd gaussian_splatting/data && zip -qr temple_colmap.zip temple
```

What it does (`prepare_dataset.py`): finds `cameras/images/points3D.{bin,txt}`,
copies them to `<out>/sparse/0/`, loads the model with `pycolmap` to list
registered images, copies **only** those images under their exact model names
into `<out>/images/`, prunes extras, refuses to continue on missing images,
and prints `model: N/N images, P points, camera MODEL WxH`.

Classical-source variant:

```bash
python gaussian_splatting/prepare_dataset.py \
    --sparse outputs/sift/sparse/0 --images outputs/sift/images \
    --out gaussian_splatting/data/temple_sift
```

| Flag | Default | Meaning |
|---|---|---|
| `--sparse` | `outputs/lightglue/sparse` | COLMAP model dir (binary or text) |
| `--images` | `datasets/templeRing/images` | image pool referenced by the model |
| `--out` | `gaussian_splatting/data/temple` | gsplat-ready output dir |

`gsplat` handles `SIMPLE_RADIAL` natively (undistorts at load; expect a
`Camera is not PINHOLE` warning). Keep `data_factor 1` — TempleRing is already
640×480.

## 3. Step 2a — train on Colab (free T4, 16 GB VRAM)

Open `gaussian_splatting/gaussian_splatting.ipynb`
(**Runtime → Change runtime type → T4 GPU**) and run top-to-bottom:

1. GPU check (`nvidia-smi`).
2. Upload `temple_colmap.zip` (or mount Drive).
3. `pip install gsplat` + trainer deps.
4. Vendor `simple_trainer.py` + helpers from the gsplat repo.
5. Train, e.g.:
   `python simple_trainer.py default --data_type colmap --data_dir data/temple --data_factor 1 --result_dir results/temple --disable_viewer`
6. Turntable video from the best checkpoint.
7. Copy ckpt + stats + video to Drive.
8. Optional PLY export (in-notebook or `export_ply.py` below).

Expect **~10–20 min** on a T4, a few hundred thousand splats, eval PSNR/SSIM
printed per eval step (every 8th image held out). Smoke-test first with
`--max_steps 3000`.

## 4. Step 2b — train on a GPU server (no Colab)

Idempotent script mirroring the notebook with the Inria implementation
(`graphdeco-inria/gaussian-splatting`: `train.py` / `render.py` / `metrics.py`).
No sudo needed; Python-OpenCV undistort replaces the COLMAP CLI.

```bash
scp gaussian_splatting/data/temple_colmap.zip user@server:~/gsplat_ws/
scp gaussian_splatting/train_server.sh user@server:~/gsplat_ws/
ssh user@server
cd ~/gsplat_ws && chmod +x train_server.sh
./train_server.sh --zip temple_colmap.zip     # first run (default 7000 iters)
./train_server.sh                             # re-run: reuses venv/repo/data
./train_server.sh --steps 3000 --no-video     # smoke test
./train_server.sh --steps 30000               # final quality
./train_server.sh --help                      # --data-dir/--model-dir/--ws/--eval/--no-eval/
                                              # --skip-undistort/--undistort-mode/--reinstall/--zip-out/...
```

Run long jobs in `tmux`/`screen` or `nohup ./train_server.sh ... > train.log 2>&1 &`.
Results: `~/gsplat_ws/gaussian-splatting/output/temple/` (`point_cloud/`,
test renders, `metrics.txt`, `turntable_<iters>.mp4`); fetch with `scp -r`.

## 5. Step 3 — view / export results

- `.pt` checkpoint → re-render in gsplat, or
- PLY via `gaussian_splatting/export_ply.py` (SH-DC → RGB, sigmoid opacity,
  exp scale; MeshLab / CloudCompare / SuperSplat compatible):

```bash
python gaussian_splatting/export_ply.py results/temple/ckpts/ckpt_29999_rank0.pt temple_splats.ply
```

![Novel-view render from the trained Gaussian Splatting model](assets/gaussian_render.png)

## 6. Comparing splats across SfM methods

Train once per upstream method (ORB, SIFT, SuperPoint+LightGlue) with separate
`prepare_dataset.py` outputs, then compare held-out PSNR/SSIM:

> Does the upstream SfM method's accuracy affect downstream render quality?

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `no COLMAP model files found` | wrong `--sparse` dir; needs `cameras/images/points3D.{bin,txt}` triple |
| `images referenced by the model are missing` | pass correct `--images` pool; names must match model 1:1 |
| `very few registered images` warning | expected for ORB (~8/47); use SIFT or LightGlue source |
| `Camera is not PINHOLE` | harmless; gsplat undistorts at load (server script undistorts to PINHOLE up front) |
| OOM on T4 | lower `--max_steps` smoke test first; keep `data_factor 1` |
