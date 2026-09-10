# Gaussian Splatting (gsplat) on our COLMAP outputs

Trains a 3D Gaussian Splatting model from the repo's SfM results using
[gsplat](https://docs.gsplat.studio/). The design target is **Google Colab
(free T4, 16 GB VRAM)** — the full plan lives in `docs/GaussianSplattingPlan.md`.

## Pipeline at a glance

```
outputs/lightglue/sparse  (COLMAP model, 47/47 imgs)
        │  prepare_dataset.py   (run locally)
        ▼
gaussian_splatting/data/temple/          ← COLMAP layout gsplat expects
        │  temple_colmap.zip  (~12 MB, upload to Colab / Drive)
        ▼
gsplat simple_trainer.py                 (run in Colab, T4)
        ▼
results/temple/  →  ckpt .pt + eval PSNR/SSIM + turntable video + PLY
```

## 1. Build the dataset (local machine)

```bash
python gaussian_splatting/prepare_dataset.py \
    --sparse outputs/lightglue/sparse \
    --out gaussian_splatting/data/temple
cd gaussian_splatting/data && zip -qr temple_colmap.zip temple
```

`prepare_dataset.py` copies the COLMAP model (`sparse/0/`) and the 47 registered
images under the exact names stored in the model, prints a summary
(47/47 images, 3,785 points, SIMPLE_RADIAL 640×480), and refuses to continue if
any referenced image is missing. To instead splat the classical ORB result
(**only 8/47 poses — much weaker**):

```bash
python gaussian_splatting/prepare_dataset.py \
    --sparse outputs/orb/sparse/0 --images outputs/orb/images \
    --out gaussian_splatting/data/temple_orb
```

## 2. Train on Colab

Open `gaussian_splatting/gsplat_colab.ipynb` in Colab
(**Runtime → Change runtime type → T4 GPU**) and run top-to-bottom:

1. GPU check
2. Upload `temple_colmap.zip` (or mount Drive)
3. `pip install gsplat` + trainer deps
4. Vendor `simple_trainer.py` + helper modules from the gsplat repo
5. Train: `python simple_trainer.py default --data_type colmap --data_dir data/temple --data_factor 1 --result_dir results/temple --disable_viewer`
6. Turntable video from the best checkpoint
7. Copy ckpt + stats + video to Google Drive
8. Optional PLY export (works in-notebook or locally with `export_ply.py`)

Expected: **~10–20 min** on the T4, a few hundred thousand splats, eval PSNR
printed at each eval step (every 8th image is held out). For a quick end-to-end
smoke test, run the trainer once with `--max_steps 3000` first.

## 2b. Alternative: train on a GPU server (no Colab)

If you have SSH access to a machine with an NVIDIA GPU, use
`gaussian_splatting/train_server.sh` — a self-contained, idempotent script that
mirrors `gaussian_splatting.ipynb` (original Inria `graphdeco-inria/gaussian-splatting`:
`train.py` / `render.py` / `metrics.py`). It creates a venv, installs torch +
the CUDA rasterizer extensions, extracts the dataset zip, undistorts with
OpenCV in pure Python when the camera model needs it (our Temple model is
`SIMPLE_RADIAL`; no sudo or COLMAP install required), trains with `--eval`, renders test views, computes metrics,
and stitches a turntable video from the renders with ffmpeg:

```bash
# from the repo:
scp gaussian_splatting/data/temple_colmap.zip user@server:~/gsplat_ws/
scp gaussian_splatting/train_server.sh user@server:~/gsplat_ws/
ssh user@server
cd ~/gsplat_ws && chmod +x train_server.sh
./train_server.sh --zip temple_colmap.zip              # first run (default 7000 iters)
./train_server.sh                                       # re-runs: reuses everything
./train_server.sh --steps 3000 --no-video               # quick smoke test
./train_server.sh --steps 30000                         # final quality
./train_server.sh --no-eval --no-video                  # skip held-out render/metrics + video
./train_server.sh --skip-undistort                      # no undistortion step at all
./train_server.sh --undistort-mode colmap               # classic CLI (must be pre-installed)
./train_server.sh --help                                # all flags (--data-dir, --model-dir,
                                                        # --ws, --reinstall, --zip-out, ...)
```

Useful for long runs over SSH: run it inside `tmux`/`screen`, or nohup it:
`nohup ./train_server.sh --zip temple_colmap.zip > train.log 2>&1 &`.
Results stay in `~/gsplat_ws/gaussian-splatting/output/temple/` (`point_cloud/`,
test renders, `metrics.txt`, `turntable_<iters>.mp4`); fetch with `scp -r`.

## 3. View results locally

- `.pt` checkpoint → re-render in gsplat, or
- `temple_splats.ply` (via `export_ply.py`) → MeshLab / CloudCompare /
  [SuperSplat](https://superspl.at) — point-splat rendering shows the model.

```bash
python gaussian_splatting/export_ply.py temple_ckpts/ckpt_29999_rank0.pt temple_splats.ply
```

## Notes

- Our model uses COLMAP `SIMPLE_RADIAL` — gsplat handles it natively and
  undistorts images at load time (expect a "Camera is not PINHOLE" warning).
- No conversion or re-estimation of poses/intrinsics is done anywhere: the
  LightGlue/pycolmap reconstruction is used as-is.
- `data_factor 1` because TempleRing images are already small (640×480).
