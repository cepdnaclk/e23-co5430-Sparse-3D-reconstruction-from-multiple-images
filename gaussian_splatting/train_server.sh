#!/usr/bin/env bash
# Train 3D Gaussian Splatting on a remote Ubuntu GPU server (no Colab).
#
# Mirrors gaussian_splatting/gaussian_splatting.ipynb, which uses the original
# Inria implementation (graphdeco-inria/gaussian-splatting: train.py /
# render.py / metrics.py). The notebook itself is left untouched; this script
# only replaces the Colab-only parts:
#   - google.colab files.upload()/download() -> --zip / --data-dir + scp
#   - matplotlib popups                   -> headless prints + ffmpeg video
#   - fresh /content VM every run         -> idempotent re-runs (venv, repo,
#                                            data and checkpoints are reused)
#
# Expected server: Ubuntu + NVIDIA GPU + SSH, internet access for pip.
# No sudo required at any step: the venv lives in $WS and undistortion is
# done with OpenCV in pure Python (no COLMAP CLI / apt packages).
# Run long trainings inside tmux/screen or with nohup.
#
# Quick start (from the repo root, or any empty work dir on the server):
#   scp gaussian_splatting/data/temple_colmap.zip user@server:~/gsplat_ws/
#   scp gaussian_splatting/train_server.sh         user@server:~/gsplat_ws/
#   ssh user@server
#   cd ~/gsplat_ws && chmod +x train_server.sh
#   ./train_server.sh --zip temple_colmap.zip              # first run
#   ./train_server.sh                                       # re-run: reuses everything
#   ./train_server.sh --steps 3000 --no-video               # smoke test
#   nohup ./train_server.sh --zip temple_colmap.zip > train.log 2>&1 &
#
# Results stay in <ws>/gaussian-splatting/output/temple/; fetch with scp -r.
set -euo pipefail

WS="$(pwd)"
ZIP=""
DATA_DIR=""
MODEL_DIR=""
ITERATIONS=7000
DO_EVAL=1
DO_VIDEO=1
SKIP_UNDISTORT=0
UNDISTORT_MODE="auto"   # auto|python|colmap (see --undistort-mode)
REINSTALL=0
UPDATE_REPO=0
ZIP_OUT=""
EXTRA_TRAIN_ARGS=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --zip PATH            Dataset zip containing a folder with images/ + sparse/0/
                        (e.g. temple_colmap.zip). Unzipped once, then reused.
  --data-dir PATH       Use a pre-extracted dataset dir instead of --zip.
                        Default: <ws>/gaussian-splatting/data/temple
  --model-dir PATH      Output dir for train.py (-m).
                        Default: <ws>/gaussian-splatting/output/temple
  --ws PATH             Work dir holding venv/, gaussian-splatting/, data_raw/.
                        Default: current directory.
  --iterations N        Total training iterations (notebook default: 7000;
                        use 30000 for final quality). Alias: --steps N.
  --steps N             Alias for --iterations.
  --eval / --no-eval    Pass / omit --eval to train.py (default: --eval, holds
                        out every 8th image for PSNR/SSIM/LPIPS).
  --no-video            Skip the ffmpeg turntable video from test renders.
  --skip-undistort      Skip the image_undistorter step even if the
                        camera model is not PINHOLE (use when you accept
                        distorted training).
  --undistort-mode M    How to undistort non-pinhole cameras: auto (default,
                        use COLMAP CLI if present, else pure Python),
                        python (OpenCV in the venv, no sudo needed), or
                        colmap (classic CLI, must already be installed).
  --reinstall           Force pip reinstall into the venv (even if it exists).
  --update-repo         git pull the gaussian-splatting checkout (default:
                        reuse as-is for reproducibility).
  --zip-out PATH        Zip the finished model dir to PATH.
  --extra-args "..."    Extra string appended to the train.py call.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip) ZIP="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --ws) WS="$2"; shift 2 ;;
    --iterations|--steps) ITERATIONS="$2"; shift 2 ;;
    --eval) DO_EVAL=1; shift ;;
    --no-eval) DO_EVAL=0; shift ;;
    --no-video) DO_VIDEO=0; shift ;;
    --skip-undistort) SKIP_UNDISTORT=1; shift ;;
    --undistort-mode) UNDISTORT_MODE="$2"; shift 2 ;;
    --reinstall) REINSTALL=1; shift ;;
    --update-repo) UPDATE_REPO=1; shift ;;
    --zip-out) ZIP_OUT="$2"; shift 2 ;;
    --extra-args) EXTRA_TRAIN_ARGS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REPO_DIR="$WS/gaussian-splatting"
VENV_DIR="$WS/venv"
DATA_RAW="$WS/data_raw"
case "$UNDISTORT_MODE" in
  auto|python|colmap) ;;
  *) echo "Unknown --undistort-mode: $UNDISTORT_MODE (use auto|python|colmap)" >&2; exit 2 ;;
esac
DEFAULT_DATA_DIR="$REPO_DIR/data/temple"
DEFAULT_MODEL_DIR="$REPO_DIR/output/temple"
[[ -z "$DATA_DIR" ]] && DATA_DIR="$DEFAULT_DATA_DIR"
[[ -z "$MODEL_DIR" ]] && MODEL_DIR="$DEFAULT_MODEL_DIR"

log() { echo "[train_server] $*"; }
die() { echo "[train_server] ERROR: $*" >&2; exit 1; }

log "work dir : $WS"
log "repo dir : $REPO_DIR"
log "venv     : $VENV_DIR"
log "data dir : $DATA_DIR"
log "model dir: $MODEL_DIR"
log "iters    : $ITERATIONS (eval=$DO_EVAL video=$DO_VIDEO)"

# --- 0. GPU sanity check (notebook cell: !nvidia-smi) ---
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
else
  log "WARNING: nvidia-smi not found; continuing anyway (training needs an NVIDIA GPU)."
fi
command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v git >/dev/null 2>&1 || die "git not found (needed to clone gaussian-splatting)"

mkdir -p "$WS"

# --- 1. Clone the Inria repo (notebook cell: git clone --recursive) ---
if [[ ! -d "$REPO_DIR/.git" ]]; then
  log "Cloning graphdeco-inria/gaussian-splatting ..."
  git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git "$REPO_DIR"
else
  log "Repo already present, reusing ($REPO_DIR)"
  if [[ "$UPDATE_REPO" == "1" ]]; then
    log "Updating repo (--update-repo) ..."
    git -C "$REPO_DIR" pull --recurse-submodules || log "WARNING: git pull failed, continuing with existing checkout"
  fi
fi
[[ -d "$REPO_DIR/submodules" ]] || log "WARNING: submodules/ missing (clone without --recursive?)"
ls "$REPO_DIR/submodules" || true

# --- 2. venv + PyTorch + CUDA extensions (notebook cells: torch check, pip install) ---
if [[ ! -d "$VENV_DIR/bin" ]]; then
  log "Creating venv ..."
  python3 -m venv "$VENV_DIR"
else
  log "Venv already present, reusing"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# Install (or skip if already importable and not --reinstall).
need_install=0
if [[ "$REINSTALL" == "1" ]]; then
  need_install=1
elif ! python -c "import torch, plyfile, tqdm, joblib, pycolmap" >/dev/null 2>&1; then
  need_install=1
fi
if [[ "$need_install" == "1" ]]; then
  log "Installing Python deps (torch CUDA build + trainer deps) ..."
  # CUDA wheel index for torch; cu121 is broadly driver-compatible.
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  python -m pip install plyfile tqdm joblib pycolmap pillow opencv-python-headless
else
  log "Python deps already importable, skipping pip install (use --reinstall to force)"
fi
# OpenCV is needed for the sudo-free Python undistort; install it on its own
# so a pre-existing venv without it doesn't trigger a full torch reinstall.
if ! python -c "import cv2" >/dev/null 2>&1; then
  log "Installing opencv-python-headless (pure-Python undistort) ..."
  python -m pip install opencv-python-headless
fi
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| CUDA:', torch.version.cuda)"

# Build the Inria CUDA rasterizer + kNN against this venv's torch/CUDA.
if [[ "$REINSTALL" == "1" ]] || ! python -c "import diff_gaussian_rasterization" >/dev/null 2>&1; then
  log "Building diff-gaussian-rasterization ..."
  python -m pip install "$REPO_DIR/submodules/diff-gaussian-rasterization"
else
  log "diff-gaussian-rasterization already built, skipping"
fi
if [[ "$REINSTALL" == "1" ]] || ! python -c "import simple_knn" >/dev/null 2>&1; then
  log "Building simple-knn ..."
  python -m pip install "$REPO_DIR/submodules/simple-knn"
else
  log "simple_knn already built, skipping"
fi
if [[ -d "$REPO_DIR/submodules/fused-ssim" ]]; then
  if [[ "$REINSTALL" == "1" ]] || ! python -c "import fused_ssim" >/dev/null 2>&1; then
    log "Building fused-ssim ..."
    python -m pip install "$REPO_DIR/submodules/fused-ssim" || log "WARNING: fused-ssim build failed, continuing"
  else
    log "fused-ssim already built, skipping"
  fi
fi

# --- 3. Arrange the dataset (notebook cells: upload zip -> data/temple) ---
if [[ -n "$ZIP" ]]; then
  [[ -f "$ZIP" ]] || die "zip not found: $ZIP"
  mkdir -p "$DATA_RAW"
  log "Unzipping $ZIP -> $DATA_RAW ..."
  python - "$ZIP" "$DATA_RAW" <<'EOF'
import sys, zipfile
zpath, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zpath, "r") as zf:
    zf.extractall(out)
print(f"extracted {zpath} -> {out}")
EOF
  log "Locating scene folder (images/ + sparse/0/) ..."
  SCENE_SRC="$(python - "$DATA_RAW" <<'EOF'
import os, sys
root = sys.argv[1]
for dirpath, dirnames, _ in os.walk(root):
    if "images" in dirnames and os.path.isdir(os.path.join(dirpath, "sparse", "0")):
        print(dirpath)
        break
EOF
)"
  [[ -n "$SCENE_SRC" ]] || die "could not find a folder with images/ and sparse/0/ under $DATA_RAW - check the zip structure"
  log "Found scene at: $SCENE_SRC"
  if [[ "$SCENE_SRC" != "$DATA_DIR" ]]; then
    rm -rf "$DATA_DIR"
    mkdir -p "$(dirname "$DATA_DIR")"
    cp -r "$SCENE_SRC" "$DATA_DIR"
  fi
fi
[[ -d "$DATA_DIR/images" ]] || die "images/ missing in $DATA_DIR (pass --zip or --data-dir)"
[[ -d "$DATA_DIR/sparse/0" ]] || die "sparse/0/ missing in $DATA_DIR (pass --zip or --data-dir)"
log "Dataset: $DATA_DIR ($(ls "$DATA_DIR/images" | wc -l) images)"
ls "$DATA_DIR/sparse/0"

# --- 4. Undistort if needed (notebook: pycolmap check + image_undistorter) ---
# Default needs no sudo and no COLMAP CLI: non-pinhole cameras (our Temple
# model is SIMPLE_RADIAL) are undistorted with OpenCV inside the venv, and
# the model is rewritten as distortion-free PINHOLE. Poses and 3D points are
# carried over unchanged; 2D observations in images.bin are left as-is
# (train.py only uses poses + intrinsics + pixels, so this is harmless).
if [[ "$SKIP_UNDISTORT" == "1" ]]; then
  log "Skipping undistortion check (--skip-undistort)"
else
  NEED_UNDISTORT="$(python - "$DATA_DIR" <<'EOF'
import pycolmap, os, sys
rec = pycolmap.Reconstruction(os.path.join(sys.argv[1], "sparse", "0"))
cam = next(iter(rec.cameras.values()))
print("Camera model:", cam.model, "| params:", cam.params)
print("1" if cam.model.name not in ("PINHOLE", "SIMPLE_PINHOLE") else "0")
EOF
)"
  echo "$NEED_UNDISTORT" | head -n 2
  if echo "$NEED_UNDISTORT" | tail -n 1 | grep -q "^1$"; then
    MODE="$UNDISTORT_MODE"
    if [[ "$MODE" == "auto" ]]; then
      if command -v colmap >/dev/null 2>&1; then
        MODE="colmap"
      else
        MODE="python"
      fi
    fi
    log "Camera is not PINHOLE: undistorting via $MODE ..."
    UNDISTORTED_DIR="$(dirname "$DATA_DIR")/$(basename "$DATA_DIR")_undistorted"
    if [[ "$MODE" == "colmap" ]]; then
      command -v colmap >/dev/null 2>&1 || die "colmap CLI not found. Install it, or use --undistort-mode python (default, no sudo needed)."
      colmap -h | head -n 1 || true
      rm -rf "$UNDISTORTED_DIR"
      mkdir -p "$UNDISTORTED_DIR"
      colmap image_undistorter \
        --image_path "$DATA_DIR/images" \
        --input_path "$DATA_DIR/sparse/0" \
        --output_path "$UNDISTORTED_DIR" \
        --output_type COLMAP
      # image_undistorter writes sparse/*.bin directly (not sparse/0/); match the
      # layout train.py expects, same as the repo's convert.py does.
      mkdir -p "$UNDISTORTED_DIR/sparse/0"
      for f in "$UNDISTORTED_DIR"/sparse/*; do
        [[ "$(basename "$f")" == "0" ]] && continue
        [[ -e "$f" ]] || continue
        mv "$f" "$UNDISTORTED_DIR/sparse/0/"
      done
    else
      rm -rf "$UNDISTORTED_DIR"
      python - "$DATA_DIR" "$UNDISTORTED_DIR" <<'EOF'
import os, shutil, sys
import numpy as np
import pycolmap

try:
    import cv2
except ImportError:
    sys.exit("opencv-python-headless is required for --undistort-mode python: "
             "run with --reinstall or pip install opencv-python-headless in the venv")

src, dst = sys.argv[1], sys.argv[2]
rec = pycolmap.Reconstruction(os.path.join(src, "sparse", "0"))

undistort_per_camera = {}
for cam_id, cam in rec.cameras.items():
    name = cam.model.name
    if name == "SIMPLE_RADIAL":
        # SIMPLE_RADIAL params: (f, cx, cy, k); OpenCV: K + D=[k, 0, 0, 0].
        f, cx, cy, k = (float(v) for v in cam.params)
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        D = np.array([k, 0.0, 0.0, 0.0])
        cam.model = pycolmap.CameraModelId.PINHOLE
        cam.params = [f, f, cx, cy]
        undistort_per_camera[cam_id] = (K, D)
    elif name in ("PINHOLE", "SIMPLE_PINHOLE"):
        undistort_per_camera[cam_id] = None
    else:
        sys.exit(f"unsupported camera model for python undistort: {name} "
                 "(use --undistort-mode colmap or --skip-undistort)")

os.makedirs(os.path.join(dst, "images"))
os.makedirs(os.path.join(dst, "sparse", "0"))
n = 0
for img in rec.images.values():
    kd = undistort_per_camera[img.camera_id]
    src_path = os.path.join(src, "images", img.name)
    dst_path = os.path.join(dst, "images", img.name)
    if kd is None:
        shutil.copy2(src_path, dst_path)
    else:
        K, D = kd
        im = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if im is None:
            sys.exit(f"could not read image: {src_path}")
        cv2.imwrite(dst_path, cv2.undistort(im, K, D))
    n += 1

rec.write_binary(os.path.join(dst, "sparse", "0"))
print(f"undistorted {n} images -> {dst} (PINHOLE)")
EOF
    fi
    DATA_DIR="$UNDISTORTED_DIR"
    log "Undistorted dataset: $DATA_DIR"
    python - "$DATA_DIR" <<'EOF'
import pycolmap, os, sys
rec = pycolmap.Reconstruction(os.path.join(sys.argv[1], "sparse", "0"))
cam = next(iter(rec.cameras.values()))
print("New camera model:", cam.model, "| params:", cam.params)
EOF
  else
    log "Camera already PINHOLE/SIMPLE_PINHOLE: no undistortion needed"
  fi
fi

# --- 5. Train (notebook cell: python train.py ...) ---
TRAIN_ARGS=(-s "$DATA_DIR" -m "$MODEL_DIR" --iterations "$ITERATIONS"
  --test_iterations "$ITERATIONS" --save_iterations "$ITERATIONS")
if [[ "$DO_EVAL" == "1" ]]; then
  TRAIN_ARGS+=(--eval)
fi
log "Training: python train.py ${TRAIN_ARGS[*]} $EXTRA_TRAIN_ARGS"
cd "$REPO_DIR"
# shellcheck disable=SC2086
python train.py "${TRAIN_ARGS[@]}" $EXTRA_TRAIN_ARGS

# --- 6. Render + metrics on held-out views (notebook cell: render.py, metrics.py) ---
if [[ "$DO_EVAL" == "1" ]]; then
  log "Rendering test views ..."
  python render.py -m "$MODEL_DIR"
  log "Computing metrics ..."
  python metrics.py -m "$MODEL_DIR" | tee "$MODEL_DIR/metrics.txt"
else
  log "Skipping render/metrics (--no-eval)"
fi

# --- 7. PLY location + turntable video (notebook cells: download ply + zip) ---
PLY_PATH="$MODEL_DIR/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
if [[ -f "$PLY_PATH" ]]; then
  log "PLY ready: $PLY_PATH"
else
  log "WARNING: expected PLY not found: $PLY_PATH (training may have saved a different iteration)"
  find "$MODEL_DIR/point_cloud" -name "*.ply" 2>/dev/null | head || true
fi

if [[ "$DO_VIDEO" == "1" ]]; then
  RENDER_DIR="$MODEL_DIR/test/ours_${ITERATIONS}/renders"
  VIDEO_OUT="$MODEL_DIR/turntable_${ITERATIONS}.mp4"
  if [[ -d "$RENDER_DIR" ]] && ls "$RENDER_DIR"/*.png >/dev/null 2>&1; then
    if command -v ffmpeg >/dev/null 2>&1; then
      log "Stitching turntable video -> $VIDEO_OUT ..."
      ffmpeg -y -framerate 30 -pattern_type glob -i "$RENDER_DIR/*.png" \
        -c:v libx264 -pix_fmt yuv420p "$VIDEO_OUT"
      log "Video ready: $VIDEO_OUT"
    else
      log "WARNING: ffmpeg not found, skipping video (ask your admin to install it, or re-run with --no-video)"
    fi
  else
    log "WARNING: no test renders in $RENDER_DIR, skipping video (train with --eval to get them)"
  fi
else
  log "Skipping video (--no-video)"
fi

if [[ -n "$ZIP_OUT" ]]; then
  log "Zipping $MODEL_DIR -> $ZIP_OUT ..."
  rm -f "$ZIP_OUT"
  (cd "$(dirname "$MODEL_DIR")" && zip -qr "$ZIP_OUT" "$(basename "$MODEL_DIR")")
  log "Zip ready: $ZIP_OUT"
fi

log "Done. Fetch results with: scp -r <user>@<server>:$MODEL_DIR ./"
log "PLY: $PLY_PATH"
