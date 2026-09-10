"""Prepare a COLMAP-format dataset folder for gsplat's simple_trainer.py.

Reads one of this repo's SfM outputs (COLMAP sparse model + the images it
references) and writes the flat layout gsplat expects:

    <out>/
    ├── images/          # images referenced by the model, original resolution
    └── sparse/0/        # cameras.bin, images.bin, points3D.bin (or .txt)

gsplat's COLMAP parser loads `sparse/0` via pycolmap (binary or text both
work) and reads images from `images/`, matching them by the name stored in
the model, so filenames are copied 1:1. SIMPLE_RADIAL / PINHOLE models are
both supported by gsplat (undistortion happens at load time); nothing is
re-estimated here.

Examples:
    python gaussian_splatting/prepare_dataset.py \
        --sparse outputs/lightglue/sparse --out gaussian_splatting/data/temple
    python gaussian_splatting/prepare_dataset.py \
        --sparse outputs/orb/sparse/0 --images outputs/orb/images \
        --out gaussian_splatting/data/temple_orb
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
MODEL_FILES_TXT = ("cameras.txt", "images.txt", "points3D.txt")
SKIP_PATTERNS = ("database.db", "colmap.LOG", "rigs.bin", "frames.bin")


def find_model_files(sparse_dir: Path) -> list[Path]:
    """Locate the COLMAP model files (binary preferred, text as fallback)."""
    binary = [sparse_dir / f for f in MODEL_FILES if (sparse_dir / f).exists()]
    text = [sparse_dir / f for f in MODEL_FILES_TXT if (sparse_dir / f).exists()]
    if len(binary) == 3:
        return binary
    if len(text) == 3:
        return text
    if binary or text:
        missing = set(MODEL_FILES + MODEL_FILES_TXT) - {
            p.name for p in binary + text
        }
        raise SystemExit(
            f"[prepare] incomplete COLMAP model in {sparse_dir}: missing {sorted(missing)}"
        )
    raise SystemExit(
        f"[prepare] no COLMAP model files found in {sparse_dir} "
        f"(looked for {MODEL_FILES} and {MODEL_FILES_TXT})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sparse",
        type=Path,
        default=Path("outputs/lightglue/sparse"),
        help="COLMAP sparse model dir (binary or text)",
    )
    ap.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Image folder referenced by the model (default: datasets/templeRing/images)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("gaussian_splatting/data/temple"),
        help="Output dataset dir for gsplat",
    )
    args = ap.parse_args()

    images_dir = args.images or Path("datasets/templeRing/images")

    # --- copy the model ---
    model_files = find_model_files(args.sparse)
    out_sparse = args.out / "sparse" / "0"
    out_sparse.mkdir(parents=True, exist_ok=True)
    for src in model_files:
        shutil.copy2(src, out_sparse / src.name)

    # --- load it and check which images are actually registered ---
    import pycolmap

    recon = pycolmap.Reconstruction(str(out_sparse))
    registered = sorted(im.name for im in recon.images.values())
    n_points = len(recon.points3D)
    cam = next(iter(recon.cameras.values()))
    print(
        f"[prepare] model: {len(registered)}/{len(registered)} images registered, "
        f"{n_points} points, camera {cam.model_name} {cam.width}x{cam.height}"
    )
    if len(recon.images) < len(recon.cameras) * 4 and len(registered) < 10:
        print(
            "[prepare] WARNING: very few registered images — this model will "
            "produce a poor splat (outputs/orb registers only 8/47)."
        )

    # --- copy the images, keeping the exact names stored in the model ---
    out_images = args.out / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in registered:
        src = images_dir / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, out_images / name)
    if missing:
        raise SystemExit(
            f"[prepare] {len(missing)} images referenced by the model are missing "
            f"from {images_dir}: {missing[:5]}..."
        )
    extras = sorted(p.name for p in out_images.iterdir()) if out_images.exists() else []
    extras = [n for n in extras if n not in set(registered)]
    if extras:
        for name in extras:
            (out_images / name).unlink()

    print(f"[prepare] copied {len(registered)} images -> {out_images}")
    print(f"[prepare] wrote model -> {out_sparse}")
    print(f"[prepare] dataset ready: {args.out}")
    print(
        "[prepare] next: zip it and upload to Colab (or train locally if you have a GPU)"
    )


if __name__ == "__main__":
    main()
