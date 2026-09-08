"""
Full SfM pipeline for the Temple Ring dataset using SuperPoint + LightGlue
for correspondences and pycolmap (via hloc) for incremental
reconstruction + bundle adjustment.

Steps:
  1. Extract SuperPoint features for every image (hloc.extract_features)
  2. Build an exhaustive pair list (hloc.pairs_from_exhaustive)
  3. Match pairs with LightGlue (hloc.match_features)
  4. Build a COLMAP database and run incremental mapping + bundle
     adjustment (hloc.reconstruction, backed by pycolmap)
  5. Export the sparse point cloud to PLY and camera poses to a text file

Run with:  python -m pipeline.run_pipeline
"""
import argparse
import sys
from pathlib import Path

import pycolmap
from hloc import extract_features, match_features, pairs_from_exhaustive
from hloc import reconstruction as hloc_reconstruction

from . import config as cfg
from .export_results import export_point_cloud, export_poses


def get_image_list(dataset_dir: Path, pattern: str) -> list[str]:
    images = sorted(p.name for p in dataset_dir.glob(pattern))
    if not images:
        sys.exit(
            f"No images matching '{pattern}' found in {dataset_dir}.\n"
            f"Check pipeline/config.py -> DATASET_DIR / IMAGE_GLOB."
        )
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=cfg.DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=cfg.OUTPUT_DIR)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse features/matches already on disk instead of recomputing.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "features.h5"
    matches_path = output_dir / "matches.h5"
    pairs_path = output_dir / "pairs-exhaustive.txt"
    sfm_dir = output_dir / "sparse"

    images = get_image_list(dataset_dir, cfg.IMAGE_GLOB)
    print(f"[1/5] Found {len(images)} images in {dataset_dir}")

    # ---- 1. Feature extraction (SuperPoint) -----------------------------
    feature_conf = extract_features.confs[cfg.FEATURE_CONF]
    if args.skip_existing and features_path.exists():
        print(f"[2/5] Reusing existing features at {features_path}")
    else:
        print(f"[2/5] Extracting SuperPoint features ({cfg.FEATURE_CONF})")
        extract_features.main(
            feature_conf,
            dataset_dir,
            image_list=images,
            feature_path=features_path,
        )

    # ---- 2. Pair generation ---------------------------------------------
    print("[3/5] Building exhaustive pair list")
    pairs_from_exhaustive.main(pairs_path, image_list=images)

    # ---- 3. Matching (LightGlue) -----------------------------------------
    matcher_conf = match_features.confs[cfg.MATCHER_CONF]
    if args.skip_existing and matches_path.exists():
        print(f"[4/5] Reusing existing matches at {matches_path}")
    else:
        print(f"[4/5] Matching pairs with LightGlue ({cfg.MATCHER_CONF})")
        match_features.main(
            matcher_conf,
            pairs_path,
            features=features_path,
            matches=matches_path,
        )

    # ---- 4. Incremental SfM + bundle adjustment (pycolmap via hloc) -----
    print("[5/5] Running incremental mapping (pycolmap) + bundle adjustment")
    model = hloc_reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=dataset_dir,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
        camera_mode=pycolmap.CameraMode.SINGLE,  # one shared camera for the sequence
        image_options={"camera_model": cfg.CAMERA_MODEL},
    )

    if model is None:
        sys.exit(
            "Reconstruction failed: hloc/pycolmap could not register enough "
            "images. Check that the dataset path and image glob are correct "
            "and that matches were found for a well-connected pair graph."
        )

    print(f"\nReconstruction succeeded: {model.summary()}")

    # ---- 5. Export --------------------------------------------------------
    ply_path = output_dir / "points3D.ply"
    poses_path = output_dir / "poses.txt"
    export_point_cloud(model, ply_path)
    export_poses(model, poses_path)
    print(f"Exported sparse point cloud -> {ply_path}")
    print(f"Exported camera poses       -> {poses_path}")
    print(f"Full COLMAP model (bin)     -> {sfm_dir}")


if __name__ == "__main__":
    main()
