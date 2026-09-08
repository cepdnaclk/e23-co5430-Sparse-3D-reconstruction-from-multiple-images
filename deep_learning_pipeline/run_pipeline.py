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

    import time as _time

    timings = {}

    # ---- 1. Feature extraction (SuperPoint) -----------------------------
    feature_conf = extract_features.confs[cfg.FEATURE_CONF]
    if args.skip_existing and features_path.exists():
        print(f"[2/5] Reusing existing features at {features_path}")
        timings["feature_detection"] = 0.0
    else:
        print(f"[2/5] Extracting SuperPoint features ({cfg.FEATURE_CONF})")
        _t0 = _time.time()
        extract_features.main(
            feature_conf,
            dataset_dir,
            image_list=images,
            feature_path=features_path,
        )
        timings["feature_detection"] = _time.time() - _t0

    # ---- 2. Pair generation ---------------------------------------------
    print("[3/5] Building exhaustive pair list")
    _t0 = _time.time()
    pairs_from_exhaustive.main(pairs_path, image_list=images)
    timings["pair_generation"] = _time.time() - _t0

    # ---- 3. Matching (LightGlue) -----------------------------------------
    matcher_conf = match_features.confs[cfg.MATCHER_CONF]
    if args.skip_existing and matches_path.exists():
        print(f"[4/5] Reusing existing matches at {matches_path}")
        timings["matching"] = 0.0
    else:
        print(f"[4/5] Matching pairs with LightGlue ({cfg.MATCHER_CONF})")
        _t0 = _time.time()
        match_features.main(
            matcher_conf,
            pairs_path,
            features=features_path,
            matches=matches_path,
        )
        timings["matching"] = _time.time() - _t0

    # ---- 4. Incremental SfM + bundle adjustment (pycolmap via hloc) -----
    print("[5/5] Running incremental mapping (pycolmap) + bundle adjustment")
    _t0 = _time.time()
    model = hloc_reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=dataset_dir,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
        camera_mode=pycolmap.CameraMode.SINGLE,  # one shared camera for the sequence
        image_options={"camera_model": cfg.CAMERA_MODEL},
    )
    timings["incremental_sfm"] = _time.time() - _t0

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

    # ---- 6. Unified evaluation (M2 + pose) — duplicated separate impl ----
    # Keep evaluation separate but identical output contract (Option A).
    try:
        from .evaluate import print_m2_summary as _print_m2
        from .evaluate import (
            compute_feature_metrics_from_h5 as _feat_metrics,
            compute_structure_metrics_from_model as _struct_metrics,
            evaluate_poses as _eval_poses,
            load_gt_poses as _load_gt,
        )

        # Use default TempleRing GT if present (mirrors classical run.py)
        _gt_candidates = [
            dataset_dir.parent / "templeR_par.txt",
            Path("datasets/templeRing/templeR_par.txt"),
            Path(cfg.DATASET_DIR).parent.parent / "templeR_par.txt",
        ]
        _gt_path = None
        for _cand in _gt_candidates:
            if _cand.exists():
                _gt_path = _cand
                break

        fm = _feat_metrics(features_path, matches_path)
        _n_images = fm.pop("n_images", len(images))
        sm = _struct_metrics(sfm_dir)
        _n_reg = sm.get("num_registered", len(model.images) if hasattr(model, "images") else 0)
        _print_m2(_n_images, _n_reg, fm, sm, timings)

        if _gt_path is not None:
            try:
                _gt = _load_gt(str(_gt_path))
                # load rotations from model
                _reg = {}
                for _img in model.images.values():
                    _R = _img.cam_from_world().rotation.matrix()
                    _t = __import__("numpy").array(_img.cam_from_world().translation)
                    _reg[_img.name] = (_R, _t)
                _eval_poses(_reg, _gt)
            except Exception as _e:
                print(f"[evaluate] pose evaluation skipped: {_e}")
    except Exception as _e:
        print(f"[evaluate] unified evaluation skipped: {_e}")


if __name__ == "__main__":
    main()
