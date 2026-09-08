"""Modular classical Structure-from-Motion pipeline."""

from .bundle_adjustment import (
    ba_residuals,
    build_ba_sparsity,
    build_observations,
    bundle_adjust,
    pack_params,
    project,
    unpack_params,
)
from .exporters import export_cameras, export_colmap_text, export_ply
from .features import detect_features, make_detector, match_all, match_pair
from .geometry import (
    cheirality_mask,
    geometric_verify,
    pair_parallax_score,
    triangulate_points,
    triangulation_angle_deg,
)
from .image_io import estimate_intrinsics, load_images
from .models import Camera, Frame
from .pipeline import run_pipeline
from .reconstruction import (
    initialize_reconstruction,
    register_next_image,
    run_incremental_sfm,
    triangulate_new_points,
)
from .reporting import RUN_STATS, print_metrics_summary, record_match_count
from .visualization import export_open3d

__all__ = [
    "Camera",
    "Frame",
    "RUN_STATS",
    "ba_residuals",
    "build_ba_sparsity",
    "build_observations",
    "bundle_adjust",
    "cheirality_mask",
    "detect_features",
    "estimate_intrinsics",
    "export_cameras",
    "export_colmap_text",
    "export_open3d",
    "export_ply",
    "geometric_verify",
    "initialize_reconstruction",
    "load_images",
    "make_detector",
    "match_all",
    "match_pair",
    "pack_params",
    "pair_parallax_score",
    "print_metrics_summary",
    "project",
    "record_match_count",
    "register_next_image",
    "run_incremental_sfm",
    "run_pipeline",
    "triangulate_new_points",
    "triangulate_points",
    "triangulation_angle_deg",
    "unpack_params",
]
