"""Command-line entry point for direct classical-pipeline execution."""

import argparse

from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Folder of input images")
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument(
        "--feature_type", choices=["sift", "orb"], default="orb",
        help="orb = baseline, sift = improved classical",
    )
    parser.add_argument("--focal_mm", type=float, default=None)
    parser.add_argument("--sensor_width_mm", type=float, default=None)
    parser.add_argument("--min_inliers_init", type=int, default=60)
    parser.add_argument(
        "--export_colmap", action="store_true",
        help="Also export COLMAP-format sparse/ + images/, ready for Gaussian Splatting training",
    )
    args = parser.parse_args()
    run_pipeline(
        images=args.images, out=args.out, feature_type=args.feature_type,
        focal_mm=args.focal_mm, sensor_width_mm=args.sensor_width_mm,
        min_inliers_init=args.min_inliers_init, export_colmap=args.export_colmap,
    )
