"""Regression tests for the modular classical SfM package."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import sfm
import sfm_baseline


class CompatibilityTests(unittest.TestCase):
    def test_facade_exposes_existing_entry_points(self):
        names = (
            "load_images",
            "estimate_intrinsics",
            "make_detector",
            "detect_features",
            "match_all",
            "geometric_verify",
            "run_pipeline",
        )
        for name in names:
            self.assertIs(getattr(sfm_baseline, name), getattr(sfm, name))

    def test_facade_shares_runtime_statistics(self):
        self.assertIs(sfm_baseline.RUN_STATS, sfm.RUN_STATS)


class ImageIoTests(unittest.TestCase):
    def test_load_images_deduplicates_case_insensitive_glob_results(self):
        image1 = os.path.abspath(os.path.join("images", "frame001.png"))
        image2 = os.path.abspath(os.path.join("images", "frame002.png"))
        with patch(
            "sfm.image_io.glob.glob",
            side_effect=[[image1, image2], [], [image1, image2], [], []],
        ):
            paths = sfm.load_images("images")
        self.assertEqual(paths, sorted([image1, image2]))

    def test_estimate_intrinsics_uses_image_dimensions(self):
        intrinsics = sfm.estimate_intrinsics((480, 640))
        np.testing.assert_array_equal(
            intrinsics,
            np.array([[640.0, 0.0, 320.0], [0.0, 640.0, 240.0], [0.0, 0.0, 1.0]]),
        )


class BundleAdjustmentParameterTests(unittest.TestCase):
    def test_pack_unpack_round_trip(self):
        registered = [SimpleNamespace(R=np.eye(3), t=np.array([1.0, 2.0, 3.0]))]
        points = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        packed = sfm.pack_params(registered, points)
        camera_params, unpacked_points = sfm.unpack_params(packed, 1, 2)
        np.testing.assert_allclose(camera_params[0, :3], 0.0)
        np.testing.assert_allclose(camera_params[0, 3:], registered[0].t)
        np.testing.assert_allclose(unpacked_points, points)


if __name__ == "__main__":
    unittest.main()
