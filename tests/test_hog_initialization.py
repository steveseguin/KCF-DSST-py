"""Offline regressions using real NumPy, OpenCV and Numba feature extraction."""

import os
from pathlib import Path
import sys
import unittest

import numpy as np

# Allows the same tests to run against an unchanged source snapshot.
SOURCE_ROOT = Path(os.environ.get("KCF_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(SOURCE_ROOT))

import fhog
from tracker import KCFTracker


class HogInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = np.random.RandomState(42).randint(
            0, 256, size=(128, 160, 3)
        ).astype(np.uint8)

    def assert_finite_features(self, features):
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(np.count_nonzero(features), 0)

    def test_hog_feature_maps_for_square_and_rectangular_images(self):
        for height, width in [(32, 32), (36, 48)]:
            with self.subTest(height=height, width=width):
                feature_map = fhog.getFeatureMaps(
                    self.image[:height, :width], 4, {}
                )
                self.assertEqual(feature_map["sizeY"], height // 4)
                self.assertEqual(feature_map["sizeX"], width // 4)
                self.assertEqual(feature_map["numFeatures"], 27)
                self.assertEqual(
                    feature_map["map"].size, (height // 4) * (width // 4) * 27
                )
                self.assert_finite_features(feature_map["map"])

    def test_complete_hog_normalization_and_pca_pipeline(self):
        feature_map = fhog.getFeatureMaps(self.image[:32, :48], 4, {})
        feature_map = fhog.normalizeAndTruncate(feature_map, 0.2)
        feature_map = fhog.PCAFeatureMaps(feature_map)
        self.assertEqual(
            (feature_map["sizeY"], feature_map["sizeX"], feature_map["numFeatures"]),
            (6, 10, 31),
        )
        self.assertEqual(feature_map["map"].size, 6 * 10 * 31)
        self.assert_finite_features(feature_map["map"])

    def check_tracker(self, hog, multi_scale):
        tracker = KCFTracker(hog=hog, fixed_window=True, multi_scale=multi_scale)
        initial_roi = [40, 40, 32, 32]
        tracker.init(initial_roi, self.image)
        self.assertTrue(np.isfinite(tracker._tmpl).all())
        self.assertTrue(np.isfinite(tracker._alphaf).all())
        self.assertEqual(tracker._hogfeatures, hog or multi_scale)
        if multi_scale:
            self.assertTrue(np.isfinite(tracker.sf_num).all())
            self.assertTrue(np.isfinite(tracker.sf_den).all())

        for _ in range(3):
            roi = tracker.update(self.image)
            self.assertEqual(len(roi), 4)
            self.assertTrue(np.isfinite(roi).all())
            self.assertGreater(roi[2], 0)
            self.assertGreater(roi[3], 0)
            # Fixed-scale controls also check the location on a repeated frame.
            # This startup regression is not a DSST tracking-accuracy benchmark.
            if not multi_scale:
                np.testing.assert_allclose(roi, initial_roi, atol=1)

    def test_grayscale_tracker_control(self):
        self.check_tracker(hog=False, multi_scale=False)

    def test_hog_tracker_initialization_and_updates(self):
        self.check_tracker(hog=True, multi_scale=False)

    def test_multiscale_forces_hog_and_initializes(self):
        self.check_tracker(hog=False, multi_scale=True)

    def test_demo_configuration_initialization_and_updates(self):
        self.check_tracker(hog=True, multi_scale=True)


if __name__ == "__main__":
    unittest.main()
