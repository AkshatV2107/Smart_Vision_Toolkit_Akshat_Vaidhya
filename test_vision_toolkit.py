import sys
import unittest
from pathlib import Path
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_toolkit.pipeline import (
    load_image, to_grayscale, blur_image, detect_edges,
    threshold_image, find_objects, run_pipeline
)
from vision_toolkit.synthetic import create_sample_image

class VisionToolkitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vision_toolkit_test_"))
        self.img_path = self.tmp / "sample.png"
        create_sample_image(self.img_path)
        self.image = load_image(self.img_path)
        self.gray = to_grayscale(self.image)

    def test_01_load_image(self):
        self.assertIsNotNone(self.image)
        self.assertEqual(self.image.shape, (480, 640, 3))

    def test_02_grayscale(self):
        self.assertEqual(len(self.gray.shape), 2)
        self.assertEqual(self.gray.shape, (480, 640))

    def test_03_blur(self):
        blurred = blur_image(self.gray)
        self.assertEqual(blurred.shape, self.gray.shape)

    def test_04_edges(self):
        edges = detect_edges(blur_image(self.gray))
        self.assertGreater(int(np.count_nonzero(edges)), 0)

    def test_05_threshold(self):
        threshold = threshold_image(self.gray)
        values = set(np.unique(threshold).tolist())
        self.assertTrue(values.issubset({0, 255}))

    def test_06_contours(self):
        edges = detect_edges(blur_image(self.gray))
        annotated, objects = find_objects(edges, self.image)
        self.assertEqual(annotated.shape, self.image.shape)
        self.assertGreaterEqual(len(objects), 3)

    def test_07_pipeline_outputs(self):
        out = self.tmp / "outputs"
        result = run_pipeline(self.img_path, out)
        expected = [
            "01_original.png", "02_grayscale.png", "03_blurred.png",
            "04_edges.png", "05_threshold.png", "06_contours.png",
            "07_comparison.png"
        ]
        for name in expected:
            self.assertTrue((out / name).exists(), name)
        self.assertGreaterEqual(result["object_count"], 3)

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(VisionToolkitTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print("-" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("OVERALL: PASS" if result.wasSuccessful() else "OVERALL: FAIL")
    raise SystemExit(0 if result.wasSuccessful() else 1)
