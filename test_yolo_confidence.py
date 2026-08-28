"""Small regression checks for the configurable YOLO confidence threshold."""

import unittest

from PySide6.QtCore import QPointF

from app.infrastructure.ml.yolo_model_service import YoloModelService


class ConfidenceThresholdTests(unittest.TestCase):
    def test_normalizes_configured_threshold(self):
        self.assertEqual(0.25, YoloModelService._normalize_confidence(0.25))
        self.assertEqual(0.01, YoloModelService._normalize_confidence(-1))
        self.assertEqual(0.99, YoloModelService._normalize_confidence(2))
        self.assertEqual(0.50, YoloModelService._normalize_confidence("wrong"))

    def test_image_size_is_limited_to_supported_range(self):
        self.assertEqual(320, YoloModelService._normalize_image_size(20))
        self.assertEqual(1024, YoloModelService._normalize_image_size(1024))
        self.assertEqual(1536, YoloModelService._normalize_image_size(3000))

    def test_shlak_and_background_masks_are_excluded_or_clipped_to_core(self):
        self.assertTrue(YoloModelService._is_excluded_label("shlak"))
        self.assertTrue(YoloModelService._is_excluded_label("ШЛАК"))
        points = [QPointF(-5, 10), QPointF(25, 10), QPointF(25, 35), QPointF(-5, 35)]
        clipped = YoloModelService._clip_polygon_to_core_columns(points, [(0, 0, 20, 50)], 1.0, 1.0)
        self.assertGreaterEqual(len(clipped), 3)
        self.assertTrue(all(0 <= point.x() <= 20 and 0 <= point.y() <= 50 for point in clipped))


if __name__ == "__main__":
    unittest.main()
