"""Small regression checks for the configurable YOLO confidence threshold."""

import unittest

from PySide6.QtCore import QPointF

from app.infrastructure.ml.yolo_model_service import YoloModelService


class ConfidenceThresholdTests(unittest.TestCase):
    def test_canonical_model_label_preserves_exact_variant_without_invented_lithology(self):
        service = object.__new__(YoloModelService)
        service._facies_catalog = {"Dch@92": {"Индекс фации": "47", "Размер зерен": "Крупнозернистый"}}
        values = service._facies_attributes("Dch@92")
        self.assertEqual("Dch", values["Код фации"])
        self.assertEqual("92", values["Индекс фации"])
        self.assertEqual("Dch@92", values["Класс модели"])
        self.assertNotIn("Размер зерен", values)

    def test_ambiguous_legacy_and_wrong_case_are_not_guessed(self):
        service = object.__new__(YoloModelService)
        service._facies_catalog = {"DWCH": {"Индекс фации": "79"}}
        self.assertNotIn("Индекс фации", service._facies_attributes("Dch"))
        self.assertNotIn("Индекс фации", service._facies_attributes("dwch"))
        self.assertEqual("80", service._facies_attributes("DWCh")["Индекс фации"])

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
