"""Regression checks for the facies confidence popover data."""

import unittest

from PySide6.QtCore import QPointF, Qt

from app.domain.models import FaciesDetection
from app.ui.widgets.workspace_canvas import facies_confidence_rows, facies_visual_samples


class FaciesConfidencePopupTests(unittest.TestCase):
    def test_primary_and_nearest_variants_are_sorted_by_confidence(self):
        detection = FaciesDetection(
            label="Tl",
            confidence=0.42,
            polygon=[QPointF(0, 0), QPointF(1, 0), QPointF(1, 1)],
            alternatives={"Shf": 0.61, "Sl": 0.17},
        )

        self.assertEqual([("Shf", 0.61), ("Tl", 0.42), ("Sl", 0.17)], facies_confidence_rows(detection))
    def test_visual_samples_use_an_in_memory_crop(self):
        from PySide6.QtGui import QPixmap
        from app.domain.models import PhotoRecord

        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.white)
        record = PhotoRecord("one", "memory", pixmap, detections=[
            FaciesDetection("Tl", 0.8, [QPointF(2, 2), QPointF(20, 2), QPointF(20, 20), QPointF(2, 20)]),
        ])
        samples = facies_visual_samples([record])
        self.assertIn("Tl", samples)
        self.assertFalse(samples["Tl"].isNull())


if __name__ == "__main__":
    unittest.main()
