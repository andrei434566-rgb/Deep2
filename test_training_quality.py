from __future__ import annotations

import unittest

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.training_quality import review_queue, training_quality


_APP = QApplication.instance() or QApplication([])


class TrainingQualityTests(unittest.TestCase):
    def test_reports_small_classes_and_orders_uncertain_automatic_masks(self):
        pixmap = QPixmap(12, 12)
        records = [PhotoRecord("one", "memory", pixmap, detections=[
            FaciesDetection("A", 1.0, [QPointF(0, 0)] * 3, training_ready=True, depth_from=1, depth_to=2),
            FaciesDetection("B", 1.0, [QPointF(0, 0)] * 3, training_ready=True, depth_from=2, depth_to=3),
            FaciesDetection("A", 0.30, [QPointF(0, 0)] * 3),
            FaciesDetection("B", 0.70, [QPointF(0, 0)] * 3),
        ])]
        quality = training_quality(records, min_per_facies=2)
        self.assertEqual({"A": 1, "B": 1}, quality.underrepresented)
        queue = review_queue(records)
        self.assertEqual("A", queue[0][1].label)


if __name__ == "__main__":
    unittest.main()
