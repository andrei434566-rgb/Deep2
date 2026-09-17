from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication

from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.training_dataset import _photo_level_splits, export_training_dataset
from app.infrastructure.training_recommendation import recommend_training


_APP = QApplication.instance() or QApplication([])


def _detection(label: str, top: float = 0.0) -> FaciesDetection:
    return FaciesDetection(
        label,
        1.0,
        [
            QPointF(1, top + 1),
            QPointF(30, top + 1),
            QPointF(30, top + 10),
            QPointF(1, top + 10),
        ],
        training_ready=True,
        depth_from=top,
        depth_to=top + 1,
    )


def _record(identifier: str, detections: list[FaciesDetection], well: str = "well") -> PhotoRecord:
    pixmap = QPixmap(40, 220)
    # Different record ids represent different source photographs.  The
    # exporter hashes pixels, so identical blank test pixmaps are correctly
    # treated as copied files rather than independent evidence.
    value = sum((index + 1) * ord(char) for index, char in enumerate(identifier)) % 240
    pixmap.fill(QColor(value + 8, (value * 3) % 248 + 8, (value * 7) % 248 + 8))
    return PhotoRecord(identifier, f"{identifier}.jpg", pixmap, detections=detections, well_name=well)


class TrainingRecommendationTests(unittest.TestCase):
    def test_many_source_photos_allow_more_training_than_correlated_masks(self):
        one_photo = [_record("one", [_detection("A", i * 10) for i in range(10)] + [_detection("B", 110 + i * 10) for i in range(10)])]
        many_photos = [
            _record(str(index), [_detection("A" if index % 2 == 0 else "B")])
            for index in range(20)
        ]

        correlated = recommend_training(one_photo, known_model_classes={"A", "B"})
        diverse = recommend_training(many_photos, known_model_classes={"A", "B"})

        self.assertGreater(diverse.recommended_epochs, correlated.recommended_epochs)
        self.assertEqual("низкая", correlated.confidence)
        self.assertEqual("высокая", diverse.confidence)

    def test_new_classes_increase_the_epoch_recommendation(self):
        records = [
            _record(str(index), [_detection(("A", "B", "C")[index % 3])])
            for index in range(18)
        ]

        known = recommend_training(records, known_model_classes={"A", "B", "C"})
        with_new = recommend_training(records, known_model_classes={"A"})

        self.assertGreater(with_new.recommended_epochs, known.recommended_epochs)
        self.assertEqual(("B", "C"), with_new.new_classes)
        self.assertGreater(with_new.max_epochs, with_new.recommended_epochs)
        self.assertGreaterEqual(with_new.patience, 8)

    def test_photo_split_never_mixes_one_source_between_train_and_val(self):
        records = [_record(str(index), [_detection("A")]) for index in range(5)]
        samples = [(record, record.detections[0]) for record in records]

        split = _photo_level_splits(samples)

        self.assertIsNotNone(split)
        self.assertEqual({"train", "val"}, set(split.values()))
        self.assertEqual(len(records), len(split))

    def test_photo_split_falls_back_when_each_photo_owns_a_unique_class(self):
        records = [_record("one", [_detection("A")]), _record("two", [_detection("B")])]
        samples = [(record, record.detections[0]) for record in records]

        self.assertIsNone(_photo_level_splits(samples))

    def test_export_reports_independent_photo_validation(self):
        records = [_record(str(index), [_detection("A")]) for index in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            result = export_training_dataset(records, Path(directory) / "dataset", strict_catalog=False, balance=False)
            train_files = list((Path(result["output_dir"]) / "images" / "train").glob("*.jpg"))
            val_files = list((Path(result["output_dir"]) / "images" / "val").glob("*.jpg"))

        self.assertTrue(result["validation_grouped_by_photo"])
        self.assertEqual(4, result["train_photo_count"])
        self.assertEqual(1, result["val_photo_count"])
        self.assertEqual(4, len(train_files))
        self.assertEqual(1, len(val_files))


if __name__ == "__main__":
    unittest.main()
