"""Regression tests for CVAT COCO import variants."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from app.infrastructure.cvat_coco_import import CvatImagesMissingError, import_cvat_coco_zip, import_cvat_coco_zips


class CvatCocoImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def _image_bytes() -> bytes:
        image = np.full((12, 16, 3), 150, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        return encoded.tobytes()

    @staticmethod
    def _rle_counts(mask: np.ndarray) -> list[int]:
        values = mask.reshape(-1, order="F").astype(bool)
        counts, previous, length = [], False, 0
        for value in values:
            if bool(value) == previous:
                length += 1
            else:
                counts.append(length)
                previous, length = bool(value), 1
        counts.append(length)
        return counts

    def test_imports_embedded_polygon_and_rle_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "cvat.zip"
            mask = np.zeros((12, 16), dtype=np.uint8)
            mask[6:10, 8:13] = 1
            payload = {
                "images": [{"id": 7, "file_name": "nested/Керн 01.png"}],
                "categories": [{"id": 3, "name": "F1"}],
                "annotations": [
                    {"id": 1, "image_id": 7, "category_id": 3, "segmentation": [[1, 1, 6, 1, 6, 5, 1, 5]]},
                    {"id": 2, "image_id": 7, "category_id": 3, "segmentation": {"size": [12, 16], "counts": self._rle_counts(mask)}},
                ],
            }
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("annotations/export.json", json.dumps(payload, ensure_ascii=False))
                archive.writestr("images/default/nested/Керн 01.png", self._image_bytes())

            records, summary = import_cvat_coco_zip(archive_path, root / "project" / "images")

            self.assertEqual(1, len(records))
            self.assertEqual(2, len(records[0].detections))
            self.assertEqual(2, summary["contours"])
            self.assertTrue(all(item.training_ready for item in records[0].detections))

    def test_imports_annotation_only_zip_from_selected_image_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "annotations.zip"
            image_root = root / "source" / "deep"
            image_root.mkdir(parents=True)
            (image_root / "core.png").write_bytes(self._image_bytes())
            payload = {
                "images": [{"id": 1, "file_name": "deep/core.png"}],
                "categories": [{"id": 1, "name": "F2"}],
                "annotations": [{"image_id": 1, "category_id": 1, "bbox": [2, 3, 5, 4]}],
            }
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("anything.json", json.dumps(payload))

            with self.assertRaises(CvatImagesMissingError):
                import_cvat_coco_zip(archive_path, root / "first")
            records, summary = import_cvat_coco_zip(archive_path, root / "second", root / "source")

            self.assertEqual(1, summary["images"])
            self.assertEqual(1, len(records[0].detections))

    def test_imports_archives_sequentially_and_removes_exact_duplicate_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, image_name in enumerate(("same.png", "same.png"), start=1):
                archive_path = root / f"job_{index}.zip"
                payload = {
                    "images": [{"id": 1, "file_name": image_name}],
                    "categories": [{"id": 1, "name": "F1"}],
                    "annotations": [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]}],
                }
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("annotations.json", json.dumps(payload))
                    archive.writestr(f"images/{image_name}", self._image_bytes())
                paths.append(archive_path)

            records, summary = import_cvat_coco_zips(paths, root / "project" / "images")

            self.assertEqual(1, len(records))
            self.assertEqual(2, summary["archives"])
            self.assertEqual(1, summary["duplicate_images"])


if __name__ == "__main__":
    unittest.main()
