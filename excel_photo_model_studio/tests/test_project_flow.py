from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from openpyxl import Workbook

from excel_photo_model_studio.dataset import build_dataset
from excel_photo_model_studio.project import create_project, load_annotations, refresh_project, set_annotation_approvals


class ProjectFlowTests(unittest.TestCase):
    def test_excel_photos_review_and_dataset_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excel = root / "description.xlsx"
            photos = root / "photos"
            photos.mkdir()
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["Скважина", "Кровля", "Подошва", "Класс", "Краткое описание"])
            sheet.append(["W-1", 100.0, 102.0, "Sand", "Описание песчаника."])
            sheet.append(["W-2", 100.0, 102.0, "Sand", "Описание песчаника."])
            workbook.save(excel)
            for well in ("W-1", "W-2"):
                image = np.full((240, 160, 3), (20, 80, 150), dtype=np.uint8)
                cv2.rectangle(image, (55, 20), (105, 220), (115, 115, 115), -1)
                ok, encoded = cv2.imencode(".jpg", image)
                self.assertTrue(ok)
                (photos / f"{well} 100-102.jpg").write_bytes(encoded.tobytes())

            report = create_project(excel, photos, root / "project")
            self.assertTrue((root / "project" / "table_cache.json").is_file())
            detected = json.loads((root / "project" / "detected_columns.json").read_text(encoding="utf-8"))
            self.assertTrue(all(len(item["depth_ranges"]) == 1 for item in detected.values()))
            with patch(
                "excel_photo_model_studio.project.read_many_tables",
                side_effect=AssertionError("unchanged Excel must be loaded from the project cache"),
            ):
                refresh_project(root / "project")
            annotations = load_annotations(root / "project")
            self.assertEqual(2, report["confirmed_photos"])
            self.assertEqual(0, report["photos_without_intervals"])
            self.assertEqual(0, report["photos_without_core_columns"])
            self.assertEqual(0, report["uncovered_facies_intervals"])
            self.assertEqual(0, report["uncovered_description_intervals"])
            self.assertGreaterEqual(len(annotations), 2)
            set_annotation_approvals(
                root / "project", {row["annotation_id"]: True for row in annotations}
            )
            dataset = build_dataset(root / "project", root / "dataset")
            manifest = json.loads((root / "dataset" / "dataset_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(["Sand"], dataset["class_names"])
        self.assertEqual(1, manifest["val_photo_count"])
        self.assertEqual(1, manifest["train_photo_count"])


if __name__ == "__main__":
    unittest.main()
