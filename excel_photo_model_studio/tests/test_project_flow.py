from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from openpyxl import Workbook

from excel_photo_model_studio.dataset import build_dataset
from excel_photo_model_studio.matching import read_photo_map
from excel_photo_model_studio.models import DescriptionRow, Match, PhotoRecord
from excel_photo_model_studio.project import (
    _write_facies_inventory, create_project, load_annotations,
    refresh_project, set_annotation_approvals,
)


class ProjectFlowTests(unittest.TestCase):
    def test_all_ten_pages_and_all_core_columns_receive_facies_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excel = root / "description.xlsx"
            photos_dir = root / "photos"
            photos_dir.mkdir()
            workbook = Workbook()
            sheet = workbook.active
            sheet.append([
                "Скважина", "Интервал фации по бурению Кровля",
                "Интервал фации по бурению Подошва", "Толщина фации, м",
                "Название фации", "Краткое описание",
                "Интервал отбора керна Кровля", "Интервал отбора керна Подошва",
            ])
            sheet.append(["W-1", 100.0, 150.0, 50.0, "Sand", "Песчаник серый.", 100.0, 150.0])
            workbook.save(excel)

            for index in range(10):
                image = np.full((900, 850, 3), (25, 90, 160), dtype=np.uint8)
                for column in range(5):
                    left = 60 + column * 150
                    cv2.rectangle(image, (left, 80), (left + 90, 820), (115, 115, 115), -1)
                sequence = index * 2 + 1
                stem = f"W-1 page-{sequence:04d}"
                if index == 0:
                    stem += " 100.00-105.00"
                elif index == 9:
                    stem += " 145.00-150.00"
                ok, encoded = cv2.imencode(".jpg", image)
                self.assertTrue(ok)
                (photos_dir / f"{stem}.jpg").write_bytes(encoded.tobytes())

            report = create_project(excel, photos_dir, root / "project")
            inventory_path = root / "project" / "photo_inventory.csv"
            with inventory_path.open("r", encoding="utf-8-sig", newline="") as source:
                photo_inventory = list(csv.DictReader(source, delimiter=";"))
            facies_inventory_path = root / "project" / "facies_inventory.csv"
            with facies_inventory_path.open("r", encoding="utf-8-sig", newline="") as source:
                facies_inventory = list(csv.DictReader(source, delimiter=";"))

        self.assertEqual(10, report["photos"])
        self.assertEqual(10, report["confirmed_photos"])
        self.assertEqual(0, report["photos_without_masks"])
        self.assertEqual(50, report["annotations"])
        self.assertEqual(0, report["excel_facies_rows_without_photo_match"])
        self.assertEqual(10, len(photo_inventory))
        self.assertTrue(all(item["status"] == "READY_FOR_REVIEW" for item in photo_inventory))
        self.assertEqual("MATCHED", facies_inventory[0]["status"])
        self.assertEqual(10, len(facies_inventory[0]["matched_photos"].split(", ")))

    def test_facies_inventory_exposes_rows_without_matching_photo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "facies_inventory.csv"
            photo = PhotoRecord(Path("W-1 100-101.jpg"), "W-1", 100.0, 101.0, "filename", True)
            rows = [
                DescriptionRow("W-1", 100.0, 101.0, "A", "Data", 2, target_text="First."),
                DescriptionRow(
                    "W-1", 101.0, 102.0, "B", "Data", 3,
                    thickness=0.5, thickness_valid=False, target_text="Second.",
                ),
                DescriptionRow("W-2", 200.0, 201.0, "C", "Data", 4, target_text="Third."),
            ]
            matches = [Match(photo, rows[0], 100.0, 101.0)]

            unmatched = _write_facies_inventory(path, rows, [photo], matches)
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                inventory = list(csv.DictReader(source, delimiter=";"))

        self.assertEqual(2, unmatched)
        self.assertEqual(["MATCHED", "INVALID_FACIES_THICKNESS", "NO_PHOTO_OVERLAP"], [
            item["status"] for item in inventory
        ])

    def test_excel_photos_review_and_dataset_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excel = root / "description.xlsx"
            photos = root / "photos"
            photos.mkdir()
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["Скважина", "Кровля", "Подошва", "Класс", "Краткое описание"])
            sheet.append(["W-1", 100.0, 101.0, "Sand", "Описание песчаника."])
            sheet.append(["W-2", 100.0, 101.0, "Sand", "Описание песчаника."])
            workbook.save(excel)
            for well in ("W-1", "W-2"):
                image = np.full((240, 160, 3), (20, 80, 150), dtype=np.uint8)
                shade = 115 if well == "W-1" else 130
                cv2.rectangle(image, (55, 20), (105, 220), (shade, shade, shade), -1)
                ok, encoded = cv2.imencode(".jpg", image)
                self.assertTrue(ok)
                (photos / f"{well} 100-101.jpg").write_bytes(encoded.tobytes())

            report = create_project(excel, photos, root / "project")
            photo_map = read_photo_map(root / "project" / "photo_map.csv")
            self.assertTrue(all(photo.depth_basis == "drilling" for photo in photo_map))
            self.assertTrue((root / "project" / "table_cache.json").is_file())
            self.assertTrue((root / "project" / "photo_inventory.csv").is_file())
            self.assertTrue((root / "project" / "facies_inventory.csv").is_file())
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
            self.assertEqual(2, report["photos_with_facies"])
            self.assertEqual(0, report["photos_without_masks"])
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
