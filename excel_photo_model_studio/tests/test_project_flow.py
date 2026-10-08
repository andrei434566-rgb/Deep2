from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from openpyxl import Workbook

from excel_photo_model_studio.dataset import build_dataset
from excel_photo_model_studio.matching import read_photo_map, write_photo_map
from excel_photo_model_studio.models import DescriptionRow, Match, PhotoRecord
from excel_photo_model_studio.project import (
    _write_facies_inventory, _write_table_cache, create_project, load_annotations,
    refresh_project, set_annotation_approvals, write_verified_columns,
)


class ProjectFlowTests(unittest.TestCase):
    def test_table_cache_falls_back_when_windows_denies_atomic_rename(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()

            with patch.object(Path, "replace", side_effect=PermissionError("rename denied")):
                _write_table_cache(project, [], [], [], [])

            cached = json.loads((project / "table_cache.json").read_text(encoding="utf-8"))
            self.assertEqual("excel-photo-table-cache-v3", cached["schema"])
            self.assertFalse((project / "table_cache.json.tmp").exists())

    def test_refresh_reuses_guided_core_boxes_instead_of_running_detector_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excel = root / "description.xlsx"
            photos = root / "photos"
            photos.mkdir()
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["Скважина", "Кровля", "Подошва", "Толщина фации", "Код", "Краткое описание"])
            sheet.append(["W-1", 100.0, 101.0, 1.0, "Dch", "Каналы"])
            workbook.save(excel)

            image = np.full((240, 180, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (50, 20), (125, 220), (100, 100, 100), -1)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            photo = photos / "W-1 100.00-101.00.jpg"
            photo.write_bytes(encoded.tobytes())

            project = root / "project"
            create_project(excel, photos, project)
            confirmed_boxes = [(45, 18, 130, 222)]
            write_verified_columns(
                project, {photo: confirmed_boxes}, {photo: "left_to_right"},
            )

            with patch(
                "excel_photo_model_studio.project.detect_core_columns_from_path",
                side_effect=AssertionError("guided geometry must be reused"),
            ) as detector:
                report = refresh_project(project)

            detector.assert_not_called()
            self.assertEqual(1, report["verified_column_photos"])
            detected = json.loads((project / "detected_columns.json").read_text(encoding="utf-8"))
            self.assertEqual([list(confirmed_boxes[0])], detected[str(photo.resolve())]["boxes"])
            annotation = load_annotations(project)[0]
            polygon = json.loads(annotation["polygon_json"])
            # The manually confirmed box is reused and the training mask has
            # a clean, straight four-corner outline.
            self.assertEqual(4, len(polygon))
            self.assertEqual(2, len({point[0] for point in polygon}))
            self.assertGreaterEqual(min(point[0] for point in polygon), 45.0)
            self.assertLessEqual(max(point[0] for point in polygon), 130.0)
            self.assertEqual(0, report["photos_without_masks"])

    def test_refresh_keeps_manual_interval_and_reprojects_facies_on_all_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excel = root / "description.xlsx"
            photos = root / "photos"
            photos.mkdir()
            workbook = Workbook()
            sheet = workbook.active
            sheet.append([
                "Скважина", "Интервал фации по бурению Кровля",
                "Интервал фации по бурению Подошва", "Толщина фации, м",
                "Индекс фации", "Название фации", "Краткое описание",
            ])
            sheet.append([
                "W-1", 100.5, 101.0, 0.5, "Mstf", "Mstf",
                "Неровномерный песчаник",
            ])
            sheet.append([
                "W-1", 101.0, 102.0, 1.0, "Dch", "Dch",
                "Каналы песчаника",
            ])
            workbook.save(excel)
            image = np.full((240, 180, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (50, 20), (125, 220), (100, 100, 100), -1)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            photo_path = photos / "W-1 100.00-102.00.jpg"
            photo_path.write_bytes(encoded.tobytes())
            next_photo_path = photos / "W-1 101.00-102.00.jpg"
            next_photo_path.write_bytes(encoded.tobytes())

            project = root / "project"
            create_project(excel, photos, project)
            photo_map = read_photo_map(project / "photo_map.csv")
            photo = next(item for item in photo_map if item.path == photo_path.resolve())
            corrected = replace(
                photo, top=100.5, base=101.0, source="manual",
                mapping_confirmed=True, column_depths=(), column_ocr_checked=False,
                depth_basis="unknown",
            )
            write_photo_map(
                project / "photo_map.csv",
                [corrected if item.path == photo.path else item for item in photo_map],
            )
            config_path = project / "project.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["use_ocr"] = True
            config_path.write_text(json.dumps(config), encoding="utf-8")
            boxes = {path: [(45, 18, 130, 222)] for path in (photo_path, next_photo_path)}
            write_verified_columns(project, boxes, {path: "left_to_right" for path in boxes})

            def read_depth_labels(*_args, metadata, reference_interval, **_kwargs):
                metadata["depth_basis"] = "drilling"
                if reference_interval and reference_interval[0] >= 101:
                    return ((0.5, 101.0, 102.0),)
                return ((0.5, 100.0, 102.0),)

            with (
                patch("excel_photo_model_studio.photos._configure_tesseract", return_value=True),
                patch("excel_photo_model_studio.vision.detect_core_columns", return_value=[(45, 18, 130, 222)]),
                patch("excel_photo_model_studio.vision.extract_core_column_depths", side_effect=read_depth_labels),
            ):
                report = refresh_project(project)

            refreshed_map = read_photo_map(project / "photo_map.csv")
            refreshed = next(item for item in refreshed_map if item.path == photo.path)
            annotations = load_annotations(project)

        self.assertEqual((100.5, 101.0), (refreshed.top, refreshed.base))
        self.assertEqual("manual", refreshed.source)
        self.assertEqual((), refreshed.column_depths)
        self.assertEqual(2, report["photos_with_facies"])
        self.assertEqual(2, len(annotations), report)
        intervals_by_photo = {
            Path(item["photo"]).name: (float(item["depth_top"]), float(item["depth_base"]))
            for item in annotations
        }
        self.assertEqual((100.5, 101.0), intervals_by_photo[photo_path.name])
        self.assertEqual((101.0, 102.0), intervals_by_photo[next_photo_path.name])

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
