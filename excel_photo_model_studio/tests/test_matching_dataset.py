from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from excel_photo_model_studio.dataset import build_dataset
from excel_photo_model_studio.matching import match_photos, read_photo_map, suggest_missing_intervals, write_photo_map
from excel_photo_model_studio.models import (
    COLUMN_ORDER_RIGHT_TO_LEFT, DescriptionRow, PhotoRecord,
)
from excel_photo_model_studio.photos import parse_filename


class MatchingTests(unittest.TestCase):
    def test_ocr_interval_is_auto_confirmed_only_when_it_matches_excel_core_interval(self):
        rows = [DescriptionRow(
            well="67ПО", top=4105.0, base=4108.65, label="Dch", sheet="седимент", row=5,
            core_top=4104.9, core_base=4116.9, target_text="Песчаник серый.",
        )]
        good = PhotoRecord(
            Path("Рис. 5.1-5.20 Восточно-Тазовское, скв№ 67ПО-0001.jpg"),
            top=4104.9, base=4116.55, source="ocr",
        )
        false_pair = PhotoRecord(
            Path("Рис. 5.1-5.20 Восточно-Тазовское, скв№ 67ПО-0003.jpg"),
            top=4105.0, base=4106.0, source="ocr",
        )

        resolved = suggest_missing_intervals([good, false_pair], rows)

        self.assertEqual("67ПО", resolved[0].well)
        self.assertTrue(resolved[0].mapping_confirmed)
        self.assertEqual("ocr_verified", resolved[0].source)
        self.assertFalse(resolved[1].mapping_confirmed)

    def test_photo_map_preserves_column_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photo_map.csv"
            record = PhotoRecord(
                Path(directory) / "photo.jpg", "W-1", 100.0, 101.0,
                "manual", True, COLUMN_ORDER_RIGHT_TO_LEFT,
            )
            write_photo_map(path, [record])

            loaded = read_photo_map(path)

        self.assertEqual(COLUMN_ORDER_RIGHT_TO_LEFT, loaded[0].column_order)

    def test_filename_and_overlap_matching(self):
        photo = parse_filename(Path("Р-31 3002,00–3004,96 (1).jpg"))
        rows = [
            DescriptionRow("Р31", 3001.0, 3002.5, "A", "Data", 2),
            DescriptionRow("Р31", 3002.5, 3004.0, "B", "Data", 3),
            DescriptionRow("X-1", 3002.5, 3004.0, "C", "Data", 4),
        ]
        matches, unresolved = match_photos([photo], rows)
        self.assertEqual([], unresolved)
        self.assertEqual(["A", "B"], [item.description.label for item in matches])
        self.assertAlmostEqual(3002.0, matches[0].overlap_top)

    def test_row_with_failed_facies_thickness_check_is_not_masked(self):
        photo = PhotoRecord(Path("W-1 100-112.jpg"), "W-1", 100.0, 112.0, "filename", True)
        rows = [DescriptionRow(
            "W-1", 100.0, 112.0, "Dch", "Data", 2,
            thickness=3.65, thickness_valid=False,
        )]

        matches, unresolved = match_photos([photo], rows)

        self.assertEqual([], matches)
        self.assertEqual([photo], unresolved)

    def test_dataset_uses_only_approved_rows_and_separates_photos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            annotations = []
            for index in range(3):
                image_path = root / f"photo_{index}.jpg"
                image = np.full((80, 60, 3), 80 + index * 30, dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", image)
                self.assertTrue(ok)
                image_path.write_bytes(encoded.tobytes())
                annotations.append({
                    "annotation_id": f"a{index}", "photo": str(image_path), "preview": "", "well": f"W-{index}",
                    "photo_top": "100", "photo_base": "101", "depth_top": "100", "depth_base": "101",
                    "label": "Sand", "polygon_json": json.dumps([[5, 5], [50, 5], [50, 70], [5, 70]]),
                    "image_width": "60", "image_height": "80", "source_sheet": "Data", "source_row": str(index + 2),
                    "approved": "1",
                    "target_text": "Песчаник светло-серый, слоистый.",
                })
            path = project / "annotations.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=annotations[0].keys(), delimiter=";")
                writer.writeheader()
                writer.writerows(annotations)

            result = build_dataset(project, root / "dataset")

            manifest = json.loads((root / "dataset" / "dataset_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(3, result["photo_count"])
        self.assertEqual(1, manifest["val_photo_count"])
        self.assertEqual(["Sand"], manifest["class_names"])
        self.assertEqual(3, manifest["caption_count"])


if __name__ == "__main__":
    unittest.main()
