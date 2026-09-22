from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from excel_photo_model_studio.dataset import build_dataset
from excel_photo_model_studio.matching import (
    match_photos, read_photo_map, suggest_missing_intervals,
    uncovered_photo_description_intervals, uncovered_photo_intervals, write_photo_map,
)
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

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=500):
            resolved = suggest_missing_intervals([good, false_pair], rows)

        self.assertEqual("67ПО", resolved[0].well)
        self.assertTrue(resolved[0].mapping_confirmed)
        self.assertEqual("ocr_sequenced", resolved[0].source)
        self.assertEqual((4104.9, 4109.9), (resolved[0].top, resolved[0].base))
        self.assertFalse(resolved[1].mapping_confirmed)

    def test_sequences_ocr_pages_by_depth_and_carries_long_facies_between_them(self):
        rows = [DescriptionRow(
            well="W-1", top=102.0, base=105.03, label="Dch", sheet="Data", row=2,
            core_top=100.0, core_base=106.03, thickness=3.03,
            target_text="Одно описание всей фации длиной 3,03 м.",
        )]
        records = [
            PhotoRecord(Path(f"core-{suffix}.jpg"), "W-1", 100.0, 106.03, "ocr", False)
            for suffix in ("0003", "0001", "0005")
        ]

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=300):
            sequenced = suggest_missing_intervals(records, rows)
        matches, _ = match_photos(sequenced, rows)

        self.assertEqual(
            [(100.0, 103.0), (103.0, 106.0), (106.0, 106.03)],
            [(item.top, item.base) for item in sequenced],
        )
        self.assertEqual(["core-0001.jpg", "core-0003.jpg"], [item.photo.path.name for item in matches])
        self.assertEqual([(102.0, 103.0), (103.0, 105.03)], [
            (item.overlap_top, item.overlap_base) for item in matches
        ])
        self.assertTrue(all(
            item.description.target_text == "Одно описание всей фации длиной 3,03 м."
            for item in matches
        ))

    def test_sequences_every_photo_when_ocr_misses_first_and_last_pages(self):
        rows = [
            DescriptionRow(
                well="W-1", top=100.0, base=112.0, label="A", sheet="Data", row=2,
                core_top=100.0, core_base=112.0, target_text="Первая фация.",
            ),
            DescriptionRow(
                well="W-1", top=113.0, base=130.94, label="B", sheet="Data", row=3,
                core_top=113.0, core_base=130.94, target_text="Вторая фация.",
            ),
            DescriptionRow(
                well="W-1", top=131.0, base=145.0, label="C", sheet="Data", row=4,
                core_top=131.0, core_base=145.0, target_text="Третья фация.",
            ),
        ]
        capacities = {
            "core-0001": 500, "core-0003": 500, "core-0005": 200,
            "core-0007": 500, "core-0009": 500, "core-0011": 500, "core-0013": 294,
            "core-0015": 500, "core-0017": 500, "core-0019": 400,
        }
        records = []
        known_intervals = {
            7: (113.0, 118.0), 9: (118.0, 123.0),
            11: (123.0, 128.0), 13: (128.0, 130.94),
            15: (131.0, 136.0), 17: (136.0, 141.0),
        }
        for suffix in range(1, 20, 2):
            path = Path(f"core-{suffix:04d}.jpg")
            if suffix in known_intervals:
                top, base = known_intervals[suffix]
                records.append(PhotoRecord(path, "W-1", top, base, "ocr_sequenced", True))
            else:
                records.append(PhotoRecord(path, "W-1", source="ocr_not_found"))

        with patch(
            "excel_photo_model_studio.matching._photo_core_capacity_cm",
            side_effect=lambda path: capacities[path.stem],
        ):
            sequenced = suggest_missing_intervals(records, rows)
        matches, unresolved = match_photos(sequenced, rows)

        self.assertEqual(10, len(sequenced))
        self.assertTrue(all(item.mapping_confirmed for item in sequenced))
        self.assertTrue(all(item.source == "excel_sequenced" for item in sequenced))
        self.assertEqual((100.0, 105.0), (sequenced[0].top, sequenced[0].base))
        self.assertEqual((110.0, 112.0), (sequenced[2].top, sequenced[2].base))
        self.assertEqual((113.0, 118.0), (sequenced[3].top, sequenced[3].base))
        self.assertEqual((128.0, 130.94), (sequenced[6].top, sequenced[6].base))
        self.assertEqual((141.0, 145.0), (sequenced[9].top, sequenced[9].base))
        self.assertEqual([], unresolved)
        self.assertEqual(10, len(matches))
        self.assertTrue(all(not uncovered_photo_intervals(photo, matches) for photo in sequenced))

    def test_reports_any_centimetre_not_covered_by_a_facies(self):
        photo = PhotoRecord(Path("W-1 100-103.jpg"), "W-1", 100.0, 103.0, "filename", True)
        rows = [
            DescriptionRow("W-1", 100.0, 101.5, "A", "Data", 2),
            DescriptionRow("W-1", 101.51, 103.0, "B", "Data", 3),
        ]

        matches, _ = match_photos([photo], rows)

        self.assertEqual([(101.5, 101.51)], uncovered_photo_intervals(photo, matches))

    def test_reports_core_part_whose_facies_has_no_short_description(self):
        photo = PhotoRecord(Path("W-1 100-102.jpg"), "W-1", 100.0, 102.0, "filename", True)
        rows = [
            DescriptionRow(
                "W-1", 100.0, 101.0, "A", "Data", 2,
                target_text="Обязательное описание первой фации.",
            ),
            DescriptionRow("W-1", 101.0, 102.0, "B", "Data", 3),
        ]

        matches, _ = match_photos([photo], rows)

        self.assertEqual([], uncovered_photo_intervals(photo, matches))
        self.assertEqual(
            [(101.0, 102.0)],
            uncovered_photo_description_intervals(photo, matches),
        )

    def test_dataset_is_blocked_when_any_photo_has_no_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            (project / "report.json").write_text(json.dumps({
                "photos": 10,
                "confirmed_photos": 9,
                "photos_without_intervals": 1,
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "фото без обязательного интервала: 1"):
                build_dataset(project, root / "dataset")

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
