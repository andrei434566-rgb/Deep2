from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from excel_photo_model_studio.photos import (
    discover_photos, enrich_core_column_depths, extract_depth_interval, extract_well, parse_filename,
)
from excel_photo_model_studio.models import PhotoRecord


class PhotoOcrParsingTests(unittest.TestCase):
    def test_discovery_keeps_every_supported_photo_in_nested_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            for index in range(10):
                parent = root if index % 2 == 0 else root / "nested"
                suffix = ".png" if index == 0 else ".JPG"
                (parent / f"core-{index:04d}{suffix}").write_bytes(b"test")
            (root / "ignore.txt").write_text("not a photo", encoding="utf-8")

            photos = discover_photos(root)

        self.assertEqual(10, len(photos))
        self.assertEqual(10, len({photo.path for photo in photos}))

    def test_prefers_adjusted_caption_interval_over_raw_sampling_and_column_numbers(self):
        text = (
            "Глубина по керну 4105.00 4106.00 4107.00 4107.94 4108.94. "
            "Глубина с увязкой 4104.90 4105.90 4106.90 4107.84 4108.84. "
            "Интервал отбора керна с 4104,90 до 4116,55 м, после увязки 4104,90-4116,45 м."
        )

        self.assertEqual((4104.9, 4116.45), extract_depth_interval(text))

    def test_prefers_post_alignment_range_over_full_sampling_interval(self):
        text = (
            "Интервал отбора керна 4106.00-4116.55 м, "
            "после увязки 4106.00-4114.65 м."
        )

        self.assertEqual(
            (4106.0, 4114.65),
            extract_depth_interval(text, ((4105.0, 4117.0),)),
        )

    def test_accepts_photo_page_interval_inside_excel_core_interval(self):
        self.assertEqual(
            (4109.94, 4114.65),
            extract_depth_interval(
                "Интервал отбора керна 4109.94-4114.65 м",
                ((4105.0, 4117.0),),
            ),
        )

    def test_caption_records_depth_coordinate_basis(self):
        for caption, basis in (
            ("Интервал отбора керна 4105.00-4117.00 м", "drilling"),
            ("Интервал отбора керна 4105.00-4117.00 м, после увязки 4104.90-4116.90 м", "gis"),
        ):
            metadata = {}
            self.assertIsNotNone(extract_depth_interval(caption, metadata=metadata))
            self.assertEqual(basis, metadata["depth_basis"])

    def test_extracts_well_and_removes_figure_suffix(self):
        self.assertEqual("67ПО", extract_well("Восточно-Тазовское м-е, скв№ 67ПО-0001"))

    def test_filename_parser_does_not_read_figure_number_as_depth(self):
        record = parse_filename(Path("Рис. 15.1-20 Восточно-Тазовское, скв№ 67ПО-0001.jpg"))

        self.assertEqual("67ПО", record.well)
        self.assertFalse(record.has_interval)

    def test_filename_parser_still_reads_depth_after_a_figure_reference(self):
        record = parse_filename(Path(
            "Рис. 15.1-20 Восточно-Тазовское, скв№ 67ПО 4104,90-4105,90.jpg"
        ))

        self.assertEqual((4104.9, 4105.9), (record.top, record.base))

    def test_manual_photo_retries_column_ocr_and_uses_complete_physical_depth_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "core.jpg"
            path.write_bytes(b"image")
            record = PhotoRecord(
                path, "W-1", 4105.0, 4109.84, "manual", True,
                column_ocr_checked=False,
            )
            image = np.zeros((100, 100, 3), dtype=np.uint8)

            def extract(_image, _columns, *, metadata, **_kwargs):
                metadata["depth_basis"] = "gis"
                return ((0.5, 4104.9, 4109.84),)

            with (
                patch("excel_photo_model_studio.photos._configure_tesseract", return_value=True),
                patch("cv2.imdecode", return_value=image),
                patch("excel_photo_model_studio.vision.detect_core_columns", return_value=[(40, 10, 60, 90)]),
                patch("excel_photo_model_studio.vision.extract_core_column_depths", side_effect=extract),
            ):
                resolved = enrich_core_column_depths([record], ((4104.9, 4116.9),))[0]

        self.assertEqual((4104.9, 4109.84), (resolved.top, resolved.base))
        self.assertEqual("gis", resolved.depth_basis)
        self.assertTrue(resolved.column_ocr_checked)
        self.assertEqual(((0.5, 4104.9, 4109.84),), resolved.column_depths)

    def test_rejects_numbers_from_ruler_and_figure_caption_as_depth_pair(self):
        self.assertIsNone(extract_depth_interval(
            "Рис. 22, шкала 0 10 20 30 40, страница 410",
            ((4104.9, 4116.9),),
        ))

    def test_uses_excel_core_interval_to_reject_larger_false_ocr_pair(self):
        text = "4105.00 4304.00 интервал отбора керна 4104.90 4116.55"
        self.assertEqual(
            (4104.9, 4116.55),
            extract_depth_interval(text, ((4104.9, 4116.9),)),
        )

    def test_ignores_ruler_rows_when_recovering_caption_depths(self):
        text = (
            "Глубина по керну 4105.00 4106.00 4107.00 4108.00\n"
            "Шкала 0 10 20 30 40 50 60 70 80 90 100\n"
            "Интервал отбора керна с 4104,90 до 4116,55 м"
        )
        self.assertEqual(
            (4104.9, 4116.55),
            extract_depth_interval(text, ((4104.9, 4116.9),)),
        )

    def test_does_not_pair_depths_across_unrelated_ocr_lines(self):
        text = "Интервал отбора керна\n4104.90\nШкала 0 10 20 30 40\n4116.55"
        self.assertIsNone(extract_depth_interval(text, ((4104.9, 4116.9),)))

    def test_does_not_accept_bare_numeric_ruler_without_excel_or_caption(self):
        self.assertIsNone(extract_depth_interval("Шкала 40 50 60 70 80 90 100"))
        self.assertEqual((4104.9, 4116.55), extract_depth_interval("4104.90-4116.55 м"))


if __name__ == "__main__":
    unittest.main()
