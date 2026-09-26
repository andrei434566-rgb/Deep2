from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from excel_photo_model_studio.dataset import _report_blockers, _split_sources, build_dataset
from excel_photo_model_studio.matching import (
    match_photos, read_photo_map, suggest_missing_intervals,
    uncovered_photo_description_intervals, uncovered_photo_intervals, write_photo_map,
)
from excel_photo_model_studio.models import (
    COLUMN_ORDER_RIGHT_TO_LEFT, DescriptionRow, PhotoRecord,
)
from excel_photo_model_studio.photos import parse_filename


class MatchingTests(unittest.TestCase):
    def test_explicit_gis_photo_is_not_rewritten_by_drilling_sequence(self):
        row = DescriptionRow("W-1", 100, 102, "A", "Data", 2, core_top=100, core_base=102)
        record = PhotoRecord(Path("page.jpg"), "W-1", 99.9, 101.9, "ocr_verified", True, depth_basis="gis")

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=200):
            resolved = suggest_missing_intervals([record], [row])

        self.assertEqual((99.9, 101.9), (resolved[0].top, resolved[0].base))
        self.assertEqual("gis", resolved[0].depth_basis)

    def test_trusted_page_ocr_must_agree_with_sequential_position(self):
        row = DescriptionRow("W-1", 100, 102, "A", "Data", 2, core_top=100, core_base=102)
        records = [
            PhotoRecord(Path("page-1.jpg"), "W-1", 101, 102, "ocr_verified", True),
            PhotoRecord(Path("page-2.jpg"), "W-1"),
        ]

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=100):
            resolved = suggest_missing_intervals(records, [row])

        self.assertEqual((101, 102), (resolved[0].top, resolved[0].base))
        self.assertFalse(resolved[1].mapping_confirmed)

    def test_does_not_mix_shifted_gis_neighbour_with_drilling_facies(self):
        rows = [
            DescriptionRow("W-1", 100, 101, "A", "Data", 2, gis_top=99.5, gis_base=100.5),
            DescriptionRow("W-1", 101, 102, "B", "Data", 3, gis_top=100.5, gis_base=101.5),
        ]
        photo = PhotoRecord(Path("core.jpg"), "W-1", 100, 101, "manual", True)

        matches, _ = match_photos([photo], rows)

        self.assertEqual([("A", 100, 101)], [
            (item.description.label, item.overlap_top, item.overlap_base) for item in matches
        ])

    def test_explicit_photo_gis_basis_uses_gis_for_every_facies(self):
        rows = [
            DescriptionRow("W-1", 100, 101, "A", "Data", 2, gis_top=99.5, gis_base=100.5),
            DescriptionRow("W-1", 101, 102, "B", "Data", 3, gis_top=100.5, gis_base=101.5),
        ]
        photo = PhotoRecord(Path("core.jpg"), "W-1", 100, 101, "manual", True, depth_basis="gis")

        matches, _ = match_photos([photo], rows)

        self.assertEqual([("A", 100, 100.5), ("B", 100.5, 101)], [
            (item.description.label, item.overlap_top, item.overlap_base) for item in matches
        ])
        self.assertTrue(all(item.description.metadata["interval_source"] == "gis" for item in matches))

    def test_inferred_photo_coordinate_basis_is_attached_to_match(self):
        row = DescriptionRow(
            well="W-1", top=100.0, base=101.0, gis_top=200.0, gis_base=201.0,
            thickness=1.0, label="Sand", sheet="Data", row=2,
        )
        photo = PhotoRecord(Path("core.jpg"), "W-1", 200.0, 201.0, "manual", True)

        matches, unresolved = match_photos([photo], [row])

        self.assertEqual([], unresolved)
        self.assertEqual(1, len(matches))
        self.assertEqual("gis", matches[0].photo.depth_basis)

    def test_unknown_well_cannot_match_two_wells_at_the_same_depth(self):
        photo = PhotoRecord(Path("core.jpg"), top=100, base=101)
        rows = [DescriptionRow(well, 100, 101, "A", "Data", i) for i, well in enumerate(("W-1", "W-2"))]

        matches, unresolved = match_photos([photo], rows)

        self.assertEqual([], matches)
        self.assertEqual([photo], unresolved)

    def test_sequence_cannot_squeeze_a_three_meter_photo_into_three_centimeters(self):
        row = DescriptionRow("W-1", 100, 103.03, "A", "Data", 2, core_top=100, core_base=103.03)
        records = [PhotoRecord(Path(f"page-{i}.jpg"), "W-1") for i in range(2)]

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=300):
            resolved = suggest_missing_intervals(records, [row])

        self.assertTrue(all(not item.mapping_confirmed for item in resolved))
        self.assertTrue(all(not item.has_interval for item in resolved))

    def test_uses_gis_limits_only_when_drilling_does_not_match_that_facies(self):
        row = DescriptionRow(
            well="W-1", top=100.0, base=101.0, gis_top=200.0, gis_base=201.0,
            thickness=1.0, label="Sand", sheet="Data", row=2,
            target_text="Песчаник серый.",
        )
        photo = PhotoRecord(Path("core.jpg"), "W-1", 200.0, 201.0, "manual", True)

        matches, unresolved = match_photos([photo], [row])

        self.assertEqual([], unresolved)
        self.assertEqual(1, len(matches))
        self.assertEqual((200.0, 201.0), (matches[0].description.top, matches[0].description.base))
        self.assertEqual("gis", matches[0].description.metadata["interval_source"])
        self.assertEqual("Песчаник серый.", matches[0].description.target_text)

    def test_drilling_match_wins_over_overlapping_gis_fallback(self):
        row = DescriptionRow(
            well="W-1", top=100.0, base=101.0, gis_top=100.5, gis_base=101.5,
            thickness=1.0, label="Sand", sheet="Data", row=2,
        )
        photo = PhotoRecord(Path("core.jpg"), "W-1", 100.0, 101.5, "manual", True)

        matches, unresolved = match_photos([photo], [row])

        self.assertEqual([], unresolved)
        self.assertEqual(1, len(matches))
        self.assertEqual((100.0, 101.0), (matches[0].description.top, matches[0].description.base))
        self.assertNotIn("interval_source", matches[0].description.metadata)

    def test_rejects_gis_fallback_when_its_span_disagrees_with_facies_thickness(self):
        row = DescriptionRow(
            well="W-1", top=100.0, base=101.0, gis_top=200.0, gis_base=202.0,
            thickness=1.0, label="Sand", sheet="Data", row=2,
        )
        photo = PhotoRecord(Path("core.jpg"), "W-1", 200.0, 202.0, "manual", True)

        matches, unresolved = match_photos([photo], [row])

        self.assertEqual([], matches)
        self.assertEqual([photo], unresolved)

    def test_sequences_missing_pages_around_verified_filename_anchors(self):
        rows = [DescriptionRow(
            well="W-1", top=100.0, base=110.0, label="Sand", sheet="Data", row=2,
            core_top=100.0, core_base=110.0, target_text="Песчаник.",
        )]
        records = [PhotoRecord(
            Path(f"core-{index:04d}.jpg"), "W-1", source="not_found",
        ) for index in range(10)]
        records[0] = PhotoRecord(records[0].path, "W-1", 100.0, 101.0, "filename", True)
        records[9] = PhotoRecord(records[9].path, "W-1", 109.0, 110.0, "filename", True)

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=100):
            sequenced = suggest_missing_intervals(records, rows)

        self.assertTrue(all(item.mapping_confirmed for item in sequenced))
        self.assertEqual((100.0, 101.0), (sequenced[0].top, sequenced[0].base))
        self.assertEqual((109.0, 110.0), (sequenced[-1].top, sequenced[-1].base))

    def test_manual_anchor_is_preserved_while_other_pages_are_sequenced(self):
        rows = [DescriptionRow(
            well="W-1", top=100.0, base=110.0, label="Sand", sheet="Data", row=2,
            core_top=100.0, core_base=110.0, target_text="Песчаник.",
        )]
        records = [PhotoRecord(
            Path(f"core-{index:04d}.jpg"), "W-1", source="not_found",
        ) for index in range(10)]
        records[4] = PhotoRecord(records[4].path, "W-1", 104.0, 105.0, "manual", True)

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=100):
            sequenced = suggest_missing_intervals(records, rows)

        self.assertTrue(all(item.mapping_confirmed for item in sequenced))
        self.assertEqual("manual", sequenced[4].source)
        self.assertEqual((104.0, 105.0), (sequenced[4].top, sequenced[4].base))

    def test_rejects_conflicting_manual_anchor_without_overwriting_it(self):
        rows = [DescriptionRow(
            well="W-1", top=100.0, base=110.0, label="Sand", sheet="Data", row=2,
            core_top=100.0, core_base=110.0, target_text="Песчаник.",
        )]
        records = [PhotoRecord(
            Path(f"core-{index:04d}.jpg"), "W-1", source="not_found",
        ) for index in range(10)]
        records[0] = PhotoRecord(records[0].path, "W-1", 105.0, 106.0, "manual", True)

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=100):
            sequenced = suggest_missing_intervals(records, rows)

        self.assertEqual("manual", sequenced[0].source)
        self.assertEqual((105.0, 106.0), (sequenced[0].top, sequenced[0].base))
        self.assertTrue(all(not item.mapping_confirmed for item in sequenced[1:]))

    def test_wrong_filename_interval_outside_excel_is_recovered_from_full_sequence(self):
        rows = [DescriptionRow(
            well="W-1", top=100.0, base=110.0, label="Sand", sheet="Data", row=2,
            core_top=100.0, core_base=110.0, target_text="Песчаник.",
        )]
        photo = PhotoRecord(Path("W-1 120-130.jpg"), "W-1", 120.0, 130.0, "filename", True)

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=1000):
            sequenced = suggest_missing_intervals([photo], rows)

        self.assertEqual(1, len(sequenced))
        self.assertEqual("excel_sequenced", sequenced[0].source)
        self.assertEqual((100.0, 110.0), (sequenced[0].top, sequenced[0].base))

    def test_sequences_filename_only_pages_when_ocr_is_disabled(self):
        rows = [DescriptionRow(
            well="67ПО", top=100.0, base=110.0, label="Sand", sheet="Data", row=2,
            core_top=100.0, core_base=110.0, target_text="Песчаник.",
        )]
        records = [
            PhotoRecord(
                Path(f"Рис. 5.1-20 Восточно-Тазовское, скв№ 67ПО-{suffix:04d}.jpg"),
                "67ПО", source="filename",
            )
            for suffix in range(1, 20, 2)
        ]

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=100):
            sequenced = suggest_missing_intervals(records, rows)

        self.assertEqual(10, len(sequenced))
        self.assertTrue(all(item.mapping_confirmed for item in sequenced))
        self.assertEqual("excel_sequenced", sequenced[0].source)
        self.assertEqual((100.0, 101.0), (sequenced[0].top, sequenced[0].base))
        self.assertEqual((109.0, 110.0), (sequenced[-1].top, sequenced[-1].base))

    def test_complete_excel_sequence_overrides_unconfirmed_ocr_anchor(self):
        rows = [DescriptionRow(
            well="W-1", top=100.0, base=110.0, label="Sand", sheet="Data", row=2,
            core_top=100.0, core_base=110.0, target_text="Песчаник.",
        )]
        records = [
            PhotoRecord(Path(f"core-{index:04d}.jpg"), "W-1", source="ocr_not_found")
            for index in range(10)
        ]
        # A weak, unconfirmed OCR hit on one page must not veto an exact
        # complete sequence through the Excel core range.
        records[4] = PhotoRecord(
            records[4].path, "W-1", 200.0, 201.0, "ocr", False,
        )

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=100):
            sequenced = suggest_missing_intervals(records, rows)

        self.assertEqual(10, len(sequenced))
        self.assertTrue(all(item.mapping_confirmed for item in sequenced))
        self.assertEqual((104.0, 105.0), (sequenced[4].top, sequenced[4].base))

    def test_weak_ocr_on_an_earlier_core_range_does_not_drop_later_pages(self):
        rows = [
            DescriptionRow(
                "W-1", 100.0, 102.0, "A", "Data", 2,
                core_top=100.0, core_base=102.0, target_text="Первая фация.",
            ),
            DescriptionRow(
                "W-1", 110.0, 112.0, "B", "Data", 3,
                core_top=110.0, core_base=112.0, target_text="Вторая фация.",
            ),
        ]
        records = [
            PhotoRecord(Path(f"core-{index:04d}.jpg"), "W-1", source="ocr_not_found")
            for index in range(4)
        ]
        records[2] = PhotoRecord(records[2].path, "W-1", 100.0, 102.0, "ocr", False)

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=100):
            sequenced = suggest_missing_intervals(records, rows)

        self.assertEqual(
            [(100.0, 101.0), (101.0, 102.0), (110.0, 111.0), (111.0, 112.0)],
            [(item.top, item.base) for item in sequenced],
        )
        self.assertTrue(all(item.mapping_confirmed for item in sequenced))

    def test_partial_ocr_group_suggests_page_ranges_but_cannot_confirm_their_start(self):
        rows = [DescriptionRow(
            well="W-1", top=100.0, base=110.0, label="Sand", sheet="Data", row=2,
            core_top=100.0, core_base=110.0, target_text="Песчаник.",
        )]
        records = [
            PhotoRecord(Path(f"core-{index:04d}.jpg"), "W-1", 100.0, 110.0, "ocr")
            for index in range(2)
        ]

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=100):
            sequenced = suggest_missing_intervals(records, rows)

        self.assertTrue(all(not item.mapping_confirmed for item in sequenced))
        self.assertEqual([(100.0, 101.0), (101.0, 102.0)], [
            (item.top, item.base) for item in sequenced
        ])
        self.assertTrue(all(item.source == "ocr_sequenced" for item in sequenced))

    def test_short_core_fragment_is_not_assigned_a_full_meter_excel_interval(self):
        rows = [DescriptionRow(
            well="W-1", top=4144.0, base=4145.0, label="Sand", sheet="Data", row=2,
            core_top=4144.0, core_base=4145.0,
        )]
        photo = PhotoRecord(Path("W-1 photo.jpg"), "W-1", source="not_found")

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=20):
            resolved = suggest_missing_intervals([photo], rows)

        self.assertEqual(1, len(resolved))
        self.assertFalse(resolved[0].has_interval)
        self.assertFalse(resolved[0].mapping_confirmed)

    def test_partial_photos_are_packed_by_measured_capacity_into_excel_core_range(self):
        rows = [DescriptionRow(
            well="W-1", top=4144.0, base=4145.0, label="Sand", sheet="Data", row=2,
            core_top=4144.0, core_base=4145.0,
        )]
        photos = [
            PhotoRecord(Path(f"W-1 page-{index}.jpg"), "W-1", source="not_found")
            for index in range(2)
        ]

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", return_value=50):
            resolved = suggest_missing_intervals(photos, rows)

        self.assertEqual([(4144.0, 4144.5), (4144.5, 4145.0)], [
            (record.top, record.base) for record in resolved
        ])
        self.assertTrue(all(record.source == "excel_sequenced" for record in resolved))

    def test_dataset_reports_photos_without_masks_as_a_blocker(self):
        self.assertEqual(
            ["фото без масок в датасете: 2"],
            _report_blockers({"photos": 10, "photos_without_masks": 2}),
        )

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
        # A sampling interval in a caption does not establish that this is
        # its first page when the folder's measured capacity is incomplete.
        self.assertFalse(resolved[0].mapping_confirmed)
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

        with patch("excel_photo_model_studio.matching._photo_core_capacity_cm", side_effect=lambda path: 3 if path.stem.endswith("0005") else 300):
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
                column_depths=((0.25, 100.0, 100.4),), column_ocr_checked=True,
                depth_basis="gis",
            )
            write_photo_map(path, [record])

            loaded = read_photo_map(path)

        self.assertEqual(COLUMN_ORDER_RIGHT_TO_LEFT, loaded[0].column_order)
        self.assertEqual(((0.25, 100.0, 100.4),), loaded[0].column_depths)
        self.assertTrue(loaded[0].column_ocr_checked)
        self.assertEqual("gis", loaded[0].depth_basis)

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


class DatasetSplitTests(unittest.TestCase):
    def test_validation_falls_back_to_photo_split_when_well_split_loses_all_classes(self):
        by_photo = {
            photo: [{"label": label, "well": well}]
            for photo, label, well in (
                ("a", "A", "w1"), ("b", "A", "w1"),
                ("c", "B", "w1"), ("d", "B", "w1"),
                ("e", "C", "w2"), ("f", "C", "w2"),
                ("g", "D", "w2"), ("h", "D", "w2"),
            )
        }

        split, strategy = _split_sources(by_photo)

        self.assertEqual("photo_fallback", strategy)
        self.assertIn("val", split.values())
        self.assertIn("train", split.values())


if __name__ == "__main__":
    unittest.main()
