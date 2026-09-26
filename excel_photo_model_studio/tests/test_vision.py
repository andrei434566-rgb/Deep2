from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from excel_photo_model_studio.models import (
    Annotation, COLUMN_ORDER_LEFT_TO_RIGHT, COLUMN_ORDER_RIGHT_TO_LEFT,
    DescriptionRow, Match, PhotoRecord,
)
from excel_photo_model_studio.vision import (
    calibrate_core_columns, core_photo_capacity_centimeters, detect_column_order, detect_core_columns, project_matches,
    _column_depths_by_box, _depth_label_value, _pair_column_depth_rows, read_image, render_previews,
)


class VisionTests(unittest.TestCase):
    def test_detects_column_order_from_top_and_bottom_markers(self):
        for count in (3, 4, 5):
            with self.subTest(column_count=count):
                columns = [(80 + index * 150, 100, 180 + index * 150, 700) for index in range(count)]
                width = columns[-1][2] + 80
                right_to_left = np.full((800, width, 3), 255, dtype=np.uint8)
                cv2.rectangle(right_to_left, (columns[-1][0] + 20, 55), (columns[-1][2] - 20, 80), (0, 0, 0), -1)
                cv2.rectangle(right_to_left, (columns[0][0] + 20, 720), (columns[0][2] - 20, 745), (0, 0, 0), -1)
                self.assertEqual(COLUMN_ORDER_RIGHT_TO_LEFT, detect_column_order(right_to_left, columns))

                left_to_right = np.full((800, width, 3), 255, dtype=np.uint8)
                cv2.rectangle(left_to_right, (columns[0][0] + 20, 55), (columns[0][2] - 20, 80), (0, 0, 0), -1)
                cv2.rectangle(left_to_right, (columns[-1][0] + 20, 720), (columns[-1][2] - 20, 745), (0, 0, 0), -1)
                self.assertEqual(COLUMN_ORDER_LEFT_TO_RIGHT, detect_column_order(left_to_right, columns))

    def test_detects_three_to_five_core_columns(self):
        for count in (3, 4, 5):
            with self.subTest(column_count=count):
                width = count * 150 + 100
                image = np.full((900, width, 3), (25, 90, 160), dtype=np.uint8)
                for index in range(count):
                    left = 60 + index * 150
                    cv2.rectangle(image, (left, 80), (left + 90, 820), (115, 115, 115), -1)

                boxes = detect_core_columns(image)

                self.assertEqual(count, len(boxes))

    def test_project_matches_respects_manual_right_to_left_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo_path = root / "W-1 100-103.jpg"
            image = np.full((800, 600, 3), 255, dtype=np.uint8)
            for left in (80, 250, 420):
                cv2.rectangle(image, (left, 100), (left + 100, 700), (115, 115, 115), -1)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            photo_path.write_bytes(encoded.tobytes())
            photo = PhotoRecord(
                path=photo_path, well="W-1", top=100.0, base=103.0,
                source="filename", mapping_confirmed=True,
                column_order=COLUMN_ORDER_RIGHT_TO_LEFT,
            )
            row = DescriptionRow("W-1", 100.0, 100.01, "Sand", "Data", 2)

            annotations, _columns, orders = project_matches([Match(photo, row, 100.0, 100.01)])

        self.assertEqual(COLUMN_ORDER_RIGHT_TO_LEFT, orders[photo_path])
        self.assertEqual(1, len(annotations))
        self.assertGreaterEqual(min(point[0] for point in annotations[0].polygon), 400)
        ys = [point[1] for point in annotations[0].polygon]
        self.assertGreaterEqual(max(ys) - min(ys), 2.0)

    def test_rejects_narrow_ruler_beside_three_core_columns(self):
        image = np.full((1000, 800, 3), (25, 90, 160), dtype=np.uint8)
        cv2.rectangle(image, (70, 80), (90, 900), (120, 120, 120), -1)
        for left in (220, 360, 500):
            cv2.rectangle(image, (left, 70), (left + 90, 910), (115, 115, 115), -1)

        boxes = detect_core_columns(image)

        self.assertEqual(3, len(boxes))
        self.assertTrue(all(right - left >= 80 for left, _, right, _ in boxes))

    def test_detects_pale_report_columns_instead_of_left_depth_ruler(self):
        image = np.full((1200, 1000, 3), 255, dtype=np.uint8)
        cv2.line(image, (105, 110), (105, 1030), (40, 40, 40), 3)
        for y in range(120, 1030, 18):
            cv2.line(image, (105, y), (138 if y % 90 else 160, y), (50, 50, 50), 2)
        expected_lefts = (270, 420, 570, 720)
        for index, left in enumerate(expected_lefts):
            cv2.rectangle(image, (left, 150), (left + 92, 1040), (247, 247, 247), -1)
            for y in range(185 + index * 7, 1020, 73):
                cv2.line(image, (left, y), (left + 91, y + 12), (185, 185, 185), 2)
        # Printed depth grid lines connect the physical columns in a component
        # image, which is the layout that previously left only the ruler.
        for y in range(250, 1000, 180):
            cv2.line(image, (145, y), (900, y), (80, 80, 80), 2)

        boxes = detect_core_columns(image)

        self.assertEqual(4, len(boxes))
        self.assertTrue(all(left > 230 for left, _, _, _ in boxes))
        self.assertTrue(all(right - left >= 80 for left, _, right, _ in boxes))

    def test_detects_one_short_partial_core_but_not_the_ruler(self):
        image = np.full((900, 700, 3), 255, dtype=np.uint8)
        cv2.line(image, (75, 90), (75, 800), (30, 30, 30), 3)
        for y in range(100, 800, 16):
            cv2.line(image, (75, y), (112, y), (60, 60, 60), 2)
        cv2.rectangle(image, (315, 390), (440, 690), (246, 246, 246), -1)
        for y in range(410, 680, 35):
            cv2.line(image, (315, y), (440, y + 8), (175, 175, 175), 2)

        boxes = detect_core_columns(image)

        self.assertEqual(1, len(boxes))
        left, top, right, bottom = boxes[0]
        self.assertGreater(left, 280)
        self.assertLess(top, 420)
        self.assertGreater(right, 420)
        self.assertGreater(bottom - top, 250)

    def test_does_not_treat_a_depth_ruler_as_a_single_core(self):
        image = np.full((900, 700, 3), 255, dtype=np.uint8)
        cv2.line(image, (75, 90), (75, 800), (30, 30, 30), 3)
        for y in range(100, 800, 16):
            cv2.line(image, (75, y), (112, y), (60, 60, 60), 2)

        self.assertEqual([], detect_core_columns(image))

    def test_column_starts_at_core_not_at_depth_number_above_it(self):
        image = np.full((1000, 700, 3), 255, dtype=np.uint8)
        cv2.putText(
            image, "4130.00", (275, 145), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (25, 25, 25), 2, cv2.LINE_AA,
        )
        cv2.rectangle(image, (275, 210), (410, 850), (242, 242, 242), -1)
        for y in range(235, 830, 47):
            cv2.line(image, (275, y), (410, y + 9), (160, 160, 160), 2)
        cv2.putText(
            image, "4131.00", (275, 915), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (25, 25, 25), 2, cv2.LINE_AA,
        )

        boxes = detect_core_columns(image)

        self.assertEqual(1, len(boxes))
        _, top, _, bottom = boxes[0]
        self.assertGreaterEqual(top, 195)
        self.assertLessEqual(top, 220)
        self.assertGreaterEqual(bottom, 835)
        self.assertLessEqual(bottom, 865)

    def test_close_depth_digits_are_excluded_from_core_geometry(self):
        for gap in (1, 3, 6, 12):
            with self.subTest(gap_pixels=gap):
                image = np.full((1000, 700, 3), 255, dtype=np.uint8)
                cv2.putText(image, "4130.00", (275, 200 - gap), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (25, 25, 25), 2, cv2.LINE_AA)
                cv2.rectangle(image, (275, 200), (410, 850), (242, 242, 242), -1)
                boxes = detect_core_columns(image)
                self.assertEqual(1, len(boxes))
                self.assertGreaterEqual(boxes[0][1], 197)
                self.assertLessEqual(boxes[0][1], 203)

    def test_calibrates_depth_inside_each_detected_column(self):
        columns = [(10, 20, 50, 120), (70, 20, 110, 220), (130, 20, 170, 120)]

        calibrated = calibrate_core_columns(columns, 100.0, 104.0)

        self.assertEqual((100.0, 101.0), calibrated[0][1:])
        self.assertEqual((101.0, 102.0), calibrated[1][1:])
        self.assertEqual((102.0, 103.0), calibrated[2][1:])

    def test_calibrates_full_metre_columns_and_last_three_centimetres(self):
        columns = [(20 + index * 80, 30, 70 + index * 80, 430) for index in range(5)]

        calibrated = calibrate_core_columns(columns, 100.0, 104.03)

        self.assertEqual([
            (100.0, 101.0), (101.0, 102.0), (102.0, 103.0),
            (103.0, 104.0), (104.0, 104.03),
        ], [(top, base) for _, top, base in calibrated])

    def test_capacity_counts_one_full_and_one_half_column_as_150_cm(self):
        columns = [(20, 20, 70, 420), (100, 20, 150, 220)]

        self.assertEqual(150, core_photo_capacity_centimeters(columns))

    def test_capacity_uses_ruler_for_a_single_short_core_fragment(self):
        image = np.full((1000, 700, 3), 255, dtype=np.uint8)
        for y in range(100, 901, 80):
            cv2.line(image, (100, y), (650, y), (35, 35, 35), 2)
        columns = [(300, 100, 400, 260)]

        self.assertEqual(20, core_photo_capacity_centimeters(columns, image))

    def test_partial_column_depth_calibration_uses_actual_pixel_capacity(self):
        image = np.full((1000, 700, 3), 255, dtype=np.uint8)
        for y in range(100, 901, 80):
            cv2.line(image, (100, y), (650, y), (35, 35, 35), 2)
        columns = [(300, 100, 400, 260)]

        calibrated = calibrate_core_columns(columns, 4144.1, 4144.3, image=image)

        self.assertEqual([(4144.1, 4144.3)], [item[1:] for item in calibrated])

    def test_printed_grid_rejects_squeezing_complete_columns_to_shorter_caption(self):
        image = np.full((1000, 900, 3), 255, dtype=np.uint8)
        for y in range(100, 901, 80):
            cv2.line(image, (100, y), (850, y), (35, 35, 35), 2)
        columns = [(150 + 130 * index, 100, 240 + 130 * index, 900) for index in range(5)]
        self.assertEqual([], calibrate_core_columns(columns, 100.0, 104.1, image))
        self.assertEqual([], calibrate_core_columns(columns, 100.0, 106.0, image))
        calibrated = calibrate_core_columns(columns, 100.0, 105.0, image)
        self.assertEqual(5, len(calibrated))
        self.assertEqual((104.0, 105.0), calibrated[-1][1:])

    def test_single_column_without_depth_grid_uses_safe_full_lane_fallback(self):
        image = np.full((900, 700, 3), 255, dtype=np.uint8)
        columns = [(300, 100, 400, 400)]

        self.assertEqual(100, core_photo_capacity_centimeters(columns, image))

    def test_calibration_prefers_per_column_depth_labels(self):
        image = np.full((1000, 500, 3), 255, dtype=np.uint8)
        columns = [(100, 100, 180, 900), (260, 100, 340, 900)]
        labels = ((0.28, 4109.94, 4110.93), (0.60, 4110.93, 4111.73))

        calibrated = calibrate_core_columns(
            columns, 4109.90, 4111.80, image=image, column_depths=labels,
        )

        self.assertEqual([(4109.94, 4110.93), (4110.93, 4111.73)], [item[1:] for item in calibrated])
        self.assertEqual(179, core_photo_capacity_centimeters(columns, image, labels))

    def test_ocr_depth_rows_assign_each_header_footer_pair_to_its_column(self):
        columns = [(100, 100, 180, 900), (260, 100, 340, 900)]
        tokens = [
            (4109.94, 110.0, 70.0, 90.0), (4110.93, 270.0, 70.0, 90.0),
            (4110.93, 110.0, 930.0, 90.0), (4111.73, 270.0, 930.0, 90.0),
        ]

        ranges = _pair_column_depth_rows(tokens, columns, 500, 1000)

        self.assertEqual(1, len(ranges))
        self.assertEqual({0: (4109.94, 4110.93), 1: (4110.93, 4111.73)}, ranges[0])

    def test_partial_column_reads_its_footer_label_at_the_bottom_of_the_page(self):
        columns = [(200, 100, 300, 300)]
        tokens = [
            (4144.0, 210.0, 70.0, 90.0),
            (4144.2, 210.0, 930.0, 90.0),
        ]

        ranges = _pair_column_depth_rows(tokens, columns, 700, 1000)

        self.assertEqual([{0: (4144.0, 4144.2)}], ranges)

    def test_depth_label_parser_handles_decimal_comma_and_ocr_zeros(self):
        self.assertAlmostEqual(4114.65, _depth_label_value("4I14,65"))

    def test_depth_label_parser_accepts_shallow_and_deep_wells(self):
        for text, depth in (("12,03", 12.03), ("100.01", 100.01), ("0.20", 0.2), ("12345.67", 12345.67)):
            self.assertEqual(depth, _depth_label_value(text))
        for text in ("Fig.4105", "№12345", "20cm", "5.1-20"):
            self.assertIsNone(_depth_label_value(text))

    def test_column_depth_labels_are_not_reused_for_two_boxes(self):
        columns = [(100, 100, 160, 900), (160, 100, 220, 900)]
        self.assertEqual({}, _column_depths_by_box(columns, ((0.32, 100.0, 101.0),), 500))
        self.assertEqual({}, _column_depths_by_box(
            columns, ((0.32, 100.0, 101.0), (0.32, 101.0, 102.0)), 500,
        ))

    def test_stale_or_overlapping_column_labels_cannot_fall_back_to_guessed_masks(self):
        columns = [(100, 100, 160, 900), (260, 100, 320, 900)]
        image = np.full((1000, 500, 3), 255, dtype=np.uint8)
        for labels in (
            ((0.26, 100.0, 101.0),),
            ((0.26, 100.0, 101.0), (0.58, 100.5, 101.5)),
        ):
            self.assertEqual([], calibrate_core_columns(columns, 100.0, 102.0, image, labels))
            self.assertEqual(0, core_photo_capacity_centimeters(columns, image, labels))

    def test_source_identity_distinguishes_same_workbook_name_in_different_wells(self):
        first = DescriptionRow("W", 100, 101, "Sand", "Data", 2, source_file="well_a/description.xlsx")
        second = DescriptionRow("W", 100, 101, "Sand", "Data", 2, source_file="well_b/description.xlsx")
        self.assertNotEqual(first.source_id, second.source_id)

    def test_ocr_rows_do_not_cross_pair_drilling_and_gis(self):
        columns = [(100, 150, 180, 900), (260, 150, 340, 900)]
        tokens = []
        for y, depths in ((70, (100, 101)), (90, (99.9, 100.9)), (930, (101, 102)), (950, (100.9, 101.9))):
            tokens.extend((depth, x, y, 90) for depth, x in zip(depths, (140, 300)))
        ranges = _pair_column_depth_rows(tokens, columns, 500, 1000)
        self.assertEqual([
            {0: (100, 101), 1: (101, 102)},
            {0: (99.9, 100.9), 1: (100.9, 101.9)},
        ], ranges)

    def test_ocr_row_labels_pair_same_basis_even_when_one_row_is_missing(self):
        columns = [(100, 150, 180, 900)]
        tokens = [(100, 140, 70, 90), (99.9, 140, 90, 90), (100.9, 140, 950, 90)]
        # With no semantic evidence the incomplete two-row table is ambiguous.
        self.assertEqual([], _pair_column_depth_rows(tokens, columns, 500, 1000))
        bases = {}
        ranges = _pair_column_depth_rows(
            tokens, columns, 500, 1000,
            label_tokens=[("Глубина по керну", 40, 70), ("Глубина с увязкой", 40, 90), ("Глубина с увязкой", 40, 950)],
            candidate_bases=bases,
        )
        self.assertEqual([{0: (99.9, 100.9)}], ranges)
        self.assertEqual({0: "gis"}, bases)

    def test_one_centimetre_column_is_not_rejected_by_float_rounding(self):
        self.assertEqual([{0: (100.0, 100.01)}], _pair_column_depth_rows(
            [(100.0, 140, 70, 90), (100.01, 140, 950, 90)], [(100, 150, 180, 200)], 500, 1000,
        ))

    def test_does_not_create_full_photo_mask_when_columns_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo_path = root / "W-1 100-102.jpg"
            image = np.full((300, 300, 3), 255, dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            photo_path.write_bytes(encoded.tobytes())
            photo = PhotoRecord(photo_path, "W-1", 100.0, 102.0, "filename", True)
            row = DescriptionRow("W-1", 100.0, 101.0, "Sand", "Data", 2)

            annotations, columns, _orders = project_matches([Match(photo, row, 100.0, 101.0)])

        self.assertEqual([], annotations)
        self.assertEqual([], columns[photo_path])

    def test_render_preview_applies_visible_interval_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / "W-1 100-102.jpg"
            original = np.full((200, 120, 3), 110, dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", original)
            self.assertTrue(ok)
            photo.write_bytes(encoded.tobytes())
            annotation = Annotation(
                annotation_id="mask-1", photo_path=photo, well="W-1",
                photo_top=100.0, photo_base=102.0, depth_top=100.5, depth_base=101.5,
                label="Sand", polygon=((20, 50), (100, 50), (100, 150), (20, 150)),
                image_width=120, image_height=200, source_sheet="Sheet1", source_row=2,
                source_file=str(root / "description.xlsx"), target_text="Sandstone",
            )

            outputs = render_previews([annotation], root / "previews")
            preview = read_image(outputs[photo])

        difference = np.abs(preview[80:130, 35:85].astype(np.int16) - original[80:130, 35:85]).mean()
        self.assertGreater(difference, 8.0)

    def test_large_preview_is_downscaled_and_reused_from_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / "W-1 100-102.jpg"
            original = np.full((2500, 1000, 3), 120, dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", original)
            self.assertTrue(ok)
            photo.write_bytes(encoded.tobytes())
            annotation = Annotation(
                annotation_id="mask-large", photo_path=photo, well="W-1",
                photo_top=100.0, photo_base=102.0, depth_top=100.0, depth_base=100.1,
                label="Tcr", polygon=((100, 100), (900, 100), (900, 160), (100, 160)),
                image_width=1000, image_height=2500, source_sheet="Data", source_row=2,
            )
            destination = root / "previews"
            first = render_previews([annotation], destination)
            preview = read_image(first[photo])
            with patch(
                "excel_photo_model_studio.vision.read_image",
                side_effect=AssertionError("cached preview must not decode the source again"),
            ):
                second = render_previews([annotation], destination)

        self.assertLessEqual(max(preview.shape[:2]), 1800)
        self.assertEqual(first[photo].name, second[photo].name)


if __name__ == "__main__":
    unittest.main()
