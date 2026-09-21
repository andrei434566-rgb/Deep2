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
    detect_column_order, detect_core_columns, project_matches, read_image, render_previews,
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
            row = DescriptionRow("W-1", 100.0, 100.0001, "Sand", "Data", 2)

            annotations, _columns, orders = project_matches([Match(photo, row, 100.0, 100.0001)])

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
