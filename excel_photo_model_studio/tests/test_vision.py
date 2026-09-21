from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from excel_photo_model_studio.models import Annotation
from excel_photo_model_studio.vision import detect_core_columns, read_image, render_previews


class VisionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
