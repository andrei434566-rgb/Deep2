from __future__ import annotations

import unittest

import cv2
import numpy as np

from excel_photo_model_studio.vision import detect_core_columns


class VisionTests(unittest.TestCase):
    def test_rejects_narrow_ruler_beside_three_core_columns(self):
        image = np.full((1000, 800, 3), (25, 90, 160), dtype=np.uint8)
        cv2.rectangle(image, (70, 80), (90, 900), (120, 120, 120), -1)
        for left in (220, 360, 500):
            cv2.rectangle(image, (left, 70), (left + 90, 910), (115, 115, 115), -1)

        boxes = detect_core_columns(image)

        self.assertEqual(3, len(boxes))
        self.assertTrue(all(right - left >= 80 for left, _, right, _ in boxes))


if __name__ == "__main__":
    unittest.main()
