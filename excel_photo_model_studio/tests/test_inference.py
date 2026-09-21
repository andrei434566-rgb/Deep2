from __future__ import annotations

import unittest

import numpy as np

from excel_photo_model_studio.inference import polygon_depth_interval


class InferenceTests(unittest.TestCase):
    def test_maps_masks_across_ordered_three_to_five_columns(self):
        for count in (3, 4, 5):
            with self.subTest(column_count=count):
                columns = [
                    (500 - index * 120, 100, 580 - index * 120, 700)
                    for index in range(count)
                ]
                target = columns[1]
                polygon = np.array([
                    [target[0], target[1]], [target[2], target[1]],
                    [target[2], target[3]], [target[0], target[3]],
                ], dtype=np.float32)

                interval = polygon_depth_interval(polygon, columns, 100.0, 100.0 + count)

                self.assertEqual((101.0, 102.0), interval)


if __name__ == "__main__":
    unittest.main()
