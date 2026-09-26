from __future__ import annotations

import unittest
import tempfile
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from excel_photo_model_studio.inference import (
    analyze_photos_to_excel, polygon_depth_interval, _uncovered_intervals, _resolve_prediction_overlaps,
)
from excel_photo_model_studio.models import PhotoRecord


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

    def test_background_mask_is_not_attached_to_nearest_core(self):
        polygon = np.array([[1, 100], [20, 100], [20, 700], [1, 700]], dtype=np.float32)
        self.assertIsNone(polygon_depth_interval(polygon, [(100, 100, 180, 700)], 100, 101))

    def test_uncovered_centimetres_and_overlap_resolution(self):
        self.assertEqual([(100.2, 100.21)], _uncovered_intervals(100, 101, [(100, 100.2), (100.21, 101)]))
        self.assertEqual([(100.0, 101.0)], _uncovered_intervals(100, 101, [(102, 103)]))
        rows = [
            {"facies_top": 100.0, "facies_base": 100.7, "confidence": .6, "prediction_index": 1},
            {"facies_top": 100.5, "facies_base": 101.0, "confidence": .9, "prediction_index": 2},
        ]
        result = _resolve_prediction_overlaps(rows)
        self.assertEqual([(100.0, 100.5), (100.5, 101.0)], [(row["facies_top"], row["facies_base"]) for row in result])

    def _analysis_context(self, root, records, polygons):
        from contextlib import ExitStack
        stack = ExitStack()
        model_path = root / "best.pt"
        model_path.touch()
        result = SimpleNamespace(
            masks=SimpleNamespace(xy=polygons),
            boxes=SimpleNamespace(cls=torch.zeros(len(polygons)), conf=torch.full((len(polygons),), .95)),
        )
        visual = SimpleNamespace(names={0: "Tcr"}, predict=lambda **kwargs: [result])
        fake = types.ModuleType("ultralytics")
        fake.YOLO = lambda _: visual
        stack.enter_context(patch.dict(sys.modules, {"ultralytics": fake}))
        stack.enter_context(patch("torch.load", return_value={"core_description_checkpoint": {"schema": "fake"}}))
        stack.enter_context(patch("excel_photo_model_studio.inference.discover_photos", return_value=records))
        stack.enter_context(patch("excel_photo_model_studio.inference.enrich_core_column_depths", side_effect=lambda rows: rows))
        stack.enter_context(patch("excel_photo_model_studio.inference.read_image", return_value=np.full((100, 100, 3), 180, dtype=np.uint8)))
        stack.enter_context(patch("excel_photo_model_studio.inference.detect_core_columns", return_value=[(10, 10, 50, 90)]))
        stack.enter_context(patch("excel_photo_model_studio.inference.detect_column_order", return_value="left_to_right"))
        text_factory = stack.enter_context(patch("excel_photo_model_studio.inference.DescriptionGenerator"))
        text_factory.return_value.generate.return_value = "Песчаник слоистый."
        export = stack.enter_context(patch("excel_photo_model_studio.inference.export_standardized_workbook"))
        return stack, model_path, text_factory, export

    def test_unresolved_photo_blocks_partial_well_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [PhotoRecord(root / "ok.jpg", "W", 100, 101), PhotoRecord(root / "unknown.jpg")]
            stack, model, _, export = self._analysis_context(root, records, [])
            with stack, self.assertRaisesRegex(ValueError, "unknown.jpg"):
                analyze_photos_to_excel(model, root, root / "out.xlsx")
            export.assert_not_called()

    def test_mask_gap_blocks_partial_well_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [PhotoRecord(root / "a.jpg", "W", 100, 101)]
            polygon = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32)
            stack, model, _, export = self._analysis_context(root, records, [polygon])
            with stack, self.assertRaisesRegex(ValueError, "не покрывают"):
                analyze_photos_to_excel(model, root, root / "out.xlsx")
            export.assert_not_called()

    def test_full_well_reuses_text_model_and_numbers_layers_globally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [PhotoRecord(root / "b.jpg", "W", 101, 102), PhotoRecord(root / "a.jpg", "W", 100, 101)]
            polygon = np.array([[10, 10], [50, 10], [50, 90], [10, 90]], dtype=np.float32)
            stack, model, text_factory, export = self._analysis_context(root, records, [polygon])
            with stack:
                info = analyze_photos_to_excel(model, root, root / "out.xlsx")
            text_factory.assert_called_once()
            self.assertEqual(2, text_factory.return_value.generate.call_count)
            self.assertEqual(2, info["photos"])
            rows = export.call_args.args[0]
            self.assertEqual([1, 2], [row["layer_no"] for row in rows])
            self.assertEqual([100, 101], [row["facies_top"] for row in rows])


if __name__ == "__main__":
    unittest.main()
