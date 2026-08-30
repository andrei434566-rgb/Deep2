"""Tests for the explicitly synthetic seven-facies DEMO route."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.infrastructure.kern_analyzer_pipeline import KernAnalyzerDemoPipeline


class KernAnalyzerDemoPipelineTests(unittest.TestCase):
    def test_covers_every_detected_core_pixel_with_seven_demo_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "photo.jpg"
            image = np.full((300, 360, 3), 245, dtype=np.uint8)
            boxes = [(20, 20, 100, 280), (140, 20, 220, 280), (260, 20, 340, 280)]
            for left, top, right, bottom in boxes:
                image[top:bottom, left:right] = 75
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            encoded.tofile(str(source))
            review = Path(temporary) / "_core_tape_review"
            review.mkdir()
            (review / "manifest.json").write_text(json.dumps({
                "source_folder": str(Path(temporary).resolve()),
                "columns": [
                    {"source_relative": source.name, "source_rectangle_px": {"left": left, "top": top, "right": right, "bottom": bottom}}
                    for left, top, right, bottom in boxes
                ],
            }), encoding="utf-8")

            result = KernAnalyzerDemoPipeline().run(source, Path(temporary) / "demo", seed=42)

            self.assertEqual(3, result.columns_detected)
            self.assertEqual(7, len(result.classes))
            self.assertGreaterEqual(result.rectangles_created, 7)
            mask_path = result.output_dir / "class_masks" / "photo_demo_mask.png"
            mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(mask)
            for left, top, right, bottom in boxes:
                self.assertTrue(np.all(mask[top:bottom, left:right] > 0))
            self.assertTrue((result.output_dir / "DEMO_7_фаций.xlsx").is_file())
            manifest = json.loads((result.output_dir / "demo_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("NOT TRAINING DATA", manifest["pipeline"])


if __name__ == "__main__":
    unittest.main()
