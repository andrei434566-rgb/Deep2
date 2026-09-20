from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from excel_photo_model_studio.dataset import build_dataset
from excel_photo_model_studio.matching import match_photos
from excel_photo_model_studio.models import DescriptionRow, PhotoRecord
from excel_photo_model_studio.photos import parse_filename


class MatchingTests(unittest.TestCase):
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
