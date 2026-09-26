from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from excel_photo_model_studio.catalog import catalog_summary, load_project_catalog, register_project
from excel_photo_model_studio.dataset import build_dataset


class CatalogTests(unittest.TestCase):
    def test_separate_well_projects_accumulate_into_one_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "training_catalog.json"
            projects = []
            for index in range(2):
                project = root / f"project_{index}"
                project.mkdir()
                (project / "project.json").write_text("{}", encoding="utf-8")
                (project / "report.json").write_text(json.dumps({
                    "photos": 1, "annotations": 1, "approved_annotations": 1,
                }), encoding="utf-8")
                photo_folder = project / "photos"
                photo_folder.mkdir()
                photo = photo_folder / f"photo_{index}.jpg"
                image = np.full((80, 60, 3), 100 + index * 20, dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", image)
                self.assertTrue(ok)
                photo.write_bytes(encoded.tobytes())
                row = {
                    "annotation_id": f"a{index}", "photo": str(photo), "preview": "",
                    "well": f"W-{index}", "photo_top": "100", "photo_base": "101",
                    "depth_top": "100", "depth_base": "101", "label": "Tcr",
                    "polygon_json": json.dumps([[5, 5], [50, 5], [50, 70], [5, 70]]),
                    "image_width": "60", "image_height": "80", "source_sheet": "Data",
                    "source_row": "2", "source_file": "source.xlsx", "target_text": "Описание",
                    "association": "A", "environment": "B", "field_name": "Field", "approved": "1",
                }
                with (project / "annotations.csv").open("w", encoding="utf-8-sig", newline="") as target:
                    writer = csv.DictWriter(target, fieldnames=row.keys(), delimiter=";")
                    writer.writeheader()
                    writer.writerow(row)
                tracked = (photo, project / "annotations.csv")
                (project / "report.json").write_text(json.dumps({
                    "photos": 1, "annotations": 1, "approved_annotations": 1,
                    "validation_snapshot": {
                        "files": {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in tracked},
                        "photos_dir": str(photo_folder), "photos": [str(photo.resolve())],
                    },
                }), encoding="utf-8")
                register_project(project, catalog)
                register_project(project, catalog)
                projects.append(project)

            loaded = load_project_catalog(catalog)
            summary = catalog_summary(catalog)
            result = build_dataset(loaded, root / "dataset")

        self.assertEqual(len(projects), len(loaded))
        self.assertEqual([item.name for item in projects], [item.name for item in loaded])
        self.assertEqual(2, summary["projects"])
        self.assertEqual(2, result["project_count"])
        self.assertEqual(2, result["photo_count"])


if __name__ == "__main__":
    unittest.main()
