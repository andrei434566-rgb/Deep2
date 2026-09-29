from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from excel_photo_model_studio.catalog import (
    catalog_summary, confirm_project_for_training, load_confirmed_project_catalog,
)
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
                    "facies_index": "Tcr", "facies_name": "Песчаник",
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
                    "blocking_errors": 0,
                    "validation_snapshot": {
                        "files": {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in tracked},
                        "photos_dir": str(photo_folder), "photos": [str(photo.resolve())],
                    },
                }), encoding="utf-8")
                confirm_project_for_training(project, catalog)
                confirm_project_for_training(project, catalog)
                projects.append(project)

            loaded = load_confirmed_project_catalog(catalog)
            summary = catalog_summary(catalog)
            result = build_dataset(loaded, root / "dataset")
            cataloged_sources = {
                item["source_project"] for item in json.loads(catalog.read_text(encoding="utf-8"))["projects"]
            }
            cached_manifests_exist = all((item / "cache_manifest.json").is_file() for item in loaded)

        self.assertEqual(len(projects), len(loaded))
        self.assertEqual(2, len(loaded))
        self.assertTrue(all(item.parent.name == "confirmed_wells" for item in loaded))
        self.assertEqual({str(item.resolve()) for item in projects}, {
            str(Path(source).resolve()) for source in cataloged_sources
        })
        self.assertEqual(2, summary["projects"])
        self.assertEqual(2, result["project_count"])
        self.assertEqual(2, result["photo_count"])
        self.assertTrue(cached_manifests_exist)

    def test_reconfirming_changed_project_adds_new_snapshot_without_losing_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "training_catalog.json"
            project = root / "project"
            project.mkdir()
            (project / "project.json").write_text("{}", encoding="utf-8")
            photos = project / "photos"
            photos.mkdir()
            photo = photos / "core.jpg"
            image = np.full((80, 60, 3), 120, dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            photo.write_bytes(encoded.tobytes())
            row = {
                "annotation_id": "a1", "photo": str(photo), "preview": "", "well": "W-1",
                "photo_top": "100", "photo_base": "101", "depth_top": "100", "depth_base": "101",
                "label": "Dch", "facies_index": "Dch", "facies_name": "Каналы",
                "polygon_json": json.dumps([[5, 5], [50, 5], [50, 70], [5, 70]]),
                "image_width": "60", "image_height": "80", "source_sheet": "Data",
                "source_row": "2", "source_file": "source.xlsx", "target_text": "Первое описание",
                "association": "", "environment": "", "field_name": "", "approved": "1",
            }
            annotations = project / "annotations.csv"
            with annotations.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=row.keys(), delimiter=";")
                writer.writeheader()
                writer.writerow(row)
            report = {
                "photos": 1, "annotations": 1, "approved_annotations": 1,
                "blocking_errors": 0,
                "validation_snapshot": {
                    "files": {str(item): [item.stat().st_size, item.stat().st_mtime_ns] for item in (photo, annotations)},
                    "photos_dir": str(photos), "photos": [str(photo.resolve())],
                },
            }
            (project / "report.json").write_text(json.dumps(report), encoding="utf-8")
            confirm_project_for_training(project, catalog)
            original_snapshot = load_confirmed_project_catalog(catalog)[0]

            row["target_text"] = "Исправленное описание"
            with annotations.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=row.keys(), delimiter=";")
                writer.writeheader()
                writer.writerow(row)
            annotation_key = next(
                key for key in report["validation_snapshot"]["files"]
                if Path(key).name.casefold() == annotations.name.casefold()
            )
            report["validation_snapshot"]["files"][annotation_key] = [
                annotations.stat().st_size, annotations.stat().st_mtime_ns,
            ]
            (project / "report.json").write_text(json.dumps(report), encoding="utf-8")
            confirm_project_for_training(project, catalog)
            refreshed = load_confirmed_project_catalog(catalog)[0]

            self.assertNotEqual(original_snapshot, refreshed)
            self.assertTrue(original_snapshot.is_dir())
            with (refreshed / "annotations.csv").open("r", encoding="utf-8-sig", newline="") as source:
                cached_row = next(csv.DictReader(source, delimiter=";"))
            self.assertEqual("Исправленное описание", cached_row["target_text"])
            self.assertEqual(1, len(json.loads(catalog.read_text(encoding="utf-8"))["projects"]))

            row["approved"] = "0"
            with annotations.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=row.keys(), delimiter=";")
                writer.writeheader()
                writer.writerow(row)
            report["approved_annotations"] = 0
            report["validation_snapshot"]["files"][annotation_key] = [
                annotations.stat().st_size, annotations.stat().st_mtime_ns,
            ]
            (project / "report.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "не подтверждены маски"):
                confirm_project_for_training(project, catalog)
            self.assertEqual([], load_confirmed_project_catalog(catalog))
            self.assertTrue(refreshed.is_dir(), "deactivated snapshots remain recoverable")

    def test_unconfirmed_project_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "unconfirmed"
            project.mkdir()
            (project / "project.json").write_text("{}", encoding="utf-8")
            (project / "annotations.csv").write_text("annotation_id;approved\na1;0\n", encoding="utf-8-sig")
            (project / "report.json").write_text(json.dumps({
                "annotations": 1, "approved_annotations": 0, "blocking_errors": 0,
                "validation_snapshot": {"files": {}, "photos_dir": str(root), "photos": []},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "не подтверждены маски"):
                confirm_project_for_training(project, root / "training_catalog.json")
            self.assertFalse((root / "confirmed_wells").exists())


if __name__ == "__main__":
    unittest.main()
