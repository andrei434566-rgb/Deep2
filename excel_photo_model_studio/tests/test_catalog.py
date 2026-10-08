from __future__ import annotations

import csv
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from excel_photo_model_studio.catalog import (
    catalog_overview, catalog_summary, confirm_project_for_training,
    load_confirmed_project_catalog,
)
from excel_photo_model_studio.cli import main as run_cli
from excel_photo_model_studio.dataset import build_dataset
from excel_photo_model_studio.project import set_annotation_approvals


class CatalogTests(unittest.TestCase):
    def test_one_approved_mask_is_cached_and_dataset_built_despite_other_project_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "training_catalog.json"
            project = root / "projects" / "project"
            project.mkdir(parents=True)
            (project / "project.json").write_text("{}", encoding="utf-8")
            photo_folder = project / "photos"
            photo_folder.mkdir()
            photos = []
            annotations = []
            for index in range(2):
                photo = photo_folder / f"photo_{index}.jpg"
                image = np.full((80, 60, 3), 90 + index * 30, dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", image)
                self.assertTrue(ok)
                photo.write_bytes(encoded.tobytes())
                photos.append(photo)
                annotations.append({
                    "annotation_id": f"a{index}", "photo": str(photo), "preview": "",
                    "well": "W-1", "photo_top": "100", "photo_base": "101",
                    "depth_top": "100", "depth_base": "101", "label": "Dch",
                    "facies_index": "Dch", "facies_name": "Каналы",
                    "polygon_json": json.dumps([[5, 5], [50, 5], [50, 70], [5, 70]]),
                    "image_width": "60", "image_height": "80", "source_sheet": "Data",
                    "source_row": str(index + 2), "source_file": "source.xlsx",
                    "target_text": "Песчаник светло-серый.", "approved": "1" if index == 0 else "0",
                })
            annotations_path = project / "annotations.csv"
            with annotations_path.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=annotations[0].keys(), delimiter=";")
                writer.writeheader()
                writer.writerows(annotations)
            report = {
                "photos": 2, "confirmed_photos": 1, "annotations": 2,
                "approved_annotations": 1, "blocking_errors": 4,
                "projection_errors": 1, "photos_without_masks": 1,
                "validation_snapshot": {
                    "files": {
                        str(path.resolve()): [path.stat().st_size, path.stat().st_mtime_ns]
                        for path in (*photos, annotations_path)
                    },
                    "photos_dir": str(photo_folder),
                    "photos": sorted(str(photo.resolve()) for photo in photos),
                },
            }
            (project / "report.json").write_text(json.dumps(report), encoding="utf-8")

            primary_cache_root = root / "confirmed_wells"
            original_mkdir = Path.mkdir

            def deny_primary_cache_child(path, *args, **kwargs):
                if path.parent == primary_cache_root and path.name.startswith("project_"):
                    raise PermissionError("simulated AppData subdirectory restriction")
                return original_mkdir(path, *args, **kwargs)

            with patch(
                "excel_photo_model_studio.catalog._snapshot_confirmed_project",
                side_effect=PermissionError("simulated first-attempt AppData failure"),
            ):
                with self.assertRaises(PermissionError):
                    confirm_project_for_training(project, catalog)

            with patch.object(Path, "mkdir", new=deny_primary_cache_child):
                cached_projects = load_confirmed_project_catalog(catalog)
            self.assertEqual(project / "confirmed_wells", cached_projects[0].parent)
            cache_manifest_path = cached_projects[0] / "cache_manifest.json"
            initial_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            initial_manifest.pop("photo_names", None)  # Existing 0.6.8 caches have only the hashed path.
            cache_manifest_path.write_text(json.dumps(initial_manifest), encoding="utf-8")
            overview = catalog_overview(catalog)
            result = build_dataset(cached_projects, root / "dataset")
            cli_destination = root / "dataset_from_catalog"
            with redirect_stdout(io.StringIO()):
                cli_result = run_cli([
                    "dataset", "--catalog", str(catalog), "--output", str(cli_destination),
                ])
            cli_manifest = json.loads((cli_destination / "dataset_manifest.json").read_text(encoding="utf-8"))

            with (cached_projects[0] / "annotations.csv").open(
                "r", encoding="utf-8-sig", newline="",
            ) as source:
                cached_rows = list(csv.DictReader(source, delimiter=";"))
            cache_manifest = json.loads((cached_projects[0] / "cache_manifest.json").read_text(encoding="utf-8"))
            first_snapshot = cached_projects[0]

            set_annotation_approvals(project, {"a1": True})
            confirm_project_for_training(project, catalog)
            updated_projects = load_confirmed_project_catalog(catalog)
            updated_overview = catalog_overview(catalog)
            first_snapshot_exists = first_snapshot.is_dir()
            with (updated_projects[0] / "annotations.csv").open(
                "r", encoding="utf-8-sig", newline="",
            ) as source:
                updated_rows = list(csv.DictReader(source, delimiter=";"))

        self.assertEqual(1, len(cached_projects))
        self.assertEqual(1, len(cached_rows), "unchecked masks must not leak into the approved snapshot")
        self.assertEqual("1", cached_rows[0]["approved"])
        self.assertTrue(cache_manifest["partial_confirmation"])
        self.assertEqual(1, overview["summary"]["photos"])
        self.assertEqual(1, overview["summary"]["annotations"])
        self.assertEqual(1, len(overview["wells"][0]["mask_details"]))
        self.assertEqual("photo_0.jpg", overview["wells"][0]["mask_details"][0]["photo"])
        self.assertEqual("100", overview["wells"][0]["mask_details"][0]["depth_top"])
        self.assertEqual("Каналы", overview["wells"][0]["mask_details"][0]["facies_name"])
        self.assertEqual(1, result["photo_count"])
        self.assertEqual(1, result["annotation_count"])
        self.assertFalse(result["training_ready"], "a one-photo set must not pretend to have independent validation")
        self.assertEqual(0, result["val_photo_count"])
        self.assertEqual(1, result["train_photo_count"])
        self.assertEqual(0, cli_result, "the application's catalog export route must also allow one approved interval")
        self.assertEqual(1, cli_manifest["photo_count"])
        self.assertEqual(1, cli_manifest["annotation_count"])
        self.assertFalse(cli_manifest["training_ready"])
        self.assertEqual(2, len(updated_rows), "later approvals should extend the current well snapshot")
        self.assertNotEqual(first_snapshot, updated_projects[0])
        self.assertTrue(first_snapshot_exists, "older snapshots remain recoverable")
        self.assertEqual(2, updated_overview["summary"]["photos"])
        self.assertEqual(2, updated_overview["summary"]["annotations"])
        self.assertEqual(2, len(updated_overview["wells"][0]["mask_details"]))

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
            overview = catalog_overview(catalog)
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
        self.assertEqual(["W-0", "W-1"], [
            name for entry in overview["wells"] for name in entry["well_names"]
        ])
        self.assertTrue(all(entry["masks"] == 1 for entry in overview["wells"]))
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
