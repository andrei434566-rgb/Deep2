from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from excel_photo_model_studio.archive_queue import (
    discover_archive, load_archive_queue, merge_archive_queue,
    prepare_archive_queue, save_archive_queue,
)
from unittest.mock import patch


class ArchiveQueueTests(unittest.TestCase):
    def test_separate_roots_match_well_ids_and_keep_all_nested_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, photos = root / "tables", root / "photos"
            tables.mkdir()
            photos.mkdir()
            (tables / "Описание керна скв 671ПО.xlsx").write_bytes(b"excel")
            (tables / "Описание керна скв 672ПО.xlsx").write_bytes(b"excel")
            for well in ("671ПО", "672ПО"):
                for page in ("001", "002"):
                    folder = photos / well[:3] / page
                    folder.mkdir(parents=True)
                    (folder / f"Рис. скв № {well}.jpg").write_bytes(b"photo")

            result = discover_archive(tables, photos)

            self.assertEqual(2, result["workbooks_found"])
            self.assertEqual(4, result["photos_found"])
            self.assertEqual(2, len(result["entries"]))
            self.assertEqual({"671", "672"}, {Path(item["photos"]).name for item in result["entries"]})
            self.assertTrue(all(item["photo_count"] == 2 for item in result["entries"]))

    def test_mixed_latin_cyrillic_well_letters_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, photos = root / "tables", root / "photos"
            tables.mkdir()
            photos.mkdir()
            (tables / "Описание скв 671ПО.xlsx").write_bytes(b"excel")
            folder = photos / "671ПO"  # Last letter is Latin O.
            folder.mkdir()
            (folder / "керн скв 671ПO.jpg").write_bytes(b"photo")

            result = discover_archive(tables, photos)

            self.assertEqual(str(folder.resolve()), result["entries"][0]["photos"])

    def test_shared_photo_folder_is_not_silently_assigned_to_multiple_excels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, photos = root / "tables", root / "photos"
            tables.mkdir()
            photos.mkdir()
            for well in ("A1", "B2"):
                (tables / f"скв {well}.xlsx").write_bytes(b"excel")
                (photos / f"скв {well}.jpg").write_bytes(b"photo")

            result = discover_archive(tables, photos)

            self.assertTrue(all(not item["photos"] for item in result["entries"]))
            self.assertTrue(all(item["reason"] for item in result["entries"]))

    def test_one_to_one_explicit_well_mismatch_is_not_auto_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, photos = root / "tables", root / "photos"
            tables.mkdir()
            photos.mkdir()
            (tables / "скв A1.xlsx").write_bytes(b"excel")
            folder = photos / "B2"
            folder.mkdir()
            (folder / "скв B2.jpg").write_bytes(b"photo")

            result = discover_archive(tables, photos)

            self.assertEqual("", result["entries"][0]["photos"])

    def test_generic_excel_name_is_paired_by_well_inside_workbook(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, photos = root / "tables", root / "photos"
            tables.mkdir()
            photos.mkdir()
            for well in ("A1", "B2"):
                workbook = Workbook()
                sheet = workbook.active
                sheet.append(["Скважина", "Кровля", "Подошва", "Класс", "Краткое описание"])
                sheet.append([well, 100.0, 100.5, "Sand", "Песчаник светлый"])
                workbook.save(tables / f"description_{well[-1]}.xlsx")
                folder = photos / well
                folder.mkdir()
                (folder / f"скв {well}.jpg").write_bytes(b"photo")

            result = discover_archive(tables, photos)

            self.assertEqual(2, len(result["entries"]))
            self.assertEqual({"A1", "B2"}, {Path(item["photos"]).name for item in result["entries"]})
            self.assertTrue(all(item["excel_rows"] == 1 for item in result["entries"]))

    def test_rescan_preserves_manual_pair_and_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, photos = root / "tables", root / "photos"
            tables.mkdir()
            photos.mkdir()
            book = tables / "one.xlsx"
            book.write_bytes(b"excel")
            (photos / "one.jpg").write_bytes(b"photo")
            project = root / "project"
            project.mkdir()
            (project / "project.json").write_text(json.dumps({
                "excel_paths": [str(book)], "photos_dir": str(photos),
            }), encoding="utf-8")
            previous = discover_archive(tables, photos)
            previous["entries"][0].update(
                photos=str(photos), manual_pair=True, project=str(project),
            )
            queue_file = root / "queue.json"
            save_archive_queue(previous, queue_file)

            merged = merge_archive_queue(discover_archive(tables, photos), load_archive_queue(queue_file))

            self.assertEqual(str(photos), merged["entries"][0]["photos"])
            self.assertEqual(str(project), merged["entries"][0]["project"])
            self.assertTrue(merged["entries"][0]["manual_pair"])

    def test_batch_preparation_continues_after_one_error_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photos = root / "photos"
            photos.mkdir()
            entries = []
            for index in (1, 2):
                workbook = root / f"well{index}.xlsx"
                workbook.write_bytes(b"excel")
                entries.append({"excel": str(workbook), "photos": str(photos), "project": ""})
            queue = {"entries": entries}
            calls = []

            def fake_create(excel, photo_dir, project_dir, *, use_ocr):
                calls.append(Path(excel).name)
                if Path(excel).stem == "well1":
                    raise ValueError("не прочитан заголовок")
                project_dir.mkdir(parents=True)
                (project_dir / "project.json").write_text(json.dumps({
                    "excel_paths": [str(excel)], "photos_dir": str(photo_dir),
                }), encoding="utf-8")
                return {"excel_rows": 4, "photos": 2, "annotations": 3, "blocking_errors": 1}

            queue_path = root / "queue.json"
            with patch("excel_photo_model_studio.project.create_project", side_effect=fake_create):
                result = prepare_archive_queue(
                    queue, use_ocr=True, project_root=root / "projects",
                    queue_path=queue_path,
                )
                again = prepare_archive_queue(
                    result, use_ocr=True, project_root=root / "projects",
                    queue_path=queue_path,
                )

            self.assertEqual(["well1.xlsx", "well2.xlsx", "well1.xlsx"], calls)
            self.assertIn("не прочитан заголовок", again["entries"][0]["error"])
            self.assertEqual(3, again["entries"][1]["mask_count"])
            self.assertEqual(1, again["entries"][1]["blocking_errors"])
            self.assertEqual(
                again["entries"][1]["project"], load_archive_queue(queue_path)["entries"][1]["project"],
            )


if __name__ == "__main__":
    unittest.main()
