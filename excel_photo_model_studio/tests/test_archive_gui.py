from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from excel_photo_model_studio.gui import MainWindow


EMPTY_CATALOG = {
    "summary": {"projects": 0, "photos": 0, "annotations": 0, "approved_annotations": 0},
    "wells": [],
}


class ArchiveGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_archive_row_opens_correct_sources_and_resumes_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excel = root / "well.xlsx"
            excel.write_bytes(b"dummy")
            photos = root / "photos"
            photos.mkdir()
            project = root / "project"
            project.mkdir()
            (project / "project.json").write_text(json.dumps({
                "excel_paths": [str(excel)], "photos_dir": str(photos),
            }), encoding="utf-8")
            queue = {
                "excel_root": str(root), "photo_root": str(photos),
                "entries": [{"excel": str(excel), "photos": str(photos),
                             "photo_count": 1, "reason": "", "project": str(project)}],
            }
            with (
                patch("excel_photo_model_studio.gui.load_archive_queue", return_value=queue),
                patch("excel_photo_model_studio.gui.load_confirmed_project_catalog", return_value=[]),
                patch("excel_photo_model_studio.gui.catalog_overview", return_value=EMPTY_CATALOG),
            ):
                window = MainWindow()
                try:
                    self.assertEqual("6. Массовый архив", window.tabs.tabText(5))
                    self.assertEqual("Проверить маски", window.archive_table.item(0, 3).text())
                    window.archive_table.selectRow(0)
                    with patch.object(window, "_start_project_process") as start:
                        window._archive_open_selected()
                    self.assertEqual(str(excel), window.excel.edit.text())
                    self.assertEqual(str(photos), window.photos.edit.text())
                    self.assertEqual(0, window.tabs.currentIndex())
                    self.assertEqual(project, start.call_args.args[1])
                finally:
                    window.close()

    def test_confirmed_stage_requires_catalog_snapshot_not_checkbox_only(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            (project / "project.json").write_text("{}", encoding="utf-8")
            entry = {"excel": "well.xlsx", "photos": directory, "project": str(project)}
            self.assertEqual("Проверить маски", MainWindow._archive_stage(
                object(), entry, set(),
            ))
            self.assertEqual("Подтверждено ✓", MainWindow._archive_stage(
                object(), entry, {os.path.normcase(str(project))},
            ))

    def test_training_rejects_dataset_built_before_new_well_was_confirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "dataset_manifest.json").write_text(json.dumps({
                "projects": [str(root / "old_snapshot")], "class_names": ["Sand"],
            }), encoding="utf-8")
            with (
                patch("excel_photo_model_studio.gui.load_archive_queue", return_value={"entries": []}),
                patch("excel_photo_model_studio.gui.load_confirmed_project_catalog", return_value=[]),
                patch("excel_photo_model_studio.gui.catalog_overview", return_value=EMPTY_CATALOG),
            ):
                window = MainWindow()
            try:
                window.dataset_output.set_value(dataset)
                window.model_output.set_value(root / "model")
                with (
                    patch("excel_photo_model_studio.gui.load_confirmed_project_catalog",
                          return_value=[root / "new_snapshot"]),
                    patch.object(window, "_error") as show_error,
                ):
                    window._start_training()
                self.assertIn("новые маски", show_error.call_args.args[0])
                self.assertIsNone(window.training_worker)
            finally:
                window.close()

    def test_build_and_train_waits_for_ready_joint_dataset(self):
        with (
            patch("excel_photo_model_studio.gui.load_archive_queue", return_value={"entries": []}),
            patch("excel_photo_model_studio.gui.load_confirmed_project_catalog", return_value=[]),
            patch("excel_photo_model_studio.gui.catalog_overview", return_value=EMPTY_CATALOG),
        ):
            window = MainWindow()
        try:
            with patch.object(window, "_archive_build_dataset", return_value=True) as build:
                window._archive_build_and_train()
            build.assert_called_once()
            self.assertTrue(window._auto_train_after_dataset)
            with patch.object(window, "_start_training") as train:
                window._dataset_completed({
                    "training_ready": True,
                    "train_caption_count": 20,
                    "val_caption_count": 3,
                })
                window._dataset_worker_finished(object())
            train.assert_called_once()
            self.assertFalse(window._auto_train_after_dataset)
        finally:
            window.close()

    def test_build_and_train_does_not_start_on_incomplete_data(self):
        with (
            patch("excel_photo_model_studio.gui.load_archive_queue", return_value={"entries": []}),
            patch("excel_photo_model_studio.gui.load_confirmed_project_catalog", return_value=[]),
            patch("excel_photo_model_studio.gui.catalog_overview", return_value=EMPTY_CATALOG),
        ):
            window = MainWindow()
        try:
            with patch.object(window, "_archive_build_dataset", return_value=True):
                window._archive_build_and_train()
            with patch.object(window, "_start_training") as train:
                window._dataset_completed({
                    "training_ready": True,
                    "train_caption_count": 2,
                    "val_caption_count": 0,
                })
                window._dataset_worker_finished(object())
            train.assert_not_called()
            self.assertIn("Автозапуск обучения не выполнен", window.train_log.toPlainText())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
