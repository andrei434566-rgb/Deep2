from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from openpyxl import Workbook
from PySide6.QtWidgets import QApplication

from excel_photo_model_studio.gui import MainWindow
from excel_photo_model_studio.matching import read_photo_map, write_photo_map
from excel_photo_model_studio.models import PhotoRecord
from excel_photo_model_studio.project import create_project


class GuiPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_matching_tab_shows_photo_with_interval_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excel = root / "description.xlsx"
            photos = root / "photos"
            photos.mkdir()
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["Скважина", "Кровля", "Подошва", "Класс", "Краткое описание"])
            sheet.append(["W-1", 100.0, 102.0, "Sand", "Sandstone description"])
            workbook.save(excel)
            image = np.full((240, 160, 3), (20, 80, 150), dtype=np.uint8)
            cv2.rectangle(image, (55, 20), (105, 220), (115, 115, 115), -1)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            (photos / "W-1 100-102.jpg").write_bytes(encoded.tobytes())
            project = root / "project"
            create_project(excel, photos, project)

            window = MainWindow()
            window.show()
            self.application.processEvents()
            window.current_project = project
            window._load_photo_map()
            window._load_review()
            self.application.processEvents()

            self.assertEqual(1, window.photo_table.rowCount())
            self.assertTrue(window.ocr.isChecked())
            self.assertEqual("yolo11n-seg.yaml", window.architecture.currentData())
            self.assertEqual("auto", window.photo_table.cellWidget(0, 5).currentData())
            self.assertIn("Колонок керна найдено: 1", window.matching_preview_title.text())
            self.assertIn("Интервалы колонок: 1) 100–101 м", window.matching_preview_title.text())
            self.assertIn("Порядок:", window.matching_preview_title.text())
            self.assertIn("Найдено интервалов", window.matching_preview_title.text())
            self.assertIsNotNone(window.matching_preview.pixmap())
            self.assertFalse(window.matching_preview.pixmap().isNull())
            tooltip = window.matching_preview.tooltip_for_image_point(80, 120)
            self.assertIn(
                "Краткое описание: Sandstone description",
                tooltip,
            )
            self.assertIn("Участок на фото: 100–101 м", tooltip)
            self.assertIn("Полный интервал фации: 100–102 м", tooltip)
            report = json.loads((project / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(any("длиннее вместимости найденного керна" in issue["message"] for issue in report["issues"]))
            window.close()

    def test_recalculate_retains_ocr_coordinate_basis_unless_depth_is_edited(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            photo = project / "core.jpg"
            image = np.full((200, 200, 3), 255, dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            photo.write_bytes(encoded.tobytes())
            original = PhotoRecord(
                photo, "W-1", 100.0, 101.0, "ocr_verified", True,
                column_depths=((0.5, 100.0, 101.0),), column_ocr_checked=True,
                depth_basis="gis",
            )
            write_photo_map(project / "photo_map.csv", [original])
            window = MainWindow()
            window.current_project = project
            window._load_photo_map()
            with patch.object(window, "_start_project_process"):
                window._save_photo_map()
            retained = read_photo_map(project / "photo_map.csv")[0]
            self.assertEqual("gis", retained.depth_basis)
            self.assertEqual(original.column_depths, retained.column_depths)
            self.assertIn("по ГИС / с увязкой", window.matching_preview_title.text())

            window.photo_table.item(0, 3).setText("99.9")
            with patch.object(window, "_start_project_process"):
                window._save_photo_map()
            edited = read_photo_map(project / "photo_map.csv")[0]
            self.assertEqual("unknown", edited.depth_basis)
            self.assertEqual((), edited.column_depths)
            self.assertFalse(edited.column_ocr_checked)
            self.assertEqual("manual", edited.source)
            self.assertIn("Система глубин: не определена", window.matching_preview_title.text())
            window.close()

    def test_report_exposes_projection_failures_and_incomplete_facies(self):
        window = MainWindow()
        report = {
            "project_dir": "example", "excel_rows": 1, "photos": 1,
            "confirmed_photos": 1, "unconfirmed_photos": 0,
            "annotations": 1, "approved_annotations": 0, "blocking_errors": 2,
            "projection_errors": 1, "facies_rows_with_incomplete_masks": 1,
            "issues": [{"severity": "error", "source": "core.jpg", "message": "Колонка керна не покрыта масками."}],
        }
        window._show_report(report)
        message = window.project_log.toPlainText()
        self.assertIn("Обучение заблокировано", message)
        self.assertIn("Ошибок привязки масок к колонкам керна: 1", message)
        self.assertIn("Фаций Excel с неполным покрытием масками: 1", message)
        self.assertEqual(["Колонка керна не покрыта масками."], window.matching_photo_issues["core.jpg"])
        window.close()


if __name__ == "__main__":
    unittest.main()
