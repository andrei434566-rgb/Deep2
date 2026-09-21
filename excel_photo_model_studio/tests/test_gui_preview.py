from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from openpyxl import Workbook
from PySide6.QtWidgets import QApplication

from excel_photo_model_studio.gui import MainWindow
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
            sheet.append(["Скважина", "Кровля", "Подошва", "Класс"])
            sheet.append(["W-1", 100.0, 102.0, "Sand"])
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
            self.assertIn("Найдено интервалов", window.matching_preview_title.text())
            self.assertIsNotNone(window.matching_preview.pixmap())
            self.assertFalse(window.matching_preview.pixmap().isNull())
            window.close()


if __name__ == "__main__":
    unittest.main()
