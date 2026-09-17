"""UI regression check for re-annotation after a facies-model change."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox

from app.domain.models import PhotoRecord
from app.ui.windows.main_window import MainWindow


_APP = QApplication.instance() or QApplication([])


class ModelResegmentationTests(unittest.TestCase):
    def test_selecting_model_offers_and_runs_full_reannotation(self):
        window = MainWindow()
        record = PhotoRecord("one", "memory.jpg", QPixmap(10, 10))
        window._records = [record]
        with tempfile.TemporaryDirectory() as directory:
            model_path = (Path(directory) / "candidate.pt").resolve()
            model_path.write_bytes(b"test model placeholder")
            with (
                patch("app.ui.windows.main_window.QFileDialog.getOpenFileName", return_value=(str(model_path), "")),
                patch("app.ui.windows.main_window.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes),
                patch.object(window, "run_segmentation") as run_segmentation,
            ):
                window.select_detection_model()

            self.assertEqual(model_path, window._selected_model_path.resolve())
            run_segmentation.assert_called_once_with(window._records)
        window.close()


if __name__ == "__main__":
    unittest.main()
