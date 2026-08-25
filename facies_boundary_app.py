"""Desktop demo: upload a core photo and inspect automatic facies boundaries.

Run with:
    python facies_boundary_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.infrastructure.ml.rule_based_facies import RuleBasedFaciesDetector, TextureInterval


class FaciesBoundaryWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DeepCore — границы текстурных пакетов")
        self.resize(1440, 860)
        self._detector = RuleBasedFaciesDetector()
        self._source_bgr: np.ndarray | None = None
        self._annotated_bgr: np.ndarray | None = None
        self._source_path: Path | None = None

        central = QWidget(self)
        layout = QVBoxLayout(central)
        controls = QHBoxLayout()
        open_button = QPushButton("Загрузить фото", self)
        open_button.clicked.connect(self.open_image)
        save_button = QPushButton("Сохранить разметку", self)
        save_button.clicked.connect(self.save_result)
        self._summary = QLabel("Загрузите фотографию керна. Справа появятся только границы текстурных пакетов.", self)
        self._summary.setWordWrap(True)
        controls.addWidget(open_button)
        controls.addWidget(save_button)
        controls.addWidget(self._summary, 1)
        layout.addLayout(controls)

        image_layout = QHBoxLayout()
        self._original_label = self._create_image_label("Исходное фото")
        self._result_label = self._create_image_label("Автоматические пакеты")
        image_layout.addWidget(self._scrollable(self._original_label), 1)
        image_layout.addWidget(self._scrollable(self._result_label), 1)
        layout.addLayout(image_layout, 1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Алгоритм не предполагает фацию: он показывает только устойчивые смены текстуры.")

    @staticmethod
    def _create_image_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(500, 600)
        label.setStyleSheet("QLabel { background: #f4f4f4; border: 1px solid #cfcfcf; }")
        return label

    @staticmethod
    def _scrollable(label: QLabel) -> QScrollArea:
        area = QScrollArea()
        area.setWidget(label)
        area.setWidgetResizable(True)
        return area

    def open_image(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите фотографию керна",
            str(self._source_path.parent if self._source_path else Path.home()),
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not file_name:
            return
        image = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.critical(self, "Ошибка", "Не удалось открыть изображение.")
            return
        try:
            annotated, intervals = self._detector.analyse(image)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка анализа", str(exc))
            return

        self._source_bgr = image
        self._annotated_bgr = annotated
        self._source_path = Path(file_name)
        self._set_image(self._original_label, image)
        self._set_image(self._result_label, annotated)
        self._summary.setText(self._summary_text(intervals))
        self.statusBar().showMessage(f"Готово: выделено пакетов — {len(intervals)}.")

    def save_result(self) -> None:
        if self._annotated_bgr is None or self._source_path is None:
            QMessageBox.information(self, "Нет результата", "Сначала загрузите и обработайте фотографию.")
            return
        default_path = self._source_path.with_name(f"{self._source_path.stem}_facies_rule_based.png")
        file_name, _ = QFileDialog.getSaveFileName(self, "Сохранить разметку", str(default_path), "PNG (*.png)")
        if not file_name:
            return
        success, buffer = cv2.imencode(".png", self._annotated_bgr)
        if not success or not buffer.tofile(file_name) is None:
            QMessageBox.critical(self, "Ошибка", "Не удалось сохранить PNG.")
            return
        self.statusBar().showMessage(f"Сохранено: {file_name}")

    @staticmethod
    def _set_image(label: QLabel, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap)
        label.resize(pixmap.size())

    @staticmethod
    def _summary_text(intervals: list[TextureInterval]) -> str:
        if not intervals:
            return "Столбики керна не найдены. Нужна фотография с хорошо видимым керном на светлом или контрастном фоне."
        columns = len({item.column_index for item in intervals})
        return f"Найдено столбиков керна: {columns}; текстурных пакетов: {len(intervals)}. Цвет разметки не обозначает фацию."


def main() -> int:
    app = QApplication(sys.argv)
    window = FaciesBoundaryWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
