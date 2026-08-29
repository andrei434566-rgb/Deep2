"""Interactive local SAM annotator for building a facies reference library.

Left-click points belong to the desired core fragment; right-click points belong
to the background.  SAM turns those prompts into a mask, and the accepted
masked crop is saved into ``<output root>/<facies name>/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "models" / "sam" / "sam_vit_b_01ec64.pth"


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось открыть изображение: {path}")
    return image


class ImageCanvas(QWidget):
    """Display an image and collect SAM foreground/background point prompts."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(760, 680)
        self.setMouseTracking(True)
        self.image_rgb: np.ndarray | None = None
        self.points: list[tuple[float, float, int]] = []
        self.mask: np.ndarray | None = None
        self._overlay: np.ndarray | None = None
        self._qimage: QImage | None = None

    def set_image(self, image_bgr: np.ndarray) -> None:
        self.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._qimage = self._array_to_qimage(self.image_rgb, QImage.Format.Format_RGB888)
        self.points.clear()
        self.set_mask(None)
        self.update()

    def set_mask(self, mask: np.ndarray | None) -> None:
        self.mask = mask
        self._overlay = None
        if mask is not None:
            overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
            overlay[mask] = (0, 210, 255, 115)  # cyan with transparency
            self._overlay = overlay
        self.update()

    def clear_prompts(self) -> None:
        self.points.clear()
        self.set_mask(None)

    def source_point(self, position) -> tuple[float, float] | None:
        if self.image_rgb is None:
            return None
        rect = self._image_rect()
        if not rect.contains(position):
            return None
        x = (position.x() - rect.left()) / rect.width() * self.image_rgb.shape[1]
        y = (position.y() - rect.top()) / rect.height() * self.image_rgb.shape[0]
        return float(np.clip(x, 0, self.image_rgb.shape[1] - 1)), float(np.clip(y, 0, self.image_rgb.shape[0] - 1))

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        source = self.source_point(event.position())
        if source is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.points.append((*source, 1))
            self.set_mask(None)
        elif event.button() == Qt.MouseButton.RightButton:
            self.points.append((*source, 0))
            self.set_mask(None)

    def _image_rect(self) -> QRectF:
        if self.image_rgb is None:
            return QRectF()
        image_height, image_width = self.image_rgb.shape[:2]
        scale = min(self.width() / image_width, self.height() / image_height)
        width, height = image_width * scale, image_height * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    @staticmethod
    def _array_to_qimage(array: np.ndarray, image_format: QImage.Format) -> QImage:
        height, width = array.shape[:2]
        return QImage(array.data, width, height, array.strides[0], image_format)

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#20242b"))
        if self._qimage is None:
            painter.setPen(QColor("#d9e1ec"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Откройте фото керна")
            return
        rect = self._image_rect()
        painter.drawImage(rect, self._qimage)
        if self._overlay is not None:
            painter.drawImage(rect, self._array_to_qimage(self._overlay, QImage.Format.Format_RGBA8888))

        for x, y, label in self.points:
            display_x = rect.left() + x / self.image_rgb.shape[1] * rect.width()
            display_y = rect.top() + y / self.image_rgb.shape[0] * rect.height()
            painter.setPen(QPen(QColor("#2fe37a") if label else QColor("#f05252"), 2))
            painter.setBrush(QColor("#2fe37a") if label else QColor("#f05252"))
            painter.drawEllipse(int(display_x - 5), int(display_y - 5), 10, 10)


class SamCropAnnotator(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SAM — разметка и вырезка фрагментов керна")
        self.resize(1180, 780)
        self.predictor = None
        self.source_path: Path | None = None
        self.source_bgr: np.ndarray | None = None

        self.canvas = ImageCanvas()
        self.status = QLabel("Загрузка локальной модели SAM…")
        self.output_root = QLineEdit(str(ROOT / "dataset_facies"))
        self.facies_name = QLineEdit("фация_1")

        open_button = QPushButton("Открыть фото")
        open_button.clicked.connect(self.open_image)
        self.predict_button = QPushButton("Построить маску  [M]")
        self.predict_button.clicked.connect(self.predict_mask)
        clear_button = QPushButton("Очистить точки  [C]")
        clear_button.clicked.connect(self.clear_prompts)
        save_button = QPushButton("Сохранить фрагмент  [S]")
        save_button.clicked.connect(self.save_crop)
        browse_button = QPushButton("Выбрать…")
        browse_button.clicked.connect(self.choose_output_root)

        controls = QWidget()
        form = QFormLayout(controls)
        form.addRow(open_button)
        form.addRow(QLabel("ЛКМ — точка керна; ПКМ — фон."))
        form.addRow(QLabel("После кликов нажмите M, затем S."))
        form.addRow(self.predict_button)
        form.addRow(clear_button)
        root_row = QHBoxLayout()
        root_row.addWidget(self.output_root)
        root_row.addWidget(browse_button)
        form.addRow("Корень датасета", root_row)
        form.addRow("Папка фации", self.facies_name)
        form.addRow(save_button)
        form.addRow(self.status)
        form.addRow(QLabel("Сохраняются JPG-фрагмент и PNG-маска с одинаковым именем."))

        layout = QHBoxLayout()
        layout.addWidget(self.canvas, 1)
        layout.addWidget(controls)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        for text, shortcut, handler in (
            ("Открыть", "Ctrl+O", self.open_image),
            ("Маска", "M", self.predict_mask),
            ("Очистить", "C", self.clear_prompts),
            ("Сохранить", "S", self.save_crop),
        ):
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(handler)
            self.addAction(action)

        self._load_model()

    def _load_model(self) -> None:
        if not DEFAULT_CHECKPOINT.is_file():
            QMessageBox.critical(self, "SAM не найден", f"Нет checkpoint:\n{DEFAULT_CHECKPOINT}")
            self.status.setText("Модель не найдена")
            return
        QApplication.processEvents()
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = sam_model_registry["vit_b"](checkpoint=str(DEFAULT_CHECKPOINT))
            model.to(device=device)
            self.predictor = SamPredictor(model)
            self.status.setText(f"SAM готов ({device.upper()}). Откройте фото.")
        except Exception as error:
            self.status.setText("Не удалось запустить SAM")
            QMessageBox.critical(self, "Ошибка SAM", str(error))

    def open_image(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Выберите фото керна", str(self.source_path.parent if self.source_path else ROOT), "Изображения (*.jpg *.jpeg *.png *.tif *.tiff)")
        if not selected:
            return
        if self.predictor is None:
            QMessageBox.warning(self, "SAM ещё не готов", "Подождите загрузки модели и повторите.")
            return
        try:
            self.source_path = Path(selected)
            self.source_bgr = read_image(self.source_path)
            self.canvas.set_image(self.source_bgr)
            self.status.setText("SAM кодирует фото; это может занять некоторое время на CPU…")
            QApplication.processEvents()
            self.predictor.set_image(cv2.cvtColor(self.source_bgr, cv2.COLOR_BGR2RGB))
            self.status.setText("Фото готово. ЛКМ: керн, ПКМ: фон; M: маска; S: сохранить.")
        except Exception as error:
            QMessageBox.critical(self, "Не удалось открыть фото", str(error))

    def clear_prompts(self) -> None:
        self.canvas.clear_prompts()
        self.status.setText("Точки очищены.")

    def predict_mask(self) -> None:
        if self.predictor is None or self.source_bgr is None:
            QMessageBox.warning(self, "Нет фото", "Сначала откройте фото.")
            return
        if not self.canvas.points:
            QMessageBox.warning(self, "Нет точек", "Поставьте хотя бы одну зелёную точку на нужном фрагменте.")
            return
        points = np.asarray([(x, y) for x, y, _ in self.canvas.points], dtype=np.float32)
        labels = np.asarray([label for _, _, label in self.canvas.points], dtype=np.int32)
        try:
            self.status.setText("SAM строит маску…")
            QApplication.processEvents()
            masks, scores, _ = self.predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
            index = int(np.argmax(scores))
            self.canvas.set_mask(masks[index].astype(bool))
            self.status.setText(f"Маска готова. Оценка SAM: {float(scores[index]):.0%}. Проверьте и сохраните S.")
        except Exception as error:
            QMessageBox.critical(self, "Ошибка маски", str(error))

    def choose_output_root(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Корень папок фаций", self.output_root.text())
        if directory:
            self.output_root.setText(directory)

    def save_crop(self) -> None:
        if self.source_bgr is None or self.source_path is None or self.canvas.mask is None:
            QMessageBox.warning(self, "Нечего сохранять", "Постройте маску фрагмента перед сохранением.")
            return
        name = self.facies_name.text().strip()
        if not name or any(symbol in name for symbol in '<>:"/\\|?*'):
            QMessageBox.warning(self, "Имя папки", "Укажите корректное название фации.")
            return
        ys, xs = np.where(self.canvas.mask)
        if not len(xs):
            QMessageBox.warning(self, "Пустая маска", "Постройте другую маску.")
            return
        pad = 8
        x1, x2 = max(0, int(xs.min()) - pad), min(self.source_bgr.shape[1], int(xs.max()) + pad + 1)
        y1, y2 = max(0, int(ys.min()) - pad), min(self.source_bgr.shape[0], int(ys.max()) + pad + 1)
        mask = self.canvas.mask[y1:y2, x1:x2]
        crop = self.source_bgr[y1:y2, x1:x2]
        masked_crop = np.zeros_like(crop)
        masked_crop[mask] = crop[mask]

        destination = Path(self.output_root.text()).expanduser() / name
        destination.mkdir(parents=True, exist_ok=True)
        stem = self.source_path.stem
        number = 1
        while (destination / f"{stem}_{number:03d}.jpg").exists():
            number += 1
        image_path = destination / f"{stem}_{number:03d}.jpg"
        mask_path = destination / f"{stem}_{number:03d}_mask.png"
        success, encoded = cv2.imencode(".jpg", masked_crop, [cv2.IMWRITE_JPEG_QUALITY, 96])
        if not success:
            raise RuntimeError("Не удалось закодировать JPG")
        encoded.tofile(str(image_path))
        success, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
        if success:
            encoded.tofile(str(mask_path))
        self.status.setText(f"Сохранено: {image_path.name}")
        self.canvas.clear_prompts()


def main() -> int:
    app = QApplication(sys.argv)
    window = SamCropAnnotator()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
