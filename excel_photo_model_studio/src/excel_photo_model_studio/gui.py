from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import Qt, QProcess
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from .dataset import build_dataset
from .matching import read_photo_map, write_photo_map
from .models import PhotoRecord
from .project import create_project, load_annotations, refresh_project, set_annotation_approvals
from .tabular import as_float


class PathField(QWidget):
    def __init__(self, *, directory: bool = False, file_filter: str = "Все файлы (*)", parent=None):
        super().__init__(parent)
        self.directory = directory
        self.file_filter = file_filter
        self.edit = QLineEdit(self)
        button = QPushButton("Выбрать…", self)
        button.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(button)

    def value(self) -> Path:
        return Path(self.edit.text().strip())

    def set_value(self, value: str | Path) -> None:
        self.edit.setText(str(value))

    def _browse(self) -> None:
        if self.directory:
            value = QFileDialog.getExistingDirectory(self, "Выберите папку", self.edit.text())
        else:
            value, _ = QFileDialog.getOpenFileName(self, "Выберите файл", self.edit.text(), self.file_filter)
        if value:
            self.edit.setText(value)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Excel Photo Model Studio")
        self.resize(1180, 760)
        self.current_project: Path | None = None
        self.matching_preview_paths: dict[str, str] = {}
        self.matching_preview_details: dict[str, list[str]] = {}
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)
        self._build_project_tab()
        self._build_review_tab()
        self._build_train_tab()
        self.process: QProcess | None = None

    def _build_project_tab(self) -> None:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.excel = PathField(file_filter="Таблицы (*.xlsx *.xlsm *.xltx *.xltm *.xls *.csv *.tsv)")
        self.photos = PathField(directory=True)
        self.ocr = QCheckBox("Использовать Tesseract для фото без интервала в имени")
        form.addRow("Файл Excel/CSV:", self.excel)
        form.addRow("Папка фотографий:", self.photos)
        form.addRow("OCR:", self.ocr)
        layout.addLayout(form)
        row = QHBoxLayout()
        create = QPushButton("Сопоставить и показать маски")
        create.clicked.connect(self._create_project)
        save_map = QPushButton("Пересчитать после исправлений")
        save_map.clicked.connect(self._save_photo_map)
        row.addWidget(create)
        row.addWidget(save_map)
        row.addStretch(1)
        layout.addLayout(row)
        self.photo_table = QTableWidget(0, 6)
        self.photo_table.setHorizontalHeaderLabels(("OK", "Файл", "Скважина", "Начало", "Конец", "Источник"))
        self.photo_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.photo_table.setMaximumHeight(260)
        self.photo_table.itemSelectionChanged.connect(self._show_matching_preview)
        self.project_log = QTextEdit()
        self.project_log.setReadOnly(True)
        content = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self.photo_table)
        left.addWidget(self.project_log, 1)
        content.addLayout(left, 3)
        right = QVBoxLayout()
        self.matching_preview_title = QLabel("После сопоставления здесь появится фото с найденными интервалами.")
        self.matching_preview_title.setWordWrap(True)
        right.addWidget(self.matching_preview_title)
        self.matching_preview = QLabel("Выберите фотографию в таблице.")
        self.matching_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.matching_preview.setWordWrap(True)
        self.matching_preview.setMinimumWidth(380)
        right.addWidget(self.matching_preview, 1)
        content.addLayout(right, 2)
        layout.addLayout(content, 1)
        self.tabs.addTab(page, "1. Сопоставление")

    def _build_review_tab(self) -> None:
        page = QWidget(self)
        layout = QHBoxLayout(page)
        left = QVBoxLayout()
        controls = QHBoxLayout()
        load = QPushButton("Загрузить таблицу проверки")
        load.clicked.connect(self._load_review)
        save = QPushButton("Сохранить подтверждения")
        save.clicked.connect(self._save_review)
        controls.addWidget(load)
        controls.addWidget(save)
        controls.addStretch(1)
        left.addLayout(controls)
        self.review_table = QTableWidget(0, 9)
        self.review_table.setHorizontalHeaderLabels(("OK", "Фото", "Скважина", "От", "До", "Фация", "Текст №22", "Excel", "ID"))
        self.review_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.review_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.review_table.itemSelectionChanged.connect(self._show_preview)
        left.addWidget(self.review_table, 1)
        layout.addLayout(left, 3)
        self.preview = QLabel("Выберите строку. Цветная область — автоматически построенная маска.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setWordWrap(True)
        self.preview.setMinimumWidth(380)
        layout.addWidget(self.preview, 2)
        self.tabs.addTab(page, "2. Проверка масок")

    def _build_train_tab(self) -> None:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.dataset_output = PathField(directory=True)
        self.base_model = PathField(file_filter="PyTorch weights (*.pt)")
        self.model_output = PathField(directory=True)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(50)
        self.patience = QSpinBox()
        self.patience.setRange(1, 1000)
        self.patience.setValue(12)
        self.description_epochs = QSpinBox()
        self.description_epochs.setRange(1, 10000)
        self.description_epochs.setValue(40)
        form.addRow("Новая папка датасета:", self.dataset_output)
        form.addRow("Локальная базовая seg-модель .pt:", self.base_model)
        form.addRow("Новая папка результата:", self.model_output)
        form.addRow("Эпохи (верхний предел):", self.epochs)
        form.addRow("Early stopping patience:", self.patience)
        form.addRow("Эпохи модели текста №22:", self.description_epochs)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        build = QPushButton("Собрать датасет")
        build.clicked.connect(self._build_dataset)
        train = QPushButton("Обучить комплект best.pt + текст №22")
        train.clicked.connect(self._start_training)
        buttons.addWidget(build)
        buttons.addWidget(train)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        layout.addWidget(self.train_log, 1)
        self.tabs.addTab(page, "3. Датасет и best.pt")

    def _create_project(self) -> None:
        try:
            excel_path = self.excel.value()
            photos_path = self.photos.value()
            if not self.excel.edit.text().strip() or not excel_path.is_file():
                raise ValueError("Выберите один существующий файл Excel или CSV.")
            if excel_path.suffix.lower() not in {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls", ".csv", ".tsv"}:
                raise ValueError("Поддерживаются файлы XLSX, XLSM, XLTX, XLTM, XLS, CSV и TSV.")
            if not self.photos.edit.text().strip() or not photos_path.is_dir():
                raise ValueError("Выберите существующую папку с фотографиями.")
            project_dir = self._automatic_project_dir(excel_path)
            result = create_project(excel_path, photos_path, project_dir, use_ocr=self.ocr.isChecked())
        except Exception as exc:
            return self._error(str(exc))
        self.current_project = project_dir
        self.matching_preview_paths.clear()
        self.matching_preview_details.clear()
        self.dataset_output.set_value(project_dir / "dataset")
        self.model_output.set_value(project_dir / "model_candidate")
        self._show_report(result)
        self._load_photo_map()
        self._load_review()

    def _load_photo_map(self) -> None:
        try:
            records = read_photo_map(self._project_dir() / "photo_map.csv")
        except Exception as exc:
            return self._error(str(exc))
        self.photo_table.setRowCount(len(records))
        for row, record in enumerate(records):
            confirmed = QTableWidgetItem()
            confirmed.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
            confirmed.setCheckState(Qt.CheckState.Checked if record.mapping_confirmed else Qt.CheckState.Unchecked)
            self.photo_table.setItem(row, 0, confirmed)
            values = (
                str(record.path), record.well,
                "" if record.top is None else f"{record.top:g}",
                "" if record.base is None else f"{record.base:g}", record.source,
            )
            for column, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                if column in {1, 5}:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.photo_table.setItem(row, column, item)
        if records:
            self.photo_table.selectRow(0)

    def _save_photo_map(self) -> None:
        records = []
        for row in range(self.photo_table.rowCount()):
            path = Path(self.photo_table.item(row, 1).text())
            well = self.photo_table.item(row, 2).text().strip()
            top = as_float(self.photo_table.item(row, 3).text())
            base = as_float(self.photo_table.item(row, 4).text())
            confirmed = self.photo_table.item(row, 0).checkState() == Qt.CheckState.Checked
            if confirmed and (not well or top is None or base is None or base <= top):
                return self._error(f"{path.name}: для подтверждения нужны скважина и корректный интервал.")
            records.append(PhotoRecord(
                path=path, well=well, top=top, base=base,
                source=self.photo_table.item(row, 5).text().strip() or "manual",
                mapping_confirmed=confirmed,
            ))
        try:
            project_dir = self._project_dir()
            write_photo_map(project_dir / "photo_map.csv", records)
            result = refresh_project(project_dir)
        except Exception as exc:
            return self._error(str(exc))
        self._show_report(result)
        self._load_review()

    def _show_report(self, report: dict) -> None:
        lines = [
            f"Служебная папка создана автоматически: {report['project_dir']}",
            f"Excel/CSV-файлов: {report.get('excel_files', 1)}", f"Строк Excel: {report['excel_rows']}", f"Фото: {report['photos']}",
            f"Подтверждены интервалы фото: {report['confirmed_photos']}",
            f"Нужно подтвердить интервалы: {report['unconfirmed_photos']}",
            f"Спроецировано масок: {report['annotations']}",
            f"Масок с целевым текстом №22: {report.get('text_targets', 0)}",
            f"Подтверждено масок: {report['approved_annotations']}",
        ]
        if report.get("issues"):
            lines.append("\nПроверить:")
            lines.extend(f"- {item['source']}: {item['message']}" for item in report["issues"])
        lines.append("\nНеизвестные интервалы исправьте прямо в таблице выше, отметьте OK и нажмите «Пересчитать после исправлений».")
        self.project_log.setPlainText("\n".join(lines))

    def _load_review(self) -> None:
        try:
            rows = load_annotations(self._project_dir())
        except Exception as exc:
            return self._error(str(exc))
        self.matching_preview_paths.clear()
        self.matching_preview_details.clear()
        self.review_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            approved = QTableWidgetItem()
            approved.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
            approved.setCheckState(Qt.CheckState.Checked if row.get("approved") == "1" else Qt.CheckState.Unchecked)
            approved.setData(Qt.ItemDataRole.UserRole, row.get("preview", ""))
            self.review_table.setItem(row_index, 0, approved)
            values = (
                Path(row["photo"]).name, row["well"], row["depth_top"], row["depth_base"],
                row["label"], row.get("target_text", ""),
                f"{Path(row.get('source_file', '')).name}:{row['source_sheet']}!{row['source_row']}",
                row["annotation_id"],
            )
            for column, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.review_table.setItem(row_index, column, item)
            photo_key = self._photo_key(row["photo"])
            if row.get("preview"):
                self.matching_preview_paths[photo_key] = row["preview"]
            self.matching_preview_details.setdefault(photo_key, []).append(
                f"{row['depth_top']}–{row['depth_base']} м: {row['label']}"
            )
        if rows:
            self.review_table.selectRow(0)
        self._select_first_masked_photo()

    def _save_review(self) -> None:
        approvals = {}
        for row in range(self.review_table.rowCount()):
            approvals[self.review_table.item(row, 8).text()] = self.review_table.item(row, 0).checkState() == Qt.CheckState.Checked
        try:
            report = set_annotation_approvals(self._project_dir(), approvals)
        except Exception as exc:
            return self._error(str(exc))
        self.project_log.append(f"Сохранено подтверждений: {report['approved_annotations']}")

    def _show_preview(self) -> None:
        row = self.review_table.currentRow()
        if row < 0:
            return
        preview = self.review_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        self._set_preview_pixmap(self.preview, preview, "Предпросмотр не найден.")

    def _show_matching_preview(self) -> None:
        row = self.photo_table.currentRow()
        if row < 0 or self.photo_table.item(row, 1) is None:
            self.matching_preview_title.setText("После сопоставления здесь появится фото с найденными интервалами.")
            self.matching_preview.setText("Выберите фотографию в таблице.")
            return
        photo_path = self.photo_table.item(row, 1).text()
        photo_key = self._photo_key(photo_path)
        preview_path = self.matching_preview_paths.get(photo_key)
        details = self.matching_preview_details.get(photo_key, [])
        if preview_path:
            visible_details = "\n".join(details[:4])
            suffix = f"\nЕщё интервалов: {len(details) - 4}" if len(details) > 4 else ""
            self.matching_preview_title.setText(
                f"Найдено интервалов: {len(details)}\n{visible_details}{suffix}"
            )
            self._set_preview_pixmap(self.matching_preview, preview_path, "Не удалось открыть фото с масками.")
        else:
            self.matching_preview_title.setText("На этом фото совпадающие интервалы пока не найдены.")
            self._set_preview_pixmap(self.matching_preview, photo_path, "Не удалось открыть исходную фотографию.")

    def _select_first_masked_photo(self) -> None:
        for row in range(self.photo_table.rowCount()):
            item = self.photo_table.item(row, 1)
            if item is not None and self._photo_key(item.text()) in self.matching_preview_paths:
                self.photo_table.selectRow(row)
                self._show_matching_preview()
                return
        self._show_matching_preview()

    @staticmethod
    def _photo_key(path: str | Path) -> str:
        return os.path.normcase(os.path.abspath(str(path)))

    @staticmethod
    def _set_preview_pixmap(label: QLabel, path: str | Path, error_text: str) -> None:
        pixmap = QPixmap()
        try:
            loaded = pixmap.loadFromData(Path(path).read_bytes())
        except OSError:
            loaded = False
        if not loaded or pixmap.isNull():
            label.setText(error_text)
            return
        label.setPixmap(pixmap.scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "photo_table"):
            self._show_matching_preview()
        if hasattr(self, "review_table"):
            self._show_preview()

    def _build_dataset(self) -> None:
        try:
            result = build_dataset(self._project_dir(), self.dataset_output.value())
        except Exception as exc:
            return self._error(str(exc))
        self.train_log.setPlainText(json.dumps({key: value for key, value in result.items() if key != "samples"}, ensure_ascii=False, indent=2))

    def _start_training(self) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            return self._error("Обучение уже запущено.")
        command = [
            "train",
            "--dataset", str(self.dataset_output.value()), "--base-model", str(self.base_model.value()),
            "--output", str(self.model_output.value()), "--epochs", str(self.epochs.value()),
            "--patience", str(self.patience.value()),
            "--description-epochs", str(self.description_epochs.value()),
        ]
        if not getattr(sys, "frozen", False):
            launcher = Path(__file__).resolve().parents[2] / "run.py"
            command.insert(0, str(launcher))
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(command)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_training_output)
        self.process.finished.connect(self._training_finished)
        self.train_log.clear()
        self.process.start()

    def _read_training_output(self) -> None:
        if self.process:
            self.train_log.append(bytes(self.process.readAllStandardOutput()).decode(errors="replace"))

    def _training_finished(self, code: int, _status) -> None:
        self.train_log.append(f"\nПроцесс завершён, код {code}.")

    def _automatic_project_dir(self, excel_path: Path) -> Path:
        local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        safe_stem = re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", excel_path.stem).strip("_") or "project"
        unique_name = f"{safe_stem}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        return local_app_data / "ExcelPhotoModelStudio" / "projects" / unique_name

    def _project_dir(self) -> Path:
        if self.current_project is None:
            raise RuntimeError("Сначала выберите Excel, папку фотографий и выполните сопоставление.")
        return self.current_project

    def _error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка", message)


def run_gui() -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return application.exec()
