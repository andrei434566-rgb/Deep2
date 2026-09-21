from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QPointF, Qt, QProcess
from PySide6.QtGui import QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QToolTip, QVBoxLayout, QWidget,
)

from .catalog import catalog_summary, default_catalog_path, register_project
from .matching import read_photo_map, write_photo_map
from .models import (
    COLUMN_ORDER_AUTO, COLUMN_ORDER_LEFT_TO_RIGHT, COLUMN_ORDER_RIGHT_TO_LEFT,
    PhotoRecord, normalize_column_order,
)
from .project import load_annotations, set_annotation_approvals
from .tabular import as_float


class PathField(QWidget):
    def __init__(self, *, directory: bool = False, save: bool = False, file_filter: str = "Все файлы (*)", parent=None):
        super().__init__(parent)
        self.directory = directory
        self.save = save
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
        elif self.save:
            value, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", self.edit.text(), self.file_filter)
        else:
            value, _ = QFileDialog.getOpenFileName(self, "Выберите файл", self.edit.text(), self.file_filter)
        if value:
            self.edit.setText(value)


class MaskPreviewLabel(QLabel):
    """Image preview which shows the matched short description over a mask."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._regions: list[dict] = []
        self._last_tip = ""
        self.setMouseTracking(True)

    def set_regions(self, regions: list[dict]) -> None:
        self._regions = list(regions)
        self._last_tip = ""

    def tooltip_for_image_point(self, x: float, y: float) -> str:
        candidates: list[tuple[float, dict]] = []
        point = QPointF(float(x), float(y))
        for region in self._regions:
            try:
                polygon = QPolygonF([QPointF(float(px), float(py)) for px, py in region["polygon"]])
            except (KeyError, TypeError, ValueError):
                continue
            if polygon.containsPoint(point, Qt.FillRule.OddEvenFill):
                bounds = polygon.boundingRect()
                candidates.append((max(1.0, bounds.width() * bounds.height()), region))
        if not candidates:
            return ""
        region = min(candidates, key=lambda item: item[0])[1]
        description = str(region.get("target_text", "")).strip() or "Описание отсутствует"
        label = str(region.get("label", "")).strip()
        interval = f"{region.get('depth_top', '')}–{region.get('depth_base', '')} м"
        parts = [interval]
        if label:
            parts.append(f"Фация: {label}")
        parts.append(f"Краткое описание: {description}")
        return "\n".join(parts)

    def mouseMoveEvent(self, event) -> None:
        pixmap = self.pixmap()
        text = ""
        if pixmap is not None and not pixmap.isNull() and self._regions:
            x_offset = (self.width() - pixmap.width()) / 2.0
            y_offset = (self.height() - pixmap.height()) / 2.0
            local_x = event.position().x() - x_offset
            local_y = event.position().y() - y_offset
            if 0 <= local_x < pixmap.width() and 0 <= local_y < pixmap.height():
                width = max(float(region.get("image_width", 0) or 0) for region in self._regions)
                height = max(float(region.get("image_height", 0) or 0) for region in self._regions)
                if width > 0 and height > 0:
                    text = self.tooltip_for_image_point(
                        local_x * width / pixmap.width(),
                        local_y * height / pixmap.height(),
                    )
        if text and text != self._last_tip:
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        elif not text and self._last_tip:
            QToolTip.hideText()
        self._last_tip = text
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        QToolTip.hideText()
        self._last_tip = ""
        super().leaveEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Excel Photo Model Studio")
        self.resize(1180, 760)
        self.current_project: Path | None = None
        self.matching_preview_paths: dict[str, str] = {}
        self.matching_preview_details: dict[str, list[str]] = {}
        self.matching_preview_regions: dict[str, list[dict]] = {}
        self.matching_column_orders: dict[str, str] = {}
        self.matching_column_counts: dict[str, int] = {}
        self.matching_column_depths: dict[str, list[str]] = {}
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)
        self._build_project_tab()
        self._build_review_tab()
        self._build_train_tab()
        self._build_analyze_tab()
        self.project_process: QProcess | None = None
        self.dataset_process: QProcess | None = None
        self.process: QProcess | None = None
        self.analysis_process: QProcess | None = None
        self._pending_project_action = ""
        self._ensure_training_output_paths()

    def _build_project_tab(self) -> None:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.excel = PathField(file_filter="Таблицы (*.xlsx *.xlsm *.xltx *.xltm *.xls *.csv *.tsv)")
        self.photos = PathField(directory=True)
        self.ocr = QCheckBox("Автоматически читать интервал из подписи фото (Tesseract)")
        self.ocr.setChecked(True)
        form.addRow("Файл Excel/CSV:", self.excel)
        form.addRow("Папка фотографий:", self.photos)
        form.addRow("OCR:", self.ocr)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.create_button = QPushButton("Сопоставить и показать маски")
        self.create_button.clicked.connect(self._create_project)
        self.recalculate_button = QPushButton("Пересчитать после исправлений")
        self.recalculate_button.clicked.connect(self._save_photo_map)
        row.addWidget(self.create_button)
        row.addWidget(self.recalculate_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.project_progress = QProgressBar()
        self.project_progress.setRange(0, 0)
        self.project_progress.setVisible(False)
        layout.addWidget(self.project_progress)
        self.photo_table = QTableWidget(0, 7)
        self.photo_table.setHorizontalHeaderLabels((
            "OK", "Файл", "Скважина", "Начало", "Конец", "Порядок колонок", "Источник",
        ))
        self.photo_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.photo_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.photo_table.setMaximumHeight(280)
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
        self.matching_preview_title.setStyleSheet("font-size: 14px; font-weight: 600;")
        right.addWidget(self.matching_preview_title)
        self.matching_preview = MaskPreviewLabel("Выберите фотографию в таблице.")
        self.matching_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.matching_preview.setWordWrap(True)
        self.matching_preview.setMinimumWidth(380)
        self.matching_preview.setStyleSheet("border: 3px solid #ff8c00; background: #202020; color: white;")
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
        self.review_table.setHorizontalHeaderLabels(("OK", "Фото", "Скважина", "От", "До", "Фация", "Краткое описание", "Excel", "ID"))
        self.review_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.review_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.review_table.itemSelectionChanged.connect(self._show_preview)
        left.addWidget(self.review_table, 1)
        layout.addLayout(left, 3)
        self.preview = MaskPreviewLabel("Выберите строку. Цветная область — автоматически построенная маска.")
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
        self.model_output = PathField(directory=True)
        self.architecture = QComboBox()
        self.architecture.addItem("YOLO11 nano segmentation — быстрее", "yolo11n-seg.yaml")
        self.architecture.addItem("YOLO11 small segmentation", "yolo11s-seg.yaml")
        self.architecture.addItem("YOLO11 medium segmentation — тяжелее", "yolo11m-seg.yaml")
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
        form.addRow("Новая папка результата:", self.model_output)
        form.addRow("Архитектура со случайными весами:", self.architecture)
        form.addRow("Эпохи (верхний предел):", self.epochs)
        form.addRow("Early stopping patience:", self.patience)
        form.addRow("Эпохи модели «Краткое описание»:", self.description_epochs)
        layout.addLayout(form)
        self.catalog_status = QLabel()
        self.catalog_status.setWordWrap(True)
        layout.addWidget(self.catalog_status)
        buttons = QHBoxLayout()
        self.dataset_button = QPushButton("Собрать датасет")
        self.dataset_button.clicked.connect(self._build_dataset)
        train = QPushButton("Обучить с нуля единый best.pt: слои + фации + описание")
        train.clicked.connect(self._start_training)
        buttons.addWidget(self.dataset_button)
        buttons.addWidget(train)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        layout.addWidget(self.train_log, 1)
        self.tabs.addTab(page, "3. Датасет и best.pt")
        self._update_catalog_status()

    def _build_analyze_tab(self) -> None:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.analysis_model = PathField(file_filter="Единая модель best.pt (*.pt)")
        self.analysis_photos = PathField(directory=True)
        self.analysis_output = PathField(save=True, file_filter="Excel (*.xlsx)")
        self.analysis_confidence = QSpinBox()
        self.analysis_confidence.setRange(1, 99)
        self.analysis_confidence.setValue(25)
        self.analysis_confidence.setSuffix(" %")
        form.addRow("Созданный best.pt:", self.analysis_model)
        form.addRow("Папка нового керна:", self.analysis_photos)
        form.addRow("Итоговый Excel из 22 столбцов:", self.analysis_output)
        form.addRow("Минимальная уверенность:", self.analysis_confidence)
        layout.addLayout(form)
        analyze = QPushButton("Распознать фации, сформировать столбец 22 и создать Excel")
        analyze.clicked.connect(self._start_analysis)
        layout.addWidget(analyze)
        self.analysis_log = QTextEdit()
        self.analysis_log.setReadOnly(True)
        layout.addWidget(self.analysis_log, 1)
        self.tabs.addTab(page, "4. Новый керн → Excel")

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
        except Exception as exc:
            return self._error(str(exc))
        command = [
            "create", "--excel", str(excel_path), "--photos", str(photos_path),
            "--project", str(project_dir),
        ]
        if self.ocr.isChecked():
            command.append("--ocr")
        self._start_project_process(command, project_dir, "create")

    def _load_photo_map(self) -> None:
        try:
            records = read_photo_map(self._project_dir() / "photo_map.csv")
        except Exception as exc:
            return self._error(str(exc))
        self.photo_table.blockSignals(True)
        self.photo_table.setUpdatesEnabled(False)
        self.photo_table.setRowCount(len(records))
        for row, record in enumerate(records):
            confirmed = QTableWidgetItem()
            confirmed.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
            confirmed.setCheckState(Qt.CheckState.Checked if record.mapping_confirmed else Qt.CheckState.Unchecked)
            self.photo_table.setItem(row, 0, confirmed)
            values = (
                str(record.path), record.well,
                "" if record.top is None else f"{record.top:g}",
                "" if record.base is None else f"{record.base:g}",
            )
            for column, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                if column == 1:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.photo_table.setItem(row, column, item)
            order = QComboBox(self.photo_table)
            order.addItem("Авто (Верх/Низ/цифры)", COLUMN_ORDER_AUTO)
            order.addItem("Слева → направо", COLUMN_ORDER_LEFT_TO_RIGHT)
            order.addItem("Справа → налево", COLUMN_ORDER_RIGHT_TO_LEFT)
            selected = order.findData(normalize_column_order(record.column_order))
            order.setCurrentIndex(max(0, selected))
            order.currentIndexChanged.connect(self._show_matching_preview)
            self.photo_table.setCellWidget(row, 5, order)
            source = QTableWidgetItem(record.source)
            source.setFlags(source.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.photo_table.setItem(row, 6, source)
        self.photo_table.setUpdatesEnabled(True)
        self.photo_table.blockSignals(False)
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
            order_widget = self.photo_table.cellWidget(row, 5)
            column_order = order_widget.currentData() if isinstance(order_widget, QComboBox) else COLUMN_ORDER_AUTO
            if confirmed and (not well or top is None or base is None or base <= top):
                return self._error(f"{path.name}: для подтверждения нужны скважина и корректный интервал.")
            records.append(PhotoRecord(
                path=path, well=well, top=top, base=base,
                source=self.photo_table.item(row, 6).text().strip() or "manual",
                mapping_confirmed=confirmed,
                column_order=column_order,
            ))
        try:
            project_dir = self._project_dir()
            write_photo_map(project_dir / "photo_map.csv", records)
        except Exception as exc:
            return self._error(str(exc))
        self._start_project_process(["refresh", "--project", str(project_dir)], project_dir, "refresh")

    def _start_project_process(self, command: list[str], project_dir: Path, action: str) -> None:
        if self.project_process and self.project_process.state() != QProcess.ProcessState.NotRunning:
            return self._error("Сопоставление этой скважины уже выполняется.")
        self.current_project = project_dir
        self._pending_project_action = action
        self.create_button.setEnabled(False)
        self.recalculate_button.setEnabled(False)
        self.project_progress.setVisible(True)
        self.project_log.setPlainText(
            "Обработка выполняется в отдельном процессе. Окно остаётся доступным; "
            "скорость зависит от количества и размера фотографий."
        )
        arguments = self._with_launcher(command)
        self.project_process = QProcess(self)
        self.project_process.setProgram(sys.executable)
        self.project_process.setArguments(arguments)
        self.project_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.project_process.readyReadStandardOutput.connect(self._read_project_output)
        self.project_process.finished.connect(self._project_finished)
        self.project_process.start()

    def _read_project_output(self) -> None:
        if self.project_process:
            text = bytes(self.project_process.readAllStandardOutput()).decode(errors="replace").strip()
            if text:
                self.project_log.append(text)

    def _project_finished(self, code: int, _status) -> None:
        self._read_project_output()
        self.create_button.setEnabled(True)
        self.recalculate_button.setEnabled(True)
        self.project_progress.setVisible(False)
        if code != 0:
            return self._error("Обработка не завершена. Подробности показаны в журнале.")
        try:
            project_dir = self._project_dir()
            report = json.loads((project_dir / "report.json").read_text(encoding="utf-8"))
        except Exception as exc:
            return self._error(str(exc))
        if self._pending_project_action == "create":
            self.matching_preview_paths.clear()
            self.matching_preview_details.clear()
            self.matching_preview_regions.clear()
            self.matching_column_orders.clear()
            self.matching_column_counts.clear()
            self.matching_column_depths.clear()
            try:
                register_project(project_dir)
            except OSError as exc:
                self.project_log.append(f"Не удалось обновить внутренний каталог: {exc}")
            self._ensure_training_output_paths()
        self._show_report(report)
        self._load_photo_map()
        self._load_review()
        self._update_catalog_status()

    def _show_report(self, report: dict) -> None:
        lines = [
            f"Служебная папка создана автоматически: {report['project_dir']}",
            f"Excel/CSV-файлов: {report.get('excel_files', 1)}", f"Строк Excel: {report['excel_rows']}", f"Фото: {report['photos']}",
            f"Подтверждены интервалы фото: {report['confirmed_photos']}",
            f"Нужно подтвердить интервалы: {report['unconfirmed_photos']}",
            f"Автоматически подтверждено OCR: {report.get('ocr_verified_photos', 0)}",
            f"Спроецировано масок: {report['annotations']}",
            f"Строк с ошибкой толщины фации: {report.get('invalid_thickness_rows', 0)}",
            f"Строк Excel с «Кратким описанием»: {report.get('excel_text_targets', 0)}",
            f"Масок с «Кратким описанием»: {report.get('text_targets', 0)}",
            f"Подтверждено масок: {report['approved_annotations']}",
        ]
        mappings = report.get("column_mappings", [])
        if mappings:
            lines.append("\nРаспознанные столбцы Excel (по названиям):")
            lines.extend(
                f"- {item.get('sheet', '')}: интервал фации по бурению "
                f"{item.get('facies_top') or '?'}–{item.get('facies_base') or '?'}, "
                f"толщина фации {item.get('facies_thickness') or '?'}, "
                f"«Краткое описание» {item.get('target_text') or '?'}"
                for item in mappings
            )
        if report.get("issues"):
            lines.append("\nПроверить:")
            lines.extend(f"- {item['source']}: {item['message']}" for item in report["issues"])
        lines.append("\nНеизвестные интервалы исправьте прямо в таблице выше, отметьте OK и нажмите «Пересчитать после исправлений».")
        lines.append(
            "Порядок колонок определяется автоматически по отметкам «Верх/Низ» или крайним цифрам. "
            "Если направление определено неверно, выберите его в таблице вручную и нажмите «Пересчитать»."
        )
        self.project_log.setPlainText("\n".join(lines))

    def _load_review(self) -> None:
        try:
            rows = load_annotations(self._project_dir())
        except Exception as exc:
            return self._error(str(exc))
        self.matching_preview_paths.clear()
        self.matching_preview_details.clear()
        self.matching_preview_regions.clear()
        self._load_detected_column_orders()
        self.review_table.blockSignals(True)
        self.review_table.setUpdatesEnabled(False)
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
            try:
                polygon = json.loads(row.get("polygon_json", "[]"))
            except (TypeError, ValueError):
                polygon = []
            if polygon:
                self.matching_preview_regions.setdefault(photo_key, []).append({
                    "polygon": polygon,
                    "image_width": as_float(row.get("image_width")) or 0,
                    "image_height": as_float(row.get("image_height")) or 0,
                    "depth_top": row.get("depth_top", ""),
                    "depth_base": row.get("depth_base", ""),
                    "label": row.get("label", ""),
                    "target_text": row.get("target_text", ""),
                })
        self.review_table.setUpdatesEnabled(True)
        self.review_table.blockSignals(False)
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
        self._update_catalog_status()

    def _show_preview(self) -> None:
        row = self.review_table.currentRow()
        if row < 0:
            self.preview.set_regions([])
            return
        preview = self.review_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        photo_item = self.review_table.item(row, 1)
        photo_key = self._photo_key(photo_item.text()) if photo_item is not None else ""
        self.preview.set_regions(self.matching_preview_regions.get(photo_key, []))
        self._set_preview_pixmap(self.preview, preview, "Предпросмотр не найден.")

    def _show_matching_preview(self) -> None:
        row = self.photo_table.currentRow()
        if row < 0 or self.photo_table.item(row, 1) is None:
            self.matching_preview_title.setText("После сопоставления здесь появится фото с найденными интервалами.")
            self.matching_preview.set_regions([])
            self.matching_preview.setText("Выберите фотографию в таблице.")
            return
        photo_path = self.photo_table.item(row, 1).text()
        photo_key = self._photo_key(photo_path)
        preview_path = self.matching_preview_paths.get(photo_key)
        details = self.matching_preview_details.get(photo_key, [])
        self.matching_preview.set_regions(self.matching_preview_regions.get(photo_key, []))
        order_widget = self.photo_table.cellWidget(row, 5)
        requested_order = order_widget.currentData() if isinstance(order_widget, QComboBox) else COLUMN_ORDER_AUTO
        effective_order = self.matching_column_orders.get(photo_key)
        column_count = self.matching_column_counts.get(photo_key)
        column_depths = self.matching_column_depths.get(photo_key, [])
        columns_text = f"Колонок керна найдено: {column_count}" if column_count else "Колонки керна ещё не определены"
        depth_text = f"\nИнтервалы колонок: {'; '.join(column_depths)}" if column_depths else ""
        if requested_order == COLUMN_ORDER_AUTO:
            order_text = (
                f"Порядок: {self._column_order_label(effective_order)} (определено автоматически)"
                if effective_order else "Порядок: авто, результат появится после пересчёта"
            )
        elif effective_order and effective_order != requested_order:
            order_text = (
                f"Выбрано: {self._column_order_label(requested_order)} — "
                "нажмите «Пересчитать», чтобы обновить маски"
            )
        else:
            order_text = f"Порядок: {self._column_order_label(requested_order)} (задан вручную)"
        if preview_path:
            visible_details = "\n".join(details[:4])
            suffix = f"\nЕщё интервалов: {len(details) - 4}" if len(details) > 4 else ""
            self.matching_preview_title.setText(
                f"{columns_text}{depth_text}\n{order_text}\nНайдено интервалов: {len(details)}\n"
                f"{visible_details}{suffix}\nНаведите курсор на маску — появится краткое описание."
            )
            self._set_preview_pixmap(self.matching_preview, preview_path, "Не удалось открыть фото с масками.")
        else:
            self.matching_preview_title.setText(
                f"{columns_text}{depth_text}\n{order_text}\nНа этом фото совпадающие интервалы пока не найдены."
            )
            self._set_preview_pixmap(self.matching_preview, photo_path, "Не удалось открыть исходную фотографию.")

    def _load_detected_column_orders(self) -> None:
        self.matching_column_orders.clear()
        self.matching_column_counts.clear()
        self.matching_column_depths.clear()
        try:
            data = json.loads((self._project_dir() / "detected_columns.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, RuntimeError):
            return
        for photo_path, value in data.items():
            if not isinstance(value, dict) or not value.get("order"):
                continue
            try:
                self.matching_column_orders[self._photo_key(photo_path)] = normalize_column_order(value["order"])
            except ValueError:
                continue
            boxes = value.get("boxes", [])
            if isinstance(boxes, list) and boxes:
                self.matching_column_counts[self._photo_key(photo_path)] = len(boxes)
            depth_ranges = value.get("depth_ranges", [])
            if isinstance(depth_ranges, list):
                labels = []
                for index, interval in enumerate(depth_ranges, start=1):
                    if not isinstance(interval, dict):
                        continue
                    top = as_float(interval.get("top"))
                    base = as_float(interval.get("base"))
                    if top is not None and base is not None:
                        labels.append(f"{index}) {top:g}–{base:g} м")
                if labels:
                    self.matching_column_depths[self._photo_key(photo_path)] = labels

    @staticmethod
    def _column_order_label(value: str | None) -> str:
        return {
            COLUMN_ORDER_LEFT_TO_RIGHT: "слева → направо",
            COLUMN_ORDER_RIGHT_TO_LEFT: "справа → налево",
            COLUMN_ORDER_AUTO: "авто",
        }.get(value, "не определён")

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
        if self.dataset_process and self.dataset_process.state() != QProcess.ProcessState.NotRunning:
            return self._error("Сборка датасета уже выполняется.")
        try:
            destination = self.dataset_output.value()
            if not self.dataset_output.edit.text().strip():
                raise ValueError("Укажите новую папку датасета.")
            if catalog_summary()["projects"] < 1:
                raise ValueError("Сначала обработайте хотя бы одну скважину.")
        except Exception as exc:
            return self._error(str(exc))
        self.dataset_process = QProcess(self)
        self.dataset_process.setProgram(sys.executable)
        self.dataset_process.setArguments(self._with_launcher([
            "dataset", "--catalog", str(default_catalog_path()), "--output", str(destination),
        ]))
        self.dataset_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.dataset_process.readyReadStandardOutput.connect(self._read_dataset_output)
        self.dataset_process.finished.connect(self._dataset_finished)
        self.dataset_button.setEnabled(False)
        self.train_log.setPlainText("Сборка датасета выполняется в отдельном процессе…")
        self.dataset_process.start()

    def _read_dataset_output(self) -> None:
        if self.dataset_process:
            text = bytes(self.dataset_process.readAllStandardOutput()).decode(errors="replace").strip()
            if text:
                self.train_log.append(text)

    def _dataset_finished(self, code: int, _status) -> None:
        self._read_dataset_output()
        self.dataset_button.setEnabled(True)
        self.train_log.append(f"\nСборка датасета завершена, код {code}.")

    def _start_training(self) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            return self._error("Обучение уже запущено.")
        command = [
            "train",
            "--dataset", str(self.dataset_output.value()),
            "--output", str(self.model_output.value()), "--epochs", str(self.epochs.value()),
            "--patience", str(self.patience.value()),
            "--description-epochs", str(self.description_epochs.value()),
            "--architecture", str(self.architecture.currentData()),
        ]
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(self._with_launcher(command))
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
        if code == 0:
            self.analysis_model.set_value(self.model_output.value() / "best.pt")

    def _start_analysis(self) -> None:
        if self.analysis_process and self.analysis_process.state() != QProcess.ProcessState.NotRunning:
            return self._error("Анализ уже запущен.")
        model = self.analysis_model.value()
        photos = self.analysis_photos.value()
        output = self.analysis_output.value()
        if not self.analysis_model.edit.text().strip() or not model.is_file():
            return self._error("Выберите существующий best.pt, созданный этой системой.")
        if not self.analysis_photos.edit.text().strip() or not photos.is_dir():
            return self._error("Выберите папку с фотографиями нового керна.")
        if not self.analysis_output.edit.text().strip() or output.suffix.lower() != ".xlsx":
            return self._error("Укажите новый итоговый файл с расширением .xlsx.")
        command = [
            "analyze", "--model", str(model), "--photos", str(photos),
            "--output-excel", str(output),
            "--confidence", f"{self.analysis_confidence.value() / 100:.2f}",
        ]
        self.analysis_process = QProcess(self)
        self.analysis_process.setProgram(sys.executable)
        self.analysis_process.setArguments(self._with_launcher(command))
        self.analysis_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.analysis_process.readyReadStandardOutput.connect(self._read_analysis_output)
        self.analysis_process.finished.connect(self._analysis_finished)
        self.analysis_log.clear()
        self.analysis_process.start()

    def _read_analysis_output(self) -> None:
        if self.analysis_process:
            self.analysis_log.append(bytes(self.analysis_process.readAllStandardOutput()).decode(errors="replace"))

    def _analysis_finished(self, code: int, _status) -> None:
        self.analysis_log.append(f"\nАнализ завершён, код {code}.")

    @staticmethod
    def _with_launcher(command: list[str]) -> list[str]:
        if getattr(sys, "frozen", False):
            return command
        launcher = Path(__file__).resolve().parents[2] / "run.py"
        return [str(launcher), *command]

    def _automatic_project_dir(self, excel_path: Path) -> Path:
        local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        safe_stem = re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", excel_path.stem).strip("_") or "project"
        unique_name = f"{safe_stem}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        return local_app_data / "ExcelPhotoModelStudio" / "projects" / unique_name

    def _ensure_training_output_paths(self) -> None:
        if self.dataset_output.edit.text().strip() and self.model_output.edit.text().strip():
            return
        local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        run = local_app_data / "ExcelPhotoModelStudio" / "training_runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        if not self.dataset_output.edit.text().strip():
            self.dataset_output.set_value(run / "dataset")
        if not self.model_output.edit.text().strip():
            self.model_output.set_value(run / "model")

    def _update_catalog_status(self) -> None:
        if not hasattr(self, "catalog_status"):
            return
        summary = catalog_summary()
        self.catalog_status.setText(
            "Внутренняя обучающая база: "
            f"скважин — {summary['projects']}, фото — {summary['photos']}, "
            f"масок — {summary['annotations']}, подтверждено — {summary['approved_annotations']}. "
            "Каждая скважина выбирается отдельно; общая папка не требуется."
        )

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
