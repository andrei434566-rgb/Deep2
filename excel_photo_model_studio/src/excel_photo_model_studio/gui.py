from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QPointF, Qt, QProcess, QThread, Signal
from PySide6.QtGui import QColor, QImageReader, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QToolTip, QVBoxLayout, QWidget,
)

from .catalog import catalog_summary, default_catalog_path, register_project
from .depth import format_depth
from .matching import match_photos, read_photo_map, suggest_missing_intervals, write_photo_map
from .models import (
    COLUMN_ORDER_AUTO, COLUMN_ORDER_LEFT_TO_RIGHT, COLUMN_ORDER_RIGHT_TO_LEFT,
    PhotoRecord, normalize_column_order,
)
from .photos import discover_photos, enrich_core_column_depths
from .project import load_annotations, set_annotation_approvals
from .tabular import as_float, read_many_tables, read_workbook_sheets
from .vision import detect_core_columns_from_path, project_matches


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
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        image_width = max((float(item.get("image_width", 0) or 0) for item in self._regions), default=0)
        image_height = max((float(item.get("image_height", 0) or 0) for item in self._regions), default=0)
        if image_width <= 0 or image_height <= 0:
            return
        x_offset = (self.width() - pixmap.width()) / 2.0
        y_offset = (self.height() - pixmap.height()) / 2.0
        x_scale = pixmap.width() / image_width
        y_scale = pixmap.height() / image_height
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Facies overlays are painted first so the cyan core-column contours
        # remain visible on top of them during the verification workflow.
        for region in self._regions:
            if region.get("kind") != "facies":
                continue
            try:
                polygon = QPolygonF([
                    QPointF(x_offset + float(x) * x_scale, y_offset + float(y) * y_scale)
                    for x, y in region["polygon"]
                ])
            except (KeyError, TypeError, ValueError):
                continue
            fill = QColor(str(region.get("color", "#ff9f1c")))
            fill.setAlpha(115)
            pen = QPen(QColor(str(region.get("color", "#ff9f1c"))), 2.0)
            painter.setPen(pen)
            painter.setBrush(fill)
            painter.drawPolygon(polygon)
        for region in self._regions:
            if region.get("kind") != "core_column":
                continue
            try:
                polygon = QPolygonF([
                    QPointF(x_offset + float(x) * x_scale, y_offset + float(y) * y_scale)
                    for x, y in region["polygon"]
                ])
            except (KeyError, TypeError, ValueError):
                continue
            pen = QPen(QColor("#00e5ff"), 2.0, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 229, 255, 28))
            painter.drawPolygon(polygon)
            painter.drawText(polygon.boundingRect().topLeft() + QPointF(3, 16), str(region.get("label", "Керн")))
        painter.end()

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
        if region.get("kind") == "core_column":
            return f"Распознанная колонка керна: {region.get('label', '')}".strip()
        description = str(region.get("target_text", "")).strip() or "Описание отсутствует"
        label = str(region.get("label", "")).strip()
        try:
            interval = (
                f"Участок на фото: {format_depth(region.get('depth_top'))}–"
                f"{format_depth(region.get('depth_base'))} м"
            )
        except ValueError:
            interval = f"Участок на фото: {region.get('depth_top', '')}–{region.get('depth_base', '')} м"
        parts = [interval]
        if region.get("facies_top") not in {None, ""} and region.get("facies_base") not in {None, ""}:
            try:
                parts.append(
                    f"Полный интервал фации: {format_depth(region['facies_top'])}–"
                    f"{format_depth(region['facies_base'])} м"
                )
            except ValueError:
                pass
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


class _DiagnosticWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self) -> None:
        try:
            self.completed.emit(self.function(self.progress.emit))
        except Exception as exc:
            self.failed.emit(str(exc))


class StepVerificationDialog(QDialog):
    """Interactive checkpoints over the exact Excel/photo/mask pipeline."""

    def __init__(self, excel_path: Path, photos_dir: Path, *, use_ocr: bool, parent=None):
        super().__init__(parent)
        self.excel_path = Path(excel_path)
        self.photos_dir = Path(photos_dir)
        self.use_ocr = bool(use_ocr)
        self.setWindowTitle("Пошаговая сверка Excel → фото → маски")
        self.resize(1180, 820)
        self._worker: _DiagnosticWorker | None = None
        self._stage = 0
        self._current_photo = 0
        self.rows = []
        self.mappings = []
        self.issues = []
        self.excel_files = []
        self.raw_sheets: dict[str, list[list]] = {}
        self.photos: list[PhotoRecord] = []
        self.columns: dict[Path, list[tuple[int, int, int, int]]] = {}
        self.column_errors: dict[Path, str] = {}
        self.column_confirmed: set[Path] = set()
        self.matches = []
        self.unresolved = []
        self.annotations = []
        self.projected_columns: dict[Path, list[tuple[int, int, int, int]]] = {}
        self.orders: dict[Path, str] = {}
        self.mask_confirmed: set[Path] = set()
        self.audit: list[str] = []
        self._failure = ""

        root = QVBoxLayout(self)
        self.stage_title = QLabel()
        self.stage_title.setStyleSheet("font-size: 18px; font-weight: 650;")
        root.addWidget(self.stage_title)
        self.status = QLabel("Подготовка…")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.pages = QStackedWidget(self)
        root.addWidget(self.pages, 1)
        self._build_excel_page()
        self._build_photo_page()
        self._build_interval_page()
        self._build_matching_page()
        self._build_mask_page()
        self._build_summary_page()

        buttons = QHBoxLayout()
        self.back_button = QPushButton("← Назад")
        self.back_button.clicked.connect(self._go_back)
        self.action_button = QPushButton()
        self.action_button.clicked.connect(self._go_forward)
        self.issue_button = QPushButton("Зафиксировать ошибку на этом этапе")
        self.issue_button.clicked.connect(self._report_issue)
        self.cancel_button = QPushButton("Закрыть")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.back_button)
        buttons.addWidget(self.issue_button)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.action_button)
        root.addLayout(buttons)
        self._set_stage(0)
        self._run_async("Читаю книгу и определяю столбцы…", self._read_excel, self._excel_loaded)

    def _build_excel_page(self) -> None:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        self.mapping_summary = QLabel()
        self.mapping_summary.setWordWrap(True)
        layout.addWidget(self.mapping_summary)
        self.excel_views = QTabWidget(page)
        source_page = QWidget(self.excel_views)
        source_layout = QVBoxLayout(source_page)
        sheet_row = QHBoxLayout()
        sheet_row.addWidget(QLabel("Исходный лист:"))
        self.sheet_selector = QComboBox(source_page)
        self.sheet_selector.currentIndexChanged.connect(self._show_raw_sheet)
        sheet_row.addWidget(self.sheet_selector, 1)
        self.raw_sheet_info = QLabel("Цветом выделены исходные столбцы, выбранные для сопоставления.")
        self.raw_sheet_info.setWordWrap(True)
        source_layout.addLayout(sheet_row)
        source_layout.addWidget(self.raw_sheet_info)
        self.raw_sheet_table = QTableWidget(0, 0, source_page)
        self.raw_sheet_table.setAlternatingRowColors(True)
        self.raw_sheet_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)
        source_layout.addWidget(self.raw_sheet_table, 1)
        self.excel_views.addTab(source_page, "Исходный Excel — выделенные колонки")
        self.excel_table = QTableWidget(0, 9, page)
        self.excel_table.setHorizontalHeaderLabels((
            "Файл / лист / строка", "Скважина", "От, м", "До, м", "Толщина, м",
            "Фация", "Краткое описание", "Источник интервала", "Интервал керна",
        ))
        self.excel_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.excel_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.excel_table.setAlternatingRowColors(True)
        self.excel_views.addTab(self.excel_table, "Строки после чтения")
        layout.addWidget(self.excel_views, 1)
        self.pages.addWidget(page)

    def _build_photo_page(self) -> None:
        page = QWidget(self)
        layout = QHBoxLayout(page)
        left = QVBoxLayout()
        self.column_photo_info = QLabel()
        self.column_photo_info.setWordWrap(True)
        self.column_photo_info.setMinimumWidth(300)
        left.addWidget(self.column_photo_info)
        nav = QHBoxLayout()
        self.column_prev = QPushButton("← Фото")
        self.column_prev.clicked.connect(lambda: self._navigate_photo(-1))
        self.column_next = QPushButton("Фото →")
        self.column_next.clicked.connect(lambda: self._navigate_photo(1))
        nav.addWidget(self.column_prev)
        nav.addWidget(self.column_next)
        left.addLayout(nav)
        self.column_confirm = QPushButton("Столбики на этом фото распознаны верно")
        self.column_confirm.clicked.connect(self._confirm_columns)
        left.addWidget(self.column_confirm)
        left.addStretch(1)
        layout.addLayout(left, 1)
        self.column_preview = self._new_preview()
        layout.addWidget(self.column_preview, 2)
        self.pages.addWidget(page)

    def _build_interval_page(self) -> None:
        page = QWidget(self)
        layout = QHBoxLayout(page)
        self.interval_table = QTableWidget(0, 6, page)
        self.interval_table.setHorizontalHeaderLabels(("OK", "Фото", "Скважина", "Начало, м", "Конец, м", "Источник"))
        self.interval_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.interval_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.interval_table.itemSelectionChanged.connect(self._show_interval_photo)
        layout.addWidget(self.interval_table, 3)
        right = QVBoxLayout()
        self.interval_info = QLabel("Проверьте и при необходимости исправьте скважину и границы каждого фото.")
        self.interval_info.setWordWrap(True)
        right.addWidget(self.interval_info)
        self.interval_preview = self._new_preview()
        right.addWidget(self.interval_preview, 1)
        layout.addLayout(right, 2)
        self.pages.addWidget(page)

    def _build_matching_page(self) -> None:
        page = QWidget(self)
        layout = QHBoxLayout(page)
        self.match_table = QTableWidget(0, 6, page)
        self.match_table.setHorizontalHeaderLabels(("OK", "Фото", "Интервал фото", "Система", "Фаций найдено", "Совпавшие интервалы Excel"))
        self.match_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.match_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.match_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.match_table.itemSelectionChanged.connect(self._show_matching_photo)
        layout.addWidget(self.match_table, 3)
        right = QVBoxLayout()
        self.match_info = QLabel()
        self.match_info.setWordWrap(True)
        right.addWidget(self.match_info)
        self.match_preview = self._new_preview()
        right.addWidget(self.match_preview, 1)
        layout.addLayout(right, 2)
        self.pages.addWidget(page)

    def _build_mask_page(self) -> None:
        page = QWidget(self)
        layout = QHBoxLayout(page)
        left = QVBoxLayout()
        self.mask_info = QLabel()
        self.mask_info.setWordWrap(True)
        left.addWidget(self.mask_info)
        nav = QHBoxLayout()
        self.mask_prev = QPushButton("← Фото")
        self.mask_prev.clicked.connect(lambda: self._navigate_photo(-1))
        self.mask_next = QPushButton("Фото →")
        self.mask_next.clicked.connect(lambda: self._navigate_photo(1))
        nav.addWidget(self.mask_prev)
        nav.addWidget(self.mask_next)
        left.addLayout(nav)
        self.mask_confirm = QPushButton("Маски на этом фото верны")
        self.mask_confirm.clicked.connect(self._confirm_masks)
        left.addWidget(self.mask_confirm)
        left.addStretch(1)
        layout.addLayout(left, 1)
        self.mask_preview = self._new_preview()
        layout.addWidget(self.mask_preview, 2)
        self.pages.addWidget(page)

    def _build_summary_page(self) -> None:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        self.summary_text = QTextEdit(page)
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text, 1)
        self.pages.addWidget(page)

    @staticmethod
    def _new_preview() -> MaskPreviewLabel:
        preview = MaskPreviewLabel("Здесь появится выбранное фото.")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setWordWrap(True)
        preview.setMinimumSize(480, 540)
        preview.setStyleSheet("border: 2px solid #ff8c00; background: #202020; color: white;")
        return preview

    def _set_stage(self, stage: int) -> None:
        self._stage = stage
        titles = {
            0: "Этап 1 из 5 — чтение Excel",
            1: "Этап 2 из 5 — поиск столбиков керна на каждом фото",
            2: "Этап 3 из 5 — интервалы фото и глубины",
            3: "Этап 4 из 5 — сопоставление интервалов с фациями Excel",
            4: "Этап 5 из 5 — проекция и проверка масок",
            5: "Итог пошаговой сверки",
        }
        self.stage_title.setText(titles[stage])
        self.pages.setCurrentIndex(stage)
        self.back_button.setEnabled(stage > 0 and stage < 5 and not self._busy())
        self.issue_button.setVisible(stage < 5)
        self.action_button.setVisible(stage < 5 or not self._failure)
        self.cancel_button.setText("Закрыть")
        labels = {
            0: "Проверьте распознанные поля и строки Excel, затем подтвердите этот этап.",
            1: "Подтверждается каждое фото отдельно. Бирюзовый контур — результат распознавания колонок.",
            2: "Предложенные глубины можно исправить. Используйте точку или запятую как десятичный знак.",
            3: "Проверьте, какие строки фаций Excel сопоставились с интервалом каждого фото.",
            4: "Проверьте геометрию масок на фото; наведите курсор на маску, чтобы прочитать описание.",
            5: "Сверка завершена. При ошибке основной расчёт не запускается.",
        }
        if not self._busy():
            self.status.setText(labels[stage])
        action_labels = {
            0: "Excel прочитан верно — продолжить",
            1: "Колонки подтверждены — перейти к глубинам",
            2: "Подтвердить интервалы и сверить фации",
            3: "Совпадения верны — построить маски",
            4: "Маски проверены — завершить сверку",
            5: "Запустить полный расчёт проекта",
        }
        self.action_button.setText(action_labels[stage])
        if stage == 5 and self._failure:
            self.action_button.setVisible(False)
        if stage == 1:
            self._render_column_photo()
        elif stage == 2:
            self.interval_table.resizeColumnsToContents()
            self._show_interval_photo()
        elif stage == 3:
            self.match_table.resizeColumnsToContents()
            self._show_matching_photo()
        elif stage == 4:
            self._render_mask_photo()
        elif stage == 5:
            self._render_summary()

    def _busy(self) -> bool:
        return self._worker is not None

    def _run_async(self, message: str, function, on_complete) -> None:
        if self._worker is not None:
            return
        self.status.setText(message)
        self.action_button.setEnabled(False)
        self.back_button.setEnabled(False)
        self.issue_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        worker = _DiagnosticWorker(function, self)
        self._worker = worker
        worker.progress.connect(self.status.setText)
        worker.completed.connect(lambda result: self._async_completed(on_complete, result))
        worker.failed.connect(self._async_failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _async_completed(self, callback, result) -> None:
        self._worker = None
        self.action_button.setEnabled(True)
        self.issue_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        callback(result)
        self._set_stage(self._stage)

    def _async_failed(self, message: str) -> None:
        self._worker = None
        self.action_button.setEnabled(True)
        self.back_button.setEnabled(0 < self._stage < 5)
        self.issue_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.status.setText(f"Этап не выполнен: {message}")
        QMessageBox.critical(self, "Ошибка пошаговой сверки", message)

    def _read_excel(self, progress):
        progress("Читаю Excel штатным парсером приложения…")
        parsed = read_many_tables(self.excel_path)
        raw_sheets = read_workbook_sheets(self.excel_path)
        return (*parsed, raw_sheets)

    def _excel_loaded(self, result) -> None:
        self.rows, self.mappings, self.issues, self.excel_files, raw_sheets = result
        self.raw_sheets = {name: cells for name, cells in raw_sheets}
        fields = (
            ("well", "Скважина"), ("interval", "Интервал фации"),
            ("top", "Верх фации"), ("base", "Низ фации"),
            ("facies_thickness", "Толщина фации"), ("core_top", "Верх интервала керна"),
            ("core_base", "Низ интервала керна"), ("class_code", "Код фации"),
            ("class_index", "Индекс фации"), ("label", "Название фации"),
            ("description", "Описание"), ("target_text", "Краткое описание"),
            ("gis_interval", "Интервал по ГИС"), ("gis_top", "Верх по ГИС"),
            ("gis_base", "Низ по ГИС"),
        )
        mapping_lines = [f"Строки с фациями: {len(self.rows)}. Листы Excel: {len(self.mappings)}."]
        for mapping in self.mappings:
            source = Path(mapping.source_file).name if mapping.source_file else self.excel_path.name
            resolved = [f"{label}: {self._excel_column(getattr(mapping, role))}" for role, label in fields if getattr(mapping, role)]
            mapping_lines.append(
                f"{source} / лист «{mapping.sheet}», строка заголовков {mapping.header_row + 1}: "
                + "; ".join(resolved)
            )
        if self.issues:
            mapping_lines.append(f"Замечаний парсера: {len(self.issues)}. Первое: {self.issues[0].message}")
        self.mapping_summary.setText("\n".join(mapping_lines))
        self.sheet_selector.blockSignals(True)
        self.sheet_selector.clear()
        for index, mapping in enumerate(self.mappings):
            self.sheet_selector.addItem(f"{Path(mapping.source_file).name if mapping.source_file else self.excel_path.name} — {mapping.sheet}", index)
        self.sheet_selector.blockSignals(False)
        self._show_raw_sheet()
        self.excel_table.setRowCount(len(self.rows))
        for index, row in enumerate(self.rows):
            values = (
                f"{Path(row.source_file).name if row.source_file else row.sheet} / {row.sheet}!{row.row}",
                row.well, format_depth(row.top), format_depth(row.base),
                "" if row.thickness is None else format_depth(row.thickness), row.label,
                row.target_text or row.description, row.metadata.get("interval_source", "drilling"),
                "" if row.core_top is None or row.core_base is None else f"{format_depth(row.core_top)}–{format_depth(row.core_base)}",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.excel_table.setItem(index, column, item)
        self.excel_table.resizeColumnsToContents()
        self.audit.append(f"Excel: прочитано {len(self.rows)} строк; распознано отображений листов: {len(self.mappings)}.")

    def _show_raw_sheet(self, *_args) -> None:
        mapping_index = self.sheet_selector.currentData()
        if mapping_index is None or not self.mappings:
            self.raw_sheet_table.setRowCount(0)
            self.raw_sheet_table.setColumnCount(0)
            self.raw_sheet_info.setText("Не найдено листов Excel с распознанной схемой.")
            return
        mapping = self.mappings[int(mapping_index)]
        source_rows = self.raw_sheets.get(mapping.sheet, [])
        if not source_rows:
            self.raw_sheet_table.setRowCount(0)
            self.raw_sheet_table.setColumnCount(0)
            self.raw_sheet_info.setText(f"Исходная сетка листа «{mapping.sheet}» недоступна для показа.")
            return

        role_specs = (
            ("well", "Скважина", "#d9eaf7"),
            ("interval", "Интервал фации", "#e2e2e2"),
            ("top", "Кровля фации", "#bee3f8"),
            ("base", "Подошва фации", "#a8dadc"),
            ("facies_thickness", "Толщина фации", "#ffe0b2"),
            ("core_top", "Кровля интервала керна", "#d9ed92"),
            ("core_base", "Подошва интервала керна", "#c7e9b0"),
            ("class_code", "Код фации", "#e9d8fd"),
            ("class_index", "Индекс фации", "#d6bcfa"),
            ("label", "Название фации", "#e4c1f9"),
            ("description", "Описание", "#fff1a8"),
            ("target_text", "Краткое описание", "#ffd166"),
            ("gis_interval", "Интервал по ГИС", "#f7cad0"),
            ("gis_top", "Кровля по ГИС", "#f4acb7"),
            ("gis_base", "Подошва по ГИС", "#ee8c99"),
        )
        selected_columns: dict[int, tuple[str, QColor]] = {}
        for role, label, color in role_specs:
            column = getattr(mapping, role)
            if column and column not in selected_columns:
                selected_columns[column] = (label, QColor(color))
        column_count = max(
            max((len(row) for row in source_rows), default=0),
            max(selected_columns, default=0),
        )
        header_row = max(0, min(int(mapping.header_row), len(source_rows) - 1))
        first_row = max(0, header_row - 3)
        last_row = min(len(source_rows), max(header_row + 51, first_row + 60))
        shown_rows = source_rows[first_row:last_row]
        header_values = source_rows[header_row]
        column_headers = []
        header_roles = []
        for column_index in range(1, column_count + 1):
            header_text = str(header_values[column_index - 1] or "").strip() if column_index <= len(header_values) else ""
            role_info = selected_columns.get(column_index)
            role_text = f"\n◆ {role_info[0]}" if role_info else ""
            column_headers.append(f"{self._excel_column(column_index)}\n{header_text}{role_text}")
            header_roles.append(role_info)

        table = self.raw_sheet_table
        table.clear()
        table.setColumnCount(column_count)
        table.setRowCount(len(shown_rows))
        table.setHorizontalHeaderLabels(column_headers)
        table.setVerticalHeaderLabels([str(first_row + offset + 1) for offset in range(len(shown_rows))])
        table.verticalHeader().setDefaultSectionSize(27)
        for column_index, role_info in enumerate(header_roles):
            table.setColumnWidth(column_index, 112)
            if role_info:
                header_item = table.horizontalHeaderItem(column_index)
                header_item.setBackground(role_info[1].darker(112))
                header_item.setToolTip(f"Колонка Excel {self._excel_column(column_index + 1)} выбрана как: {role_info[0]}")
                table.setColumnWidth(column_index, 165 if role_info[0] in {"Краткое описание", "Описание", "Название фации"} else 132)
        for row_index, source_row in enumerate(shown_rows):
            for column_index in range(column_count):
                value = source_row[column_index] if column_index < len(source_row) else ""
                item = QTableWidgetItem("" if value is None else str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                role_info = selected_columns.get(column_index + 1)
                if role_info:
                    item.setBackground(role_info[1])
                    item.setToolTip(f"Выбранный столбец: {role_info[0]}")
                table.setItem(row_index, column_index, item)
        selected_labels = ", ".join(
            f"{self._excel_column(column)} — {label}" for column, (label, _color) in selected_columns.items()
        )
        self.raw_sheet_info.setText(
            f"Лист «{mapping.sheet}», строки {first_row + 1}–{last_row} из {len(source_rows)}. "
            f"Подсвечены выбранные столбцы: {selected_labels or 'нет'}.")
        if selected_columns:
            first_column = min(selected_columns)
            anchor_row = max(0, min(header_row - first_row, len(shown_rows) - 1))
            table.scrollToItem(table.item(anchor_row, first_column - 1), QTableWidget.ScrollHint.PositionAtCenter)

    @staticmethod
    def _excel_column(index: int | None) -> str:
        if not index:
            return "—"
        result = ""
        number = int(index)
        while number:
            number, remainder = divmod(number - 1, 26)
            result = chr(65 + remainder) + result
        return result

    def _discover_and_detect(self, progress):
        expected = sorted({
            (float(row.core_top), float(row.core_base)) for row in self.rows
            if row.core_top is not None and row.core_base is not None and row.core_base > row.core_top
        })
        progress("Ищу все изображения в выбранной папке…")
        photos = discover_photos(self.photos_dir, use_ocr=self.use_ocr, expected_intervals=expected)
        if not photos:
            raise ValueError("В папке не найдены поддерживаемые изображения.")
        if self.use_ocr:
            progress("Читаю OCR-глубины отдельных колонок керна…")
            photos = enrich_core_column_depths(photos, expected)
        columns = {}
        errors = {}
        for index, photo in enumerate(photos, start=1):
            progress(f"Распознаю столбики керна: фото {index} из {len(photos)} — {photo.path.name}")
            try:
                columns[photo.path] = detect_core_columns_from_path(photo.path)
            except (OSError, ValueError) as exc:
                columns[photo.path] = []
                errors[photo.path] = str(exc)
        return photos, columns, errors

    def _columns_loaded(self, result) -> None:
        self.photos, self.columns, self.column_errors = result
        self.column_confirmed.clear()
        self._current_photo = 0
        self.audit.append(f"Поиск фото: найдено изображений {len(self.photos)}; детекция керна запущена для каждого.")

    def _fill_image(self, preview: MaskPreviewLabel, path: Path, regions: list[dict]) -> None:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        source_size = reader.size()
        image_width, image_height = source_size.width(), source_size.height()
        if image_width <= 0 or image_height <= 0:
            preview.setText("Не удалось прочитать размеры фото.")
            preview.set_regions([])
            return
        max_width = max(200, preview.width() - 18)
        max_height = max(200, preview.height() - 18)
        scale = min(max_width / image_width, max_height / image_height, 1.0)
        from PySide6.QtCore import QSize
        reader.setScaledSize(QSize(max(1, round(image_width * scale)), max(1, round(image_height * scale))))
        image = reader.read()
        if image.isNull():
            preview.setText("Не удалось открыть фото.")
            preview.set_regions([])
            return
        preview.setPixmap(QPixmap.fromImage(image))
        normalized = []
        for region in regions:
            normalized.append({**region, "image_width": image_width, "image_height": image_height})
        preview.set_regions(normalized)

    def _column_regions(self, path: Path) -> list[dict]:
        return [{
            "kind": "core_column", "label": f"Керн {index}",
            "polygon": ((left, top), (right, top), (right, bottom), (left, bottom)),
        } for index, (left, top, right, bottom) in enumerate(self.columns.get(path, []), start=1)]

    def _render_column_photo(self) -> None:
        if not self.photos:
            return
        self._current_photo %= len(self.photos)
        photo = self.photos[self._current_photo]
        boxes = self.columns.get(photo.path, [])
        confirmation = "Подтверждено" if photo.path in self.column_confirmed else "Ожидает проверки"
        failure = self.column_errors.get(photo.path, "")
        self.column_photo_info.setText(
            f"Фото {self._current_photo + 1} из {len(self.photos)}\n{photo.path.name}\n\n"
            f"Найдено физических столбиков: {len(boxes)}\nСтатус: {confirmation}"
            + (f"\nОшибка чтения: {failure}" if failure else "")
            + "\n\nПроверьте, что выделен каждый столбик керна, включая половинные и одиночные. "
            "Линейка и подписи не должны быть выделены как керн."
        )
        self.column_confirm.setText(
            "Отменить подтверждение этого фото" if photo.path in self.column_confirmed
            else "Столбики на этом фото распознаны верно"
        )
        self.column_prev.setEnabled(self._current_photo > 0)
        self.column_next.setEnabled(self._current_photo < len(self.photos) - 1)
        self._fill_image(self.column_preview, photo.path, self._column_regions(photo.path))

    def _confirm_columns(self) -> None:
        if not self.photos:
            return
        photo = self.photos[self._current_photo]
        if not self.columns.get(photo.path):
            return QMessageBox.warning(self, "Нет столбиков", "На этом фото не найдено ни одного столбика керна. Зафиксируйте ошибку на этапе.")
        if photo.path in self.column_confirmed:
            self.column_confirmed.remove(photo.path)
        else:
            self.column_confirmed.add(photo.path)
            self.audit.append(f"Колонки: подтверждено фото {photo.path.name} — найдено {len(self.columns[photo.path])} столбиков.")
        self._render_column_photo()
        approved = len(self.column_confirmed)
        self.status.setText(f"Подтверждено столбиков на фото: {approved} из {len(self.photos)}.")

    def _navigate_photo(self, delta: int) -> None:
        if not self.photos:
            return
        self._current_photo = max(0, min(len(self.photos) - 1, self._current_photo + delta))
        if self._stage == 1:
            self._render_column_photo()
        elif self._stage == 4:
            self._render_mask_photo()

    def _prepare_intervals(self, progress):
        progress("Распределяю глубины по всей последовательности фото…")
        return suggest_missing_intervals(self.photos, self.rows)

    def _intervals_loaded(self, photos: list[PhotoRecord]) -> None:
        self.photos = photos
        self.interval_table.setRowCount(len(photos))
        for index, photo in enumerate(photos):
            ok = QTableWidgetItem()
            ok.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
            ok.setCheckState(Qt.CheckState.Unchecked)
            self.interval_table.setItem(index, 0, ok)
            path_item = QTableWidgetItem(str(photo.path))
            path_item.setFlags(path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            path_item.setData(Qt.ItemDataRole.UserRole, photo)
            self.interval_table.setItem(index, 1, path_item)
            self.interval_table.setItem(index, 2, QTableWidgetItem(photo.well))
            self.interval_table.setItem(index, 3, QTableWidgetItem("" if photo.top is None else format_depth(photo.top)))
            self.interval_table.setItem(index, 4, QTableWidgetItem("" if photo.base is None else format_depth(photo.base)))
            source = QTableWidgetItem(photo.source)
            source.setFlags(source.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.interval_table.setItem(index, 5, source)
        if photos:
            self.interval_table.selectRow(0)
        self.audit.append("Интервалы: подготовлены предложения глубин для всех найденных фото.")

    def _show_interval_photo(self) -> None:
        row = self.interval_table.currentRow()
        if row < 0 or row >= len(self.photos):
            return
        photo = self.photos[row]
        top_item = self.interval_table.item(row, 3)
        base_item = self.interval_table.item(row, 4)
        self.interval_info.setText(
            f"{photo.path.name}\nПредложенный источник: {photo.source}. "
            f"Глубина фото: {top_item.text() if top_item else ''}–{base_item.text() if base_item else ''} м.\n"
            "При исправлении укажите весь интервал этого фото, а не интервал отдельной фации."
        )
        self._fill_image(self.interval_preview, photo.path, self._column_regions(photo.path))

    def _confirm_interval_rows(self) -> list[PhotoRecord]:
        result = []
        for index, original in enumerate(self.photos):
            if self.interval_table.item(index, 0).checkState() != Qt.CheckState.Checked:
                raise ValueError(f"Не подтверждён интервал фото: {original.path.name}.")
            well = self.interval_table.item(index, 2).text().strip()
            top = as_float(self.interval_table.item(index, 3).text())
            base = as_float(self.interval_table.item(index, 4).text())
            if not well or top is None or base is None or base <= top:
                raise ValueError(f"{original.path.name}: задайте скважину и корректные глубины от/до.")
            edited = top != original.top or base != original.base or well != original.well
            result.append(replace(
                original, well=well, top=top, base=base,
                source="manual" if edited else original.source,
                mapping_confirmed=True,
                depth_basis="unknown" if edited else original.depth_basis,
            ))
        return result

    def _match_confirmed_intervals(self, progress):
        progress("Сопоставляю подтверждённые интервалы с фациями Excel…")
        return match_photos(self._confirmed_photos, self.rows)

    def _matches_loaded(self, result) -> None:
        self.matches, self.unresolved = result
        selected_photos = {match.photo.path: match.photo for match in self.matches}
        self.photos = [selected_photos.get(photo.path, photo) for photo in self.photos]
        self._confirmed_photos = self.photos
        by_photo: dict[Path, list] = {}
        for match in self.matches:
            by_photo.setdefault(match.photo.path, []).append(match)
        self.match_table.setRowCount(len(self.photos))
        for index, photo in enumerate(self.photos):
            matches = by_photo.get(photo.path, [])
            ok = QTableWidgetItem()
            ok.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
            ok.setCheckState(Qt.CheckState.Unchecked)
            self.match_table.setItem(index, 0, ok)
            path_item = QTableWidgetItem(photo.path.name)
            path_item.setData(Qt.ItemDataRole.UserRole, photo)
            path_item.setFlags(path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.match_table.setItem(index, 1, path_item)
            interval_item = QTableWidgetItem(f"{format_depth(photo.top)}–{format_depth(photo.base)}")
            interval_item.setFlags(interval_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.match_table.setItem(index, 2, interval_item)
            basis = next((match.photo.depth_basis for match in matches), photo.depth_basis)
            basis_item = QTableWidgetItem(basis)
            basis_item.setFlags(basis_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.match_table.setItem(index, 3, basis_item)
            unique = {(match.description.source_id, match.description.label) for match in matches}
            count_item = QTableWidgetItem(str(len(unique)))
            count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.match_table.setItem(index, 4, count_item)
            ranges = [
                f"{match.description.label}: {format_depth(match.overlap_top)}–{format_depth(match.overlap_base)}"
                for match in matches
            ]
            details = [
                f"{match.description.label} [{format_depth(match.overlap_top)}–{format_depth(match.overlap_base)} м]: "
                f"{match.description.target_text or match.description.description or 'описание отсутствует'} "
                f"(Excel: {Path(match.description.source_file).name or match.description.sheet}!{match.description.row})"
                for match in matches
            ]
            match_item = QTableWidgetItem("; ".join(ranges) if ranges else "НЕТ СОВПАДЕНИЙ")
            match_item.setFlags(match_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            match_item.setToolTip("\n".join(details) if details else "Для этого фото пересечений с интервалами фаций нет.")
            self.match_table.setItem(index, 5, match_item)
        if self.photos:
            self.match_table.selectRow(0)
        self.audit.append(
            f"Сопоставление: фации Excel найдены для {len(by_photo)} из {len(self.photos)} фото; "
            f"строк совпадений: {len(self.matches)}."
        )

    def _show_matching_photo(self) -> None:
        row = self.match_table.currentRow()
        if row < 0 or row >= len(self.photos):
            return
        photo = self.photos[row]
        matches = [item for item in self.matches if item.photo.path == photo.path]
        labels = list(dict.fromkeys(item.description.label for item in matches))
        self.match_info.setText(
            f"{photo.path.name}\nИнтервал: {format_depth(photo.top)}–{format_depth(photo.base)} м; "
            f"система: {matches[0].photo.depth_basis if matches else photo.depth_basis}.\n"
            f"Фаций сопоставлено: {len(labels)}. "
            + (", ".join(labels) if labels else "Ни одна строка Excel не пересеклась с интервалом фото.")
        )
        self._fill_image(self.match_preview, photo.path, self._column_regions(photo.path))

    def _project_confirmed_matches(self, progress):
        progress("Проецирую интервалы фаций на физические столбики керна…")
        annotations, columns, orders = project_matches(self.matches)
        return annotations, columns, orders

    def _projection_loaded(self, result) -> None:
        self.annotations, self.projected_columns, self.orders = result
        self.mask_confirmed.clear()
        self._current_photo = 0
        self.audit.append(f"Проекция: создано участков-масок: {len(self.annotations)}.")

    def _facies_regions(self, path: Path) -> list[dict]:
        colors = ("#ff9f1c", "#f04438", "#13c2c2", "#b34de3", "#e6b800", "#4b9b55")
        return [{
            "kind": "facies", "label": item.label,
            "polygon": item.polygon, "depth_top": item.depth_top, "depth_base": item.depth_base,
            "facies_top": item.facies_top, "facies_base": item.facies_base,
            "target_text": item.target_text,
            "color": colors[index % len(colors)],
        } for index, item in enumerate(annotation for annotation in self.annotations if annotation.photo_path == path)]

    def _render_mask_photo(self) -> None:
        if not self.photos:
            return
        self._current_photo %= len(self.photos)
        photo = self.photos[self._current_photo]
        annotations = [item for item in self.annotations if item.photo_path == photo.path]
        original_count = len(self.columns.get(photo.path, []))
        was_projected = photo.path in self.projected_columns
        projected_count = len(self.projected_columns.get(photo.path, []))
        facies_count = len({(item.source_file, item.source_sheet, item.source_row) for item in annotations})
        status = "Подтверждено" if photo.path in self.mask_confirmed else "Ожидает проверки"
        mismatch = (
            "\nВНИМАНИЕ: детектор колонок на этапе проекции вернул другое число."
            if was_projected and projected_count != original_count else ""
        )
        projection_count = str(projected_count) if was_projected else "не выполнялась (нет совпавших фаций)"
        self.mask_info.setText(
            f"Фото {self._current_photo + 1} из {len(self.photos)}\n{photo.path.name}\n\n"
            f"Колонок на этапе распознавания: {original_count}; при проекции: {projection_count}.\n"
            f"Строк фаций с масками: {facies_count}; сегментов масок: {len(annotations)}.\n"
            f"Статус: {status}{mismatch}\n\n"
            + ("Маски не построены — проверьте этап сопоставления." if not annotations else "Бирюзовый контур — керн, цветные области — интервалы фаций.")
        )
        if photo.path in self.mask_confirmed:
            self.mask_confirm.setText("Отменить подтверждение этого фото")
        elif annotations:
            self.mask_confirm.setText("Маски на этом фото верны")
        else:
            self.mask_confirm.setText("Подтвердить: масок на фото нет")
        self.mask_prev.setEnabled(self._current_photo > 0)
        self.mask_next.setEnabled(self._current_photo < len(self.photos) - 1)
        regions = self._column_regions(photo.path) + self._facies_regions(photo.path)
        self._fill_image(self.mask_preview, photo.path, regions)

    def _confirm_masks(self) -> None:
        if not self.photos:
            return
        path = self.photos[self._current_photo].path
        if path in self.mask_confirmed:
            self.mask_confirmed.remove(path)
        else:
            self.mask_confirmed.add(path)
            count = sum(item.photo_path == path for item in self.annotations)
            self.audit.append(f"Маски: подтверждено фото {path.name}; сегментов {count}.")
        self._render_mask_photo()
        self.status.setText(f"Подтверждено фото с масками: {len(self.mask_confirmed)} из {len(self.photos)}.")

    def _go_forward(self) -> None:
        if self._busy():
            return
        if self._stage == 0:
            self.audit.append("Подтверждение: пользователь проверил распознанные заголовки и значения Excel.")
            self._set_stage(1)
            self._run_async("Ищу изображения и выделяю столбики керна на каждом…", self._discover_and_detect, self._columns_loaded)
        elif self._stage == 1:
            if len(self.column_confirmed) != len(self.photos):
                return QMessageBox.warning(self, "Нужно проверить все фото", f"Подтверждено {len(self.column_confirmed)} из {len(self.photos)} фото.")
            self.audit.append(f"Колонки: подтверждены все {len(self.photos)} фото.")
            self._set_stage(2)
            self._run_async("Распределяю интервалы по фото…", self._prepare_intervals, self._intervals_loaded)
        elif self._stage == 2:
            try:
                self._confirmed_photos = self._confirm_interval_rows()
            except ValueError as exc:
                return QMessageBox.warning(self, "Проверьте интервалы", str(exc))
            manual = sum(item.source == "manual" for item in self._confirmed_photos)
            self.audit.append(f"Интервалы: подтверждено {len(self._confirmed_photos)} фото; вручную исправлено: {manual}.")
            self.photos = self._confirmed_photos
            self._set_stage(3)
            self._run_async("Сопоставляю глубины с фациями Excel…", self._match_confirmed_intervals, self._matches_loaded)
        elif self._stage == 3:
            if self.match_table.rowCount() != len(self.photos) or any(
                self.match_table.item(index, 0).checkState() != Qt.CheckState.Checked
                for index in range(self.match_table.rowCount())
            ):
                return QMessageBox.warning(self, "Нужно проверить все фото", "Подтвердите результат сопоставления по каждому фото, в том числе отсутствие совпадений.")
            self.audit.append(f"Сопоставление: пользователь проверил результаты для всех {len(self.photos)} фото.")
            self._set_stage(4)
            self._run_async("Проецирую маски на фото…", self._project_confirmed_matches, self._projection_loaded)
        elif self._stage == 4:
            if len(self.mask_confirmed) != len(self.photos):
                return QMessageBox.warning(self, "Нужно проверить все фото", f"Подтверждено масок: {len(self.mask_confirmed)} из {len(self.photos)} фото.")
            self.audit.append(f"Маски: просмотрены и подтверждены для всех {len(self.photos)} фото.")
            self._set_stage(5)
        elif self._stage == 5 and not self._failure:
            self.accept()

    def _go_back(self) -> None:
        if self._busy():
            return
        if self._stage > 0:
            self._set_stage(self._stage - 1)

    def _report_issue(self) -> None:
        reason, accepted = QInputDialog.getText(
            self, "Зафиксировать ошибку", "Что именно не совпало?",
        )
        if not accepted:
            return
        stage_name = self.stage_title.text()
        context = ""
        if self._stage == 1 and self.photos:
            photo = self.photos[self._current_photo]
            context = f" Фото: {photo.path.name}; распознано колонок: {len(self.columns.get(photo.path, []))}."
        elif self._stage == 4 and self.photos:
            photo = self.photos[self._current_photo]
            masks = sum(item.photo_path == photo.path for item in self.annotations)
            context = f" Фото: {photo.path.name}; сегментов масок: {masks}."
        elif self._stage == 2:
            table = self.interval_table if self._stage == 2 else self.match_table
            row = table.currentRow()
            if 0 <= row < len(self.photos):
                top = self.interval_table.item(row, 3).text()
                base = self.interval_table.item(row, 4).text()
                context = f" Фото: {self.photos[row].path.name}; интервал: {top}–{base} м."
        elif self._stage == 3:
            row = self.match_table.currentRow()
            if 0 <= row < len(self.photos):
                context = f" Фото: {self.photos[row].path.name}; совпавших строк Excel: {self.match_table.item(row, 4).text()}."
        self._failure = f"{stage_name}.{context} Причина: {reason.strip() or 'не указана'}"
        self.audit.append("ОШИБКА ЗАФИКСИРОВАНА: " + self._failure)
        self._set_stage(5)

    def _render_summary(self) -> None:
        state = "Сверка пройдена." if not self._failure else "Сверка остановлена на этапе с ошибкой."
        lines = [state, "", *self.audit]
        if self._failure:
            lines.extend(("", self._failure, "Исправьте указанный этап и запустите сверку заново."))
        else:
            lines.extend((
                "", "Следующий шаг запустит стандартное создание проекта с теми же исходными Excel и фото.",
                "Пошаговая сверка — диагностика: найденные здесь расхождения перечислены выше и не скрываются автоматическим переходом к обучению.",
            ))
        self.summary_text.setPlainText("\n".join(lines))

    def closeEvent(self, event) -> None:
        if self._busy():
            QMessageBox.information(self, "Подождите", "Текущий этап ещё выполняется.")
            event.ignore()
            return
        super().closeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Excel Photo Model Studio")
        self.resize(1180, 760)
        self.current_project: Path | None = None
        self._pending_verified_photo_records: list[PhotoRecord] | None = None
        self.matching_preview_paths: dict[str, str] = {}
        self.matching_preview_details: dict[str, list[str]] = {}
        self.matching_preview_regions: dict[str, list[dict]] = {}
        self.matching_column_boxes: dict[str, list[list[int]]] = {}
        self.matching_column_orders: dict[str, str] = {}
        self.matching_column_counts: dict[str, int] = {}
        self.matching_column_depths: dict[str, list[str]] = {}
        self.matching_photo_issues: dict[str, list[str]] = {}
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
        self.discovery_process: QProcess | None = None
        self._pending_project_action = ""
        self._ensure_training_output_paths()

    def _build_project_tab(self) -> None:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.excel = PathField(file_filter="Таблицы (*.xlsx *.xlsm *.xltx *.xltm *.xls *.csv *.tsv)")
        self.photos = PathField(directory=True)
        self.ocr = QCheckBox("Автоматически читать глубину фото и колонок керна (Tesseract)")
        self.ocr.setChecked(True)
        form.addRow("Файл Excel/CSV:", self.excel)
        form.addRow("Папка фотографий:", self.photos)
        form.addRow("OCR:", self.ocr)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.create_button = QPushButton("Сопоставить и показать маски")
        self.create_button.clicked.connect(self._create_project)
        self.verify_button = QPushButton("Пошаговая сверка этапов…")
        self.verify_button.clicked.connect(self._open_step_verification)
        self.recalculate_button = QPushButton("Пересчитать после исправлений")
        self.recalculate_button.clicked.connect(self._save_photo_map)
        row.addWidget(self.create_button)
        row.addWidget(self.verify_button)
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
        queue_title = QLabel(
            "Автоматический режим: добавьте пары «таблица + папка фото» для каждой скважины. "
            "Дальше приложение само сопоставит интервалы, проверит покрытие, соберёт датасет и обучит модель."
        )
        queue_title.setWordWrap(True)
        layout.addWidget(queue_title)
        self.automatic_queue = QTableWidget(0, 2)
        self.automatic_queue.setHorizontalHeaderLabels(("Файл Excel", "Папка фотографий"))
        self.automatic_queue.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.automatic_queue.setMaximumHeight(170)
        layout.addWidget(self.automatic_queue)
        queue_buttons = QHBoxLayout()
        discover_wells = QPushButton("Автонаходить Excel и фото в папке")
        discover_wells.clicked.connect(self._discover_automatic_wells)
        self.discovery_button = discover_wells
        add_well = QPushButton("Добавить скважину (Excel + фото)")
        add_well.clicked.connect(self._add_automatic_well)
        remove_well = QPushButton("Убрать выбранную")
        remove_well.clicked.connect(self._remove_automatic_well)
        self.automatic_train_button = QPushButton("Автоматически обработать всё и создать best.pt")
        self.automatic_train_button.clicked.connect(self._start_automatic_training)
        queue_buttons.addWidget(discover_wells)
        queue_buttons.addWidget(add_well)
        queue_buttons.addWidget(remove_well)
        queue_buttons.addWidget(self.automatic_train_button)
        queue_buttons.addStretch(1)
        layout.addLayout(queue_buttons)
        self.catalog_status = QLabel()
        self.catalog_status.setWordWrap(True)
        layout.addWidget(self.catalog_status)
        buttons = QHBoxLayout()
        self.dataset_button = QPushButton("Собрать датасет")
        self.dataset_button.clicked.connect(self._build_dataset)
        self.standard_train_button = QPushButton("Обучить с нуля единый best.pt: слои + фации + описание")
        self.standard_train_button.clicked.connect(self._start_training)
        buttons.addWidget(self.dataset_button)
        buttons.addWidget(self.standard_train_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        layout.addWidget(self.train_log, 1)
        self.tabs.addTab(page, "3. Датасет и best.pt")
        self._load_automatic_queue()
        self._update_catalog_status()

    def _automatic_queue_path(self) -> Path:
        local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return local_app_data / "ExcelPhotoModelStudio" / "automatic_training_queue.json"

    def _load_automatic_queue(self) -> None:
        try:
            items = json.loads(self._automatic_queue_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            items = []
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, dict) and item.get("excel") and item.get("photos"):
                self._append_automatic_well(item["excel"], item["photos"])

    def _append_automatic_well(self, excel: str, photos: str) -> None:
        row = self.automatic_queue.rowCount()
        self.automatic_queue.insertRow(row)
        for column, value in enumerate((excel, photos)):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.automatic_queue.setItem(row, column, item)

    def _save_automatic_queue(self) -> None:
        items = []
        for row in range(self.automatic_queue.rowCount()):
            excel = self.automatic_queue.item(row, 0)
            photos = self.automatic_queue.item(row, 1)
            if excel and photos:
                items.append({"excel": excel.text(), "photos": photos.text()})
        path = self._automatic_queue_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _add_automatic_well(self) -> None:
        excel, _ = QFileDialog.getOpenFileName(
            self, "Выберите Excel для скважины", str(Path.home()),
            "Таблицы (*.xlsx *.xlsm *.xltx *.xltm *.xls *.csv *.tsv)",
        )
        if not excel:
            return
        photos = QFileDialog.getExistingDirectory(self, "Выберите папку фото этой скважины", str(Path(excel).parent))
        if not photos:
            return
        for row in range(self.automatic_queue.rowCount()):
            if (self.automatic_queue.item(row, 0).text() == excel
                    and self.automatic_queue.item(row, 1).text() == photos):
                return self._error("Эта пара Excel + фото уже есть в очереди.")
        self._append_automatic_well(excel, photos)
        self._save_automatic_queue()

    def _discover_automatic_wells(self) -> None:
        if self.discovery_process and self.discovery_process.state() != QProcess.ProcessState.NotRunning:
            return self._error("Поиск архива уже выполняется.")
        root = QFileDialog.getExistingDirectory(self, "Выберите папку с архивом Excel и фото", str(Path.home()))
        if not root:
            return
        self.discovery_button.setEnabled(False)
        self.discovery_button.setText("Ищу Excel и фото…")
        self.discovery_process = QProcess(self)
        self.discovery_process.setProgram(sys.executable)
        self.discovery_process.setArguments(self._with_launcher(["discover-wells", "--root", root]))
        self.discovery_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.discovery_process.finished.connect(self._automatic_discovery_finished)
        self.discovery_process.start()

    def _automatic_discovery_finished(self, code: int, _status) -> None:
        self.discovery_button.setEnabled(True)
        self.discovery_button.setText("Автонаходить Excel и фото в папке")
        if not self.discovery_process:
            return
        output = bytes(self.discovery_process.readAllStandardOutput()).decode(errors="replace").strip()
        if code != 0:
            return self._error(output or "Не удалось просканировать архив.")
        try:
            result = json.loads(output)
        except ValueError:
            return self._error("Сканер вернул некорректный результат. Подробности: " + output[:500])
        existing = {
            (self.automatic_queue.item(row, 0).text(), self.automatic_queue.item(row, 1).text())
            for row in range(self.automatic_queue.rowCount())
        }
        added = 0
        for pair in result["pairs"]:
            key = (pair["excel"], pair["photos"])
            if key not in existing:
                self._append_automatic_well(*key)
                existing.add(key)
                added += 1
        if added:
            self._save_automatic_queue()
        message = (
            f"Найдено Excel: {result['workbooks_found']}; фотографий: {result['photos_found']}; "
            f"однозначных пар добавлено: {added}."
        )
        if result["unmatched"]:
            examples = "\n".join(
                f"• {Path(item['excel']).name}: {item['reason']}"
                for item in result["unmatched"][:8]
            )
            tail = f"\n…и ещё {len(result['unmatched']) - 8}" if len(result["unmatched"]) > 8 else ""
            message += f"\n\nНе сопоставлено автоматически — проверьте эти случаи вручную:\n{examples}{tail}"
        QMessageBox.information(self, "Автопоиск завершён", message)

    def _remove_automatic_well(self) -> None:
        rows = sorted({index.row() for index in self.automatic_queue.selectedIndexes()}, reverse=True)
        for row in rows:
            self.automatic_queue.removeRow(row)
        self._save_automatic_queue()

    def _start_automatic_training(self) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            return self._error("Обучение уже запущено.")
        wells = []
        for row in range(self.automatic_queue.rowCount()):
            excel = self.automatic_queue.item(row, 0)
            photos = self.automatic_queue.item(row, 1)
            if excel and photos:
                wells.append({"excel": excel.text(), "photos": photos.text()})
        if not wells:
            return self._error("Добавьте хотя бы одну пару Excel + папка фото.")
        if not self.dataset_output.edit.text().strip() or not self.model_output.edit.text().strip():
            return self._error("Укажите папки для нового датасета и результата модели.")
        try:
            self._save_automatic_queue()
            local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            job_dir = local_app_data / "ExcelPhotoModelStudio" / "automatic_jobs" / f"job_{uuid4().hex[:12]}"
            job_dir.mkdir(parents=True, exist_ok=False)
            manifest = job_dir / "manifest.json"
            manifest.write_text(json.dumps({"wells": wells}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            return self._error(f"Не удалось сохранить очередь автоматического обучения: {exc}")
        command = [
            "auto-train", "--manifest", str(manifest),
            "--dataset", str(self.dataset_output.value()), "--output", str(self.model_output.value()),
            "--architecture", str(self.architecture.currentData()),
            "--epochs", str(self.epochs.value()), "--patience", str(self.patience.value()),
            "--description-epochs", str(self.description_epochs.value()),
        ]
        if not self.ocr.isChecked():
            command.append("--no-ocr")
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(self._with_launcher(command))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_training_output)
        self.process.finished.connect(self._training_finished)
        self.train_log.clear()
        self.train_log.append(f"Поставлено наборов скважин: {len(wells)}. Автоматический цикл запущен.")
        self.automatic_train_button.setEnabled(False)
        self.dataset_button.setEnabled(False)
        self.standard_train_button.setEnabled(False)
        self.process.start()

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

    def _open_step_verification(self) -> None:
        excel_path = self.excel.value()
        photos_path = self.photos.value()
        if not self.excel.edit.text().strip() or not excel_path.is_file():
            return self._error("Выберите один существующий файл Excel или CSV.")
        if excel_path.suffix.lower() not in {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls", ".csv", ".tsv"}:
            return self._error("Поддерживаются файлы XLSX, XLSM, XLTX, XLTM, XLS, CSV и TSV.")
        if not self.photos.edit.text().strip() or not photos_path.is_dir():
            return self._error("Выберите существующую папку с фотографиями.")
        dialog = StepVerificationDialog(
            excel_path, photos_path, use_ocr=self.ocr.isChecked(), parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._pending_verified_photo_records = list(dialog.photos)
            self._create_project()

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
                "" if record.top is None else format_depth(record.top),
                "" if record.base is None else format_depth(record.base),
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
            source.setData(Qt.ItemDataRole.UserRole, (
                record.well,
                "" if record.top is None else format_depth(record.top),
                "" if record.base is None else format_depth(record.base),
            ))
            source.setData(Qt.ItemDataRole.UserRole + 1, record)
            source.setToolTip(f"Система глубин: {self._depth_basis_label(record.depth_basis)}")
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
            source_item = self.photo_table.item(row, 6)
            original = source_item.data(Qt.ItemDataRole.UserRole) if source_item else None
            original_record = source_item.data(Qt.ItemDataRole.UserRole + 1) if source_item else None
            current = (
                well,
                "" if top is None else format_depth(top),
                "" if base is None else format_depth(base),
            )
            source = source_item.text().strip() if source_item else "not_found"
            interval_edited = original is not None and current != tuple(original)
            if interval_edited:
                source = "manual"
            retry_column_ocr = interval_edited or (
                source == "manual" and confirmed
                and isinstance(original_record, PhotoRecord)
                and not original_record.column_depths
            )
            records.append(PhotoRecord(
                path=path, well=well, top=top, base=base,
                source=source or "manual",
                mapping_confirmed=confirmed,
                column_order=column_order,
                column_depths=(
                    original_record.column_depths
                    if isinstance(original_record, PhotoRecord) and not interval_edited else ()
                ),
                column_ocr_checked=(
                    False if retry_column_ocr else (
                        original_record.column_ocr_checked if isinstance(original_record, PhotoRecord) else False
                    )
                ),
                depth_basis=(
                    original_record.depth_basis
                    if isinstance(original_record, PhotoRecord) and not interval_edited else "unknown"
                ),
            ))
        try:
            project_dir = self._project_dir()
            write_photo_map(project_dir / "photo_map.csv", records)
        except Exception as exc:
            return self._error(str(exc))
        self._show_matching_preview()
        self._start_project_process(["refresh", "--project", str(project_dir)], project_dir, "refresh")

    def _start_project_process(self, command: list[str], project_dir: Path, action: str) -> None:
        if self.project_process and self.project_process.state() != QProcess.ProcessState.NotRunning:
            return self._error("Сопоставление этой скважины уже выполняется.")
        self.current_project = project_dir
        self._pending_project_action = action
        self.create_button.setEnabled(False)
        self.verify_button.setEnabled(False)
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
        self.verify_button.setEnabled(True)
        self.recalculate_button.setEnabled(True)
        self.project_progress.setVisible(False)
        if code != 0:
            self._pending_verified_photo_records = None
            return self._error("Обработка не завершена. Подробности показаны в журнале.")
        try:
            project_dir = self._project_dir()
            report = json.loads((project_dir / "report.json").read_text(encoding="utf-8"))
        except Exception as exc:
            return self._error(str(exc))
        if self._pending_project_action == "create" and self._pending_verified_photo_records is not None:
            try:
                write_photo_map(project_dir / "photo_map.csv", self._pending_verified_photo_records)
            except Exception as exc:
                return self._error(f"Не удалось перенести подтверждённые интервалы в проект: {exc}")
            self._pending_verified_photo_records = None
            self.matching_preview_paths.clear()
            self.matching_preview_details.clear()
            self.matching_preview_regions.clear()
            self.matching_column_boxes.clear()
            self.matching_column_orders.clear()
            self.matching_column_counts.clear()
            self.matching_column_depths.clear()
            self._start_project_process(
                ["refresh", "--project", str(project_dir)], project_dir, "verified_refresh",
            )
            return
        if self._pending_project_action in {"create", "verified_refresh"}:
            self.matching_preview_paths.clear()
            self.matching_preview_details.clear()
            self.matching_preview_regions.clear()
            self.matching_column_boxes.clear()
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
        self._set_photo_issues(report)
        lines = [
            "Обучение заблокировано: сопоставление содержит ошибки."
            if report.get("blocking_errors", 0) else "Проверка сопоставления пройдена.",
            f"Служебная папка создана автоматически: {report['project_dir']}",
            f"Excel/CSV-файлов: {report.get('excel_files', 1)}", f"Строк Excel: {report['excel_rows']}", f"Фото: {report['photos']}",
            f"Подтверждены интервалы фото: {report['confirmed_photos']}",
            f"Нужно подтвердить интервалы: {report['unconfirmed_photos']}",
            f"Фото без обязательного интервала: {report.get('photos_without_intervals', 0)}",
            f"Фото без распознанного керна: {report.get('photos_without_core_columns', 0)}",
            f"Фото с найденными фациями: {report.get('photos_with_facies', 0)} из {report.get('photos', 0)}",
            f"Фото без масок в датасете: {report.get('photos_without_masks', 0)}",
            f"Полный список фото и причин пропуска: {report.get('photo_inventory', 'photo_inventory.csv')}",
            f"Автоматически восстановлено интервалов фото: "
            f"{report.get('auto_sequenced_photos', report.get('ocr_verified_photos', 0))}",
            f"Спроецировано масок: {report['annotations']}",
            f"Ошибок привязки масок к колонкам керна: {report.get('projection_errors', 0)}",
            f"Фаций Excel с неполным покрытием масками: {report.get('facies_rows_with_incomplete_masks', 0)}",
            f"Непокрытых фациями участков: {report.get('uncovered_facies_intervals', 0)}",
            f"Участков без «Краткого описания»: "
            f"{report.get('uncovered_description_intervals', 0)}",
            f"Интервалов керна Excel без фото: "
            f"{report.get('uncovered_excel_core_intervals', 0)}",
            f"Строк с ошибкой толщины фации: {report.get('invalid_thickness_rows', 0)}",
            f"Строк Excel с «Кратким описанием»: {report.get('excel_text_targets', 0)}",
            f"Фаций сопоставлено по резервному интервалу ГИС: {report.get('gis_fallback_matches', 0)}",
            f"Строк фаций без «Краткого описания»: "
            f"{report.get('facies_rows_without_description', 0)}",
            f"Строк фаций без сопоставленного фото: "
            f"{report.get('excel_facies_rows_without_photo_match', 0)}",
            f"Подробная сверка каждой строки Excel: "
            f"{report.get('facies_inventory', 'facies_inventory.csv')}",
            f"Масок с «Кратким описанием»: {report.get('text_targets', 0)}",
            f"Подтверждено масок: {report['approved_annotations']}",
            f"Блокирующих ошибок: {report.get('blocking_errors', 0)}",
        ]
        mappings = report.get("column_mappings", [])
        if mappings:
            lines.append("\nРаспознанные столбцы Excel (по названиям):")
            lines.extend(
                (
                    f"- {item.get('sheet', '')}: основной интервал фации "
                    f"{item.get('facies_top') or '?'}–{item.get('facies_base') or '?'}, "
                    + (
                        f"резерв ГИС {item.get('gis_facies_top')}–{item.get('gis_facies_base')}, "
                        if item.get("gis_facies_top") is not None and item.get("gis_facies_base") is not None
                        else "резерв ГИС не найден, "
                    )
                    + f"толщина фации {item.get('facies_thickness') or '?'}, "
                    + f"«Краткое описание» {item.get('target_text') or '?'}"
                )
                for item in mappings
            )
        if report.get("issues"):
            lines.append("\nПроверить:")
            lines.extend(f"- {item['source']}: {item['message']}" for item in report["issues"])
        if report.get("unconfirmed_photos", 0):
            lines.append("\nНеизвестные интервалы можно исправить в таблице выше, отметить OK и нажать «Пересчитать после исправлений». Изменённые вручную глубины проверяются заново; их система глубин больше не считается определённой OCR.")
        lines.append(
            "Порядок колонок определяется автоматически по отметкам «Верх/Низ» или крайним цифрам. "
            "Если направление определено неверно, выберите его в таблице вручную и нажмите «Пересчитать»."
        )
        self.project_log.setPlainText("\n".join(lines))

    def _set_photo_issues(self, report: dict) -> None:
        self.matching_photo_issues.clear()
        for issue in report.get("issues", []):
            if issue.get("severity") == "error" and issue.get("source") and issue.get("message"):
                self.matching_photo_issues.setdefault(str(issue["source"]), []).append(str(issue["message"]))

    @staticmethod
    def _depth_basis_label(basis: str) -> str:
        return {"drilling": "по бурению / керну", "gis": "по ГИС / с увязкой"}.get(basis, "не определена")

    def _load_review(self) -> None:
        try:
            rows = load_annotations(self._project_dir())
        except Exception as exc:
            return self._error(str(exc))
        try:
            self._set_photo_issues(json.loads((self._project_dir() / "report.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            self.matching_photo_issues.clear()
        self.matching_preview_paths.clear()
        self.matching_preview_details.clear()
        self.matching_preview_regions.clear()
        self.matching_column_boxes.clear()
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
                    "facies_top": row.get("facies_top", ""),
                    "facies_base": row.get("facies_base", ""),
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
        regions = list(self.matching_preview_regions.get(photo_key, []))
        source_size = QImageReader(photo_path).size()
        image_width, image_height = source_size.width(), source_size.height()
        for index, box in enumerate(self.matching_column_boxes.get(photo_key, []), start=1):
            if len(box) != 4 or image_width <= 0 or image_height <= 0:
                continue
            left, top, right, bottom = (int(value) for value in box)
            regions.append({
                "kind": "core_column",
                "polygon": ((left, top), (right, top), (right, bottom), (left, bottom)),
                "image_width": image_width,
                "image_height": image_height,
                "label": f"Керн {index}",
            })
        self.matching_preview.set_regions(regions)
        order_widget = self.photo_table.cellWidget(row, 5)
        requested_order = order_widget.currentData() if isinstance(order_widget, QComboBox) else COLUMN_ORDER_AUTO
        effective_order = self.matching_column_orders.get(photo_key)
        column_count = self.matching_column_counts.get(photo_key)
        column_depths = self.matching_column_depths.get(photo_key, [])
        source_item = self.photo_table.item(row, 6)
        original_record = source_item.data(Qt.ItemDataRole.UserRole + 1) if source_item else None
        basis = original_record.depth_basis if isinstance(original_record, PhotoRecord) else "unknown"
        original_values = source_item.data(Qt.ItemDataRole.UserRole) if source_item else None
        if original_values and (
            as_float(self.photo_table.item(row, 3).text()) != as_float(original_values[1])
            or as_float(self.photo_table.item(row, 4).text()) != as_float(original_values[2])
        ):
            basis = "unknown"
        basis_text = f"\nСистема глубин: {self._depth_basis_label(basis)}"
        photo_issues = list(dict.fromkeys(
            issue for source in (photo_path, str(Path(photo_path).resolve()), Path(photo_path).name)
            for issue in self.matching_photo_issues.get(source, [])
        ))
        issue_text = "\nПроверка не пройдена: " + " ".join(photo_issues[:2]) if photo_issues else ""
        if len(photo_issues) > 2:
            issue_text += f" Ещё ошибок: {len(photo_issues) - 2}; подробности в журнале."
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
                f"{columns_text}{depth_text}{basis_text}\n{order_text}{issue_text}\nНайдено интервалов: {len(details)}\n"
                f"{visible_details}{suffix}\nНаведите курсор на маску — появится краткое описание."
            )
            self._set_preview_pixmap(self.matching_preview, preview_path, "Не удалось открыть фото с масками.")
        else:
            confirmed = self.photo_table.item(row, 0).checkState() == Qt.CheckState.Checked
            top = as_float(self.photo_table.item(row, 3).text())
            base = as_float(self.photo_table.item(row, 4).text())
            if not confirmed or top is None or base is None or base <= top:
                reason = (
                    "Маски пока не построены: заполните начало и конец интервала, отметьте OK "
                    "и нажмите «Пересчитать после исправлений»."
                )
            elif photo_issues:
                reason = "Маски не построены." + issue_text
            else:
                reason = (
                    f"Маски для интервала {format_depth(top)}–{format_depth(base)} м не построены. "
                    "Проверьте привязку глубин, геометрию колонок и фации Excel; подробности в журнале."
                )
            self.matching_preview_title.setText(
                f"{columns_text}{depth_text}{basis_text}\n{order_text}\n{reason}\n"
                "Бирюзовый контур показывает найденную колонку керна."
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
                self.matching_column_boxes[self._photo_key(photo_path)] = boxes
            depth_ranges = value.get("depth_ranges", [])
            if isinstance(depth_ranges, list):
                labels = []
                for index, interval in enumerate(depth_ranges, start=1):
                    if not isinstance(interval, dict):
                        continue
                    top = as_float(interval.get("top"))
                    base = as_float(interval.get("base"))
                    if top is not None and base is not None:
                        labels.append(f"{index}) {format_depth(top)}–{format_depth(base)} м")
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
        if hasattr(self, "automatic_train_button"):
            self.automatic_train_button.setEnabled(True)
        if hasattr(self, "dataset_button"):
            self.dataset_button.setEnabled(True)
        if hasattr(self, "standard_train_button"):
            self.standard_train_button.setEnabled(True)
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
