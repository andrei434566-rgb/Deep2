"""Editable lithology and facies colours with conventional symbols."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from app.domain.lithology import (
    LITHOLOGY_PATTERN_TITLES,
    default_lithology_palette,
    normalize_lithology_palette,
)
from app.domain.facies_palette import normalize_facies_palette


class LithologyPaletteDialog(QDialog):
    """Small persistent legend editor used by the assembled-core track."""

    HEADERS = ("Название", "Условный знак", "Цвет", "Узор")

    def __init__(
        self,
        palette: list[dict[str, str]],
        parent=None,
        *,
        title: str = "Литологическая палетка",
        description: str | None = None,
        defaults: list[dict[str, str]] | None = None,
        normalizer=normalize_lithology_palette,
    ):
        super().__init__(parent)
        self._normalizer = normalizer
        self._defaults = [dict(item) for item in (defaults or default_lithology_palette())]
        self.setWindowTitle(title)
        self.setMinimumSize(760, 480)

        layout = QVBoxLayout(self)
        note = QLabel(description or (
            "Настройте названия пород, сокращения, цвета и условные узоры. "
            "После сохранения палетка останется доступной при следующем запуске."
        ))
        note.setWordWrap(True)
        layout.addWidget(note)

        self.table = QTableWidget(0, len(self.HEADERS), self)
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        controls = QHBoxLayout()
        add = QPushButton("Добавить элемент", self)
        add.clicked.connect(self._add_row)
        remove = QPushButton("Удалить выбранную", self)
        remove.clicked.connect(self._remove_selected)
        choose_color = QPushButton("Выбрать цвет…", self)
        choose_color.clicked.connect(self._choose_color)
        defaults = QPushButton("Вернуть стандартную", self)
        defaults.clicked.connect(self._restore_defaults)
        controls.addWidget(add)
        controls.addWidget(remove)
        controls.addWidget(choose_color)
        controls.addStretch(1)
        controls.addWidget(defaults)
        layout.addLayout(controls)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for item in self._normalizer(palette) or self._defaults:
            self._add_row(item)

    def _add_row(self, item: dict[str, str] | None = None) -> None:
        item = dict(item or {})
        row = self.table.rowCount()
        self.table.insertRow(row)
        name_item = QTableWidgetItem(str(item.get("name") or "Новая порода"))
        name_item.setData(Qt.ItemDataRole.UserRole, str(item.get("key") or ""))
        self.table.setItem(row, 0, name_item)
        self.table.setItem(row, 1, QTableWidgetItem(str(item.get("symbol") or "LITH")))
        self._set_color(row, str(item.get("color") or "#808080"))
        patterns = QComboBox(self.table)
        for value, title in LITHOLOGY_PATTERN_TITLES.items():
            patterns.addItem(title, value)
        index = patterns.findData(str(item.get("pattern") or "solid"))
        patterns.setCurrentIndex(max(0, index))
        self.table.setCellWidget(row, 3, patterns)
        self.table.setCurrentCell(row, 0)

    def _remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()}, reverse=True)
        if not rows and self.table.currentRow() >= 0:
            rows = [self.table.currentRow()]
        for row in rows:
            self.table.removeRow(row)

    def _choose_color(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        current = QColor(self.table.item(row, 2).text() if self.table.item(row, 2) else "#808080")
        color = QColorDialog.getColor(current, self, f"Цвет · {self.windowTitle()}")
        if color.isValid():
            self._set_color(row, color.name(QColor.NameFormat.HexRgb))

    def _set_color(self, row: int, value: str) -> None:
        color = QColor(value)
        if not color.isValid():
            color = QColor("#808080")
        item = QTableWidgetItem(color.name(QColor.NameFormat.HexRgb))
        item.setBackground(color)
        item.setForeground(QColor("#ffffff") if color.lightness() < 125 else QColor("#202532"))
        self.table.setItem(row, 2, item)

    def _restore_defaults(self) -> None:
        self.table.setRowCount(0)
        for item in self._defaults:
            self._add_row(item)

    def _accept_if_valid(self) -> None:
        if not self.palette():
            QMessageBox.warning(self, self.windowTitle(), "Добавьте хотя бы одну строку с названием.")
            return
        self.accept()

    def palette(self) -> list[dict[str, str]]:
        rows = []
        for row in range(self.table.rowCount()):
            pattern_box = self.table.cellWidget(row, 3)
            row_data = {
                "name": self.table.item(row, 0).text() if self.table.item(row, 0) else "",
                "symbol": self.table.item(row, 1).text() if self.table.item(row, 1) else "",
                "color": self.table.item(row, 2).text() if self.table.item(row, 2) else "",
                "pattern": str(pattern_box.currentData() if isinstance(pattern_box, QComboBox) else "solid"),
            }
            source_key = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) if self.table.item(row, 0) else ""
            if source_key:
                row_data["key"] = str(source_key)
            rows.append(row_data)
        return self._normalizer(rows)


class FaciesPaletteDialog(LithologyPaletteDialog):
    """The same editor, seeded by facies currently present in photo memory."""

    def __init__(self, palette: list[dict[str, str]], defaults: list[dict[str, str]], parent=None):
        super().__init__(
            palette,
            parent,
            title="Палетка фаций",
            description=(
                "Классы собраны из масок на загруженных фотографиях. Можно изменить отображаемое "
                "название, цвет, сокращение и узор; исходная метка модели останется связанной с классом."
            ),
            defaults=defaults,
            normalizer=normalize_facies_palette,
        )
