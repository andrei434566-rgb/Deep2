"""One pre-flight table for mapping anonymous core photographs to depth ranges."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout

from app.infrastructure.excel_core_description import CoreInterval


class PhotoIntervalMappingDialog(QDialog):
    """Makes every photo-to-depth binding visible before masks are created."""

    HEADERS = ("Файл фото", "Скважина", "Начало, м", "Конец, м", "Источник")

    def __init__(self, rows: list[tuple[Path, CoreInterval | None, str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Проверка интервалов фотографий")
        self.resize(900, 520)
        self._paths = [path for path, _, _ in rows]
        layout = QVBoxLayout(self)
        note = QLabel(
            "Маски и литологическое описание будут созданы только по пересечению интервалов. "
            "Название файла не используется как идентификатор: проверьте/впишите скважину, начало и конец для каждой фотографии.",
            self,
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(len(rows), len(self.HEADERS), self)
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for row, (path, interval, source) in enumerate(rows):
            values = (path.name, interval.well if interval else "", f"{interval.top:g}" if interval else "", f"{interval.base:g}" if interval else "", source)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in {0, 4}:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, item)
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def mappings(self) -> list[tuple[Path, CoreInterval]]:
        result: list[tuple[Path, CoreInterval]] = []
        for row, path in enumerate(self._paths):
            try:
                well = self.table.item(row, 1).text().strip()
                top = float(self.table.item(row, 2).text().strip().replace(",", "."))
                base = float(self.table.item(row, 3).text().strip().replace(",", "."))
            except (AttributeError, ValueError) as exc:
                raise ValueError(f"{path.name}: заполните скважину, начало и конец интервала.") from exc
            if not well or base <= top:
                raise ValueError(f"{path.name}: нужен номер скважины и интервал, где конец больше начала.")
            result.append((path, CoreInterval(well, top, base)))
        return result
