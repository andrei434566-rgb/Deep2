"""Editor for the physical core-column bounds on one photo."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout


class CoreColumnsDialog(QDialog):
    """Small numeric editor: robust on large photos and precise enough for trays."""

    HEADERS = ("Лево", "Верх", "Право", "Низ")

    def __init__(self, columns: list[dict[str, float]], image_size: tuple[int, int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Границы колонок керна")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        width, height = image_size
        layout.addWidget(QLabel(
            "Укажите только прямоугольники с керном. Всё за их пределами (фон, линейки, подписи и шлак) "
            "не попадёт в сегментацию. Координаты — пиксели на текущем фото "
            f"({width} × {height})."
        ))
        self.table = QTableWidget(0, len(self.HEADERS), self)
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        controls = QHBoxLayout()
        add = QPushButton("Добавить колонку", self)
        add.clicked.connect(self._add_row)
        remove = QPushButton("Удалить выбранную", self)
        remove.clicked.connect(self._remove_selected)
        controls.addWidget(add)
        controls.addWidget(remove)
        controls.addStretch(1)
        layout.addLayout(controls)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for column in columns:
            self._add_row(column)
        if not columns:
            self._add_row({"left": 0, "top": 0, "right": width, "bottom": height})

    def _add_row(self, column: dict[str, float] | None = None):
        column = column or {}
        row = self.table.rowCount()
        self.table.insertRow(row)
        for index, key in enumerate(("left", "top", "right", "bottom")):
            item = QTableWidgetItem(f"{float(column.get(key, 0)):.1f}")
            self.table.setItem(row, index, item)

    def _remove_selected(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def columns(self) -> list[dict[str, float]]:
        result: list[dict[str, float]] = []
        for row in range(self.table.rowCount()):
            try:
                left, top, right, bottom = [float(self.table.item(row, index).text().replace(",", ".")) for index in range(4)]
            except (AttributeError, ValueError):
                continue
            if right > left and bottom > top:
                result.append({"left": left, "top": top, "right": right, "bottom": bottom})
        return result
