"""Editor for a loaded LAS/RIGIS curve and its core interval."""

from __future__ import annotations

import math

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout


class LogEditorDialog(QDialog):
    """Lets an interpreter align a log to core and correct sampled curve values."""

    MAX_EDITABLE_ROWS = 1500

    def __init__(self, log_data: dict, depth_from: float, depth_to: float, title: str, parent=None, settings_key: str = "gis"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(590, 620)
        self.settings = QSettings("DeepCore", "DeepCore2")
        self.settings_key = settings_key
        self._data = {
            **log_data,
            "curves": [dict(curve) for curve in log_data.get("curves", [])],
            "depths": list(log_data.get("depths", []) or []),
            "values": {name: list(values or []) for name, values in (log_data.get("values", {}) or {}).items()},
        }
        self._curve_names = [
            curve.get("mnemonic")
            for curve in self._data["curves"]
            if curve.get("mnemonic") and curve.get("mnemonic") != self._data.get("depth_curve")
        ]
        self._row_indexes: list[int] = []
        self._current_curve = ""

        layout = QVBoxLayout(self)
        notice = QLabel("Укажите интервал керна, затем при необходимости исправьте значения выбранной кривой.")
        notice.setWordWrap(True)
        notice.setStyleSheet("color: #687087;")
        layout.addWidget(notice)
        form = QFormLayout()
        self.depth_from = QLineEdit(f"{depth_from:g}")
        self.depth_to = QLineEdit(f"{depth_to:g}")
        self.depth_from.setPlaceholderText("верх керна, м")
        self.depth_to.setPlaceholderText("низ керна, м")
        self.shift_step = QDoubleSpinBox()
        self.shift_step.setDecimals(3)
        self.shift_step.setRange(0.001, 1_000_000)
        self.shift_step.setValue(float(self.settings.value(f"last/{settings_key}_shift_step", 0.1)))
        self.shift_step.setSuffix(" м")
        self.curve_combo = QComboBox()
        self.curve_combo.addItems(self._curve_names)
        self.curve_name = QLineEdit()
        self.curve_visible = QCheckBox("\u041f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u0442\u044c \u0432 \u043f\u043b\u0430\u043d\u0448\u0435\u0442\u0435")
        self.curve_name.setPlaceholderText("например, GR_MAIN")
        form.addRow("Верх керна, м:", self.depth_from)
        form.addRow("Низ керна, м:", self.depth_to)
        form.addRow("Шаг сдвига:", self.shift_step)
        form.addRow("Кривая:", self.curve_combo)
        form.addRow("Название кривой:", self.curve_name)
        form.addRow("\u0412\u0438\u0434\u0438\u043c\u043e\u0441\u0442\u044c:", self.curve_visible)
        layout.addLayout(form)
        shift_actions = QHBoxLayout()
        up_button = QPushButton("▲ Сдвинуть выше")
        down_button = QPushButton("▼ Сдвинуть ниже")
        up_button.clicked.connect(lambda: self._shift_interval(-1))
        down_button.clicked.connect(lambda: self._shift_interval(1))
        shift_actions.addWidget(up_button)
        shift_actions.addWidget(down_button)
        layout.addLayout(shift_actions)
        self.table_hint = QLabel()
        self.table_hint.setStyleSheet("color: #687087;")
        layout.addWidget(self.table_hint)
        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Глубина, м", "Значение"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.curve_combo.currentTextChanged.connect(self._change_curve)
        self.curve_name.editingFinished.connect(self._rename_current_curve)
        self.curve_visible.toggled.connect(self._set_current_curve_visible)
        if self._curve_names:
            last_curve = str(self.settings.value(f"last/{settings_key}_curve", ""))
            self.curve_combo.setCurrentText(last_curve if last_curve in self._curve_names else self._curve_names[0])
            self._change_curve(self.curve_combo.currentText())

    def values(self) -> tuple[dict, float, float]:
        self._store_current_table()
        top = self._parse_depth(self.depth_from.text())
        base = self._parse_depth(self.depth_to.text())
        if top >= base:
            raise ValueError("Низ керна должен быть больше его верха")
        self.settings.setValue(f"last/{self.settings_key}_curve", self._current_curve)
        self.settings.setValue(f"last/{self.settings_key}_shift_step", self.shift_step.value())
        return self._data, top, base

    def _change_curve(self, curve_name: str) -> None:
        self._store_current_table()
        self._current_curve = curve_name
        self.curve_name.setText(curve_name)
        curve_meta = next((curve for curve in self._data["curves"] if curve.get("mnemonic") == curve_name), {})
        self.curve_visible.blockSignals(True)
        self.curve_visible.setChecked(bool(curve_meta.get("visible", True)))
        self.curve_visible.blockSignals(False)
        depths = self._data["depths"]
        step = max(1, math.ceil(len(depths) / self.MAX_EDITABLE_ROWS))
        self._row_indexes = list(range(0, len(depths), step))
        if self._row_indexes and self._row_indexes[-1] != len(depths) - 1:
            self._row_indexes.append(len(depths) - 1)
        values = self._data["values"].setdefault(curve_name, [])
        self.table.setRowCount(len(self._row_indexes))
        for row, index in enumerate(self._row_indexes):
            depth = depths[index] if index < len(depths) else None
            value = values[index] if index < len(values) else None
            depth_item = QTableWidgetItem("" if depth is None else f"{float(depth):g}")
            depth_item.setFlags(depth_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, depth_item)
            self.table.setItem(row, 1, QTableWidgetItem("" if value is None else f"{float(value):g}"))
        suffix = "" if step == 1 else f" Показана каждая {step}-я точка; остальные данные сохраняются без изменений."
        self.table_hint.setText(f"Можно исправить {len(self._row_indexes)} значений.{suffix}")

    def _set_current_curve_visible(self, visible: bool) -> None:
        for curve in self._data["curves"]:
            if curve.get("mnemonic") == self._current_curve:
                curve["visible"] = bool(visible)
                return

    def _rename_current_curve(self) -> None:
        old_name = self._current_curve
        new_name = self.curve_name.text().strip().upper()
        if not old_name or not new_name or old_name == new_name:
            return
        if new_name in self._data["values"]:
            self.curve_name.setText(old_name)
            return
        try:
            self._store_current_table()
        except ValueError:
            self.curve_name.setText(old_name)
            return
        values = self._data["values"].pop(old_name, [])
        self._data["values"][new_name] = values
        for curve in self._data["curves"]:
            if curve.get("mnemonic") == old_name:
                curve["mnemonic"] = new_name
        index = self._curve_names.index(old_name)
        self._curve_names[index] = new_name
        self._current_curve = new_name
        self.curve_combo.blockSignals(True)
        self.curve_combo.setItemText(index, new_name)
        self.curve_combo.setCurrentIndex(index)
        self.curve_combo.blockSignals(False)
        self.curve_name.setText(new_name)

    def _store_current_table(self) -> None:
        if not self._current_curve:
            return
        values = self._data["values"].setdefault(self._current_curve, [])
        if len(values) < len(self._data["depths"]):
            values.extend([None] * (len(self._data["depths"]) - len(values)))
        for row, index in enumerate(self._row_indexes):
            item = self.table.item(row, 1)
            text = str(item.text() if item else "").strip().replace(",", ".")
            if not text:
                values[index] = None
                continue
            try:
                values[index] = float(text)
            except ValueError as exc:
                raise ValueError(f"Некорректное значение в строке {row + 1}") from exc

    def _shift_interval(self, direction: int) -> None:
        try:
            offset = float(direction) * self.shift_step.value()
            self.depth_from.setText(f"{self._parse_depth(self.depth_from.text()) + offset:g}")
            self.depth_to.setText(f"{self._parse_depth(self.depth_to.text()) + offset:g}")
        except ValueError:
            # The normal validation on Save will explain incomplete values.
            return

    @staticmethod
    def _parse_depth(text: str) -> float:
        try:
            return float(str(text or "").strip().replace(",", "."))
        except ValueError as exc:
            raise ValueError("Глубина должна быть числом") from exc
