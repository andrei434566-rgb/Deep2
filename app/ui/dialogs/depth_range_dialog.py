from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QVBoxLayout


class DepthRangeDialog(QDialog):
    """Compact editor for a layer's depth interval from the stack column."""

    def __init__(self, depth_from: float | None, depth_to: float | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Привязка слоя")
        self.setMinimumWidth(270)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.depth_from = QLineEdit("" if depth_from is None else f"{depth_from:g}")
        self.depth_to = QLineEdit("" if depth_to is None else f"{depth_to:g}")
        self.depth_from.setPlaceholderText("например, 2450.5")
        self.depth_to.setPlaceholderText("например, 2451.2")
        form.addRow("Начало, м:", self.depth_from)
        form.addRow("Конец, м:", self.depth_to)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[float | None, float | None]:
        depth_from, depth_to = self._parse(self.depth_from.text()), self._parse(self.depth_to.text())
        if (depth_from is None) != (depth_to is None):
            raise ValueError("Укажите обе границы интервала или очистите оба поля")
        if depth_from is not None and depth_to is not None and depth_to <= depth_from:
            raise ValueError("Конец слоя должен быть больше начала")
        return depth_from, depth_to

    @staticmethod
    def _parse(value: str) -> float | None:
        value = str(value or "").strip().replace(",", ".")
        if not value:
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError("Глубина должна быть числом") from exc
