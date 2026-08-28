from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout


class PhotoIntervalDialog(QDialog):
    """Manual fallback when a core photograph has no parsable file name/caption."""

    def __init__(self, photo_name: str, well: str = "", top: float | None = None, base: float | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Интервал фотографии керна")
        self.setMinimumWidth(390)
        layout = QVBoxLayout(self)
        note = QLabel(
            f"Для «{photo_name}» интервал не удалось получить из имени или подписи.\n"
            "Укажите его, чтобы сопоставить фации с Excel.",
            self,
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.well = QLineEdit(well)
        self.top = QLineEdit("" if top is None else f"{top:g}")
        self.base = QLineEdit("" if base is None else f"{base:g}")
        self.top.setPlaceholderText("например, 2450.50")
        self.base.setPlaceholderText("например, 2452.10")
        form.addRow("Скважина:", self.well)
        form.addRow("Начало, м:", self.top)
        form.addRow("Конец, м:", self.base)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, float, float]:
        well = self.well.text().strip()
        if not well:
            raise ValueError("Укажите скважину")
        try:
            top = float(self.top.text().strip().replace(",", "."))
            base = float(self.base.text().strip().replace(",", "."))
        except ValueError as exc:
            raise ValueError("Начало и конец должны быть числами") from exc
        if base <= top:
            raise ValueError("Конец интервала должен быть больше начала")
        return well, top, base
