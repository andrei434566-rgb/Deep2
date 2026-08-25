"""Well-wide depth reference, datum and range setup."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QVBoxLayout


class WellDepthDialog(QDialog):
    """Collect a displayed well range without silently converting depth systems.

    A directional survey (or a dedicated LAS depth curve) is needed to convert
    MD to TVD/TVDSS.  This dialog therefore keeps the coordinate system and its
    datum explicit, while the entered interval remains in that system.
    """

    DATUMS = [
        "RKB / KB",
        "Роторный стол (RT)",
        "Устье / поверхность (GL)",
        "Пол буровой (DF)",
        "Средний уровень моря (MSL)",
        "Морское дно",
        "Фланец обсадной колонны",
        "Пользовательский datum",
    ]
    COORDINATES = ["MD", "TVD", "TVDSS", "TVDKB", "TVDGL"]

    def __init__(
        self,
        depth_from: float,
        depth_to: float,
        references: list[str],
        selected_reference: str,
        parent=None,
        settings: dict | None = None,
    ):
        super().__init__(parent)
        settings = dict(settings or {})
        self.setWindowTitle("Глубина скважины")
        self.setMinimumWidth(390)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        unit = str(settings.get("unit") or "m")
        self.depth_from = QLineEdit(f"{depth_from:g}")
        self.depth_to = QLineEdit(f"{depth_to:g}")
        self.coordinate = QComboBox()
        self.coordinate.addItems(self.COORDINATES)
        coordinate = str(settings.get("coordinate_system") or selected_reference or "MD").upper()
        self.coordinate.setCurrentText(coordinate if coordinate in self.COORDINATES else "MD")
        self.unit = QComboBox()
        self.unit.addItems(["m", "ft"])
        self.unit.setCurrentText(unit if unit in {"m", "ft"} else "m")
        self.datum = QComboBox()
        self.datum.addItems(self.DATUMS)
        datum = str(settings.get("datum") or ("Средний уровень моря (MSL)" if coordinate == "TVDSS" else "RKB / KB"))
        self.datum.setCurrentText(datum if datum in self.DATUMS else "Пользовательский datum")
        self.datum_elevation = QLineEdit(str(settings.get("datum_elevation") or ""))
        self.source_curve = QComboBox()
        choices = list(dict.fromkeys([*references, "Не менять кривую LAS"]))
        self.source_curve.addItems(choices)
        selected_curve = str(settings.get("source_curve") or selected_reference or "")
        self.source_curve.setCurrentText(selected_curve if selected_curve in choices else "Не менять кривую LAS")
        form.addRow("Верх скважины:", self.depth_from)
        form.addRow("Низ скважины:", self.depth_to)
        form.addRow("Система координат:", self.coordinate)
        form.addRow("Единицы глубины:", self.unit)
        form.addRow("Отметка / datum:", self.datum)
        form.addRow("Высота datum над MSL:", self.datum_elevation)
        if references:
            form.addRow("Кривая глубины из LAS:", self.source_curve)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[float, float, str, dict]:
        try:
            top = float(self.depth_from.text().strip().replace(",", "."))
            base = float(self.depth_to.text().strip().replace(",", "."))
        except ValueError as exc:
            raise ValueError("Глубины должны быть числами") from exc
        if top >= base:
            raise ValueError("Низ скважины должен быть больше верха")
        elevation_text = self.datum_elevation.text().strip().replace(",", ".")
        try:
            datum_elevation = float(elevation_text) if elevation_text else None
        except ValueError as exc:
            raise ValueError("Высота datum над MSL должна быть числом") from exc
        source_curve = self.source_curve.currentText().strip()
        settings = {
            "coordinate_system": self.coordinate.currentText().strip(),
            "unit": self.unit.currentText().strip(),
            "datum": self.datum.currentText().strip(),
            "datum_elevation": datum_elevation,
            "source_curve": "" if source_curve == "Не менять кривую LAS" else source_curve,
        }
        return top, base, settings["coordinate_system"], settings
