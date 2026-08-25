from __future__ import annotations

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QInputDialog, QLabel, QLineEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget

from app.domain.facies_catalog import FACIES_CATALOG, facies_metadata, facies_title
from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS


CUSTOM_VALUE = "__deepcore_custom_value__"
FACIES_REFERENCE_FIELDS = {
    "Код фации",
    "Индекс фации",
    "Название фации",
    "Энергия среды",
    "Гидродинамический режим",
}


class FaciesDialog(QDialog):
    delete_requested = Signal()

    def __init__(
        self,
        current_facies: str,
        confidence: float,
        attributes: dict[str, str] | None = None,
        depth_from: float | None = None,
        depth_to: float | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Параметры фации")
        self.setMinimumWidth(345)
        self.resize(375, 620)
        attributes = dict(attributes or {})
        if current_facies and current_facies != "Новый контур":
            # Imported project values remain authoritative over catalogue
            # defaults, which lets an interpreter correct a single layer.
            reference = facies_metadata(current_facies)
            reference.update(attributes)
            attributes = reference
        self._preserved_attributes = {
            key: value
            for key, value in attributes.items()
            if key not in LITHOLOGY_ATTRIBUTE_OPTIONS
            and key not in {"Цвет насыщения", "Цвет литологии"}
            and key not in FACIES_REFERENCE_FIELDS
        }
        self.settings = QSettings("DeepCore", "DeepCore2")
        self.attribute_combos: dict[str, QComboBox] = {}

        layout = QVBoxLayout(self)
        confidence_label = QLabel(f"Уверенность модели: {confidence:.0%}")
        confidence_label.setStyleSheet("color: #687087;")
        layout.addWidget(confidence_label)

        form_container = QWidget()
        form = QFormLayout(form_container)
        form.setSpacing(10)
        self.facies = QComboBox()
        known_codes: set[str] = set()
        for item in FACIES_CATALOG:
            code = item["Код фации"]
            if code.casefold() in known_codes:
                continue
            known_codes.add(code.casefold())
            self.facies.addItem(facies_title(item), code)
        self.facies.addItem("Самостоятельный выбор…", CUSTOM_VALUE)
        if not current_facies or current_facies == "Новый контур":
            current_facies = str(self.settings.value("last/facies", "")).strip()
        current_index = self.facies.findData(current_facies)
        if current_facies and current_index < 0:
            self.facies.insertItem(0, current_facies, current_facies)
            current_index = 0
        if current_index >= 0:
            self.facies.setCurrentIndex(current_index)
        self.facies.activated.connect(self._choose_custom_facies)
        self.facies.setToolTip("Выберите код из справочника или «Самостоятельный выбор» для своего кода.")
        form.addRow("Фация (основная метка):", self.facies)
        self.depth_from = QLineEdit("" if depth_from is None else self._format_depth(depth_from))
        self.depth_from.setPlaceholderText("например, 2450.5")
        self.depth_to = QLineEdit("" if depth_to is None else self._format_depth(depth_to))
        self.depth_to.setPlaceholderText("например, 2451.2")
        form.addRow("Начало слоя, м:", self.depth_from)
        form.addRow("Конец слоя, м:", self.depth_to)
        for field_name, options in LITHOLOGY_ATTRIBUTE_OPTIONS.items():
            combo = QComboBox()
            combo.addItem("— не выбрано —", "")
            for option in options:
                combo.addItem(option, option)
            combo.addItem("Самостоятельный выбор…", CUSTOM_VALUE)
            value = str(attributes.get(field_name) or "").strip()
            if field_name == "Флюидонасыщение" and not value:
                value = str(self.settings.value("last/saturation", "")).strip()
            value_index = combo.findData(value)
            if value and value_index < 0:
                combo.insertItem(combo.count() - 1, value, value)
                value_index = combo.count() - 2
            combo.setCurrentIndex(value_index if value_index >= 0 else 0)
            combo.activated.connect(lambda index, field=field_name, widget=combo: self._choose_custom_attribute(field, widget, index))
            self.attribute_combos[field_name] = combo
            form.addRow(f"{field_name}:", combo)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_container)
        layout.addWidget(scroll, 1)

        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        delete_button = QPushButton("Удалить")
        delete_button.setStyleSheet("color: #b33b4d;")
        actions.addButton(delete_button, QDialogButtonBox.ButtonRole.DestructiveRole)
        actions.accepted.connect(self.accept)
        actions.rejected.connect(self.reject)
        delete_button.clicked.connect(self._request_delete)
        layout.addWidget(actions)

    def selected_facies(self) -> str:
        return str(self.facies.currentData() or self.facies.currentText()).strip()

    def selected_lithology(self) -> str:
        """Compatibility alias for code that used the old dialog API."""
        return self.selected_facies()

    def selected_attributes(self) -> dict[str, str]:
        values = facies_metadata(self.selected_facies())
        values.setdefault("Код фации", self.selected_facies())
        values.update(self._preserved_attributes)
        values.update({
            field_name: combo.currentText()
            for field_name, combo in self.attribute_combos.items()
            if combo.currentData() not in {"", CUSTOM_VALUE} and combo.currentText() != "— не выбрано —"
        })
        self.settings.setValue("last/facies", self.selected_facies())
        saturation = str(values.get("Флюидонасыщение") or "")
        if saturation:
            self.settings.setValue("last/saturation", saturation)
        return values

    def _choose_custom_facies(self, index: int) -> None:
        if self.facies.itemData(index) != CUSTOM_VALUE:
            return
        code, accepted = QInputDialog.getText(self, "Своя фация", "Введите код фации:")
        code = code.strip()
        if not accepted or not code:
            self.facies.setCurrentIndex(0 if self.facies.count() > 1 else -1)
            return
        existing = self.facies.findData(code)
        if existing < 0:
            self.facies.insertItem(self.facies.count() - 1, code, code)
            existing = self.facies.count() - 2
        self.facies.setCurrentIndex(existing)

    def _choose_custom_attribute(self, field_name: str, combo: QComboBox, index: int) -> None:
        if combo.itemData(index) != CUSTOM_VALUE:
            return
        value, accepted = QInputDialog.getText(self, f"{field_name}: свой вариант", "Введите значение:")
        value = value.strip()
        if not accepted or not value:
            combo.setCurrentIndex(0)
            return
        existing = combo.findData(value)
        if existing < 0:
            combo.insertItem(combo.count() - 1, value, value)
            existing = combo.count() - 2
        combo.setCurrentIndex(existing)

    def selected_depth_range(self) -> tuple[float | None, float | None]:
        depth_from, depth_to = self._parse_depth(self.depth_from.text()), self._parse_depth(self.depth_to.text())
        if (depth_from is None) != (depth_to is None):
            raise ValueError("Укажите обе границы интервала или очистите оба поля")
        if depth_from is not None and depth_to is not None and depth_to <= depth_from:
            raise ValueError("Конец слоя должен быть больше начала")
        return depth_from, depth_to

    @staticmethod
    def _parse_depth(value: str) -> float | None:
        value = str(value or "").strip().replace(",", ".")
        if not value:
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError("Глубина должна быть числом") from exc

    @staticmethod
    def _format_depth(value: float) -> str:
        return f"{float(value):g}"

    def _request_delete(self) -> None:
        self.delete_requested.emit()
        self.reject()
