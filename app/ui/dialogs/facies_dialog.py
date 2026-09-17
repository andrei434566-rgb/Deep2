from __future__ import annotations

import json

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget

from app.domain.facies_catalog import FACIES_MODEL_CLASSES, FACIES_REFERENCE_FIELDS, facies_metadata, facies_title, resolve_facies_class
from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS


CUSTOM_VALUE = "__kern_analyzer_custom_value__"
FACIES_METADATA_ROLE = Qt.ItemDataRole.UserRole + 1


class FaciesDialog(QDialog):
    delete_requested = Signal()
    move_up_requested = Signal()
    move_down_requested = Signal()

    def __init__(
        self,
        current_facies: str,
        confidence: float,
        attributes: dict[str, str] | None = None,
        depth_from: float | None = None,
        depth_to: float | None = None,
        parent=None,
        *,
        training_ready: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("Параметры фации")
        self.setMinimumWidth(345)
        self.resize(510, 740)
        attributes = dict(attributes or {})
        requested_index = str(attributes.get("Индекс фации") or "").strip()
        if current_facies and current_facies != "Новый контур":
            # Imported project values remain authoritative over catalogue
            # defaults, which lets an interpreter correct a single layer.
            reference = facies_metadata(current_facies, requested_index)
            reference.update(attributes)
            attributes = reference
        self._preserved_attributes = {
            key: value
            for key, value in attributes.items()
            if key not in LITHOLOGY_ATTRIBUTE_OPTIONS
            and key not in {"Цвет насыщения", "Цвет литологии"}
            and key not in FACIES_REFERENCE_FIELDS
        }
        self.settings = QSettings("Kern Analyzer", "KernAnalyzer")
        self.attribute_combos: dict[str, QComboBox] = {}

        layout = QVBoxLayout(self)
        confidence_label = QLabel(f"Оценка модели: {confidence:.0%} (не вероятность правильности)")
        confidence_label.setStyleSheet("color: #687087;")
        layout.addWidget(confidence_label)

        form_container = QWidget()
        form = QFormLayout(form_container)
        form.setSpacing(10)
        self.facies = QComboBox()
        self.facies.addItem("— фация не выбрана —", "")
        for model_class in FACIES_MODEL_CLASSES:
            item = model_class["metadata"]
            code = item["Код фации"]
            self.facies.addItem(facies_title(item), code)
            self.facies.setItemData(self.facies.count() - 1, dict(item), FACIES_METADATA_ROLE)
        self.facies.addItem("Самостоятельный выбор…", CUSTOM_VALUE)
        resolved = resolve_facies_class(current_facies, requested_index)
        if resolved:
            current_facies, requested_index = resolved["code"], resolved["index"]
        elif current_facies == "Новый контур":
            current_facies = ""
        current_index = next(
            (
                index
                for index in range(self.facies.count())
                if self.facies.itemData(index) == current_facies and (resolved or not current_facies)
                and (
                    not requested_index
                    or str((self.facies.itemData(index, FACIES_METADATA_ROLE) or {}).get("Индекс фации") or "")
                    == requested_index
                )
            ),
            -1,
        )
        if current_facies and current_index < 0:
            self.facies.insertItem(0, f"{current_facies} · уточните по справочнику", current_facies)
            current_index = 0
        if current_index >= 0:
            self.facies.setCurrentIndex(current_index)
        self.facies.activated.connect(self._choose_custom_facies)
        self.facies.currentIndexChanged.connect(self._update_facies_reference)
        self.facies.setToolTip("Выберите код из справочника или «Самостоятельный выбор» для своего кода.")
        search = QLineEdit()
        search.setPlaceholderText("Код, индекс или часть названия…")
        search.textChanged.connect(self._filter_facies)
        form.addRow("Поиск фации:", search)
        form.addRow("Фация (основная метка):", self.facies)
        self.facies_reference = QLabel()
        self.facies_reference.setWordWrap(True)
        self.facies_reference.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.facies_reference.setStyleSheet("color: #596174; background: #f5f7fa; padding: 7px; border-radius: 6px;")
        form.addRow("Определение:", self.facies_reference)
        self._update_facies_reference()
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

        self.training_confirmed = QCheckBox("Фация и границы проверены — использовать в обучении")
        self.training_confirmed.setChecked(training_ready)
        self.training_confirmed.setToolTip("Отметьте после проверки класса и контура. Сохранить черновик можно без этой отметки.")
        layout.addWidget(self.training_confirmed)
        try:
            suggestions = json.loads(str(attributes.get("__attribute_suggestions") or "{}"))
        except (ValueError, TypeError):
            suggestions = {}
        if isinstance(suggestions, dict) and suggestions:
            suggested = QLabel("Визуальные подсказки (не подтверждены): " + "; ".join(f"{k}: {v}" for k, v in suggestions.items()))
            suggested.setWordWrap(True)
            layout.addWidget(suggested)
            apply_suggestions = QPushButton("Перенести подсказки в пустые поля")
            apply_suggestions.clicked.connect(lambda: self._apply_suggestions(suggestions))
            layout.addWidget(apply_suggestions)

        order_row = QHBoxLayout()
        move_up = QPushButton("↑ Слой выше")
        move_up.setObjectName("moveFaciesLayerUp")
        move_up.setToolTip("Поднять этот слой на одну позицию в собранном столбике керна")
        move_down = QPushButton("↓ Слой ниже")
        move_down.setObjectName("moveFaciesLayerDown")
        move_down.setToolTip("Опустить этот слой на одну позицию в собранном столбике керна")
        move_up.clicked.connect(self.move_up_requested)
        move_down.clicked.connect(self.move_down_requested)
        order_row.addWidget(move_up)
        order_row.addWidget(move_down)
        layout.addLayout(order_row)

        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        delete_button = QPushButton("Удалить")
        delete_button.setStyleSheet("color: #b33b4d;")
        actions.addButton(delete_button, QDialogButtonBox.ButtonRole.DestructiveRole)
        actions.accepted.connect(self.accept)
        actions.rejected.connect(self.reject)
        delete_button.clicked.connect(self._request_delete)
        layout.addWidget(actions)

    def selected_facies(self) -> str:
        value = self.facies.currentData()
        return "" if value in (None, "", CUSTOM_VALUE) else str(value).strip()

    def _filter_facies(self, text: str) -> None:
        query = text.strip().casefold()
        for index in range(self.facies.count()):
            self.facies.view().setRowHidden(index, bool(query and query not in self.facies.itemText(index).casefold()))

    def _apply_suggestions(self, suggestions: dict) -> None:
        for field, value in suggestions.items():
            combo = self.attribute_combos.get(field)
            if combo is None or combo.currentData():
                continue
            index = combo.findData(value)
            if index < 0:
                combo.insertItem(combo.count() - 1, str(value), str(value))
                index = combo.count() - 2
            combo.setCurrentIndex(index)

    def selected_lithology(self) -> str:
        """Compatibility alias for code that used the old dialog API."""
        return self.selected_facies()

    def selected_attributes(self) -> dict[str, str]:
        selected_reference = self.facies.currentData(FACIES_METADATA_ROLE)
        values = dict(selected_reference) if isinstance(selected_reference, dict) else facies_metadata(self.selected_facies())
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

    def _update_facies_reference(self, _index: int = -1) -> None:
        metadata = self.facies.currentData(FACIES_METADATA_ROLE)
        if not isinstance(metadata, dict):
            self.facies_reference.setText("Метка не сопоставлена справочнику. Для финальной модели выберите точный код и индекс. Наблюдения по слою можно сохранить отдельно.")
            return
        rows = [
            ("Обстановка", metadata.get("Обстановка седиментации")),
            ("Ассоциация", metadata.get("Фациальная ассоциация")),
            ("Энергия", metadata.get("Энергия среды")),
            ("Режим", metadata.get("Гидродинамический режим")),
            ("Литотипы", metadata.get("Предполагаемые литотипы")),
        ]
        self.facies_reference.setText("Справочная характеристика, не измерение по фото:\n" + "\n".join(f"{title}: {value}" for title, value in rows if value))

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
