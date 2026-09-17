"""Model-independent catalogue and annotation readiness view."""

from collections import Counter, defaultdict
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout,
)

from app.domain.facies_catalog import FACIES_MODEL_CLASSES, FACIES_REFERENCE_SOURCE, resolve_facies_class
from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS


class ModelPreparationDialog(QDialog):
    check_requested = Signal()
    blueprint_requested = Signal()
    dataset_requested = Signal()
    review_requested = Signal()

    def __init__(self, records, model_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Подготовка модели фаций")
        self.resize(1000, 740)
        layout = QVBoxLayout(self)
        heading = QLabel(
            f"Справочник: {len(FACIES_MODEL_CLASSES)} классов (код + индекс). "
            + (f"Выбрана модель: {Path(model_path).name}." if model_path else "Модель не подключена — подготовка доступна.")
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)
        intro = QLabel("1. Проверьте границы и фации → 2. Проверьте выборку → 3. Экспортируйте или дообучите. "
                       "Справочник задаёт названия, но не обучает распознаванию. Дисбаланс в скважине допустим.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        counts, fields, photos = Counter(), Counter(), defaultdict(set)
        unresolved, pending = Counter(), 0
        for record in records:
            for detection in record.detections:
                if not detection.training_ready:
                    pending += 1
                    continue
                mapped = resolve_facies_class(detection.label, detection.attributes.get("Индекс фации"))
                if mapped is None:
                    unresolved[f"{detection.label} [{detection.attributes.get('Индекс фации', 'без индекса')}]"] += 1
                    continue
                key = mapped["model_label"]
                counts[key] += 1
                photos[key].add(record.identifier)
                fields[key] += sum(bool(detection.attributes.get(field)) for field in LITHOLOGY_ATTRIBUTE_OPTIONS)
        search = QLineEdit()
        search.setPlaceholderText("Поиск по коду, индексу, названию или статусу…")
        layout.addWidget(search)
        self.table = QTableWidget(len(FACIES_MODEL_CLASSES), 6)
        self.table.setHorizontalHeaderLabels(["Код / индекс", "Название", "Проверено", "Фото*", "Поля из 16", "Статус данных"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().hide()
        for row, item in enumerate(FACIES_MODEL_CLASSES):
            key, count = item["model_label"], counts[item["model_label"]]
            values = [f"{item['code']} / {item['index']}", item["name"], str(count), str(len(photos[key])),
                      f"{fields[key] / count:.1f}" if count else "—",
                      "Нет примеров" if not count else "Мало примеров" if count < 5 else "Нужна проверка выборки"]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setToolTip("\n".join(f"{k}: {v}" for k, v in item["metadata"].items()))
                self.table.setItem(row, col, cell)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        search.textChanged.connect(self._filter)
        layout.addWidget(self.table, 1)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(185)
        notes = [f"Источник: {FACIES_REFERENCE_SOURCE}",
                 f"Есть проверенные метки для {len(counts)} из {len(FACIES_MODEL_CLASSES)} классов. Ожидают проверки: {pending}.",
                 "* Пока показаны записи фото. Проверка выборки выявляет повторные изображения и разделяет обучение/контроль.",
                 "16 параметров — отдельные наблюдения. Их распознавание требует своей разметки и оценки; описание фации не заменяет измерения."]
        if unresolved:
            notes.append("Нужны точные метки из справочника: " + "; ".join(f"{k}: {v}" for k, v in unresolved.items()))
        self.details.setPlainText("\n\n".join(notes))
        layout.addWidget(self.details)
        actions = QHBoxLayout()
        for title, signal in (("Проверить выборку", self.check_requested),
                              ("Исправить следующий слой", self.review_requested),
                              ("Экспорт заготовки…", self.blueprint_requested),
                              ("Экспорт датасета…", self.dataset_requested)):
            button = QPushButton(title)
            button.clicked.connect(signal.emit)
            actions.addWidget(button)
        layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _filter(self, query):
        query = query.strip().casefold()
        for row in range(self.table.rowCount()):
            self.table.setRowHidden(row, not any(query in self.table.item(row, col).text().casefold() for col in range(self.table.columnCount())))

    def show_check(self, report: dict):
        issues = list(report.get("blocking_reasons") or [])
        warnings = list(report.get("warnings") or [])
        lines = ["Экспорт доступен." if report.get("can_export") else "Сначала исправьте выборку.",
                 f"Разделение: {report.get('split_strategy', 'не сформировано')}. "
                 f"Обучающих фото: {report.get('train_photo_count', 0)}, контрольных: {report.get('val_photo_count', 0)}.",
                 f"Исключено повторных примеров: {report.get('duplicates_removed', 0)}."]
        lines += issues + warnings
        self.details.setPlainText("\n\n".join(lines))
