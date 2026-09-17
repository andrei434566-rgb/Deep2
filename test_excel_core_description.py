"""Regression tests for flexible Excel-to-facies mapping."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpyxl import Workbook
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication

from app.infrastructure.excel_core_description import (
    CoreInterval,
    DescriptionLayer,
    _attributes_from_description,
    create_depth_bound_detections,
    photo_interval_from_filename,
    read_description_workbook,
    suggest_photo_intervals_from_excel,
    workbook_import_summary,
)


_APP = QApplication.instance() or QApplication([])


class ExcelDescriptionImportTests(unittest.TestCase):
    def _read(self, workbook: Workbook):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "description.xlsx"
            workbook.save(path)
            return read_description_workbook(path)

    def test_reads_merged_multirow_facies_interval_headers(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Описание"
        sheet.merge_cells("A1:A2")
        sheet.merge_cells("B1:B2")
        sheet.merge_cells("C1:D1")
        sheet.merge_cells("E1:E2")
        sheet.merge_cells("F1:F2")
        sheet.merge_cells("G1:G2")
        sheet["A1"] = "№ скв."
        sheet["B1"] = "Интервал фации по бурению, м"
        sheet["C1"] = "Интервал фации по бурению, м"
        sheet["C2"] = "Кровля"
        sheet["D2"] = "Подошва"
        sheet["E1"] = "Индекс фации"
        sheet["F1"] = "Название фации"
        sheet["G1"] = "Краткое описание"
        sheet.append(["Р-31", None, 3915.00, 3915.55, "Shelf", "Отложения шельфа", "Алевролит"])
        sheet.append([None, None, 3915.55, 3915.77, "TL", "Трансгрессивный слой", "Песчаник"])

        layers, issues = self._read(workbook)

        self.assertEqual([], issues)
        self.assertEqual(2, len(layers))
        self.assertEqual("Р-31", layers[1].well)
        self.assertEqual("TL", layers[1].facies_index)
        self.assertAlmostEqual(3915.55, layers[1].top)

    def test_reads_generic_from_to_columns_and_sheet_well(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Скв. Р-31"
        sheet.append(["Глубина от, м", "Глубина до, м", "Фациальный индекс", "Литофация", "Характеристика"])
        sheet.append([3002.0, 3002.45, "L", "Лагуна", "Тонкозернистый песчаник"])

        layers, issues = self._read(workbook)

        self.assertEqual([], issues)
        self.assertEqual(1, len(layers))
        self.assertEqual("Скв. Р-31", layers[0].well)
        self.assertEqual("L", layers[0].facies_index)
        self.assertEqual("Лагуна", layers[0].facies_name)

    def test_reads_single_range_column_and_does_not_import_header_numbers(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["Скважина", "Интервал фации, м", "Код фации", "Описание"])
        sheet.append([1, 2, 3, 4])
        sheet.append(["12", "1800,0–1800,5", "A", "Глинистый алевролит"])

        layers, issues = self._read(workbook)

        self.assertEqual([], issues)
        self.assertEqual(1, len(layers))
        self.assertEqual("A", layers[0].facies_code)
        self.assertAlmostEqual(1800.5, layers[0].base)

    def test_reads_short_free_form_description_and_common_attribute_headers(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Р-41"
        sheet.append(["Верх", "Низ", "Порода", "Окраска", "Описание"])
        sheet.append([2500.0, 2500.4, "Песчаник", "Серый", "Цемент: карбонатный; слоистость: волнистая"])

        layers, issues = self._read(workbook)

        self.assertEqual(1, len(layers))
        self.assertEqual("Песчаник", layers[0].facies_name)
        self.assertTrue(issues)
        attributes = _attributes_from_description(layers[0])
        self.assertEqual("Песчаник", attributes["Название породы"])
        self.assertEqual("Серый", attributes["Цвет"])
        self.assertEqual("карбонатный", attributes["Цемент"])

    def test_reads_interval_from_flexible_photo_filename(self):
        interval = photo_interval_from_filename(Path("Р-31 3002,00–3004,96 (1).jpg"))

        self.assertEqual("Р-31", interval.well)
        self.assertAlmostEqual(3002.0, interval.top)
        self.assertAlmostEqual(3004.96, interval.base)

    def test_reads_field_core_interval_facies_thickness_and_16_parameters(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Описание"
        headers = [
            "Месторождение", "№ скважины", "Отбор керна Кровля", "Отбор керна Подошва",
            "Интервал фации Кровля", "Интервал фации Подошва", "Толщина фации",
            "Название фации", "Индекс фации", "Литологическое описание (16 параметров)",
            "Цемент", "Степень цементации", "Контакт", "Ориентация контакта",
        ]
        sheet.append(headers)
        sheet.append([
            "Северное", "Р-17", 2500.0, 2505.0, 2500.0, 2501.25, 1.25,
            "Русловая", "80", "Порода: песчаник; цвет: серый", "карбонатный",
            "сильная", "резкий", "наклонный",
        ])
        sheet.append([
            None, None, None, None, 2501.25, 2503.0, 9.0,
            "Пойменная", "81", "Порода: алевролит; цвет: бурый", "глинистый",
            "слабая", "постепенный", "горизонтальный",
        ])

        layers, issues = self._read(workbook)

        self.assertEqual(2, len(layers))
        self.assertEqual("Северное", layers[1].field_name)
        self.assertEqual("Р-17", layers[1].well)
        self.assertEqual((2500.0, 2505.0), (layers[1].core_top, layers[1].core_base))
        self.assertAlmostEqual(1.75, layers[1].base - layers[1].top)
        self.assertEqual(9.0, layers[1].thickness)
        self.assertTrue(any("не совпадает" in issue.message for issue in issues))
        attributes = _attributes_from_description(layers[0])
        self.assertEqual("карбонатный", attributes["Цемент"])
        self.assertEqual("сильная", attributes["Степень цементации"])
        self.assertEqual("резкий", attributes["Контакт"])
        self.assertEqual("наклонный", attributes["Ориентация контакта"])
        self.assertEqual("Северное", attributes["Месторождение"])
        self.assertEqual("2500–2505 м", attributes["Интервал керна"])
        summary = workbook_import_summary(layers)
        self.assertEqual(["Северное"], summary["fields"])
        self.assertEqual(2, summary["facies_layers"])

    def test_excel_prefills_missing_photo_intervals_but_marks_confirmation(self):
        layers = [
            DescriptionLayer(
                well="Р-17", top=2500.0, base=2501.0, facies_name="Фация", facies_code="",
                facies_index="80", description="", attributes={}, sheet="Лист1", row=2,
                field_name="Северное", core_top=2500.0, core_base=2504.0,
            )
        ]
        rows = [
            (Path("photo_1.jpg"), None, "не найден"),
            (Path("photo_2.jpg"), None, "не найден"),
        ]

        suggested = suggest_photo_intervals_from_excel(rows, layers)

        self.assertEqual(CoreInterval("Р-17", 2500.0, 2502.0, "Северное"), suggested[0][1])
        self.assertEqual(CoreInterval("Р-17", 2502.0, 2504.0, "Северное"), suggested[1][1])
        self.assertIn("обязательно проверьте", suggested[0][2])

    def test_projected_mask_height_follows_real_layer_thickness(self):
        layers = [
            DescriptionLayer("W-1", 100.0, 102.0, "A", "", "80", "", {}, "S", 2),
            DescriptionLayer("W-1", 102.0, 105.0, "B", "", "81", "", {}, "S", 3),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "core.jpg"
            image = QImage(100, 200, QImage.Format.Format_RGB32)
            image.fill(QColor("#777777"))
            self.assertTrue(image.save(str(path), "JPG"))
            pixmap = QPixmap(str(path))
            columns = [{"left": 0.0, "top": 0.0, "right": 100.0, "bottom": 200.0}]
            with patch(
                "app.infrastructure.excel_core_description.CoreColumnRecognizer.recognize",
                return_value=columns,
            ):
                detections, issues, _ = create_depth_bound_detections(path, pixmap, 100.0, 105.0, layers)

        self.assertEqual([], issues)
        self.assertEqual(2, len(detections))
        heights = [max(p.y() for p in item.polygon) - min(p.y() for p in item.polygon) for item in detections]
        self.assertAlmostEqual(80.0, heights[0], delta=1.0)
        self.assertAlmostEqual(120.0, heights[1], delta=1.0)
        self.assertFalse(any(item.training_ready for item in detections))
        self.assertEqual((100.0, 102.0), (detections[0].depth_from, detections[0].depth_to))


if __name__ == "__main__":
    unittest.main()
