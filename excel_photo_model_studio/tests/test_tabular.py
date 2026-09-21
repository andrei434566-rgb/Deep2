from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from excel_photo_model_studio.tabular import read_table


class TableReaderTests(unittest.TestCase):
    def test_reads_reference_22_column_schema_and_target_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            headers = [
                "Месторождение", "№ скв.", "№ долбления", "Интервал отбора керна, м", "",
                "Стратиграфия", "Смещение по ГИС, м", "Интервал отбора по ГИС, м", "",
                "Проходка, м", "Вынос керна, м", "Вынос %", "Интервал фации по бурению, м", "",
                "Интервал фации по ГИС, м", "", "№ слоя", "Толщина фации, м",
                "Название фации", "Ассоциация фаций", "Обстановка осадконакопления", "Краткое описание",
            ]
            sheet.append(headers)
            for start, end in ((4, 5), (8, 9), (13, 14), (15, 16)):
                sheet.merge_cells(start_row=1, start_column=start, end_row=1, end_column=end)
            sheet.append([None, None, None, "Кровля", "Подошва", None, None, "Кровля", "Подошва", None, None, None, "Кровля", "Подошва", "Кровля", "Подошва"])
            sheet.append(list(range(1, 23)))
            sheet.append([
                "Уренгойское", "Р-31", 1, 3183.0, 3195.0, "К2m", -1, 3182.0, 3194.0,
                3, 3, 100, 3188.98, 3189.50, 3187.98, 3188.50, 1, 0.52,
                "Tcr", "Прибрежная равнина", "Прибрежная равнина", "Песчаник светло-серый, слоистый.",
            ])
            workbook.save(path)

            rows, mappings, _ = read_table(path)

        self.assertEqual(1, len(rows))
        self.assertEqual((4, 5), (mappings[0].core_top, mappings[0].core_base))
        self.assertEqual((13, 14), (mappings[0].top, mappings[0].base))
        self.assertEqual(22, mappings[0].target_text)
        self.assertEqual("Tcr", rows[0].label)
        self.assertEqual("Песчаник светло-серый, слоистый.", rows[0].target_text)

    def test_reads_merged_multiline_headers_and_forward_fills_well(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "description.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Описание"
            sheet["A1"] = "№ скважины"
            sheet["B1"] = "Интервал фации"
            sheet.merge_cells("B1:C1")
            sheet["B2"] = "Кровля"
            sheet["C2"] = "Подошва"
            sheet["D1"] = "Код фации"
            sheet["E1"] = "Индекс фации"
            sheet.append(["Р-31", 3915.0, 3915.5, "Dch", "47"])
            sheet.append([None, 3915.5, 3916.0, "DWCh", "80"])
            workbook.save(path)

            rows, mappings, issues = read_table(path)

        self.assertEqual([], [item for item in issues if item.severity == "error"])
        self.assertEqual(2, len(rows))
        self.assertEqual("Р-31", rows[1].well)
        self.assertEqual("DWCh@80", rows[1].label)
        self.assertEqual(2, mappings[0].header_row)

    def test_finds_drilling_interval_and_short_description_by_header_not_position(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extended_31_columns.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "седимент"
            sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=31)
            sheet.cell(1, 1, "Послойное седиментологическое описание керна")
            headers = [
                "Месторождение", "№ скв.", "№ долбления", "Интервал отбора керна, м", None,
                "Стратиграфия", "Смещение по ГИС, м", "Интервал отбора по ГИС, м", None,
                "Проходка, м", "Вынос керна, м", "Вынос, %", "№", "Интервал фации по бурению, м", None,
                "Интервал фации по ГИС, м", None, "№ слоя", "Толщина фации, м", "Индекс фации",
                "Название фации", "Код фации", "Индекс макрофации", "Название макрофации",
                "Код макрофации", "Индекс ассоциации фаций", "Ассоциация фаций",
                "Обстановка осадконакопления", "Комплекс осадконакопления", "Краткое описание", "Примечание",
            ]
            sheet.append(headers)
            for start, end in ((4, 5), (8, 9), (14, 15), (16, 17)):
                sheet.merge_cells(start_row=2, start_column=start, end_row=2, end_column=end)
            subheaders = [None] * 31
            for start, end in ((4, 5), (8, 9), (14, 15), (16, 17)):
                subheaders[start - 1] = "Кровля"
                subheaders[end - 1] = "Подошва"
            sheet.append(subheaders)
            sheet.append(list(range(1, 32)))
            data = [None] * 31
            values = {
                1: "Восточно-Тазовское", 2: "67ПО", 4: 4105.0, 5: 4117.0,
                14: 4105.0, 15: 4108.65, 16: 4104.90, 17: 4108.55,
                19: 3.65, 20: "Dch", 21: "Каналы распределительные", 22: 92,
                27: "Дельтовая ассоциация", 28: "Дельтовая обстановка",
                30: "Песчаник светло-серый, слоистый.", 31: "Контроль",
            }
            for column, value in values.items():
                data[column - 1] = value
            sheet.append(data)
            workbook.save(path)

            rows, mappings, issues = read_table(path)

        self.assertEqual([], [item for item in issues if item.severity == "error"])
        self.assertEqual((14, 15), (mappings[0].top, mappings[0].base))
        self.assertEqual(19, mappings[0].facies_thickness)
        self.assertEqual(30, mappings[0].target_text)
        self.assertEqual("Песчаник светло-серый, слоистый.", rows[0].target_text)
        self.assertEqual((4105.0, 4108.65), (rows[0].top, rows[0].base))
        self.assertTrue(rows[0].thickness_valid)

    def test_thickness_mismatch_blocks_row_from_masking(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_thickness.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append([
                "Скважина", "Интервал фации по бурению Кровля",
                "Интервал фации по бурению Подошва", "Толщина фации, м", "Класс",
            ])
            sheet.append(["W-1", 100.0, 112.0, 3.65, "Dch"])
            workbook.save(path)

            rows, mappings, issues = read_table(path)

        self.assertEqual((2, 3, 4), (mappings[0].top, mappings[0].base, mappings[0].facies_thickness))
        self.assertEqual(1, len(rows))
        self.assertFalse(rows[0].thickness_valid)
        self.assertTrue(any("не будет использована для маски" in item.message for item in issues))

    def test_reads_csv_with_one_interval_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "description.csv"
            path.write_text("Скважина;Интервал;Класс\nW-1;100,0-101,5;Sand\n", encoding="utf-8")

            rows, _, _ = read_table(path)

        self.assertEqual(1, len(rows))
        self.assertEqual((100.0, 101.5), (rows[0].top, rows[0].base))
        self.assertEqual("Sand", rows[0].label)


if __name__ == "__main__":
    unittest.main()
