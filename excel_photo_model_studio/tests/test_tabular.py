from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from openpyxl import Workbook, load_workbook

from excel_photo_model_studio.depth import meters_to_centimeters
from excel_photo_model_studio.tabular import (
    _expand_merged_ranges, as_float, detect_mapping, parse_interval, read_table,
    read_many_tables, read_workbook_sheets, save_mappings,
)


class TableReaderTests(unittest.TestCase):
    def test_detects_headers_in_the_third_row_and_starts_data_after_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "third_row_header.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["Отчёт по скважине"])
            sheet.append(["Сформировано автоматически"])
            sheet.append([
                "Скважина", "Интервал фации по бурению Кровля",
                "Интервал фации по бурению Подошва", "Индекс фации", "Краткое описание",
            ])
            sheet.append(["W-1", 100.0, 101.25, "Dch", "Песчаник серый."])
            workbook.save(path)

            rows, mappings, issues = read_table(path)

        self.assertEqual(3, mappings[0].header_row)
        self.assertEqual(1, len(rows))
        self.assertEqual((100.0, 101.25), (rows[0].top, rows[0].base))
        self.assertEqual([], [item for item in issues if item.severity == "error"])

    def test_detects_flat_header_in_the_fourth_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fourth_row_header.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["Отчёт по керну"])
            sheet.append(["Сформировано автоматически"])
            sheet.append(["Версия таблицы 2"])
            sheet.append([
                "Скважина", "Интервал фации по бурению Кровля",
                "Интервал фации по бурению Подошва", "Индекс фации", "Краткое описание",
            ])
            sheet.append(["W-1", 100.0, 101.25, "Dch", "Песчаник серый."])
            workbook.save(path)

            rows, mappings, issues = read_table(path)

        self.assertEqual(4, mappings[0].header_row)
        self.assertEqual(1, len(rows))
        self.assertEqual((100.0, 101.25), (rows[0].top, rows[0].base))
        self.assertFalse([issue for issue in issues if issue.severity == "error"])

    def test_header_group_in_first_three_rows_can_have_subheaders_on_row_four(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "row_four_subheader.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "седимент"
            sheet["A1"] = "Послойное описание керна"
            sheet["A3"] = "Скважина"
            sheet["B3"] = "Интервал фации по бурению, м"
            sheet.merge_cells("B3:C3")
            sheet["D3"] = "Индекс фации"
            sheet["E3"] = "Краткое описание"
            sheet["B4"] = "Кровля"
            sheet["C4"] = "Подошва"
            sheet.append([1, 2, 3, 4, 5])  # Excel's column-number ruler
            sheet.append(["W-1", 4105.0, 4105.25, "Dch", "Песчаник серый."])
            workbook.save(path)

            rows, mappings, issues = read_table(path)

        self.assertEqual(4, mappings[0].header_row)
        self.assertEqual((2, 3), (mappings[0].top, mappings[0].base))
        self.assertEqual(1, len(rows))
        self.assertEqual((4105.0, 4105.25), (rows[0].top, rows[0].base))
        self.assertEqual("Dch", rows[0].facies_index)
        self.assertEqual("Песчаник серый.", rows[0].target_text)
        self.assertFalse([issue for issue in issues if issue.severity == "error"])

    def test_multilevel_header_can_span_four_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "four_row_header.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "седимент"
            sheet["A1"] = "Скважина"
            sheet["B1"] = "Интервал фации по бурению, м"
            sheet.merge_cells("B1:C1")
            sheet["D1"] = "Индекс фации"
            sheet["E1"] = "Краткое описание"
            sheet["B4"] = "Кровля"
            sheet["C4"] = "Подошва"
            sheet.append([1, 2, 3, 4, 5])  # Excel's column-number ruler on row 5
            sheet.append(["W-1", 4105.0, 4105.25, "Dch", "Песчаник серый."])
            workbook.save(path)

            rows, mappings, issues = read_table(path)

        self.assertEqual(4, mappings[0].header_row)
        self.assertEqual((2, 3), (mappings[0].top, mappings[0].base))
        self.assertEqual(1, len(rows))
        self.assertEqual((4105.0, 4105.25), (rows[0].top, rows[0].base))
        self.assertEqual("Dch", rows[0].facies_index)
        self.assertFalse([issue for issue in issues if issue.severity == "error"])

    def test_date_number_format_does_not_hide_numeric_depths_or_zero_class_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "date_formatted_depths.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append([
                "Скважина", "Интервал фации по бурению Кровля",
                "Интервал фации по бурению Подошва", "Индекс фации", "Краткое описание",
            ])
            sheet.append(["W-1", 4105.0, 4105.25, 0, "Песчаник серый."])
            sheet["B2"].number_format = "yyyy-mm-dd"
            sheet["C2"].number_format = "yyyy-mm-dd"
            workbook.save(path)

            rows, _mappings, issues = read_table(path)

        self.assertEqual(1, len(rows))
        self.assertEqual((4105.0, 4105.25), (rows[0].top, rows[0].base))
        self.assertEqual("0", rows[0].facies_index)
        self.assertEqual("0", rows[0].label)
        self.assertFalse([issue for issue in issues if issue.severity == "error"])

    def test_xls_merged_interval_headers_are_expanded_before_mapping(self):
        rows = [
            ["Скважина", "Интервал фации по бурению", "", "Индекс фации", "Краткое описание"],
            ["", "Кровля", "Подошва", "", ""],
            ["W-1", 4105.0, 4105.25, "Dch", "Песчаник серый."],
        ]
        _expand_merged_ranges(rows, [(0, 1, 1, 3)])
        mapping = detect_mapping("седимент", rows)

        self.assertEqual("Интервал фации по бурению", rows[0][2])
        self.assertEqual((2, 3), (mapping.top, mapping.base))

    def test_missing_excel_formula_cache_is_recalculated_on_a_temporary_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formula_values.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append([
                "Скважина", "Интервал фации по бурению Кровля",
                "Интервал фации по бурению Подошва", "Индекс фации", "Краткое описание",
            ])
            sheet.append(["W-1", "=100+0", "=100+1.25", "Dch", "Песчаник серый."])
            workbook.save(path)
            workbook.close()

            def recalculate_temporary_copy(temporary_path: Path) -> bool:
                recalculated = load_workbook(temporary_path, data_only=False)
                recalculated.active["B2"] = 100.0
                recalculated.active["C2"] = 101.25
                recalculated.save(temporary_path)
                recalculated.close()
                return True

            with patch(
                "excel_photo_model_studio.tabular._recalculate_with_excel",
                side_effect=recalculate_temporary_copy,
            ) as recalculate:
                rows, _mappings, issues = read_table(path)

            original = load_workbook(path, data_only=False)
            original_formulas = (original.active["B2"].value, original.active["C2"].value)
            original.close()

        self.assertEqual(1, recalculate.call_count)
        self.assertEqual((100.0, 101.25), (rows[0].top, rows[0].base))
        self.assertEqual(("=100+0", "=100+1.25"), original_formulas)
        self.assertFalse(any("формулы" in item.message.casefold() for item in issues))

    def test_uncalculated_formula_rows_report_exact_cells_instead_of_silent_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "uncalculated.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append([
                "Скважина", "Интервал фации по бурению Кровля",
                "Интервал фации по бурению Подошва", "Индекс фации", "Краткое описание",
            ])
            sheet.append(["W-1", "=A2+99", "=A2+100", "Dch", "Песчаник серый."])
            workbook.save(path)
            workbook.close()

            with patch("excel_photo_model_studio.tabular._recalculate_with_excel", return_value=False):
                with self.assertRaisesRegex(ValueError, r"Sheet!2: Формулы.*B2.*C2"):
                    read_table(path)

    def test_row_with_missing_description_is_retained_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_description.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append([
                "Скважина", "Интервал фации по бурению Кровля",
                "Интервал фации по бурению Подошва", "Индекс фации", "Краткое описание",
            ])
            sheet.append(["W-1", 100.0, 101.0, "Dch", None])
            workbook.save(path)

            rows, _mappings, issues = read_table(path)

        self.assertEqual(1, len(rows))
        self.assertEqual("Dch", rows[0].facies_index)
        self.assertTrue(any("Краткое описание" in item.message for item in issues))

    def test_zero_parsed_rows_error_includes_exact_excel_row_cause(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_facies_index.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append([
                "Скважина", "Интервал фации по бурению Кровля",
                "Интервал фации по бурению Подошва", "Индекс фации",
            ])
            sheet.append(["W-1", 100.0, 101.0, None])
            workbook.save(path)

            with self.assertRaisesRegex(ValueError, r"Sheet!2: Нет метки класса"):
                read_many_tables(path)

    def test_source_sheet_preview_preserves_excel_cells_for_mapping_highlight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "седимент"
            sheet.append(["Интервал фации", "Краткое описание"])
            sheet.append(["4105,00–4106,00", "Песчаник"])
            workbook.save(path)
            sheets = read_workbook_sheets(path)

        self.assertEqual(["седимент"], [name for name, _rows in sheets])
        self.assertEqual("Интервал фации", sheets[0][1][0][0])
        self.assertEqual("Песчаник", sheets[0][1][1][1])

    def test_excel_depths_accept_comma_dot_and_grouped_decimal_formats(self):
        expected = 4105.25
        for value in (4105.25, "4105.25", "4105,25", "4.105,25", "4,105.25", "4 105,25"):
            with self.subTest(value=value):
                self.assertEqual(expected, as_float(value))
                self.assertEqual(410525, meters_to_centimeters(value))
        self.assertEqual(0.96, as_float(".96"))
        self.assertEqual(0.96, as_float(",96"))

    def test_facies_thickness_is_exact_to_one_centimetre_without_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exact_thickness.csv"
            path.write_text(
                "Скважина;Интервал фации по бурению Кровля;Интервал фации по бурению Подошва;"
                "Толщина фации;Индекс фации;Название фации;Краткое описание\n"
                "W-1;100;103,03;3.03;Dch;Каналы;Описание\n"
                "W-1;103,03;106,03;3;Inbay;Заливы;Описание\n"
                "W-1;106,03;109,06;3.02;Mstf;Устья;Описание\n", encoding="utf-8",
            )
            rows, _, _ = read_table(path)

        self.assertTrue(rows[0].thickness_valid)
        self.assertTrue(rows[1].thickness_valid)
        self.assertFalse(rows[2].thickness_valid)

    def test_facies_index_and_name_are_kept_as_separate_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "facies_targets.csv"
            path.write_text(
                "Скважина;Интервал фации по бурению Кровля;Интервал фации по бурению Подошва;"
                "Индекс фации;Название фации;Краткое описание\n"
                "W-1;100;101;Dch;Каналы распределительные;Песчаник\n", encoding="utf-8",
            )
            rows, _, _ = read_table(path)

        self.assertEqual("Dch", rows[0].facies_index)
        self.assertEqual("Каналы распределительные", rows[0].facies_name)
        self.assertEqual("Dch", rows[0].label)

    def test_interval_parser_does_not_split_decimal_comma_or_dot(self):
        for value in ("4105.00-4108.65", "4105,00–4108,65", "4.105,00-4.108,65"):
            with self.subTest(value=value):
                self.assertEqual((4105.0, 4108.65), parse_interval(value))

    def test_missing_drilling_cells_do_not_drop_valid_gis_facies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_drilling.csv"
            path.write_text(
                "Скважина;Интервал фации по бурению Кровля;Интервал фации по бурению Подошва;"
                "Интервал фации по ГИС Кровля;Интервал фации по ГИС Подошва;"
                "Толщина фации;Код фации;Краткое описание\n"
                "W-1;;;200;201;1;A;Песчаник.\n", encoding="utf-8",
            )
            rows, _, issues = read_table(path)

        self.assertEqual(1, len(rows))
        self.assertEqual((200, 201), (rows[0].top, rows[0].base))
        self.assertEqual("gis", rows[0].metadata["interval_source"])
        self.assertEqual("Песчаник.", rows[0].target_text)
        self.assertFalse(any(item.severity == "error" for item in issues))

    def test_combined_drilling_interval_is_not_displaced_by_gis_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.csv"
            path.write_text(
                "Скважина;Интервал фации по бурению;Интервал фации по ГИС Кровля;"
                "Интервал фации по ГИС Подошва;Толщина фации;Код фации\n"
                "W-1;100-101;200;201;1;A\n", encoding="utf-8",
            )
            rows, mappings, _ = read_table(path)

        self.assertEqual(2, mappings[0].interval)
        self.assertEqual((100, 101), (rows[0].top, rows[0].base))
        self.assertEqual((200, 201), (rows[0].gis_top, rows[0].gis_base))

    def test_merged_title_does_not_turn_facies_pair_into_core_sampling_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reordered.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["Послойное описание керна"])
            sheet.merge_cells("A1:H1")
            sheet.append([
                "Скважина", "Интервал фации по бурению Кровля", "Интервал фации по бурению Подошва",
                "Код фации", "Интервал отбора керна Кровля", "Интервал отбора керна Подошва",
                "Толщина фации", "Краткое описание",
            ])
            sheet.append(["W-1", 100, 101, "A", 100, 110, 1, "Песчаник."])
            sheet.append(["W-2", 200, 201, "B", None, None, 1, "Алевролит."])
            workbook.save(path)
            rows, mappings, _ = read_table(path)

        self.assertEqual((5, 6), (mappings[0].core_top, mappings[0].core_base))
        self.assertEqual((2, 3), (mappings[0].top, mappings[0].base))
        self.assertIsNone(rows[1].core_top)
        self.assertIsNone(rows[1].core_base)

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
        self.assertEqual("80", rows[1].label)
        self.assertEqual("80", rows[1].facies_index)
        self.assertEqual("DWCh", rows[1].facies_name)
        self.assertEqual(2, mappings[0].header_row)

    def test_facies_targets_follow_header_names_after_source_columns_are_reordered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reordered.xlsx"
            mapping_file = Path(directory) / "column_mapping.json"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Описание"
            sheet.append([
                "Скважина", "Интервал фации по бурению Кровля",
                "Интервал фации по бурению Подошва", "Индекс фации",
                "Название фации", "Краткое описание", "Толщина фации, м",
            ])
            sheet.append(["W-1", 100.00, 101.25, "Dch", "Каналы", "Гравийный песчаник.", 1.25])
            workbook.save(path)
            _rows, detected, _issues = read_table(path)
            save_mappings(mapping_file, detected)

            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Описание"
            sheet.append([
                "Краткое описание", "Скважина", "Интервал фации по бурению Подошва",
                "Название фации", "Толщина фации, м", "Интервал фации по бурению Кровля",
                "Индекс фации",
            ])
            sheet.append(["Гравийный песчаник.", "W-1", 101.25, "Каналы", 1.25, 100.00, "Dch"])
            workbook.save(path)

            rows, mappings, issues = read_table(path, mapping_file)

        self.assertFalse([issue for issue in issues if issue.severity == "error"])
        self.assertEqual(7, mappings[0].class_index)
        self.assertEqual(4, mappings[0].label)
        self.assertEqual(1, mappings[0].target_text)
        self.assertEqual((6, 3), (mappings[0].top, mappings[0].base))
        self.assertEqual("Dch", rows[0].facies_index)
        self.assertEqual("Каналы", rows[0].facies_name)
        self.assertEqual("Гравийный песчаник.", rows[0].target_text)
        self.assertEqual("Индекс фации", mappings[0].source_headers["class_index"])
        self.assertEqual("Краткое описание", mappings[0].source_headers["target_text"])

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
        self.assertEqual((16, 17), (mappings[0].gis_top, mappings[0].gis_base))
        self.assertEqual(19, mappings[0].facies_thickness)
        self.assertEqual(30, mappings[0].target_text)
        self.assertEqual("Песчаник светло-серый, слоистый.", rows[0].target_text)
        self.assertEqual((4105.0, 4108.65), (rows[0].top, rows[0].base))
        self.assertTrue(rows[0].thickness_valid)
        self.assertEqual("drilling", rows[0].metadata["interval_source"])
        self.assertEqual((4104.9, 4108.55), (rows[0].gis_top, rows[0].gis_base))
        self.assertEqual("Dch", rows[0].facies_index)
        self.assertEqual("Каналы распределительные", rows[0].facies_name)
        self.assertEqual("Dch", rows[0].label)

    def test_drilling_roof_and_base_must_share_parent_group_thickness_validates_their_span(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interval_groups.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append([
                "Скважина",
                "Интервал фации по бурению Кровля",
                "Интервал фации по ГИС Подошва",
                "Интервал фации по ГИС Кровля",
                "Интервал фации по бурению Подошва",
                "Толщина фации, м",
                "Код фации",
            ])
            sheet.append(["W-1", 100.0, 201.0, 200.0, 102.0, 2.0, "Dch"])
            workbook.save(path)

            rows, mappings, issues = read_table(path)

        self.assertEqual((2, 5), (mappings[0].top, mappings[0].base))
        self.assertEqual((4, 3), (mappings[0].gis_top, mappings[0].gis_base))
        self.assertEqual((100.0, 102.0), (rows[0].top, rows[0].base))
        self.assertTrue(rows[0].thickness_valid)
        self.assertEqual("drilling", rows[0].metadata["interval_source"])
        self.assertEqual([], [issue for issue in issues if issue.severity == "error"])

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
        self.assertEqual((100.0, 112.0), (rows[0].top, rows[0].base))
        self.assertFalse(rows[0].thickness_valid)
        self.assertTrue(any("не будет использована для маски" in item.message for item in issues))

    def test_rounds_formula_noise_to_centimetres_and_treats_3_03_as_303_cm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "centimetres.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append([
                "Скважина", "Интервал фации по бурению Кровля",
                "Интервал фации по бурению Подошва", "Толщина фации, м", "Класс",
            ])
            sheet.append(["W-1", 4129.67, 4130.139999999, 0.47, "A"])
            sheet.append(["W-1", 4130.139999999, 4133.17, 3.03, "B"])
            workbook.save(path)

            rows, _, issues = read_table(path)

        self.assertEqual([(4129.67, 4130.14), (4130.14, 4133.17)], [
            (row.top, row.base) for row in rows
        ])
        self.assertEqual([0.47, 3.03], [row.thickness for row in rows])
        self.assertTrue(all(row.thickness_valid for row in rows))
        self.assertEqual([], [item for item in issues if item.severity == "error"])

    def test_reads_csv_with_one_interval_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "description.csv"
            path.write_text("Скважина;Интервал;Класс\nW-1;100,0-101,5;Sand\n", encoding="utf-8")

            rows, _, _ = read_table(path)

        self.assertEqual(1, len(rows))
        self.assertEqual((100.0, 101.5), (rows[0].top, rows[0].base))
        self.assertEqual("Sand", rows[0].label)

    def test_reads_gis_facies_limits_as_a_separate_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drilling_and_gis.csv"
            path.write_text(
                "Скважина;Интервал фации по бурению Кровля;Интервал фации по бурению Подошва;"
                "Интервал фации по ГИС Кровля;Интервал фации по ГИС Подошва;"
                "Толщина фации;Код фации;Краткое описание\n"
                "W-1;100.00;101.00;200.00;201.00;1.00;Sand;Песчаник серый.\n",
                encoding="utf-8",
            )

            rows, mappings, issues = read_table(path)

        self.assertEqual([], [item for item in issues if item.severity == "error"])
        self.assertEqual((2, 3), (mappings[0].top, mappings[0].base))
        self.assertEqual((4, 5), (mappings[0].gis_top, mappings[0].gis_base))
        self.assertEqual((100.0, 101.0), (rows[0].top, rows[0].base))
        self.assertEqual((200.0, 201.0), (rows[0].gis_top, rows[0].gis_base))

    def test_valid_gis_thickness_turns_bad_drilling_interval_into_a_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gis_rescue.csv"
            path.write_text(
                "Скважина;Интервал фации по бурению Кровля;Интервал фации по бурению Подошва;"
                "Интервал фации по ГИС Кровля;Интервал фации по ГИС Подошва;"
                "Толщина фации;Код фации\nW-1;100.00;102.00;200.00;201.00;1.00;Sand\n",
                encoding="utf-8",
            )

            rows, _, issues = read_table(path)

        self.assertFalse(rows[0].thickness_valid)
        self.assertEqual((200.0, 201.0), (rows[0].gis_top, rows[0].gis_base))
        self.assertTrue(any(item.severity == "warning" and "резервный интервал ГИС" in item.message for item in issues))
        self.assertFalse(any(item.severity == "error" and "Толщина фации" in item.message for item in issues))

    def test_old_saved_column_map_is_supplemented_with_new_gis_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_mapping.csv"
            path.write_text(
                "Скважина;Интервал фации по бурению Кровля;Интервал фации по бурению Подошва;"
                "Интервал фации по ГИС Кровля;Интервал фации по ГИС Подошва;"
                "Толщина фации;Код фации\nW-1;100.00;101.00;200.00;201.00;1.00;Sand\n",
                encoding="utf-8",
            )
            _, detected, _ = read_table(path)
            mapping_file = Path(directory) / "column_mapping.json"
            save_mappings(mapping_file, [replace(detected[0], gis_top=None, gis_base=None)])
            old_map = json.loads(mapping_file.read_text(encoding="utf-8"))
            for saved in old_map.values():
                saved.pop("gis_interval", None)
                saved.pop("gis_top", None)
                saved.pop("gis_base", None)
            mapping_file.write_text(json.dumps(old_map), encoding="utf-8")

            rows, mappings, _ = read_table(path, mapping_file)

        self.assertEqual((2, 3), (mappings[0].top, mappings[0].base))
        self.assertEqual((4, 5), (mappings[0].gis_top, mappings[0].gis_base))
        self.assertEqual((200.0, 201.0), (rows[0].gis_top, rows[0].gis_base))

    def test_uses_gis_as_primary_interval_when_workbook_has_no_drilling_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gis_only.csv"
            path.write_text(
                "Скважина;Интервал фации по ГИС Кровля;Интервал фации по ГИС Подошва;"
                "Толщина фации;Код фации\nW-1;200.00;201.00;1.00;Sand\n",
                encoding="utf-8",
            )

            rows, mappings, issues = read_table(path)

        self.assertEqual([], [item for item in issues if item.severity == "error"])
        self.assertEqual((2, 3), (mappings[0].top, mappings[0].base))
        self.assertEqual((200.0, 201.0), (rows[0].top, rows[0].base))


if __name__ == "__main__":
    unittest.main()
