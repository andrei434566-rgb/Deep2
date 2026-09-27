from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from excel_photo_model_studio.standard_excel import export_standardized_workbook


class StandardExcelTests(unittest.TestCase):
    def test_exports_facies_targets_with_semantic_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "well.xlsx"
            export_standardized_workbook([{
                "field_name": "Уренгойское", "well": "Р-31", "core_top": 3183.0,
                "core_base": 3195.0, "facies_top": 3188.98, "facies_base": 3189.50,
                "facies_index": "Tcr", "facies_name": "Тонкослоистый песчаник",
                "description": "Песчаник светло-серый, слоистый.",
            }], output)
            workbook = load_workbook(output, data_only=True)
            sheet = workbook.active
        columns = {
            str(cell.value): cell.column
            for row in sheet.iter_rows(min_row=1, max_row=2)
            for cell in row if cell.value
        }
        self.assertEqual("Tcr", sheet.cell(4, columns["Индекс фации"]).value)
        self.assertEqual("Тонкослоистый песчаник", sheet.cell(4, columns["Название фации"]).value)
        self.assertEqual(
            "Песчаник светло-серый, слоистый.",
            sheet.cell(4, columns["Краткое описание"]).value,
        )


if __name__ == "__main__":
    unittest.main()
