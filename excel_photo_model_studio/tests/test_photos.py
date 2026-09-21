from __future__ import annotations

import unittest

from excel_photo_model_studio.photos import extract_depth_interval, extract_well


class PhotoOcrParsingTests(unittest.TestCase):
    def test_prefers_captioned_full_core_interval_over_column_numbers(self):
        text = (
            "Глубина по керну 4105.00 4106.00 4107.00 4107.94 4108.94. "
            "Глубина с увязкой 4104.90 4105.90 4106.90 4107.84 4108.84. "
            "Интервал отбора керна с 4104,90 до 4116,55 м, после увязки 4104,90-4116,45 м."
        )

        self.assertEqual((4104.9, 4116.55), extract_depth_interval(text))

    def test_extracts_well_and_removes_figure_suffix(self):
        self.assertEqual("67ПО", extract_well("Восточно-Тазовское м-е, скв№ 67ПО-0001"))

    def test_rejects_numbers_from_ruler_and_figure_caption_as_depth_pair(self):
        self.assertIsNone(extract_depth_interval(
            "Рис. 22, шкала 0 10 20 30 40, страница 410",
            ((4104.9, 4116.9),),
        ))

    def test_uses_excel_core_interval_to_reject_larger_false_ocr_pair(self):
        text = "4105.00 4304.00 интервал отбора керна 4104.90 4116.55"
        self.assertEqual(
            (4104.9, 4116.55),
            extract_depth_interval(text, ((4104.9, 4116.9),)),
        )


if __name__ == "__main__":
    unittest.main()
