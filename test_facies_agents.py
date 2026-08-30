"""Tests for the deterministic column → depth → Excel-mask agents."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from app.infrastructure.excel_core_description import CoreInterval, DescriptionLayer
from app.infrastructure.facies_agents import CoreColumn, ExcelFaciesMaskAgent, PhotoIntervalAgent, PhotoOrientationAgent, TextMark


def layer(top: float, base: float, name: str) -> DescriptionLayer:
    return DescriptionLayer(
        well="P-31", top=top, base=base, facies_name=name, facies_code="",
        facies_index="", description="", attributes={}, sheet="layers", row=1,
    )


class ExcelFaciesMaskAgentTests(unittest.TestCase):
    def test_uses_specific_filename_interval_before_wider_caption_interval(self):
        path = Path("Р-31_3002,00-3004,96.jpg")
        with patch("app.infrastructure.facies_agents.read_caption_metadata") as caption:
            interval, messages = PhotoIntervalAgent().from_photo(path, default_well="Р-31")

        self.assertEqual(("Р-31", 3002.0, 3004.96), (interval.well, interval.top, interval.base))
        caption.assert_not_called()
        self.assertTrue(any("из имени" in item.message for item in messages))

    def test_projects_excel_boundaries_across_sequential_columns(self):
        bands, columns, messages = ExcelFaciesMaskAgent().apply(
            CoreInterval("P-31", 100.0, 102.0),
            [layer(100.0, 100.5, "A"), layer(100.5, 101.5, "B"), layer(101.5, 102.0, "C")],
            [CoreColumn(10, 20, 50, 120), CoreColumn(70, 20, 110, 120)],
        )

        self.assertEqual(4, len(bands))
        self.assertEqual((100.0, 101.0), (columns[0].depth_from, columns[0].depth_to))
        self.assertEqual((101.0, 102.0), (columns[1].depth_from, columns[1].depth_to))
        self.assertEqual(["A", "B", "B", "C"], [item.label for item in bands])
        self.assertEqual((20, 70), (bands[0].top, bands[0].bottom))
        self.assertEqual((70, 120), (bands[-1].top, bands[-1].bottom))
        self.assertTrue(any(item.level == "info" for item in messages))

    def test_top_mark_on_right_reverses_column_traversal(self):
        ordered, messages = PhotoOrientationAgent().order(
            [CoreColumn(10, 20, 50, 120), CoreColumn(70, 20, 110, 120)],
            [TextMark("Верх", 74, 0, 106, 15, 95.0)],
        )

        self.assertEqual([70, 10], [column.left for column in ordered])
        self.assertTrue(all(column.direction == "down" for column in ordered))
        self.assertTrue(any(item.level == "info" for item in messages))


if __name__ == "__main__":
    unittest.main()
