"""Regression tests for the independent core-column first stage."""

from __future__ import annotations

import unittest

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPolygonF

from app.domain.models import FaciesDetection
from app.infrastructure.facies_postprocess import UNRECOGNIZED_FACIES, complete_core_column_coverage
from app.infrastructure.ml.core_column_service import assemble_core_tape, normalize_columns
from app.infrastructure.ml.rule_based_facies import RuleBasedFaciesDetector


class CoreColumnModuleTests(unittest.TestCase):
    def test_assembles_columns_into_one_tape_without_spacers(self):
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        image[1:6, 1:4] = (20, 30, 40)
        image[2:10, 7:11] = (80, 90, 100)

        tape = assemble_core_tape(image, [(1, 1, 4, 6), (7, 2, 11, 10)], target_width=6)

        self.assertEqual(6, tape.image.shape[1])
        self.assertEqual(tape.sections[0].tape_bottom, tape.sections[1].tape_top)
        self.assertEqual(tape.sections[-1].tape_bottom, tape.image.shape[0])
        self.assertTrue(np.all(tape.image[0] == (20, 30, 40)))
        self.assertTrue(np.all(tape.image[-1] == (80, 90, 100)))

    def test_normalizes_and_deduplicates_column_rectangles(self):
        columns = normalize_columns(
            [
                {"left": -5, "top": 1, "right": 20, "bottom": 80},
                {"left": 0, "top": 1, "right": 20, "bottom": 80},
                {"left": 60, "top": 3, "right": 99, "bottom": 90},
            ],
            (100, 100),
        )
        self.assertEqual(2, len(columns))
        self.assertEqual(0.0, columns[0]["left"])

    def test_facies_bands_cover_every_selected_column_without_gaps(self):
        detections = [
            self._detection("A", 0.9, 10, 30),
            self._detection("B", 0.8, 60, 80),
        ]
        columns = [
            {"left": 0, "top": 0, "right": 20, "bottom": 100},
            {"left": 40, "top": 5, "right": 60, "bottom": 55},
        ]

        bands = complete_core_column_coverage(detections, columns, fallback_label="A")

        first = [item for item in bands if QPolygonF(item.polygon).boundingRect().left() < 30]
        second = [item for item in bands if QPolygonF(item.polygon).boundingRect().left() > 30]
        self._assert_continuous(first, 0, 100)
        self._assert_continuous(second, 5, 55)
        self.assertTrue(all(item.label in {"A", "B", UNRECOGNIZED_FACIES} for item in bands))
        self.assertTrue(all(item.label == UNRECOGNIZED_FACIES for item in second))

    def test_no_evidence_leaves_filled_but_unrecognized_column(self):
        bands = complete_core_column_coverage(
            [],
            [{"left": 2, "top": 3, "right": 12, "bottom": 23}],
            fallback_label="Sandstone",
        )
        self.assertEqual(1, len(bands))
        self.assertEqual(UNRECOGNIZED_FACIES, bands[0].label)
        self.assertEqual(0.0, bands[0].confidence)
        self._assert_continuous(bands, 3, 23)

    def test_ambiguous_code_variants_do_not_merge(self):
        first = self._detection("Dch", 0.8, 0, 50)
        first.attributes["Индекс фации"] = "47"
        second = self._detection("Dch", 0.9, 50, 100)
        second.attributes["Индекс фации"] = "92"
        bands = complete_core_column_coverage([first, second], [{"left": 0, "top": 0, "right": 20, "bottom": 100}])
        self.assertEqual(["47", "92"], [item.attributes["Индекс фации"] for item in bands])
        self._assert_continuous(bands, 0, 100)

    def test_visual_structure_splits_one_column_into_continuous_intervals(self):
        image = np.full((400, 80, 3), 55, dtype=np.uint8)
        image[200:] = 195
        image[210::12] = 95

        intervals = RuleBasedFaciesDetector().split_columns(image, [(5, 0, 75, 400)])

        self.assertGreaterEqual(len(intervals), 2)
        self.assertEqual(0, intervals[0].top)
        self.assertEqual(400, intervals[-1].bottom)
        for previous, current in zip(intervals[:-1], intervals[1:]):
            self.assertEqual(previous.bottom, current.top)

    def test_interval_classification_has_priority_over_whole_photo_mask(self):
        whole = FaciesDetection(
            "Whole-photo guess", 0.99,
            [QPointF(0, 0), QPointF(20, 0), QPointF(20, 100), QPointF(0, 100)],
        )
        upper = FaciesDetection(
            "Upper facies", 0.55,
            [QPointF(0, 0), QPointF(20, 0), QPointF(20, 50), QPointF(0, 50)],
            attributes={"__structural_interval": "1:0"},
        )
        lower = FaciesDetection(
            "Lower facies", 0.60,
            [QPointF(0, 50), QPointF(20, 50), QPointF(20, 100), QPointF(0, 100)],
            attributes={"__structural_interval": "1:1"},
        )

        bands = complete_core_column_coverage(
            [whole, upper, lower],
            [{"left": 0, "top": 0, "right": 20, "bottom": 100}],
        )

        self.assertEqual(["Upper facies", "Lower facies"], [item.label for item in bands])
        self._assert_continuous(bands, 0, 100)

    def test_equal_facies_remain_separate_across_a_structural_contact(self):
        detections = [
            FaciesDetection(
                "A", 0.8,
                [QPointF(0, 0), QPointF(20, 0), QPointF(20, 50), QPointF(0, 50)],
                attributes={"__structural_interval": "1:0"},
            ),
            FaciesDetection(
                "A", 0.8,
                [QPointF(0, 50), QPointF(20, 50), QPointF(20, 100), QPointF(0, 100)],
                attributes={"__structural_interval": "1:1"},
            ),
        ]

        bands = complete_core_column_coverage(
            detections,
            [{"left": 0, "top": 0, "right": 20, "bottom": 100}],
        )

        self.assertEqual(2, len(bands))
        self._assert_continuous(bands, 0, 100)

    @staticmethod
    def _detection(label: str, confidence: float, top: float, bottom: float) -> FaciesDetection:
        return FaciesDetection(
            label,
            confidence,
            [QPointF(0, top), QPointF(20, top), QPointF(20, bottom), QPointF(0, bottom)],
        )

    def _assert_continuous(self, bands: list[FaciesDetection], expected_top: float, expected_bottom: float) -> None:
        rectangles = sorted((QPolygonF(item.polygon).boundingRect() for item in bands), key=lambda item: item.top())
        self.assertTrue(rectangles)
        self.assertAlmostEqual(expected_top, rectangles[0].top())
        self.assertAlmostEqual(expected_bottom, rectangles[-1].bottom())
        for previous, current in zip(rectangles[:-1], rectangles[1:]):
            self.assertAlmostEqual(previous.bottom(), current.top())


if __name__ == "__main__":
    unittest.main()
