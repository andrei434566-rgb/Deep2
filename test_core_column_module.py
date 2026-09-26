"""Regression tests for the independent core-column first stage."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPolygonF

from app.domain.models import FaciesDetection
from app.infrastructure.facies_postprocess import UNRECOGNIZED_FACIES, complete_core_column_coverage
from app.infrastructure.ml.core_column_service import CoreColumnRecognizer, assemble_core_tape, normalize_columns
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

    def test_column_detector_finds_narrow_short_half_core(self):
        image = np.full((1000, 900, 3), 255, dtype=np.uint8)
        for left in (180, 350, 520):
            cv2.rectangle(image, (left, 100), (left + 95, 900), (125, 125, 125), -1)
        # A narrow half-column only covers part of the usual one-metre height.
        cv2.rectangle(image, (710, 350), (722, 720), (125, 125, 125), -1)

        boxes = RuleBasedFaciesDetector._find_core_columns(image)

        self.assertEqual(4, len(boxes))
        self.assertTrue(any(left <= 710 and right >= 722 and bottom - top >= 350
                            for left, top, right, bottom in boxes))

    def test_column_detector_keeps_searching_after_partial_component_result(self):
        image = np.full((1000, 900, 3), 255, dtype=np.uint8)
        expected_lefts = (160, 340, 520, 700)
        for left in expected_lefts:
            cv2.rectangle(image, (left, 100), (left + 100, 900), (125, 125, 125), -1)

        with patch.object(RuleBasedFaciesDetector, "_core_component_boxes", return_value=[(160, 100, 261, 901)]):
            boxes = RuleBasedFaciesDetector._find_core_columns(image)

        self.assertEqual(4, len(boxes))
        self.assertTrue(all(any(abs(left - expected) <= 4 for left, _, _, _ in boxes)
                            for expected in expected_lefts))

    def test_horizontal_grid_lines_do_not_merge_lanes_or_add_a_fake_column(self):
        image = np.full((1200, 1000, 3), 255, dtype=np.uint8)
        expected_lefts = (270, 420, 570, 720)
        for index, left in enumerate(expected_lefts):
            cv2.rectangle(image, (left, 150), (left + 92, 1040), (247, 247, 247), -1)
            for y in range(185 + index * 7, 1020, 73):
                cv2.line(image, (left, y), (left + 91, y + 12), (185, 185, 185), 2)
        for y in range(250, 1000, 180):
            cv2.line(image, (145, y), (900, y), (80, 80, 80), 2)

        boxes = RuleBasedFaciesDetector._find_core_columns(image)

        self.assertEqual(4, len(boxes))
        self.assertTrue(all(left > 230 for left, _, _, _ in boxes))

    def test_depth_caption_above_core_is_not_included_in_column_start(self):
        image = np.full((1000, 700, 3), 255, dtype=np.uint8)
        cv2.putText(image, "4130.00", (275, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.rectangle(image, (275, 210), (410, 850), (242, 242, 242), -1)
        for y in range(235, 830, 47):
            cv2.line(image, (275, y), (410, y + 9), (160, 160, 160), 2)

        boxes = RuleBasedFaciesDetector._find_core_columns(image)

        self.assertEqual(1, len(boxes))
        self.assertGreaterEqual(boxes[0][1], 200)

    def test_neural_partial_result_is_completed_by_heuristic_search(self):
        image = np.full((1000, 900, 3), 255, dtype=np.uint8)
        model_box = (160, 100, 261, 520)
        heuristic_boxes = [(160, 100, 261, 901), (340, 100, 441, 901),
                           (520, 100, 621, 901), (700, 100, 801, 901)]
        recognizer = CoreColumnRecognizer.__new__(CoreColumnRecognizer)
        recognizer.model = object()
        recognizer.model_path = None
        recognizer.source_label = "test model"
        with patch.object(recognizer, "_model_boxes", return_value=[model_box]), \
                patch.object(RuleBasedFaciesDetector, "_find_core_columns", return_value=heuristic_boxes):
            boxes = recognizer.recognize(image)

        self.assertEqual(4, len(boxes))
        self.assertEqual("модель + резервный поиск", recognizer.source_label)
        self.assertGreaterEqual(boxes[0]["bottom"] - boxes[0]["top"], 800)

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
