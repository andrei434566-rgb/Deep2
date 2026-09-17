"""Regression tests for the final facies reference and movable core layers."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsView, QPushButton

from app.domain.facies_catalog import (
    FACIES_CATALOG,
    FACIES_REFERENCE_SHA256,
    facies_metadata,
)
from app.domain.facies_palette import normalize_facies_palette
from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.project_storage import load_project, save_project
from app.infrastructure.ml.yolo_model_service import YoloModelService
from app.infrastructure.core_report_export import _facies_description
from app.ui.dialogs.facies_dialog import FaciesDialog
from app.ui.widgets.workspace_canvas import StackColumnItem, WorkspaceCanvas, clear_stack_crop_cache
from app.ui.windows.main_window import MainWindow


_APP = QApplication.instance() or QApplication([])


def _layer(label: str, top: float, base: float, depth_from: float, depth_to: float) -> FaciesDetection:
    return FaciesDetection(
        label,
        0.9,
        [QPointF(0, top), QPointF(30, top), QPointF(30, base), QPointF(0, base)],
        depth_from=depth_from,
        depth_to=depth_to,
    )


class FaciesReferenceAndLayerOrderTests(unittest.TestCase):
    def test_final_workbook_reference_is_embedded_and_case_sensitive(self):
        self.assertEqual(93, len(FACIES_CATALOG))
        self.assertEqual("A8C09C73D158FD5C2EECE14820605F808E9F536090CEACD4A2320B7D30301558", FACIES_REFERENCE_SHA256)
        self.assertEqual("79", facies_metadata("DWCH")["Индекс фации"])
        self.assertEqual("80", facies_metadata("DWCh")["Индекс фации"])
        self.assertEqual("92", facies_metadata("Dch", 92)["Индекс фации"])
        self.assertIn("Debris flows", facies_metadata("DetF")["Название фации"])
        self.assertTrue(facies_metadata("Uof")["Предполагаемые литотипы"])
        palette = normalize_facies_palette([{"key": "DWCH"}, {"key": "DWCh"}])
        self.assertEqual(["DWCH", "DWCh"], [item["key"] for item in palette])
        rendered = _facies_description(facies_metadata("DetF"))
        self.assertIn("Гидродинамический режим", rendered)
        self.assertIn("Предполагаемые литотипы", rendered)

    def test_dialog_keeps_the_selected_variant_and_exposes_move_controls(self):
        dialog = FaciesDialog("Dch", 0.9, {"Индекс фации": "92"})
        try:
            self.assertEqual("92", dialog.selected_attributes()["Индекс фации"])
            self.assertIsNotNone(dialog.findChild(QPushButton, "moveFaciesLayerUp"))
            self.assertIsNotNone(dialog.findChild(QPushButton, "moveFaciesLayerDown"))
        finally:
            dialog.close()

    def test_approved_definition_overrides_an_obsolete_model_sidecar(self):
        service = object.__new__(YoloModelService)
        service._facies_catalog = {
            "Dch": {
                "Индекс фации": "92",
                "Название фации": "Старое название",
                "Размер зерен": "Крупнозернистый",
            }
        }

        attributes = service._facies_attributes("Dch")

        self.assertEqual("92", attributes["Индекс фации"])
        self.assertIn("Distributary Channels lag", attributes["Название фации"])
        self.assertNotIn("Размер зерен", attributes)

    def test_layer_move_changes_stack_order_and_keeps_depth_column_continuous(self):
        pixmap = QPixmap(30, 90)
        pixmap.fill(QColor("#665544"))
        first = _layer("A", 0, 30, 0.0, 1.0)
        second = _layer("B", 30, 60, 1.0, 3.0)
        third = _layer("C", 60, 90, 3.0, 4.0)
        record = PhotoRecord("one", "memory", pixmap, detections=[first, second, third], well_name="W-1")
        window = MainWindow()
        window._records = [record]
        try:
            with patch.object(window, "_refresh_facies_views"):
                window._move_layer(record, third, -1)
        finally:
            window.close()

        ordered = [item.label for _, item in StackColumnItem.ordered_layer_entries([record])]
        self.assertEqual(["A", "C", "B"], ordered)
        self.assertEqual([0, 2, 1], [first.stack_order, second.stack_order, third.stack_order])
        self.assertEqual((0.0, 1.0), (first.depth_from, first.depth_to))
        self.assertEqual((1.0, 2.0), (third.depth_from, third.depth_to))
        self.assertEqual((2.0, 4.0), (second.depth_from, second.depth_to))

    def test_stack_order_survives_project_save_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            pixmap = QPixmap(20, 20)
            pixmap.fill(QColor("#775533"))
            self.assertTrue(pixmap.save(str(source)))
            detection = _layer("AFp", 0, 20, 10.0, 11.0)
            detection.stack_order = 7
            record = PhotoRecord("one", str(source), pixmap, detections=[detection])

            project = root / "project"
            save_project(project, "test", [record], {})
            _, records, _, missing, _ = load_project(project)

        self.assertFalse(missing)
        self.assertEqual(7, records[0].detections[0].stack_order)

    def test_rebuild_reuses_masked_crop_and_uses_partial_viewport_updates(self):
        clear_stack_crop_cache()
        pixmap = QPixmap(30, 30)
        pixmap.fill(QColor("#887766"))
        detection = _layer("AFp", 0, 30, 0.0, 1.0)
        record = PhotoRecord("one", "memory", pixmap, detections=[detection])
        first = StackColumnItem([record])
        second = StackColumnItem([record])
        canvas = WorkspaceCanvas()
        try:
            self.assertEqual(first._placements[0][0].cacheKey(), second._placements[0][0].cacheKey())
            self.assertEqual(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate, canvas.viewportUpdateMode())
        finally:
            canvas.close()


if __name__ == "__main__":
    unittest.main()
