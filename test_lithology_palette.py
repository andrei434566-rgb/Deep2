"""Regression tests for the visible lithology track and saved palette UI."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication, QSpinBox, QToolBar

from app.domain.facies_palette import normalize_facies_palette
from app.domain.lithology import default_lithology_palette, normalize_lithology_palette
from app.domain.models import FaciesDetection, PhotoRecord
from app.ui.dialogs.lithology_palette_dialog import FaciesPaletteDialog
from app.ui.widgets.workspace_canvas import StackColumnItem, facies_color
from app.ui.windows.main_window import MainWindow


_APP = QApplication.instance() or QApplication([])


class LithologyPaletteTests(unittest.TestCase):
    def tearDown(self) -> None:
        StackColumnItem.set_lithology_palette(default_lithology_palette())
        StackColumnItem.set_facies_palette([])

    def test_palette_normalization_keeps_unique_valid_rows(self):
        palette = normalize_lithology_palette(
            [
                {"name": "Моя порода", "symbol": "MP", "color": "#abc", "pattern": "dots"},
                {"name": "моя порода", "symbol": "DUP", "color": "bad", "pattern": "unknown"},
                {"name": "Другая", "symbol": "", "color": "bad", "pattern": "unknown"},
            ]
        )

        self.assertEqual(2, len(palette))
        self.assertEqual("#aabbcc", palette[0]["color"])
        self.assertEqual("solid", palette[1]["pattern"])
        self.assertEqual("#808080", palette[1]["color"])

    def test_compact_core_column_has_clickable_lithology_track(self):
        pixmap = QPixmap(60, 100)
        pixmap.fill(QColor("#806040"))
        detection = FaciesDetection(
            "Sandstone",
            0.91,
            [QPointF(0, 0), QPointF(60, 0), QPointF(60, 100), QPointF(0, 100)],
            attributes={"Название породы": "Песчаник"},
        )
        record = PhotoRecord("one", "memory", pixmap, detections=[detection])

        item = StackColumnItem([record], well_name="Скважина 1", description_only=True)

        self.assertGreater(item.lithology_x, item.core_x + item.column_width)
        self.assertGreater(item.facies_x, item.lithology_x + item.LITHOLOGY_WIDTH)
        self.assertGreaterEqual(item.width, item.facies_x + item.FACIES_WIDTH)
        self.assertTrue(item._placements)
        placement = item._placements[0][3]
        lithology_hit = item._column_at(QPointF(item.lithology_x + 10, placement.center().y()))
        facies_hit = item._column_at(QPointF(item.facies_x + 10, placement.center().y()))
        self.assertIsNotNone(lithology_hit)
        self.assertIsNotNone(facies_hit)
        self.assertEqual("lithology", lithology_hit[2])
        self.assertEqual("facies", facies_hit[2])

    def test_custom_palette_drives_lithology_lookup(self):
        StackColumnItem.set_lithology_palette(
            [{"name": "Моя порода", "symbol": "MP", "color": "#123456", "pattern": "cross"}]
        )

        info = StackColumnItem._lithology_info("Моя порода")

        self.assertIsNotNone(info)
        self.assertEqual("MP", info["symbol"])
        self.assertEqual("cross", info["pattern"])

    def test_reference_lithology_variants_resolve_to_legend_pattern(self):
        info = StackColumnItem._lithology_info("Битуминозный и окремненный аргилит")

        self.assertIsNotNone(info)
        self.assertEqual("BCL", info["symbol"])

    def test_facies_palette_keeps_model_key_when_display_name_changes(self):
        palette = normalize_facies_palette(
            [{"key": "AFp", "name": "Конус выноса", "symbol": "AF", "color": "#123456", "pattern": "dots"}]
        )
        StackColumnItem.set_facies_palette(palette)

        info = StackColumnItem._facies_info("AFp")

        self.assertIsNotNone(info)
        self.assertEqual("Конус выноса", info["name"])
        self.assertEqual("#123456", info["color"])

    def test_facies_editor_preserves_hidden_model_key(self):
        palette = [{"key": "F-photo", "name": "Название с фото", "symbol": "F", "color": "#7169df", "pattern": "solid"}]
        dialog = FaciesPaletteDialog(palette, palette)
        try:
            dialog.table.item(0, 0).setText("Отображаемое название")
            result = dialog.palette()
        finally:
            dialog.close()

        self.assertEqual("F-photo", result[0]["key"])
        self.assertEqual("Отображаемое название", result[0]["name"])

    def test_facies_palette_defaults_use_labels_and_overlay_colors_from_memory(self):
        pixmap = QPixmap(30, 40)
        pixmap.fill(QColor("#505050"))
        detection = FaciesDetection(
            "F-photo",
            0.8,
            [QPointF(0, 0), QPointF(30, 0), QPointF(30, 40), QPointF(0, 40)],
        )
        window = MainWindow()
        try:
            window._records = [PhotoRecord("memory", "memory", pixmap, detections=[detection])]
            palette = window._facies_palette_defaults()
        finally:
            window.close()

        self.assertEqual("F-photo", palette[0]["key"])
        self.assertEqual("F-photo", palette[0]["name"])
        self.assertEqual(facies_color("F-photo").name(), palette[0]["color"])

    def test_training_controls_are_on_a_dedicated_toolbar(self):
        window = MainWindow()
        try:
            toolbar = window.findChild(QToolBar, "trainingToolbar")
            epochs = window.findChild(QSpinBox, "trainingEpochs")
            self.assertIsNotNone(toolbar)
            self.assertIsNotNone(epochs)
            self.assertIs(toolbar, epochs.parentWidget().parentWidget())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
