"""Regression checks for model-independent preparation and explicit review."""
import os
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from app.domain.facies_catalog import FACIES_MODEL_CLASSES, facies_metadata, resolve_facies_class
from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.core_report_export import _merged_layers, _lithology_description
from app.ui.dialogs.facies_dialog import FaciesDialog
from app.ui.dialogs.model_preparation_dialog import ModelPreparationDialog
from app.ui.windows.main_window import MainWindow

_APP = QApplication.instance() or QApplication([])


class ModelPreparationTests(unittest.TestCase):
    def test_taxonomy_has_stable_unique_code_index_keys(self):
        self.assertEqual(92, len(FACIES_MODEL_CLASSES))
        self.assertEqual(list(range(92)), [item['class_id'] for item in FACIES_MODEL_CLASSES])
        self.assertEqual(92, len({item['model_label'] for item in FACIES_MODEL_CLASSES}))
        self.assertEqual('92', resolve_facies_class('Dch@92')['index'])
        self.assertIsNone(resolve_facies_class('Dch'))
        self.assertIsNone(resolve_facies_class('DWCH', '80'))
        self.assertIsNone(resolve_facies_class('dwch'))
        self.assertIsNone(resolve_facies_class('ProDp', '34'))
        self.assertEqual({}, facies_metadata('Dch'))

    def test_draft_does_not_silently_choose_first_or_remembered_facies(self):
        for label in ('', 'Новый контур'):
            dialog = FaciesDialog(label, 0)
            self.assertEqual('', dialog.selected_facies())
            self.assertFalse(dialog.training_confirmed.isChecked())
            dialog.close()
        dialog = FaciesDialog('Dch', 0.8)
        self.assertNotIn('Индекс фации', dialog.selected_attributes())
        dialog.close()

    def test_reference_variant_and_search(self):
        dialog = FaciesDialog('Dch@92', 0.8, training_ready=True)
        self.assertEqual('Dch', dialog.selected_facies())
        self.assertEqual('92', dialog.selected_attributes()['Индекс фации'])
        dialog._filter_facies('DWCH')
        self.assertTrue(dialog.facies.view().isRowHidden(dialog.facies.currentIndex()))
        self.assertTrue(dialog.training_confirmed.isChecked())
        dialog.close()

    def test_no_model_needed_for_preparation(self):
        window = MainWindow()
        # User settings may legitimately remember a model from a running app;
        # isolate this test from that external desktop state.
        window._selected_model_path = None
        self.assertIsNone(window._resolve_model_path())
        dialog = ModelPreparationDialog([], None, window)
        self.assertEqual(92, dialog.table.rowCount())
        dialog._filter('Dch')
        visible = sum(not dialog.table.isRowHidden(row) for row in range(92))
        self.assertGreater(visible, 1)
        dialog.close()
        window.close()

    def test_photo_import_continues_all_batches_without_facies_weights(self):
        window = MainWindow()
        window.PHOTO_LOAD_BATCH_SIZE = 1
        window._pending_photo_imports = deque([
            (Path("one.jpg"), "W-1", None, None),
            (Path("two.jpg"), "W-1", None, None),
        ])
        pixmap = QPixmap(40, 80)
        with (
            patch("app.ui.windows.main_window.load_working_pixmap", return_value=pixmap),
            patch("app.ui.windows.main_window.QTimer.singleShot", side_effect=lambda _, callback: callback()),
            patch.object(window, "_resolve_model_path", return_value=None),
            patch.object(window, "_suggest_core_columns", return_value=[{"left": 0, "top": 0, "right": 40, "bottom": 80}]) as columns,
            patch.object(window.workspace, "add_photos"),
            patch.object(window.workspace, "update_photo_detections"),
            patch.object(window, "_show_stack"),
            patch.object(window, "_refresh_project_tree"),
            patch.object(window, "_refresh_training_bar"),
        ):
            window._load_next_photo_batch()
        self.assertEqual(2, len(window._records))
        self.assertEqual(2, columns.call_count)
        self.assertFalse(window._pending_photo_imports)
        self.assertTrue(all(record.core_columns for record in window._records))
        window.close()

    def test_training_completion_never_replaces_reviewed_layers(self):
        window = MainWindow()
        prior = Path('existing.pt')
        window._selected_model_path = prior
        with patch('app.ui.windows.main_window.QMessageBox.information'), patch.object(window, 'run_segmentation') as segment:
            window._on_fine_tune_succeeded('candidate.pt')
            window._on_fine_tune_finished()
            segment.assert_not_called()
        self.assertEqual(prior, window._selected_model_path)
        window.close()

    def test_vertex_edit_does_not_confirm_class(self):
        pixmap = QPixmap(30, 30)
        detection = FaciesDetection('DWCh', 0.6, [QPointF(0, 0), QPointF(20, 0), QPointF(20, 20)])
        record = PhotoRecord('one', 'memory', pixmap, detections=[detection])
        window = MainWindow()
        with patch.object(window, '_refresh_facies_views'), patch.object(window, '_sync_depth_range_from_polygon', return_value=False):
            window._on_facies_geometry_changed(record, detection)
        self.assertFalse(detection.training_ready)
        window.close()

    def test_report_preserves_facies_identity_and_lithological_contacts(self):
        layers = []
        for pos, (code, index, rock) in enumerate([('Dch', '47', ''), ('Dch', '92', ''), ('DWCH', '79', ''), ('DWCh', '80', ''), ('DWCh', '80', 'Песчаник')]):
            layers.append({'label':code, 'depth_from':pos, 'depth_to':pos+1,
                           'attributes':{'Код фации':code,'Индекс фации':index,'Название породы':rock}})
        self.assertEqual(5, len(_merged_layers({'layers':layers})))
        self.assertIn('не заполнены', _lithology_description({}))


if __name__ == '__main__':
    unittest.main()
