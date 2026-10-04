from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from openpyxl import Workbook
from PySide6.QtCore import QPoint, QProcess, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QLabel, QTableWidgetItem

from excel_photo_model_studio.gui import MainWindow, StepVerificationDialog
from excel_photo_model_studio.matching import read_photo_map, write_photo_map
from excel_photo_model_studio.models import PhotoRecord
from excel_photo_model_studio.project import create_project


class GuiPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_matching_tab_shows_photo_with_interval_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excel = root / "description.xlsx"
            photos = root / "photos"
            photos.mkdir()
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["Скважина", "Кровля", "Подошва", "Класс", "Краткое описание"])
            sheet.append(["W-1", 100.0, 102.0, "Sand", "Sandstone description"])
            workbook.save(excel)
            image = np.full((240, 160, 3), (20, 80, 150), dtype=np.uint8)
            cv2.rectangle(image, (55, 20), (105, 220), (115, 115, 115), -1)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            (photos / "W-1 100-102.jpg").write_bytes(encoded.tobytes())
            project = root / "project"
            create_project(excel, photos, project)

            window = MainWindow()
            window.show()
            self.application.processEvents()
            window.current_project = project
            window._load_photo_map()
            window._load_review()
            self.application.processEvents()

            self.assertEqual(1, window.photo_table.rowCount())
            self.assertTrue(window.ocr.isChecked())
            self.assertEqual("yolo11s-seg.pt", window.pretrained_weights.currentData())
            self.assertEqual("auto", window.photo_table.cellWidget(0, 5).currentData())
            self.assertIn("Колонок керна найдено: 1", window.matching_preview_title.text())
            self.assertIn("Интервалы колонок: 1) 100–101 м", window.matching_preview_title.text())
            self.assertIn("Порядок:", window.matching_preview_title.text())
            self.assertIn("Найдено интервалов", window.matching_preview_title.text())
            self.assertIsNotNone(window.matching_preview.pixmap())
            self.assertFalse(window.matching_preview.pixmap().isNull())
            tooltip = window.matching_preview.tooltip_for_image_point(80, 120)
            self.assertIn(
                "Краткое описание: Sandstone description",
                tooltip,
            )
            self.assertIn("Участок на фото: 100–101 м", tooltip)
            self.assertIn("Полный интервал фации: 100–102 м", tooltip)
            report = json.loads((project / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(any("длиннее вместимости найденного керна" in issue["message"] for issue in report["issues"]))
            window.close()

    def test_mask_review_can_confirm_all_photos_at_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excel = root / "description.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["Скважина", "Кровля", "Подошва", "Код", "Краткое описание"])
            sheet.append(["W-1", 100.0, 101.0, "Dch", "Песчаник"])
            workbook.save(excel)
            photo_records = []
            boxes = {}
            for index in range(2):
                path = root / f"page-{index}.jpg"
                image = np.full((100, 80, 3), 255, dtype=np.uint8)
                cv2.rectangle(image, (20, 10), (60, 90), (120, 120, 120), -1)
                ok, encoded = cv2.imencode(".jpg", image)
                self.assertTrue(ok)
                path.write_bytes(encoded.tobytes())
                photo_records.append(PhotoRecord(path, "W-1", 100.0, 101.0, "manual", True))
                boxes[path] = [(18, 8, 62, 92)]

            dialog = StepVerificationDialog(excel, root, use_ocr=False)
            try:
                deadline = time.monotonic() + 5
                while dialog._busy() and time.monotonic() < deadline:
                    self.application.processEvents()
                    time.sleep(0.01)
                self.application.processEvents()
                dialog.photos = photo_records
                dialog.columns = boxes
                dialog._current_photo = 0
                dialog._render_mask_photo()

                dialog.mask_confirm_all.click()

                self.assertEqual({photo.path for photo in photo_records}, dialog.mask_confirmed)
                self.assertFalse(dialog.mask_confirm_all.isEnabled())
                self.assertIn("2 из 2", dialog.status.text())
                self.assertTrue(any("подтвердил все фото" in item for item in dialog.audit))
            finally:
                dialog.close()

    def test_review_tab_can_approve_or_clear_all_mask_rows(self):
        window = MainWindow()
        try:
            window.review_table.setRowCount(3)
            for row in range(3):
                item = QTableWidgetItem()
                item.setCheckState(Qt.CheckState.Unchecked)
                window.review_table.setItem(row, 0, item)

            window._set_all_review_approvals(True)
            self.assertTrue(all(
                window.review_table.item(row, 0).checkState() == Qt.CheckState.Checked
                for row in range(3)
            ))
            window._set_all_review_approvals(False)
            self.assertTrue(all(
                window.review_table.item(row, 0).checkState() == Qt.CheckState.Unchecked
                for row in range(3)
            ))
        finally:
            window.close()

    def test_step_verification_can_open_and_switch_the_source_workbook(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photos = root / "photos"
            photos.mkdir()

            def make_workbook(path: Path, well: str):
                workbook = Workbook()
                sheet = workbook.active
                sheet.title = "седимент"
                sheet.append([
                    "Скважина", "Интервал фации по бурению Кровля",
                    "Интервал фации по бурению Подошва", "Толщина фации, м",
                    "Код фации", "Краткое описание",
                ])
                sheet.append([well, 100.0, 101.0, 1.0, "Dch", f"Песчаник {well}"])
                workbook.save(path)

            first = root / "first.xlsx"
            second = root / "second.xlsx"
            make_workbook(first, "W-1")
            make_workbook(second, "W-2")
            dialog = StepVerificationDialog(first, photos, use_ocr=False)

            def wait_for_excel_load():
                deadline = time.monotonic() + 5
                while dialog._busy() and time.monotonic() < deadline:
                    self.application.processEvents()
                    time.sleep(0.01)
                self.application.processEvents()
                self.assertFalse(dialog._busy(), "Excel worker did not finish in time")

            try:
                wait_for_excel_load()
                self.assertEqual("W-1", dialog.rows[0].well)
                self.assertEqual(2, dialog.raw_sheet_table.rowCount())
                self.assertIn("Интервал фации по бурению Кровля", dialog.raw_sheet_table.item(0, 1).text())
                self.assertIn("Кровля/начало", dialog.mapping_summary.toPlainText())
                self.assertIn("Толщина фации", dialog.mapping_summary.toPlainText())

                with patch(
                    "excel_photo_model_studio.gui.QFileDialog.getOpenFileName",
                    return_value=(str(second), ""),
                ) as choose_file:
                    dialog._choose_excel()
                self.assertEqual(
                    QFileDialog.Options(QFileDialog.Option.DontUseNativeDialog),
                    choose_file.call_args.kwargs["options"],
                )
                wait_for_excel_load()

                self.assertEqual(second.resolve(), dialog.excel_path)
                self.assertEqual(str(second.resolve()), dialog.excel_file_label.text())
                self.assertEqual("W-2", dialog.rows[0].well)
                self.assertIn("W-2", dialog.raw_sheet_table.item(1, 0).text())
            finally:
                dialog.close()

    def test_step_workbook_picker_stays_available_during_initial_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photos = root / "photos"
            photos.mkdir()

            def make_workbook(path: Path, well: str):
                workbook = Workbook()
                sheet = workbook.active
                sheet.append([
                    "Скважина", "Интервал фации по бурению Кровля",
                    "Интервал фации по бурению Подошва", "Толщина фации, м",
                    "Индекс фации", "Название фации", "Краткое описание",
                ])
                sheet.append([well, 100.0, 101.0, 1.0, "Dch", "Каналы", f"Описание {well}"])
                workbook.save(path)

            first, second = root / "first.xlsx", root / "second.xlsx"
            make_workbook(first, "W-1")
            make_workbook(second, "W-2")
            dialog = StepVerificationDialog(first, photos, use_ocr=False)
            try:
                self.assertTrue(dialog.excel_picker_button.isEnabled())
                with patch(
                    "excel_photo_model_studio.gui.QFileDialog.getOpenFileName",
                    return_value=(str(second), ""),
                ) as choose_file:
                    dialog._choose_excel()
                self.assertEqual(
                    QFileDialog.Options(QFileDialog.Option.DontUseNativeDialog),
                    choose_file.call_args.kwargs["options"],
                )
                self.assertEqual(second, dialog._pending_excel_path)

                deadline = time.monotonic() + 5
                while (dialog._busy() or dialog._pending_excel_path is not None) and time.monotonic() < deadline:
                    self.application.processEvents()
                    time.sleep(0.01)
                self.application.processEvents()

                self.assertFalse(dialog._busy(), "Excel workers did not finish in time")
                self.assertEqual(second.resolve(), dialog.excel_path)
                self.assertEqual("W-2", dialog.rows[0].well)
                self.assertEqual("Каналы", dialog.rows[0].facies_name)
                self.assertEqual("Описание W-2", dialog.rows[0].target_text)
            finally:
                dialog.close()

    def test_step_photo_review_can_remove_and_restore_an_extra_core_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excel = root / "description.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append([
                "Скважина", "Интервал фации по бурению Кровля",
                "Интервал фации по бурению Подошва", "Код фации", "Краткое описание",
            ])
            sheet.append(["W-1", 100.0, 101.0, "Dch", "Песчаник"])
            workbook.save(excel)
            photo_path = root / "W-1.jpg"
            image = np.full((300, 180, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (25, 30), (80, 270), (130, 130, 130), -1)
            cv2.rectangle(image, (100, 30), (155, 270), (130, 130, 130), -1)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            photo_path.write_bytes(encoded.tobytes())
            dialog = StepVerificationDialog(excel, root, use_ocr=False)

            try:
                deadline = time.monotonic() + 5
                while dialog._busy() and time.monotonic() < deadline:
                    self.application.processEvents()
                    time.sleep(0.01)
                self.application.processEvents()
                self.assertFalse(dialog._busy(), "Excel worker did not finish in time")

                photo = PhotoRecord(photo_path, "W-1", 100.0, 101.0, "manual", True)
                first, extra = (20, 25, 82, 275), (98, 25, 157, 275)
                dialog.photos = [photo]
                dialog.columns = {photo_path: [first, extra]}
                dialog._current_photo = 0
                dialog._stage = 1
                dialog._render_column_photo()

                # Clicking the second cyan contour selects it; deleting it
                # marks the photo unconfirmed so it must be reviewed again.
                pixmap = dialog.column_preview.pixmap()
                self.assertIsNotNone(pixmap)
                click_x = (dialog.column_preview.width() - pixmap.width()) / 2 + 127 * pixmap.width() / 180
                click_y = (dialog.column_preview.height() - pixmap.height()) / 2 + 150 * pixmap.height() / 300
                QTest.mouseClick(
                    dialog.column_preview, Qt.MouseButton.LeftButton,
                    pos=QPoint(round(click_x), round(click_y)),
                )
                self.assertEqual(1, dialog._selected_column)
                self.assertTrue(dialog.column_remove.isEnabled())
                dialog.column_remove.click()
                self.assertEqual([first], dialog.columns[photo_path])
                self.assertNotIn(photo_path, dialog.column_confirmed)
                self.assertTrue(dialog.column_undo_remove.isEnabled())

                dialog.column_undo_remove.click()
                self.assertEqual([first, extra], dialog.columns[photo_path])
                self.assertFalse(dialog.column_undo_remove.isEnabled())
            finally:
                dialog.close()

    def test_workbook_selected_in_step_dialog_is_used_for_project_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.xlsx"
            selected = root / "selected.xlsx"
            first.write_bytes(b"placeholder")
            selected.write_bytes(b"placeholder")
            photos = root / "photos"
            photos.mkdir()
            photo = photos / "core.jpg"
            confirmed_record = PhotoRecord(photo, "W-1", 100.0, 101.0, "manual", True)
            confirmed_boxes = [(10, 20, 80, 220)]
            window = MainWindow()
            window.excel.set_value(first)
            window.photos.set_value(photos)

            class AcceptedStepDialog:
                def __init__(self, *_args, **_kwargs):
                    self.excel_path = selected
                    self.photos = [confirmed_record]
                    self.columns = {photo: confirmed_boxes}
                    self.orders = {photo: "left_to_right"}

                def exec(self):
                    return QDialog.DialogCode.Accepted

            try:
                with patch("excel_photo_model_studio.gui.StepVerificationDialog", AcceptedStepDialog), \
                        patch.object(window, "_create_project") as create_project:
                    window._open_step_verification()

                self.assertEqual(selected, window.excel.value())
                self.assertEqual([confirmed_record], window._pending_verified_photo_records)
                self.assertEqual({photo: confirmed_boxes}, window._pending_verified_columns)
                self.assertEqual({photo: "left_to_right"}, window._pending_verified_orders)
                create_project.assert_called_once_with()
            finally:
                window.close()

    def test_recalculate_retains_ocr_coordinate_basis_unless_depth_is_edited(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            photo = project / "core.jpg"
            image = np.full((200, 200, 3), 255, dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            photo.write_bytes(encoded.tobytes())
            original = PhotoRecord(
                photo, "W-1", 100.0, 101.0, "ocr_verified", True,
                column_depths=((0.5, 100.0, 101.0),), column_ocr_checked=True,
                depth_basis="gis",
            )
            write_photo_map(project / "photo_map.csv", [original])
            window = MainWindow()
            window.current_project = project
            window._load_photo_map()
            with patch.object(window, "_start_project_process"):
                window._save_photo_map()
            retained = read_photo_map(project / "photo_map.csv")[0]
            self.assertEqual("gis", retained.depth_basis)
            self.assertEqual(original.column_depths, retained.column_depths)
            self.assertIn("по ГИС / с увязкой", window.matching_preview_title.text())

            window.photo_table.item(0, 3).setText("99.9")
            with patch.object(window, "_start_project_process"):
                window._save_photo_map()
            edited = read_photo_map(project / "photo_map.csv")[0]
            self.assertEqual("unknown", edited.depth_basis)
            self.assertEqual((), edited.column_depths)
            self.assertFalse(edited.column_ocr_checked)
            self.assertEqual("manual", edited.source)
            self.assertIn("Система глубин: не определена", window.matching_preview_title.text())
            window.close()

    def test_report_exposes_projection_failures_and_incomplete_facies(self):
        window = MainWindow()
        report = {
            "project_dir": "example", "excel_rows": 1, "photos": 1,
            "confirmed_photos": 1, "unconfirmed_photos": 0,
            "annotations": 1, "approved_annotations": 0, "blocking_errors": 2,
            "projection_errors": 1, "facies_rows_with_incomplete_masks": 1,
            "issues": [{"severity": "error", "source": "core.jpg", "message": "Колонка керна не покрыта масками."}],
        }
        window._show_report(report)
        message = window.project_log.toPlainText()
        self.assertIn("Общая проверка проекта нашла ошибки", message)
        self.assertIn("Ошибок привязки масок к колонкам керна: 1", message)
        self.assertIn("Фаций Excel с неполным покрытием масками: 1", message)
        self.assertEqual(["Колонка керна не покрыта масками."], window.matching_photo_issues["core.jpg"])
        window.close()

    def test_training_tab_lists_confirmed_wells_ready_to_accumulate(self):
        empty = {
            "summary": {"projects": 0, "photos": 0, "annotations": 0, "approved_annotations": 0},
            "wells": [],
        }
        with patch("excel_photo_model_studio.gui.catalog_overview", return_value=empty):
            window = MainWindow()
        try:
            self.assertIn("Пока нет подтверждённых масок", window.catalog_wells.toPlainText())
            confirmed = {
                "summary": {"projects": 1, "photos": 12, "annotations": 38, "approved_annotations": 38},
                "wells": [{
                    "well_names": ["671ПО"], "photos": 12, "masks": 38, "facies": 4,
                }],
            }
            with patch("excel_photo_model_studio.gui.catalog_overview", return_value=confirmed):
                window._update_catalog_status()

            text = window.catalog_wells.toPlainText()
            self.assertIn("Подтверждённые маски, включённые в накапливаемый датасет", text)
            self.assertIn("Скважина 671ПО — фото: 12; маски: 38; фаций: 4", text)
            self.assertIn("скважин — 1", window.catalog_status.text())
        finally:
            window.close()

    def test_yolo_export_has_own_tab_and_launches_dataset_only(self):
        overview = {
            "summary": {"projects": 1, "photos": 8, "annotations": 24, "approved_annotations": 24},
            "wells": [],
        }
        with patch("excel_photo_model_studio.gui.catalog_overview", return_value=overview):
            window = MainWindow()
        try:
            self.assertEqual("4. Экспорт YOLO Segmentation", window.tabs.tabText(3))
            self.assertEqual("5. Новый керн → Excel", window.tabs.tabText(4))
            self.assertIn("8 фото", window.export_catalog_status.text())
            self.assertIn("24 подтверждённых масок", window.export_catalog_status.text())
            tab_labels = " ".join(label.text() for label in window.tabs.widget(3).findChildren(QLabel))
            self.assertIn("не запускает обучение", tab_labels)

            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "yolo-export"
                catalog = Path(directory) / "catalog.json"
                window.export_dataset_output.setText(str(destination))
                with (
                    patch("excel_photo_model_studio.gui.catalog_overview", return_value=overview),
                    patch("excel_photo_model_studio.gui.default_catalog_path", return_value=catalog),
                    patch.object(QProcess, "start") as start,
                ):
                    window._export_dataset()

                start.assert_called_once_with()
                arguments = window.export_dataset_process.arguments()
                self.assertIn("dataset", arguments)
                self.assertNotIn("train", arguments)
                self.assertEqual(str(catalog), arguments[arguments.index("--catalog") + 1])
                self.assertEqual(str(destination), arguments[arguments.index("--output") + 1])
                self.assertFalse(destination.exists())

                existing = Path(directory) / "do-not-overwrite"
                existing.mkdir()
                window.export_dataset_output.setText(str(existing))
                with patch.object(window, "_error") as show_error:
                    window._export_dataset()
                show_error.assert_called_once()
                self.assertIn("уже существует", show_error.call_args.args[0])
        finally:
            window.close()

    def test_yolo_export_picker_creates_a_new_child_path_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            (parent / "YOLO_segmentation_dataset").mkdir()
            with patch("excel_photo_model_studio.gui.catalog_overview", return_value={
                "summary": {"projects": 0, "photos": 0, "annotations": 0, "approved_annotations": 0},
                "wells": [],
            }):
                window = MainWindow()
            try:
                with patch("excel_photo_model_studio.gui.QFileDialog.getExistingDirectory", return_value=str(parent)):
                    window._choose_export_destination()
                selected = Path(window.export_dataset_output.text())
                self.assertEqual(parent / "YOLO_segmentation_dataset_2", selected)
                self.assertFalse(selected.exists())
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
