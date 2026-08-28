from __future__ import annotations

import math
import re
import shutil
from collections import deque
from bisect import bisect_left
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QPointF, QSettings, QThread, Qt
from PySide6.QtGui import QAction, QColor, QCursor, QPolygonF
from PySide6.QtWidgets import QApplication, QColorDialog, QFileDialog, QDockWidget, QFrame, QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMenu, QMessageBox, QProgressBar, QProgressDialog, QPushButton, QSpinBox, QStatusBar, QTabWidget, QToolBar, QToolButton, QTreeWidget, QTreeWidgetItem

from app.domain.models import PhotoRecord
from app.infrastructure.io.las_parser import parse_las_file
from app.infrastructure.core_report_export import export_core_description_report
from app.infrastructure.cvat_coco_import import CvatImagesMissingError, import_cvat_coco_zip, import_cvat_coco_zips
from app.infrastructure.excel_core_description import CoreInterval, create_automatic_interval_detections, create_depth_bound_detections, layers_for_photo, photo_interval_from_filename, read_description_workbook
from app.infrastructure.facies_postprocess import postprocess_detections
from app.infrastructure.image_loading import load_working_pixmap
from app.infrastructure.lithology_attribute_service import LithologyAttributeService
from app.infrastructure.ml.fine_tune_worker import FineTuneWorker
from app.infrastructure.ml.rule_based_facies import RuleBasedFaciesDetector
from app.infrastructure.ml.yolo_model_service import SegmentationWorker
from app.infrastructure.pdf_photo_import import render_pdf_pages
from app.infrastructure.photo_caption_ocr import read_caption_metadata
from app.infrastructure.project_storage import MANIFEST_NAME, load_project, save_project
from app.infrastructure.training_dataset import MIN_TRAINING_SAMPLES, automatic_samples_count, export_training_dataset, unlabeled_manual_samples_count, verified_samples_count
from app.infrastructure.training_quality import review_queue, training_quality
from app.runtime_paths import bundled_root, user_data_root
from app.ui.dialogs.depth_range_dialog import DepthRangeDialog
from app.ui.dialogs.core_columns_dialog import CoreColumnsDialog
from app.ui.dialogs.facies_dialog import FaciesDialog
from app.ui.dialogs.log_editor_dialog import LogEditorDialog
from app.ui.dialogs.photo_interval_dialog import PhotoIntervalDialog
from app.ui.dialogs.photo_interval_mapping_dialog import PhotoIntervalMappingDialog
from app.ui.dialogs.well_depth_dialog import WellDepthDialog
from app.ui.widgets.workspace_canvas import StackColumnItem, WorkspaceCanvas


class MainWindow(QMainWindow):
    """Kern Analyzer board built with the PySide patterns of the original app."""

    TREE_KIND_ROLE = Qt.ItemDataRole.UserRole
    TREE_VALUE_ROLE = Qt.ItemDataRole.UserRole + 1
    TREE_INDEX_ROLE = Qt.ItemDataRole.UserRole + 2
    PHOTO_LOAD_BATCH_SIZE = 24

    def __init__(self):
        super().__init__()
        self._records: list[PhotoRecord] = []
        self._segmentation_thread: QThread | None = None
        self._segmentation_worker: SegmentationWorker | None = None
        self._segmentation_failed = False
        self._training_thread: QThread | None = None
        self._training_worker: FineTuneWorker | None = None
        self._training_epoch_current = 0
        self._training_epoch_total = 0
        self._fine_tune_success_model: Path | None = None
        self._activity_dialog: QProgressDialog | None = None
        self._segmentation_updated_count = 0
        self._deleted_records: list[tuple[PhotoRecord, int, object]] = []
        self._queued_records: list[PhotoRecord] = []
        self._pending_photo_imports: deque[tuple[Path, str, float | None, float | None]] = deque()
        self._facies_dialogs: list[FaciesDialog] = []
        self._selected_model_path: Path | None = None
        self._gis_data: dict | None = None
        self._rigis_data: dict | None = None
        self._settings = QSettings("Kern Analyzer", "KernAnalyzer")
        self._confidence_threshold = self._read_confidence_threshold()
        self._segmentation_profile = self._read_segmentation_profile()
        self._well_logs: dict[str, dict[str, dict]] = {}
        self._empty_wells: list[str] = []
        self._well_order: list[str] = []
        self._well_depth_ranges: dict[str, tuple[float, float]] = {}
        self._well_depth_references: dict[str, str] = {}
        self._well_depth_settings: dict[str, dict] = {}
        self._well_intervals: dict[str, tuple[float, float]] = {}
        self._well_interpretations: dict[str, list[dict]] = {}
        self._active_well_name = str(self._settings.value("last/active_well", "Скважина 1"))
        self._project_folder: Path | None = None
        self._project_title = "Новый проект"

        self.setWindowTitle("Kern Analyzer")
        self.setMinimumSize(960, 640)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("ЛКМ — переместить фото · зажатое колёсико — двигать доску · колёсико — масштаб")

        self.workspace = WorkspaceCanvas(self)
        self._connect_workspace(self.workspace, source_photos=True)
        self.sheets = QTabWidget(self)
        self.sheets.setObjectName("workspaceSheets")
        self.sheets.setDocumentMode(True)
        self.sheets.setTabPosition(QTabWidget.TabPosition.South)
        self.sheets.addTab(self.workspace, "\u041a\u0435\u0440\u043d \u0438 \u043e\u043f\u0438\u0441\u0430\u043d\u0438\u0435")
        # Пока это единственная рабочая область, нижняя вкладка только
        # дублирует очевидное содержимое и занимает полезное место.
        self.sheets.tabBar().hide()
        self.setCentralWidget(self.sheets)
        self._build_ui()
        self._build_project_sidebar()

    def _connect_workspace(self, workspace: WorkspaceCanvas, source_photos: bool = False) -> None:
        """Connect the core-description workspace to the project data model."""
        workspace.facies_selected.connect(self._begin_polygon_edit)
        workspace.facies_context_requested.connect(self._open_facies_editor)
        workspace.facies_geometry_changed.connect(self._on_facies_geometry_changed)
        workspace.facies_created.connect(self._add_manual_contour)
        workspace.column_paint_requested.connect(self._paint_stack_column)
        workspace.interpretation_interval_created.connect(self._create_interpretation_interval)
        workspace.interpretation_interval_resized.connect(self._resize_interpretation_interval)
        workspace.interpretation_interval_context_requested.connect(self._open_interpretation_interval_menu)
        workspace.well_order_changed.connect(self._set_well_order_from_canvas)
        if source_photos:
            workspace.photo_deleted.connect(self._remove_photo)

    def _build_ui(self) -> None:
        toolbar = QToolBar("Главная панель", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        title = QLabel("  KERN ANALYZER  ")
        title.setObjectName("appTitle")
        toolbar.addWidget(title)
        toolbar.addSeparator()
        toolbar.addWidget(
            self._create_toolbar_menu(
                "File",
                [
                    ("Новый проект", self.new_project, "Очистить рабочую область и начать новый проект", "Ctrl+N"),
                    ("Открыть проект…", self.open_project_dialog, "Открыть папку сохранённого проекта", "Ctrl+Shift+O"),
                    ("Сохранить проект", self.save_project, "Сохранить текущий проект", "Ctrl+S"),
                    ("Сохранить проект как…", self.save_project_as, "Сохранить проект в новой папке", "Ctrl+Shift+S"),
                    ("Загрузить фото/PDF…", self.open_images, "Выбрать изображения или PDF; каждая страница PDF станет отдельным фото", "Ctrl+O"),
                    ("Настроить колонки керна на фото…", self.configure_core_columns, "Вручную задать границы керна; фон, линейки и шлак будут исключены"),
                    ("Автоматически определить интервалы на фото…", self.detect_photo_intervals, "Выделить на фото устойчивые визуальные интервалы и заполнить строки без назначения фаций"),
                    ("Импорт разметки CVAT (COCO ZIP)…", self.import_cvat_coco, "Создать проект из фото, контуров и классов из выгрузки CVAT"),
                    ("Импорт нескольких CVAT ZIP…", self.import_cvat_coco_batch, "Последовательно объединить несколько COCO ZIP с защитой памяти и дубликатов"),
                    ("Импорт Excel + JPG для обучения…", self.import_excel_photo_batch, "Сопоставить интервалы описания Excel с подписями на фото керна"),
                    ("Сохранить послойное описание (Excel/Word/PDF)…", self.export_core_description_report, "Сформировать послойное описание по вкладке «Керн и описание»"),
                ],
                "Работа с файлами",
            )
        )
        toolbar.addWidget(
            self._create_toolbar_menu(
                "Модель",
                [
                    ("Выбрать модель сегментации…", self.select_detection_model, "Выбрать файл модели .pt"),
                    ("История дообучений…", self.show_training_history, "Сравнить сохранённые модели по validation-метрикам"),
                    ("Проверить датасет перед обучением…", self.show_training_quality, "Проверить малые классы, дубликаты и размер масок"),
                    ("Порог уверенности масок…", self.select_confidence_threshold, "Ниже порог — больше масок для ручной проверки"),
                    ("Режим сегментации…", self.select_segmentation_profile, "Быстрый, стандартный или детальный режим обработки фото"),
                    ("Проверить сомнительные маски…", self.review_uncertain_masks, "Открыть очередь автоматических масок с низкой уверенностью"),
                    ("Подобрать атрибуты пород…", self.suggest_lithology_attributes, "Отдельно от фаций предложить цвет, зернистость и слоистость внутри масок"),
                ],
                "Модель детекции и сегментации",
            )
        )
        self._new_contour_action = QAction("Новый контур", self)
        self._new_contour_action.setCheckable(True)
        self._new_contour_action.setToolTip("Нарисовать контур вручную на фотографии")
        self._new_contour_action.toggled.connect(self._set_new_contour_mode)
        toolbar.addAction(self._new_contour_action)
        add_vertex = QAction("＋ Точка", self)
        add_vertex.setToolTip("Добавить вершину в выбранный контур: нажмите кнопку, затем щёлкните по контуру")
        add_vertex.triggered.connect(self._prepare_polygon_vertex_insert)
        toolbar.addAction(add_vertex)
        delete_vertex = QAction("− Точка", self)
        delete_vertex.setToolTip("Удалить выделенную вершину контура (также работает клавиша Delete)")
        delete_vertex.triggered.connect(self._delete_selected_polygon_vertex)
        toolbar.addAction(delete_vertex)
        finish_polygon_edit = QAction("✓ Готово", self)
        finish_polygon_edit.setToolTip("Закончить редактирование контура (Esc)")
        finish_polygon_edit.triggered.connect(self._finish_polygon_edit)
        toolbar.addAction(finish_polygon_edit)
        back_to_photos = QAction("К фото", self)
        back_to_photos.setToolTip("Вернуться к исходным фотографиям с разметкой")
        back_to_photos.setShortcut("Home")
        back_to_photos.triggered.connect(self.return_to_photos)
        toolbar.addAction(back_to_photos)
        toolbar.addSeparator()
        toolbar.addWidget(self._create_training_bar())
        self._register_shortcuts()

        self.setStyleSheet(
            """
            QMainWindow { background: #fbfcfe; }
            QTabWidget#workspaceSheets::pane { border: none; background: #fbfcfe; }
            QTabWidget#workspaceSheets::tab-bar { alignment: left; }
            QTabBar::tab { background: #eef1f7; color: #596176; border: 1px solid #d9deea; border-bottom: none; padding: 8px 18px; margin-right: 2px; min-width: 130px; }
            QTabBar::tab:selected { background: #ffffff; color: #5149ca; font-weight: 700; border-top: 2px solid #655be8; }
            QTabBar::tab:hover { background: #f7f7ff; color: #5149ca; }
            QToolBar#mainToolbar { background: #ffffff; border: none; border-bottom: 1px solid #e4e8f0; spacing: 8px; padding: 9px 14px; }
            QLabel#appTitle { color: #35306f; font-weight: 800; letter-spacing: 1px; padding: 0 10px; }
            QToolButton#toolbarMenuButton { color: #262b3d; font-weight: 600; background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 7px 11px; }
            QToolButton#toolbarMenuButton:hover { background: #f0f1ff; border-color: #e4e3ff; color: #554dcc; }
            QToolBar QToolButton { color: #262b3d; font-weight: 600; background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 7px 11px; }
            QToolBar QToolButton:hover, QToolBar QToolButton:checked { background: #f0f1ff; border-color: #e4e3ff; color: #554dcc; }
            QFrame#trainingBar { background: #f5f7ff; border: 1px solid #dde2fb; border-radius: 8px; padding: 2px 4px; }
            QLabel#trainingCount { color: #4a5270; font-size: 12px; font-weight: 600; padding: 2px 5px; }
            QLabel#trainingProgressLabel { color: #554dcc; font-size: 12px; font-weight: 700; padding: 2px 3px; }
            QProgressBar#trainingProgress { min-width: 72px; max-width: 72px; max-height: 10px; border: 1px solid #cfcdf1; border-radius: 5px; background: #ebeafe; }
            QProgressBar#trainingProgress::chunk { background: #554dcc; border-radius: 4px; }
            QSpinBox#trainingEpochs { min-width: 46px; max-width: 64px; padding: 3px; border: 1px solid #cfd4e1; border-radius: 5px; background: #ffffff; color: #34394b; }
            QPushButton#trainButton { color: white; background: #554dcc; border: none; border-radius: 5px; padding: 6px 10px; font-weight: 700; }
            QPushButton#trainButton:hover { background: #433bb5; }
            QPushButton#trainButton:disabled { color: #8d95ad; background: #e2e6f1; }
            QMenu#toolbarDropdown { background: #ffffff; border: 1px solid #dde1eb; border-radius: 8px; padding: 5px; }
            QMenu#toolbarDropdown::item { color: #252a3b; padding: 9px 34px 9px 12px; border-radius: 5px; }
            QMenu#toolbarDropdown::item:selected { background: #efefff; color: #5149ca; }
            QDockWidget { color: #3a4054; font-weight: 700; background: #ffffff; }
            QDockWidget::title { background: #ffffff; padding: 10px 12px; border-bottom: 1px solid #e6e9f0; }
            QTreeWidget#projectTree { border: none; background: #fbfcfe; color: #454b60; padding: 6px; }
            QTreeWidget#projectTree::item { height: 27px; border-radius: 5px; padding: 2px 5px; }
            QTreeWidget#projectTree::item:selected { background: #e9e8ff; color: #4e46bd; }
            QStatusBar { background: #ffffff; border-top: 1px solid #e4e8f0; color: #7c8497; padding-left: 12px; }
            """
        )

    def _create_training_bar(self) -> QFrame:
        bar = QFrame(self)
        bar.setObjectName("trainingBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(5, 2, 3, 2)
        layout.setSpacing(5)
        self._training_count_label = QLabel(bar)
        self._training_count_label.setObjectName("trainingCount")
        self._training_epochs_label = QLabel("Эпохи:", bar)
        self._training_epochs_label.setObjectName("trainingCount")
        self._training_epochs = QSpinBox(bar)
        self._training_epochs.setObjectName("trainingEpochs")
        self._training_epochs.setRange(1, 500)
        self._training_epochs.setValue(int(self._settings.value("training/epochs", 20)))
        self._training_epochs.setToolTip("Сколько эпох выполнить при следующем дообучении")
        self._training_epochs.valueChanged.connect(lambda value: self._settings.setValue("training/epochs", value))
        self._training_recommendation_label = QLabel(bar)
        self._training_recommendation_label.setObjectName("trainingCount")
        self._training_recommendation_label.setToolTip("Оценка по числу вручную проверенных слоёв; её можно изменить вручную.")
        self._training_progress_label = QLabel("⏳ Обучение…", bar)
        self._training_progress_label.setObjectName("trainingProgressLabel")
        self._training_progress = QProgressBar(bar)
        self._training_progress.setObjectName("trainingProgress")
        self._training_progress.setRange(0, 0)
        self._training_progress.setTextVisible(False)
        self._training_button = QPushButton("Дообучить", bar)
        self._training_button.setObjectName("trainButton")
        self._training_button.clicked.connect(self.start_fine_tune)
        layout.addWidget(self._training_count_label)
        layout.addWidget(self._training_epochs_label)
        layout.addWidget(self._training_epochs)
        layout.addWidget(self._training_recommendation_label)
        layout.addWidget(self._training_progress_label)
        layout.addWidget(self._training_progress)
        layout.addWidget(self._training_button)
        self._refresh_training_bar()
        return bar

    def _show_activity(self, title: str, label: str, total: int) -> None:
        if self._activity_dialog is None:
            self._activity_dialog = QProgressDialog(self)
            self._activity_dialog.setCancelButton(None)
            self._activity_dialog.setAutoClose(False)
            self._activity_dialog.setAutoReset(False)
            self._activity_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self._activity_dialog.setMinimumDuration(0)
        self._activity_dialog.setWindowTitle(title)
        self._activity_dialog.setLabelText(label)
        self._activity_dialog.setRange(0, max(1, total))
        self._activity_dialog.setValue(0)
        self._activity_dialog.show()

    def _update_activity(self, label: str, current: int, total: int) -> None:
        if self._activity_dialog is None:
            return
        self._activity_dialog.setLabelText(label)
        self._activity_dialog.setRange(0, max(1, total))
        self._activity_dialog.setValue(max(0, min(current, max(1, total))))

    def _close_activity(self) -> None:
        if self._activity_dialog is not None:
            self._activity_dialog.close()

    def _training_examples_count(self) -> int:
        return verified_samples_count(self._records)

    def _automatic_examples_count(self) -> int:
        return automatic_samples_count(self._records)

    def _unlabeled_manual_examples_count(self) -> int:
        return unlabeled_manual_samples_count(self._records)

    def _refresh_training_bar(self) -> None:
        if not hasattr(self, "_training_count_label"):
            return
        count = self._training_examples_count()
        quality = training_quality(self._records)
        automatic_count = self._automatic_examples_count()
        unlabeled_count = self._unlabeled_manual_examples_count()
        needed = max(0, MIN_TRAINING_SAMPLES - count)
        self._training_count_label.setText(f"Для обучения: {count} · новые: {unlabeled_count} · авто: {automatic_count}")
        self._training_count_label.setToolTip(quality.summary)
        recommended_epochs = self._recommended_training_epochs(count)
        self._training_recommendation_label.setText(f"рекомендовано: {recommended_epochs}")
        is_busy = self._segmentation_thread is not None or self._training_thread is not None
        is_training = self._training_thread is not None
        self._training_progress_label.setVisible(is_training)
        self._training_progress.setVisible(is_training)
        if is_training:
            total = max(1, self._training_epoch_total or self._training_epochs.value())
            current = max(0, min(total, self._training_epoch_current))
            self._training_progress_label.setText(
                f"⏳ Эпоха {current} из {total} · осталось {total - current}"
            )
            self._training_progress.setRange(0, total)
            self._training_progress.setValue(current)
        self._training_epochs.setEnabled(not is_busy)
        self._training_button.setEnabled(count >= MIN_TRAINING_SAMPLES and not is_busy)
        if is_busy:
            self._training_button.setToolTip("Дождитесь завершения текущей операции")
        elif needed:
            self._training_button.setToolTip(f"Для дообучения добавьте ещё {needed} вручную проверенных слоёв")
        else:
            self._training_button.setToolTip("Собрать разметки и дообучить копию выбранной модели")

    @staticmethod
    def _recommended_training_epochs(examples: int) -> int:
        """Conservative starting point for a small, manually reviewed dataset."""
        if examples < 20:
            return 80
        if examples < 50:
            return 60
        if examples < 100:
            return 45
        if examples < 250:
            return 35
        return 25

    def _build_project_sidebar(self) -> None:
        sidebar = QDockWidget("ПРОЕКТЫ", self)
        sidebar.setObjectName("projectSidebar")
        sidebar.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        sidebar.setMinimumWidth(230)
        sidebar.setMaximumWidth(330)
        self.project_tree = QTreeWidget(sidebar)
        self.project_tree.setObjectName("projectTree")
        self.project_tree.setHeaderHidden(True)
        self.project_tree.itemActivated.connect(self._activate_project_tree_item)
        self.project_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.project_tree.customContextMenuRequested.connect(self._open_project_tree_context_menu)
        sidebar.setWidget(self.project_tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, sidebar)
        self._refresh_project_tree()

    def _create_toolbar_menu(self, title: str, actions, tooltip: str | None = None) -> QToolButton:
        """The QToolBar + QToolButton + QMenu construction used in Kern Analyzer."""
        menu = QMenu(title, self)
        menu.setObjectName("toolbarDropdown")
        for entry in actions:
            action_title, handler, action_tip = entry[:3]
            action = QAction(action_title, self)
            action.triggered.connect(handler)
            action.setToolTip(action_tip)
            if len(entry) > 3:
                action.setShortcut(entry[3])
                self.addAction(action)
            menu.addAction(action)

        button = QToolButton(self)
        button.setObjectName("toolbarMenuButton")
        button.setText(f"{title} ▾")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setMenu(menu)
        button.setToolTip(tooltip or title)
        return button

    def _register_shortcuts(self) -> None:
        """Keyboard controls stay available without adding extra top-panel menus."""
        for shortcut, handler in (
            ("Ctrl+Z", self.undo_delete),
            ("Delete", self.workspace.delete_selected_photos),
            ("Ctrl+0", self.workspace.fit_content),
        ):
            action = QAction(self)
            action.setShortcut(shortcut)
            action.triggered.connect(handler)
            self.addAction(action)

    def return_to_photos(self) -> None:
        self.sheets.setCurrentWidget(self.workspace)
        if self.workspace.focus_photos():
            self.statusBar().showMessage("Возврат к исходным фотографиям и разметке")
        else:
            self.statusBar().showMessage("Сначала загрузите фотографии")

    def _projects_root(self) -> Path:
        return user_data_root() / "projects"

    def _trained_models_root(self) -> Path:
        """Published models live here instead of inside a training-run tree."""
        return user_data_root() / "models" / "trained"

    def _pdf_pages_root(self) -> Path:
        return self._projects_root() / "pdf_pages"

    @staticmethod
    def _safe_folder_name(value: str) -> str:
        name = "".join("_" if char in '<>:"/\\|?*' else char for char in value).strip(". ")
        return name or "untitled_project"

    def _refresh_project_tree(self) -> None:
        if not hasattr(self, "project_tree"):
            return
        self.project_tree.clear()
        opened = QTreeWidgetItem([f"Открыт: {self._project_title}"])
        opened.setExpanded(True)
        self.project_tree.addTopLevelItem(opened)
        well_nodes: dict[str, QTreeWidgetItem] = {}
        for well_name in self._well_names():
            well_node = QTreeWidgetItem([f"\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410 \u00b7 {well_name}"])
            well_node.setData(0, self.TREE_KIND_ROLE, "well")
            well_node.setData(0, self.TREE_VALUE_ROLE, well_name)
            well_node.setExpanded(True)
            opened.addChild(well_node)
            well_nodes[well_name] = well_node
        for record in self._records:
            child = QTreeWidgetItem([Path(record.path).name])
            child.setData(0, self.TREE_KIND_ROLE, "photo")
            child.setData(0, self.TREE_VALUE_ROLE, record.identifier)
            child.setToolTip(0, record.path)
            well_nodes.setdefault(record.well_name, opened).addChild(child)
            for display_index, detection in enumerate(StackColumnItem._sort_by_reading_order(record.detections), start=1):
                source_index = next(
                    index for index, source_detection in enumerate(record.detections) if source_detection is detection
                )
                interval = ""
                if detection.depth_from is not None or detection.depth_to is not None:
                    top = "?" if detection.depth_from is None else f"{detection.depth_from:g}"
                    base = "?" if detection.depth_to is None else f"{detection.depth_to:g}"
                    interval = f" · {top}–{base} м"
                layer = QTreeWidgetItem([f"Слой {display_index}: {detection.label}{interval}"])
                layer.setData(0, self.TREE_KIND_ROLE, "layer")
                layer.setData(0, self.TREE_VALUE_ROLE, record.identifier)
                layer.setData(0, self.TREE_INDEX_ROLE, source_index)
                child.addChild(layer)
            child.setExpanded(True)
        for well_name in ():
            well = QTreeWidgetItem([f"{well_name} · без керна"])
            well.setData(0, self.TREE_KIND_ROLE, "well")
            well.setData(0, self.TREE_VALUE_ROLE, well_name)
            opened.addChild(well)

        saved_root = QTreeWidgetItem(["Сохранённые проекты"])
        saved_root.setExpanded(True)
        self.project_tree.addTopLevelItem(saved_root)
        root = self._projects_root()
        if root.is_dir():
            for folder in sorted((item for item in root.iterdir() if (item / MANIFEST_NAME).is_file()), key=lambda item: item.name.casefold()):
                item = QTreeWidgetItem([folder.name.removesuffix(".kern_analyzer")])
                item.setData(0, self.TREE_KIND_ROLE, "project")
                item.setData(0, self.TREE_VALUE_ROLE, str(folder))
                item.setToolTip(0, str(folder))
                saved_root.addChild(item)

    def _activate_project_tree_item(self, item: QTreeWidgetItem, _column: int) -> None:
        kind = item.data(0, self.TREE_KIND_ROLE)
        value = item.data(0, self.TREE_VALUE_ROLE)
        if kind == "project" and value:
            self._open_project(Path(str(value)))
        elif kind == "photo" and value:
            self.workspace.focus_photo(str(value))
        elif kind == "layer" and value is not None:
            layer = self._tree_layer(str(value), item.data(0, self.TREE_INDEX_ROLE))
            if layer:
                record, detection = layer
                self.workspace.focus_facies(record, detection)

    def _open_project_tree_context_menu(self, position) -> None:
        item = self.project_tree.itemAt(position)
        if item is None or item.data(0, self.TREE_KIND_ROLE) != "layer":
            return
        value = item.data(0, self.TREE_VALUE_ROLE)
        layer = self._tree_layer(str(value), item.data(0, self.TREE_INDEX_ROLE))
        if layer is None:
            return
        record, detection = layer
        menu = QMenu(self)
        edit = menu.addAction("Редактировать параметры фации")
        edit.triggered.connect(lambda: self._open_facies_editor(record, detection))
        menu.exec(self.project_tree.viewport().mapToGlobal(position))

    def _tree_layer(self, identifier: str, index) -> tuple[PhotoRecord, object] | None:
        try:
            index = int(index)
        except (TypeError, ValueError):
            return None
        record = next((item for item in self._records if item.identifier == identifier), None)
        if record is None or not 0 <= index < len(record.detections):
            return None
        return record, record.detections[index]

    def new_project(self) -> None:
        if self._records:
            answer = QMessageBox.question(
                self,
                "Новый проект",
                "Создать новый проект? Несохранённые изменения текущего проекта будут потеряны.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.workspace.clear_workspace()
        self._records = []
        self._deleted_records.clear()
        self._pending_photo_imports.clear()
        self._project_folder = None
        self._selected_model_path = None
        self._project_title = "Новый проект"
        self._gis_data = None
        self._rigis_data = None
        self._well_logs.clear()
        self._empty_wells.clear()
        self._well_order.clear()
        self._well_depth_ranges.clear()
        self._well_depth_settings.clear()
        self._well_depth_references.clear()
        self._well_intervals.clear()
        self._well_interpretations.clear()
        self._active_well_name = "Скважина 1"
        self.setWindowTitle("Kern Analyzer — Новый проект")
        self._refresh_project_tree()
        self._refresh_training_bar()
        self.statusBar().showMessage("Создан новый проект")

    def open_project_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Открыть проект Kern Analyzer", str(self._projects_root()))
        if folder:
            self._open_project(Path(folder))

    def _open_project(self, folder: Path) -> None:
        if self._segmentation_thread is not None or self._training_thread is not None:
            QMessageBox.information(self, "Подождите", "Дождитесь окончания сегментации перед открытием проекта.")
            return
        try:
            title, records, positions, missing, wells = load_project(folder)
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось открыть проект", str(exc))
            return
        self.workspace.clear_workspace()
        self._records = records
        self._deleted_records.clear()
        self._project_folder = folder
        self._project_title = title
        self._selected_model_path = self._best_project_model(folder)
        self._gis_data = None
        self._rigis_data = None
        self._well_logs.clear()
        recorded_wells = {record.well_name for record in records}
        self._well_order = [
            str(item["name"])
            for item in sorted(wells, key=lambda item: int(item.get("order", 1_000_000)))
            if str(item.get("name") or "")
        ]
        self._empty_wells = [str(item["name"]) for item in wells if str(item["name"]) not in recorded_wells]
        self._well_intervals = {
            str(item["name"]): (float((item.get("core_interval") or item["interval"])[0]), float((item.get("core_interval") or item["interval"])[1]))
            for item in wells
            if isinstance(item.get("core_interval") or item.get("interval"), list) and len(item.get("core_interval") or item.get("interval")) == 2
        }
        self._well_depth_ranges = {
            str(item["name"]): (float(item["well_depth_range"][0]), float(item["well_depth_range"][1]))
            for item in wells
            if isinstance(item.get("well_depth_range"), list) and len(item["well_depth_range"]) == 2
        }
        self._well_depth_references = {
            str(item["name"]): str(item.get("depth_reference"))
            for item in wells
            if item.get("depth_reference")
        }
        self._well_depth_settings = {
            str(item["name"]): dict(item["depth_settings"])
            for item in wells
            if isinstance(item.get("depth_settings"), dict)
        }
        self._well_interpretations = {
            str(item["name"]): [dict(interval) for interval in item.get("interpretations", []) if isinstance(interval, dict)]
            for item in wells
            if item.get("interpretations")
        }
        self._active_well_name = records[0].well_name if records else "Скважина 1"
        for record in records:
            self.workspace.restore_photo(record, positions.get(record.identifier, QPointF()))
        if records:
            self._show_stack()
        self.setWindowTitle(f"Kern Analyzer — {title}")
        self._refresh_project_tree()
        self._refresh_training_bar()
        message = f"Открыт проект: {title} · фото: {len(records)}"
        if self._selected_model_path is not None:
            message += " · выбрана последняя дообученная модель"
        if missing:
            message += f" · не найдены: {len(missing)}"
        self.statusBar().showMessage(message)

    def save_project(self) -> None:
        if self._project_folder is None:
            self.save_project_as()
            return
        self._save_current_project(self._project_folder)

    def save_project_as(self) -> None:
        title, accepted = QInputDialog.getText(self, "Сохранить проект", "Название проекта:", text=self._project_title)
        if not accepted or not title.strip():
            return
        safe_title = "".join("_" if char in '<>:"/\\|?*' else char for char in title.strip()).strip(". ")
        if not safe_title:
            QMessageBox.warning(self, "Название проекта", "Укажите допустимое название.")
            return
        folder = self._projects_root() / f"{safe_title}.kern_analyzer"
        if folder.exists() and (folder / MANIFEST_NAME).is_file():
            answer = QMessageBox.question(
                self,
                "Перезаписать проект",
                f"Проект «{safe_title}» уже существует. Перезаписать его?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._project_title = title.strip()
        self._save_current_project(folder)

    def _save_current_project(self, folder: Path) -> None:
        try:
            save_project(folder, self._project_title, self._records, self.workspace.photo_positions(), self._project_well_metadata())
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось сохранить проект", str(exc))
            return
        self._project_folder = folder
        self.setWindowTitle(f"Kern Analyzer — {self._project_title}")
        self._refresh_project_tree()
        self.statusBar().showMessage(f"Проект сохранён: {folder.name}")

    def open_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Загрузить фото или PDF",
            "",
            "Изображения и PDF (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.pdf);;Изображения (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;PDF (*.pdf);;Все файлы (*)",
        )
        if not paths:
            return

        known_wells = self._well_names()
        suggested_well = self._active_well_name if self._active_well_name in known_wells else f"Скважина {len(known_wells) + 1}"
        well_name, accepted = QInputDialog.getText(
            self,
            "Скважина для фотографий",
            "Название скважины:",
            text=suggested_well,
        )
        if not accepted or not well_name.strip():
            return
        well_name = well_name.strip()
        self._active_well_name = well_name
        self._settings.setValue("last/active_well", well_name)

        image_paths: list[Path] = []
        pdf_errors: list[str] = []
        for selected_path in map(Path, paths):
            if selected_path.suffix.casefold() != ".pdf":
                image_paths.append(selected_path)
                continue
            try:
                image_paths.extend(render_pdf_pages(selected_path, self._pdf_pages_root()))
            except Exception as exc:
                pdf_errors.append(f"{selected_path.name}: {exc}")

        for image_path in image_paths:
            photo_depth_from, photo_depth_to = self._photo_interval_from_path(image_path)
            self._pending_photo_imports.append((image_path, well_name, photo_depth_from, photo_depth_to))

        if not image_paths:
            detail = "\n".join(pdf_errors)
            QMessageBox.warning(self, "Kern Analyzer", f"Не удалось открыть выбранные изображения.\n{detail}".strip())
            return
        message = f"Поставлено в очередь фото: {len(image_paths)} · загружаю партиями по {self.PHOTO_LOAD_BATCH_SIZE}…"
        if pdf_errors:
            message += f" · PDF с ошибкой: {len(pdf_errors)}"
        self.statusBar().showMessage(message)
        self._load_next_photo_batch()

    def _load_next_photo_batch(self) -> None:
        """Read and analyse a bounded batch so a large folder stays responsive."""
        if self._segmentation_thread is not None or not self._pending_photo_imports:
            return
        records: list[PhotoRecord] = []
        while self._pending_photo_imports and len(records) < self.PHOTO_LOAD_BATCH_SIZE:
            image_path, well_name, photo_depth_from, photo_depth_to = self._pending_photo_imports.popleft()
            pixmap = load_working_pixmap(image_path)
            if pixmap.isNull():
                continue
            record = PhotoRecord(
                identifier=str(uuid4()), path=str(image_path), pixmap=pixmap, well_name=well_name,
                photo_depth_from=photo_depth_from, photo_depth_to=photo_depth_to,
            )
            self._records.append(record)
            records.append(record)
        if not records:
            self._load_next_photo_batch()
            return
        self.workspace.add_photos(records)
        self._deleted_records.clear()
        self._refresh_project_tree()
        self._refresh_training_bar()
        self.statusBar().showMessage(
            f"Загружено {len(records)} фото · в очереди: {len(self._pending_photo_imports)} · запускаю сегментацию…"
        )
        self.run_segmentation(records)

    def import_cvat_coco(self) -> None:
        """Create a clean Kern Analyzer project from a human-marked CVAT COCO ZIP."""
        if self._segmentation_thread is not None or self._training_thread is not None:
            QMessageBox.information(self, "Импорт CVAT", "Дождитесь окончания текущей операции.")
            return
        archive_path, _ = QFileDialog.getOpenFileName(
            self,
            "Импорт разметки CVAT",
            "",
            "CVAT COCO ZIP (*.zip);;Все файлы (*)",
        )
        if not archive_path:
            return
        if self._records:
            answer = QMessageBox.question(
                self,
                "Импорт CVAT",
                "Импорт создаст новый проект и заменит несохранённые данные в рабочей области. Продолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_folder = self._projects_root() / f"cvat_{Path(archive_path).stem}_{stamp}"
        try:
            records, summary = import_cvat_coco_zip(Path(archive_path), project_folder / "images")
        except CvatImagesMissingError as exc:
            images_folder = QFileDialog.getExistingDirectory(
                self,
                "В ZIP нет изображений — выберите папку исходной задачи CVAT",
                str(Path(archive_path).parent),
            )
            if not images_folder:
                QMessageBox.information(self, "Импорт CVAT", str(exc))
                return
            try:
                records, summary = import_cvat_coco_zip(
                    Path(archive_path), project_folder / "images", Path(images_folder),
                )
            except Exception as retry_exc:
                QMessageBox.critical(self, "Импорт CVAT", str(retry_exc))
                return
        except Exception as exc:
            QMessageBox.critical(self, "Импорт CVAT", str(exc))
            return

        report_path = project_folder / "cvat_import_report.txt"
        report_path.write_text(
            "\n".join([
                f"Архив: {archive_path}",
                f"COCO JSON: {summary.get('annotation_file', '')}",
                f"Фото импортировано: {summary['images']}",
                f"Контуры импортировано: {summary['contours']}",
                f"Классы: {summary['classes']}",
                f"Изображения не найдены: {summary.get('missing_images', 0)}",
                f"Изображения не открылись: {summary.get('unreadable_images', 0)}",
                f"Аннотации пропущены: {summary.get('skipped_annotations', 0)}",
                "",
            ]),
            encoding="utf-8",
        )

        well_name, accepted = QInputDialog.getText(
            self,
            "Импорт CVAT",
            "Название скважины для импортированных фото:",
            text="Скважина 1",
        )
        if not accepted or not well_name.strip():
            return
        for record in records:
            record.well_name = well_name.strip()

        self.workspace.clear_workspace()
        self._records = records
        self._deleted_records.clear()
        self._project_folder = project_folder
        self._project_title = f"CVAT · {Path(archive_path).stem}"
        self._well_logs.clear()
        self._empty_wells.clear()
        self._well_order = [well_name.strip()]
        self._well_depth_ranges.clear()
        self._well_depth_references.clear()
        self._well_depth_settings.clear()
        self._well_intervals.clear()
        self._well_interpretations.clear()
        self._active_well_name = well_name.strip()
        self.workspace.add_photos(records)
        self._save_current_project(project_folder)
        self._show_stack()
        self._refresh_project_tree()
        self._refresh_training_bar()
        QMessageBox.information(
            self,
            "Импорт CVAT завершён",
            (
                f"Фото: {summary['images']}\n"
                f"Контуры: {summary['contours']}\n"
                f"Классы: {summary['classes']}\n"
                f"Пропущено аннотаций: {summary.get('skipped_annotations', 0)}\n\n"
                "Контуры уже отмечены как проверенные. Откройте слой и заполните 16 параметров описания.\n\n"
                f"Диагностика: {report_path}"
            ),
        )

    def import_cvat_coco_batch(self) -> None:
        """Sequentially import several COCO ZIPs; never unpack the full batch in memory."""
        if self._segmentation_thread is not None or self._training_thread is not None:
            QMessageBox.information(self, "Импорт CVAT", "Дождитесь окончания текущей операции.")
            return
        archive_paths, _ = QFileDialog.getOpenFileNames(
            self, "Импорт нескольких разметок CVAT", "", "CVAT COCO ZIP (*.zip);;Все файлы (*)",
        )
        if not archive_paths:
            return
        if self._records:
            answer = QMessageBox.question(
                self, "Импорт CVAT", "Импорт создаст новый проект и заменит текущую рабочую область. Продолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        paths = [Path(path) for path in archive_paths]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_folder = self._projects_root() / f"cvat_batch_{stamp}"

        def progress(current: int, total: int, label: str) -> None:
            self._update_activity(label, current, total)
            QApplication.processEvents()

        self._show_activity("Импорт CVAT", f"Подготовка {len(paths)} ZIP…", len(paths))
        try:
            records, summary = import_cvat_coco_zips(paths, project_folder / "images", progress=progress)
        except CvatImagesMissingError as exc:
            self._close_activity()
            images_folder = QFileDialog.getExistingDirectory(
                self, "В ZIP нет изображений — выберите общую папку исходных фото CVAT", str(paths[0].parent),
            )
            if not images_folder:
                QMessageBox.information(self, "Импорт CVAT", str(exc))
                return
            self._show_activity("Импорт CVAT", f"Подготовка {len(paths)} ZIP…", len(paths))
            try:
                records, summary = import_cvat_coco_zips(
                    paths, project_folder / "images", Path(images_folder), progress,
                )
            except Exception as retry_exc:
                QMessageBox.critical(self, "Импорт CVAT", str(retry_exc))
                return
        except Exception as exc:
            QMessageBox.critical(self, "Импорт CVAT", str(exc))
            return
        finally:
            self._close_activity()

        well_name, accepted = QInputDialog.getText(
            self, "Импорт CVAT", "Название скважины для импортированных фото:", text="Скважина 1",
        )
        if not accepted or not well_name.strip():
            return
        for record in records:
            record.well_name = well_name.strip()
        report_path = project_folder / "cvat_batch_import_report.txt"
        report_path.write_text(
            "\n".join([
                *[f"Архив: {path}" for path in paths],
                f"Фото импортировано: {summary['images']}",
                f"Контуры импортировано: {summary.get('contours', 0)}",
                f"Дубликаты фото: {summary.get('duplicate_images', 0)}",
                f"Пропущено аннотаций: {summary.get('skipped_annotations', 0)}",
                "",
            ]), encoding="utf-8",
        )
        self.workspace.clear_workspace()
        self._records = records
        self._deleted_records.clear()
        self._project_folder = project_folder
        self._project_title = f"CVAT · пакет {len(paths)} ZIP"
        self._well_logs.clear()
        self._empty_wells.clear()
        self._well_order = [well_name.strip()]
        self._well_depth_ranges.clear()
        self._well_depth_references.clear()
        self._well_depth_settings.clear()
        self._well_intervals.clear()
        self._well_interpretations.clear()
        self._active_well_name = well_name.strip()
        # ``add_photos`` remains the only long-lived image holder. Archives
        # have already been closed, and photos are on disk in per-job folders.
        self.workspace.add_photos(records)
        self._save_current_project(project_folder)
        self._show_stack()
        self._refresh_project_tree()
        self._refresh_training_bar()
        QMessageBox.information(
            self, "Пакетный импорт CVAT завершён",
            f"ZIP: {len(paths)}\nФото: {summary['images']}\nКонтуры: {summary.get('contours', 0)}\n"
            f"Дубликаты: {summary.get('duplicate_images', 0)}\n\nДиагностика: {report_path}",
        )

    def import_excel_photo_batch(self) -> None:
        """Build reviewed training candidates from a depth-description Excel and JPG folder."""
        if self._segmentation_thread is not None or self._training_thread is not None:
            QMessageBox.information(self, "Импорт Excel + JPG", "Дождитесь окончания текущей операции.")
            return
        excel_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите послойное описание",
            "",
            "Excel (*.xlsx *.xlsm);;Все файлы (*)",
        )
        if not excel_path:
            return
        photos_folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с JPG-фотографиями керна",
            str(Path(excel_path).parent),
        )
        if not photos_folder:
            return
        if self._records:
            answer = QMessageBox.question(
                self,
                "Импорт Excel + JPG",
                "Импорт создаст новый проект и заменит текущую рабочую область. Продолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            layers, workbook_issues = read_description_workbook(Path(excel_path))
        except Exception as exc:
            QMessageBox.critical(self, "Импорт Excel + JPG", str(exc))
            return
        photo_paths = sorted(
            [
                path
                for path in Path(photos_folder).rglob("*")
                if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg"}
            ],
            key=lambda path: path.name.casefold(),
        )
        if not photo_paths:
            QMessageBox.warning(self, "Импорт Excel + JPG", "В выбранной папке не найдены JPG-файлы.")
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_folder = self._projects_root() / f"excel_jpg_{Path(excel_path).stem}_{stamp}"
        images_folder = project_folder / "images"
        records: list[PhotoRecord] = []
        wells_in_excel = sorted({layer.well for layer in layers}, key=str.casefold)
        default_well = wells_in_excel[0] if len(wells_in_excel) == 1 else ""
        mapping_rows = [
            (path, *self._detect_photo_interval(path, default_well))
            for path in photo_paths
        ]
        mapping_dialog = PhotoIntervalMappingDialog(mapping_rows, self)
        while mapping_dialog.exec() == mapping_dialog.DialogCode.Accepted:
            try:
                photo_mappings = mapping_dialog.mappings()
                break
            except ValueError as exc:
                QMessageBox.warning(self, "Проверка интервалов", str(exc))
        else:
            return

        report: list[str] = [
            f"Excel: {excel_path}",
            f"Фото найдено: {len(photo_paths)}",
            "Сопоставление выполнено только по подтверждённым интервалам (имена файлов не используются как идентификатор).",
            "",
        ]
        report.extend(f"[Excel] {issue.source}: {issue.message}" for issue in workbook_issues)
        for number, (photo_path, photo_interval) in enumerate(photo_mappings, start=1):
            try:
                report.append(f"[Интервал] {photo_path.name}: {photo_interval.top:g}–{photo_interval.base:g} м (подтверждено пользователем)")
                matching_layers = layers_for_photo(
                    layers,
                    photo_interval.well,
                    photo_interval.top,
                    photo_interval.base,
                )
                if not matching_layers:
                    report.append(
                        f"[Пропуск] {photo_path.name}: в Excel нет пересекающегося интервала фации."
                    )
                    continue
                output_path = images_folder / f"{number:05d}_{photo_path.name}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(photo_path, output_path)
                from PySide6.QtGui import QPixmap

                pixmap = load_working_pixmap(output_path)
                if pixmap.isNull():
                    report.append(f"[Пропуск] {photo_path.name}: JPEG не удалось открыть.")
                    continue
                detections, image_issues, depth_segments = create_depth_bound_detections(
                    output_path,
                    pixmap,
                    photo_interval.top,
                    photo_interval.base,
                    matching_layers,
                )
                report.extend(f"[Проверка] {issue.source}: {issue.message}" for issue in image_issues)
                if not detections:
                    continue
                records.append(
                    PhotoRecord(
                        identifier=str(uuid4()),
                        path=str(output_path),
                        pixmap=pixmap,
                        detections=detections,
                        well_name=photo_interval.well,
                        photo_depth_from=photo_interval.top,
                        photo_depth_to=photo_interval.base,
                        depth_segments=depth_segments,
                        core_columns=[
                            {key: value for key, value in segment.items() if key in {"left", "top", "right", "bottom"}}
                            for segment in depth_segments
                        ],
                    )
                )
            except Exception as exc:
                report.append(f"[Пропуск] {photo_path.name}: {exc}")

        project_folder.mkdir(parents=True, exist_ok=True)
        report_path = project_folder / "excel_jpg_import_report.txt"
        report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        if not records:
            QMessageBox.warning(
                self,
                "Импорт Excel + JPG",
                f"Не удалось создать ни одной разметки. Проверьте отчёт:\n{report_path}",
            )
            return

        self.workspace.clear_workspace()
        self._records = records
        self._deleted_records.clear()
        self._project_folder = project_folder
        self._project_title = f"Excel + JPG · {Path(excel_path).stem}"
        self._well_logs.clear()
        self._empty_wells.clear()
        self._well_order = sorted({record.well_name for record in records})
        self._well_depth_ranges.clear()
        self._well_depth_references.clear()
        self._well_depth_settings.clear()
        self._well_intervals.clear()
        self._well_interpretations.clear()
        self._active_well_name = records[0].well_name
        self.workspace.add_photos(records)
        self._save_current_project(project_folder)
        self._show_stack()
        self._refresh_project_tree()
        self._refresh_training_bar()
        QMessageBox.information(
            self,
            "Импорт Excel + JPG завершён",
            (
                f"Сопоставлено фото: {len(records)} из {len(photo_paths)}\n"
                f"Создано слоёв для обучения: {sum(len(record.detections) for record in records)}\n\n"
                f"Диагностика сохранена в:\n{report_path}"
            ),
        )

    def _photo_interval_for_excel_import(self, photo_path: Path, default_well: str) -> tuple[CoreInterval | None, str]:
        """Use filename first, then the printed caption, then an explicit user interval."""
        try:
            return photo_interval_from_filename(photo_path), "имя файла"
        except ValueError:
            pass
        try:
            caption = read_caption_metadata(photo_path)
            return CoreInterval(caption.well or default_well, caption.top, caption.base), "OCR подписи"
        except Exception as exc:
            ocr_reason = str(exc)
        dialog = PhotoIntervalDialog(photo_path.name, well=default_well, parent=self)
        while dialog.exec() == dialog.DialogCode.Accepted:
            try:
                well, top, base = dialog.values()
                return CoreInterval(well, top, base), "введено вручную"
            except ValueError as exc:
                QMessageBox.warning(self, "Интервал фотографии", str(exc))
        self.statusBar().showMessage(f"Пропущено фото без интервала: {photo_path.name} · OCR: {ocr_reason}")
        return None, "не задан"

    def _detect_photo_interval(self, photo_path: Path, default_well: str) -> tuple[CoreInterval | None, str]:
        """Non-interactive prefill for the batch mapping table.

        The resulting value is always displayed for human confirmation; no
        filename or OCR guess can silently bind masks to an Excel layer.
        """
        try:
            interval = photo_interval_from_filename(photo_path)
            return CoreInterval(interval.well or default_well, interval.top, interval.base), "интервал найден в имени — подтвердите"
        except ValueError:
            pass
        try:
            caption = read_caption_metadata(photo_path)
            return CoreInterval(caption.well or default_well, caption.top, caption.base), "OCR подписи — подтвердите"
        except Exception as exc:
            return None, f"интервал не найден: {exc}"

    def _remove_photo(self, identifier: str, position) -> None:
        for index, record in enumerate(self._records):
            if record.identifier != identifier:
                continue
            self._deleted_records.append((record, index, position))
            del self._records[index]
            if self._well_names():
                self._show_stack()
            self._refresh_project_tree()
            self._refresh_training_bar()
            self.statusBar().showMessage("Фото удалено · Ctrl+Z — вернуть")
            return

    def undo_delete(self) -> None:
        if not self._deleted_records:
            self.statusBar().showMessage("Нет удалённых фотографий для восстановления")
            return
        record, index, position = self._deleted_records.pop()
        self._records.insert(index, record)
        self.workspace.restore_photo(record, position)
        if self._well_names():
            self._show_stack()
        self._refresh_project_tree()
        self._refresh_training_bar()
        self.statusBar().showMessage("Фото восстановлено")

    def run_segmentation(self, records: list[PhotoRecord] | None = None) -> None:
        records = list(records or self._records)
        if not records:
            QMessageBox.information(self, "Kern Analyzer", "Сначала загрузите фотографии через File → Загрузить фотографии.")
            return
        if self._segmentation_thread is not None:
            self._queued_records.extend(records)
            self.statusBar().showMessage("Фото добавлены в очередь сегментации…")
            return

        model_path = self._resolve_model_path()
        if model_path is None:
            QMessageBox.critical(
                self,
                "Модель не найдена",
                "Не найден models/best.pt. Скопируйте модель из исходного Kern Analyzer в папку models этого проекта.",
            )
            return

        self._segmentation_failed = False
        self._segmentation_updated_count = 0
        self._show_activity("Сегментация", f"Подготовка {len(records)} фото…", len(records))
        self._refresh_training_bar()
        self._segmentation_thread = QThread(self)
        self._segmentation_worker = SegmentationWorker(
            model_path,
            [(record.path, record.pixmap.width(), record.pixmap.height(), record.core_columns or None) for record in records],
            self._confidence_threshold,
            self._segmentation_options()["image_size"],
            self._segmentation_options()["max_detections"],
        )
        self._segmentation_worker.moveToThread(self._segmentation_thread)
        self._segmentation_thread.started.connect(self._segmentation_worker.run)
        self._segmentation_worker.progress_changed.connect(self._show_segmentation_progress)
        self._segmentation_worker.image_ready.connect(self._apply_image_detections)
        self._segmentation_worker.failed.connect(self._segmentation_error)
        self._segmentation_worker.finished.connect(self._segmentation_thread.quit)
        self._segmentation_thread.finished.connect(self._on_segmentation_finished)
        self._segmentation_thread.finished.connect(self._segmentation_worker.deleteLater)
        self._segmentation_thread.finished.connect(self._segmentation_thread.deleteLater)
        self._segmentation_thread.start()

    def detect_photo_intervals(self) -> None:
        """Create editable visual interval rows when no description is available."""
        if self._segmentation_thread is not None or self._training_thread is not None:
            QMessageBox.information(self, "Автоинтервалы", "Дождитесь окончания текущей операции.")
            return
        if not self._records:
            QMessageBox.information(self, "Автоинтервалы", "Сначала добавьте фотографии керна.")
            return
        answer = QMessageBox.question(
            self,
            "Автоматически определить интервалы",
            "Текущие маски на фото будут заменены визуальными интервалами без назначенной фации. "
            "Глубины будут заполнены, если они указаны в имени фотографии. Продолжить?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        total = len(self._records)
        created = 0
        failed: list[str] = []
        self._show_activity("Определение интервалов", f"Подготовка {total} фото…", total)
        for index, record in enumerate(self._records, start=1):
            self._update_activity(f"Интервалы {index}/{total}: {Path(record.path).name}", index, total)
            try:
                photo_top, photo_base = self._record_photo_interval(record)
                detections, segments = create_automatic_interval_detections(
                    Path(record.path), record.pixmap, photo_top, photo_base,
                )
                record.detections = detections
                record.depth_segments = segments
                created += len(detections)
                self.workspace.update_photo_detections(record)
            except Exception as exc:
                failed.append(f"{Path(record.path).name}: {exc}")
        self._close_activity()
        self._show_stack()
        self._refresh_project_tree()
        self._refresh_training_bar()
        if self._project_folder is not None:
            self._save_current_project(self._project_folder)
        detail = "" if not failed else f"\n\nНе обработано: {len(failed)}. Первое: {failed[0]}"
        QMessageBox.information(
            self,
            "Автоинтервалы готовы",
            f"Создано интервалов: {created}.\nНазначьте фацию в карточке нужного интервала, чтобы добавить его в обучение.{detail}",
        )

    def _resolve_model_path(self) -> Path | None:
        root = bundled_root()
        candidates = [
            self._selected_model_path,
            self._best_project_model(self._project_folder),
            root / "models" / "best.pt",
            root.parent / "deep core" / "core-analyzer" / "models" / "best.pt",
        ]
        return next((path for path in candidates if path is not None and path.is_file()), None)

    def _best_project_model(self, project_folder: Path | None) -> Path | None:
        """Prefer the published model with best held-out mAP, never a partial run."""
        if project_folder is None:
            return None
        root = self._trained_models_root() / self._safe_folder_name(self._project_title)
        ranked: list[tuple[float, float, Path]] = []
        for info_path in root.glob("*/training_info.json"):
            try:
                import json

                info = json.loads(info_path.read_text(encoding="utf-8"))
                score = float(dict(info.get("metrics") or {}).get("mAP50-95", -1))
                model = info_path.parent / "best.pt"
                if model.is_file() and (info_path.parent / "facies_catalog.json").is_file():
                    ranked.append((score, model.stat().st_mtime, model))
            except (OSError, ValueError, TypeError):
                continue
        if ranked:
            return max(ranked, key=lambda item: (item[0], item[1]))[2]
        candidates = [
            path for path in project_folder.glob("training/runs/*/fine_tune/weights/best.pt")
            if (path.parent / "facies_catalog.json").is_file()
        ]
        return max(candidates, key=lambda path: path.stat().st_mtime, default=None)

    def select_detection_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать модель сегментации",
            str(self._selected_model_path.parent) if self._selected_model_path else "",
            "Модели YOLO (*.pt);;Все файлы (*)",
        )
        if not file_path:
            return
        self._selected_model_path = Path(file_path)
        self.statusBar().showMessage(f"Выбрана модель: {self._selected_model_path.name}")

    def show_training_history(self) -> None:
        """Show locally saved validation metrics without depending on a server."""
        root = self._trained_models_root() / self._safe_folder_name(self._project_title)
        items: list[tuple[float, str]] = []
        for info_path in root.glob("*/training_info.json"):
            try:
                import json

                info = json.loads(info_path.read_text(encoding="utf-8"))
                metrics = dict(info.get("metrics") or {})
            except (OSError, ValueError, TypeError):
                continue
            m_ap = metrics.get("mAP50-95")
            comparison = dict(info.get("comparison") or {})
            delta = comparison.get("mAP50-95")
            delta_text = "" if delta is None else f" · к исходной: {float(delta):+.1%}"
            metric_text = "нет validation-метрик" if m_ap is None else (
                f"mAP50-95: {float(m_ap):.1%} · "
                f"precision: {float(metrics.get('precision', 0)):.1%} · "
                f"recall: {float(metrics.get('recall', 0)):.1%}{delta_text}"
            )
            sample_counts = dict(info.get("dataset_summary") or {}).get("class_counts", {})
            counts_text = " · ".join(f"{name}: {count}" for name, count in sample_counts.items())
            recommendation = info.get("recommended_epochs")
            suffix = f"\n  фации: {counts_text}" if counts_text else ""
            if recommendation:
                suffix += f" · рекомендовано эпох: {recommendation}"
            items.append((float(m_ap) if m_ap is not None else -1.0, f"{info_path.parent.name}: {metric_text}{suffix}"))
        if not items:
            QMessageBox.information(self, "История дообучений", "В этом проекте ещё нет завершённых дообучений с метриками.")
            return
        lines = [text for _, text in sorted(items, reverse=True)]
        QMessageBox.information(self, "История дообучений", "\n".join(lines))

    def show_training_quality(self) -> None:
        """An explicit pre-flight review, also run automatically before training."""
        quality = training_quality(self._records)
        if quality.blocking_reasons:
            detail = "\n\nОбучение сейчас заблокировано: " + "; ".join(quality.blocking_reasons) + "."
        else:
            detail = "\n\nКритических блокировок нет. Проверьте предупреждения перед запуском."
        QMessageBox.information(self, "Проверка датасета", quality.summary + detail)

    def select_confidence_threshold(self) -> None:
        """Let an interpreter trade precision for recall without code changes."""
        value, accepted = QInputDialog.getDouble(
            self,
            "Порог уверенности масок",
            "Показывать маски с уверенностью от 0,01 до 0,99:\n"
            "меньше значение — больше кандидатов, больше ручной проверки.",
            self._confidence_threshold,
            0.01,
            0.99,
            2,
        )
        if not accepted:
            return
        self._confidence_threshold = round(float(value), 2)
        self._settings.setValue("segmentation/confidence_threshold", self._confidence_threshold)
        self.statusBar().showMessage(
            f"Порог масок: {self._confidence_threshold:.0%}. Запустите сегментацию повторно для применения."
        )

    def select_segmentation_profile(self) -> None:
        profiles = {
            "Быстро · 512 px": "fast",
            "Стандарт · 640 px": "standard",
            "Детально · 1024 px": "detailed",
        }
        current = next((index for index, value in enumerate(profiles.values()) if value == self._segmentation_profile), 1)
        selected, accepted = QInputDialog.getItem(
            self,
            "Режим сегментации",
            "Детальный режим точнее на небольших слоях, но требует больше GPU-памяти и времени:",
            list(profiles),
            current,
            False,
        )
        if not accepted:
            return
        self._segmentation_profile = profiles[selected]
        self._settings.setValue("segmentation/profile", self._segmentation_profile)
        settings = self._segmentation_options()
        self._confidence_threshold = settings["confidence"]
        self._settings.setValue("segmentation/confidence_threshold", self._confidence_threshold)
        self.statusBar().showMessage(
            f"Режим сегментации: {selected} · порог {settings['confidence']:.0%} · до {settings['max_detections']} масок. "
            "Запустите сегментацию повторно для применения."
        )

    def _read_segmentation_profile(self) -> str:
        value = str(self._settings.value("segmentation/profile", "standard"))
        return value if value in {"fast", "standard", "detailed"} else "standard"

    def _segmentation_image_size(self) -> int:
        return int(self._segmentation_options()["image_size"])

    def _segmentation_options(self) -> dict[str, int | float]:
        """One predictable quality/speed preset, including detection density."""
        profiles = {
            "fast": {"image_size": 512, "confidence": 0.60, "max_detections": 300},
            "standard": {"image_size": 640, "confidence": 0.50, "max_detections": 1000},
            "detailed": {"image_size": 1024, "confidence": 0.35, "max_detections": 2000},
        }
        return profiles[self._segmentation_profile]

    def configure_core_columns(self) -> None:
        """Persist interpreter-corrected core bounds and re-run only that photo."""
        if self._segmentation_thread is not None or self._training_thread is not None:
            QMessageBox.information(self, "Колонки керна", "Дождитесь завершения текущей обработки.")
            return
        if not self._records:
            QMessageBox.information(self, "Колонки керна", "Сначала добавьте хотя бы одну фотографию.")
            return
        choices = [f"{index + 1}. {Path(record.path).name}" for index, record in enumerate(self._records)]
        selected, accepted = QInputDialog.getItem(self, "Колонки керна", "Выберите фотографию:", choices, 0, False)
        if not accepted:
            return
        record = self._records[choices.index(selected)]
        columns = record.core_columns or self._suggest_core_columns(record)
        dialog = CoreColumnsDialog(columns, (record.pixmap.width(), record.pixmap.height()), self)
        if dialog.exec() != CoreColumnsDialog.DialogCode.Accepted:
            return
        corrected = dialog.columns()
        if not corrected:
            QMessageBox.warning(self, "Колонки керна", "Укажите хотя бы одну корректную колонку (право > лево, низ > верх).")
            return
        record.core_columns = corrected
        if self._project_folder is not None:
            self._save_current_project(self._project_folder)
        self.statusBar().showMessage("Границы колонок сохранены · повторно сегментирую это фото…")
        self.run_segmentation([record])

    @staticmethod
    def _suggest_core_columns(record: PhotoRecord) -> list[dict[str, float]]:
        """Populate the editor from the automatic detector; user remains final authority."""
        try:
            import cv2
            import numpy as np

            image = cv2.imdecode(np.frombuffer(Path(record.path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return []
            height, width = image.shape[:2]
            x_scale = record.pixmap.width() / max(1, width)
            y_scale = record.pixmap.height() / max(1, height)
            return [
                {"left": left * x_scale, "top": top * y_scale, "right": right * x_scale, "bottom": bottom * y_scale}
                for left, top, right, bottom in RuleBasedFaciesDetector._find_core_columns(image)
            ]
        except (ImportError, OSError):
            return []

    def review_uncertain_masks(self) -> None:
        """Open the highest-value automatic candidate for human verification."""
        queue = review_queue(self._records)
        if not queue:
            QMessageBox.information(
                self,
                "Проверка масок",
                "Нет непроверенных автоматических масок. Запустите сегментацию или добавьте новые фото.",
            )
            return
        record, detection = queue[0]
        self.workspace.focus_facies(record, detection)
        self._open_facies_editor(record, detection)
        QMessageBox.information(
            self,
            "Очередь ручной проверки",
            f"Открыта самая сомнительная маска. Всего в очереди: {len(queue)}.\n"
            "После сохранения откройте этот пункт снова — будет показана следующая.",
        )

    def suggest_lithology_attributes(self) -> None:
        """Run the independent attribute recognizer without replacing facies."""
        if not self._records:
            QMessageBox.information(self, "Атрибуты пород", "Сначала добавьте фото и получите маски фаций.")
            return
        answer = QMessageBox.question(
            self,
            "Подобрать атрибуты пород",
            "Будут заполнены только пустые поля «Цвет», «Зернистость» и «Слоистость» внутри существующих масок. "
            "Это визуальные рекомендации — проверьте их в карточке фации перед дообучением.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        service = LithologyAttributeService()
        changed = 0
        for record in self._records:
            for detection in record.detections:
                suggestions = service.suggest(record, detection)
                for field, value in suggestions.items():
                    if value and not str(detection.attributes.get(field) or "").strip():
                        detection.attributes[field] = value
                        changed += 1
            self.workspace.update_photo_detections(record)
        if self._project_folder is not None:
            self._save_current_project(self._project_folder)
        QMessageBox.information(
            self,
            "Атрибуты пород",
            f"Добавлено рекомендаций: {changed}. Фации и их границы не изменялись.",
        )

    def _read_confidence_threshold(self) -> float:
        try:
            value = float(self._settings.value("segmentation/confidence_threshold", 0.50))
        except (TypeError, ValueError):
            value = 0.50
        return max(0.01, min(0.99, value))

    def start_fine_tune(self) -> None:
        """Export reviewed contours and fine-tune a separate copy of the model."""
        if self._segmentation_thread is not None:
            QMessageBox.information(self, "Дообучение", "Сначала дождитесь окончания сегментации.")
            return
        if self._training_thread is not None:
            return
        examples = self._training_examples_count()
        if examples < MIN_TRAINING_SAMPLES:
            QMessageBox.information(
                self,
                "Дообучение",
                f"Нужно минимум {MIN_TRAINING_SAMPLES} вручную проверенных слоёв. Сейчас: {examples}.",
            )
            return
        quality = training_quality(self._records)
        if quality.blocking_reasons:
            QMessageBox.critical(
                self,
                "Дообучение заблокировано",
                "Обучение не начато, потому что модель начнёт игнорировать редкую фацию:\n\n"
                f"{quality.summary}\n\n"
                "Добавьте и проверьте маски редких фаций, затем повторите проверку.",
            )
            return
        if (
            quality.underrepresented
            or quality.unnamed_manual
            or quality.missing_depth
            or quality.duplicate_masks
            or quality.too_small_masks
            or quality.too_large_masks
        ):
            answer = QMessageBox.question(
                self,
                "Проверка качества разметки",
                "Перед дообучением обнаружены пункты, которые могут ухудшить качество модели:\n\n"
                f"{quality.summary}\n\n"
                "Продолжить всё равно?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        model_path = self._resolve_model_path()
        if model_path is None:
            QMessageBox.critical(
                self,
                "Модель не найдена",
                "Выберите исходную модель .pt через меню «Модель» перед дообучением.",
            )
            return
        device, device_label = FineTuneWorker._best_device()
        if device == "cpu":
            QMessageBox.critical(
                self,
                "Нужна GPU для дообучения",
                "CUDA-видеокарта не доступна в этой версии приложения. Дообучение на CPU отключено, "
                "чтобы не перегружать компьютер. Установите/соберите версию с CUDA PyTorch и обновите драйвер NVIDIA.",
            )
            return
        epochs = self._training_epochs.value()
        recommended_epochs = self._recommended_training_epochs(examples)
        answer = QMessageBox.question(
            self,
            "Запустить дообучение?",
            (
                f"Будет создан датасет из {examples} вручную проверенных слоёв и обучена "
                f"новая копия модели ({epochs} эпох, рекомендовано: {recommended_epochs}).\n"
                f"Устройство: {device_label}. Исходный файл модели не изменится."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        # A second-level timestamp can collide after a failed/retried launch.
        # Make every dataset/run/output directory unambiguously unique.
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        if self._project_folder is None:
            project_folder = self._projects_root() / f"{self._safe_folder_name(self._project_title)}_{stamp}.kern_analyzer"
            self._save_current_project(project_folder)
        elif self._project_folder is not None:
            # Copy externally selected photos into the project before the
            # dataset is exported, so this exact training session remains
            # reproducible after a file is moved or deleted outside Kern Analyzer.
            self._save_current_project(self._project_folder)
        if self._project_folder is None:
            return
        base_folder = self._project_folder
        training_root = base_folder / "training"
        published_dir = self._trained_models_root() / self._safe_folder_name(self._project_title) / stamp
        try:
            dataset = export_training_dataset(self._records, training_root / "datasets" / stamp)
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось подготовить датасет", str(exc))
            return

        self._training_thread = QThread(self)
        self._training_worker = FineTuneWorker(
            model_path,
            Path(dataset["data_yaml"]),
            training_root / "runs" / stamp,
            published_dir,
            epochs=epochs,
            dataset_summary={
                "sample_count": dataset["sample_count"],
                "class_counts": dataset["class_counts"],
                "quality": quality.summary,
            },
            recommended_epochs=recommended_epochs,
        )
        self._training_epoch_current = 0
        self._training_epoch_total = epochs
        self._fine_tune_success_model = None
        self._show_activity("Дообучение модели", f"Подготовка: 0 из {epochs} эпох…", epochs)
        self._training_worker.moveToThread(self._training_thread)
        self._training_thread.started.connect(self._training_worker.run)
        self._training_worker.progress.connect(self.statusBar().showMessage)
        self._training_worker.epoch_progress.connect(self._on_training_epoch_progress)
        self._training_worker.succeeded.connect(self._on_fine_tune_succeeded)
        self._training_worker.failed.connect(self._on_fine_tune_failed)
        self._training_worker.finished.connect(self._training_thread.quit)
        self._training_thread.finished.connect(self._on_fine_tune_finished)
        self._training_thread.finished.connect(self._training_worker.deleteLater)
        self._training_thread.finished.connect(self._training_thread.deleteLater)
        self._refresh_training_bar()
        self.statusBar().showMessage(
            f"Подготовлено {dataset['sample_count']} разметок · запускаю дообучение на {epochs} эпох…"
        )
        self._training_thread.start()

    def _on_fine_tune_succeeded(self, model_path: str) -> None:
        self._close_activity()
        self._selected_model_path = Path(model_path)
        self._fine_tune_success_model = self._selected_model_path
        self.statusBar().showMessage(f"Дообучение завершено · новая модель сохранена: {self._selected_model_path}")
        QMessageBox.information(
            self,
            "Дообучение завершено",
            "Новая модель и справочник фаций сохранены в понятной папке:\n"
            f"{self._selected_model_path.parent}",
        )
        self.statusBar().showMessage("Новая модель выбрана · повторная сегментация начнётся после корректного завершения обучения…")

    def _on_fine_tune_failed(self, message: str) -> None:
        self._close_activity()
        self._fine_tune_success_model = None
        QMessageBox.critical(self, "Ошибка дообучения", message)

    def _on_training_epoch_progress(self, current: int, total: int) -> None:
        if self._training_thread is None:
            return
        self._training_epoch_current = max(0, int(current))
        self._training_epoch_total = max(1, int(total))
        self._refresh_training_bar()
        self._update_activity(f"Дообучение: эпоха {self._training_epoch_current} из {self._training_epoch_total}", self._training_epoch_current, self._training_epoch_total)
        self.statusBar().showMessage(
            f"Дообучение: эпоха {self._training_epoch_current} из {self._training_epoch_total} "
            f"· осталось {max(0, self._training_epoch_total - self._training_epoch_current)}"
        )

    def _on_fine_tune_finished(self) -> None:
        self._close_activity()
        completed_model = self._fine_tune_success_model
        self._training_thread = None
        self._training_worker = None
        self._training_epoch_current = 0
        self._training_epoch_total = 0
        self._refresh_training_bar()
        if completed_model is not None and self._records:
            # Do not initialize another CUDA model until Ultralytics and the
            # training QThread have both exited; concurrent initialization was
            # a frequent source of post-training application crashes.
            self.statusBar().showMessage("Дообучение завершено · запускаю повторную сегментацию новой моделью…")
            self.run_segmentation(self._records)

    def import_las_file(self) -> None:
        self._import_log_file("gis")

    def import_rigis_file(self) -> None:
        self._import_log_file("rigis")

    def import_many_las_files(self) -> None:
        self._import_many_log_files("gis")

    def import_many_rigis_files(self) -> None:
        self._import_many_log_files("rigis")

    def _import_log_file(self, kind: str) -> None:
        title = "ГИС" if kind == "gis" else "РИГИС"
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            f"Загрузить LAS для {title} (можно выбрать несколько)",
            "",
            "LAS-файлы (*.las *.LAS);;Все файлы (*)",
        )
        if not paths:
            return
        self._import_log_paths(kind, paths)

    def _import_many_log_files(self, kind: str) -> None:
        title = "ГИС" if kind == "gis" else "РИГИС"
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            f"Загрузить несколько LAS для {title}",
            "",
            "LAS-файлы (*.las *.LAS);;Все файлы (*)",
        )
        if paths:
            self._import_log_paths(kind, paths)

    def _import_log_paths(self, kind: str, paths: list[str]) -> None:
        title = "ГИС" if kind == "gis" else "РИГИС"
        loaded, failures = 0, []
        for file_path in paths:
            auto_matched_well = self._well_for_log_file(file_path)
            well_name = auto_matched_well or self._choose_well(f"{title}: выбор скважины")
            if well_name is None:
                continue
            try:
                log_data = parse_las_file(file_path)
            except Exception as exc:
                failures.append(f"{Path(file_path).name}: {exc}")
                continue
            self._set_curve_display_names(log_data)
            self._set_well_log(well_name, kind, log_data, merge=True)
            if well_name not in self._well_depth_ranges:
                self._ask_for_well_depth_setup(log_data, well_name)
            if self._records_for_well(well_name) and well_name not in self._well_intervals:
                self._ask_for_core_interval(title, log_data, well_name)
            loaded += 1
        self._show_stack()
        message = f"Загружено {title}: {loaded}"
        if failures:
            message += f" · ошибок: {len(failures)}"
        self.statusBar().showMessage(message)

    def add_manual_gis_curve(self) -> None:
        well_name = self._choose_well("Новая ГИС-кривая: выбор скважины")
        if well_name is None:
            return
        log_data = self._well_log(well_name, "gis")
        if not log_data:
            QMessageBox.information(self, "Новая ГИС-кривая", "Сначала загрузите LAS для этой скважины.")
            return
        name, accepted = QInputDialog.getText(self, "Новая ГИС-кривая", "Мнемоника кривой:", text="USER")
        name = name.strip().upper()
        if not accepted or not name:
            return
        existing = {str(curve.get("mnemonic") or "") for curve in log_data.get("curves", [])}
        if name in existing:
            QMessageBox.warning(self, "Новая ГИС-кривая", "Кривая с такой мнемоникой уже существует.")
            return
        log_data.setdefault("curves", []).append({"index": len(log_data.get("curves", [])), "mnemonic": name, "unit": "", "description": "Вручную созданная кривая"})
        log_data["curves"][-1]["visible"] = True
        log_data.setdefault("values", {})[name] = [0.0] * len(log_data.get("depths", []))
        self._set_well_log(well_name, "gis", log_data)
        self._show_stack()
        self.statusBar().showMessage(f"Добавлена кривая {name} · откройте «Настроить ГИС…», чтобы вписать значения")

    def _build_gis_preview(self) -> dict | None:
        return self._build_log_preview(self._gis_data)

    def _build_rigis_preview(self) -> dict | None:
        return self._build_log_preview(self._rigis_data)

    def _well_names(self) -> list[str]:
        names = [record.well_name or "Скважина 1" for record in self._records]
        available = list(dict.fromkeys([*names, *self._empty_wells]))
        ordered = [name for name in self._well_order if name in available]
        return [*ordered, *(name for name in available if name not in ordered)]

    def _project_well_metadata(self) -> list[dict]:
        return [
            {
                "name": name,
                "order": index,
                **({"core_interval": list(self._well_intervals[name])} if name in self._well_intervals else {}),
                **({"well_depth_range": list(self._well_depth_ranges[name])} if name in self._well_depth_ranges else {}),
                **({"depth_reference": self._well_depth_references[name]} if name in self._well_depth_references else {}),
                **({"depth_settings": self._well_depth_settings[name]} if name in self._well_depth_settings else {}),
                **({"interpretations": self._well_interpretations[name]} if name in self._well_interpretations else {}),
            }
            for index, name in enumerate(self._well_names())
        ]

    def create_empty_well(self) -> None:
        default = f"Скважина {len(self._well_names()) + 1}"
        name, accepted = QInputDialog.getText(self, "Новая скважина", "Название скважины:", text=default)
        name = name.strip()
        if not accepted or not name:
            return
        if name in self._well_names():
            QMessageBox.warning(self, "Новая скважина", "Скважина с таким названием уже существует.")
            return
        self._empty_wells.append(name)
        self._active_well_name = name
        self._settings.setValue("last/active_well", name)
        self._refresh_project_tree()
        self._show_stack()
        self.configure_well_depth()
        self.statusBar().showMessage(f"Создана скважина без керна: {name} · можно загрузить ГИС или РИГИС")

    def _records_for_well(self, well_name: str) -> list[PhotoRecord]:
        return [record for record in self._records if (record.well_name or "Скважина 1") == well_name]

    def _choose_well(self, title: str) -> str | None:
        wells = self._well_names()
        if not wells:
            QMessageBox.information(self, title, "Сначала загрузите фотографии керна.")
            return None
        if len(wells) == 1:
            return wells[0]
        current = wells.index(self._active_well_name) if self._active_well_name in wells else 0
        well, accepted = QInputDialog.getItem(self, title, "Скважина:", wells, current, False)
        return str(well) if accepted and well else None

    @staticmethod
    def _file_match_key(file_path: str) -> str:
        """Normalise file names while ignoring common core/log suffixes and separators."""
        stem = Path(file_path).stem.casefold().replace("ё", "е")
        words = [word for word in re.split(r"[_\W]+", stem, flags=re.UNICODE) if word]
        ignored = {"core", "ker", "kern", "photo", "img", "image", "gis", "rigis", "las", "log", "фото", "керн", "гис", "ригис"}
        words = [word for word in words if word not in ignored]
        # A final image sequence such as ``well-12-core-01`` is ignored,
        # while the well number in ``well-12`` remains significant.
        while len(words) > 2 and words[-1].isdigit():
            words.pop()
        return "_".join(words)

    def _well_for_log_file(self, log_path: str) -> str | None:
        """Return one well only when its photo name gives an unambiguous file-name match."""
        log_key = self._file_match_key(log_path)
        if not log_key:
            return None
        scores: dict[str, int] = {}
        for well_name in self._well_names():
            candidates = [self._file_match_key(record.path) for record in self._records_for_well(well_name)]
            well_key = self._file_match_key(well_name)
            score = 0
            for key in candidates:
                if key == log_key:
                    score = max(score, 100)
                elif key and (key.startswith(log_key + "_") or log_key.startswith(key + "_")):
                    score = max(score, 80)
            if well_key == log_key:
                score = max(score, 95)
            if score:
                scores[well_name] = score
        if not scores:
            return None
        highest = max(scores.values())
        matches = [well_name for well_name, score in scores.items() if score == highest]
        return matches[0] if len(matches) == 1 else None

    def _well_log(self, well_name: str, kind: str) -> dict | None:
        return self._well_logs.get(well_name, {}).get(kind)

    def _set_well_log(self, well_name: str, kind: str, log_data: dict, merge: bool = False) -> None:
        if merge and self._well_log(well_name, kind):
            log_data = self._merge_log_data(self._well_log(well_name, kind), log_data)
        self._well_logs.setdefault(well_name, {})[kind] = log_data
        self._active_well_name = well_name
        self._settings.setValue("last/active_well", well_name)
        # Keep the established single-well fields in sync for compatibility.
        if kind == "gis":
            self._gis_data = log_data
        else:
            self._rigis_data = log_data

    @staticmethod
    def _set_curve_display_names(log_data: dict) -> None:
        """Use the LAS file name as the user-facing curve title in the tablet."""
        source_file = str(log_data.get("file_name") or Path(str(log_data.get("source_path") or "curve")).name)
        source_stem = Path(source_file).stem or "curve"
        depth_curve = str(log_data.get("depth_curve") or "")
        data_curves = [curve for curve in log_data.get("curves", []) if str(curve.get("mnemonic") or "") != depth_curve]
        for curve in data_curves:
            mnemonic = str(curve.get("mnemonic") or "")
            curve["source_file"] = source_file
            curve["display_name"] = source_stem if len(data_curves) == 1 else f"{source_stem} · {mnemonic}"

    @staticmethod
    def _merge_log_data(base: dict, incoming: dict) -> dict:
        """Append curves from another LAS file, resampling them to the first log's depth grid."""
        merged = {
            **base,
            "curves": [dict(curve) for curve in base.get("curves", [])],
            "depths": list(base.get("depths", []) or []),
            "values": {name: list(values or []) for name, values in (base.get("values", {}) or {}).items()},
        }
        target_depths = merged["depths"]
        incoming_depths = [float(value) if value is not None else None for value in incoming.get("depths", [])]
        source_depth_curve = incoming.get("depth_curve")
        existing_names = {str(curve.get("mnemonic") or "") for curve in merged["curves"]}
        for curve in incoming.get("curves", []):
            source_name = str(curve.get("mnemonic") or "")
            if not source_name or source_name == source_depth_curve:
                continue
            name = source_name
            suffix = 2
            while name in existing_names:
                name = f"{source_name}_{suffix}"
                suffix += 1
            source_values = list((incoming.get("values", {}) or {}).get(source_name, []) or [])
            curve_meta = {**curve, "index": len(merged["curves"]), "mnemonic": name, "visible": True}
            if name != source_name:
                curve_meta["display_name"] = f"{curve.get('display_name') or source_name} ({suffix - 1})"
            merged["curves"].append(curve_meta)
            merged["values"][name] = MainWindow._resample_curve(incoming_depths, source_values, target_depths)
            existing_names.add(name)
        return merged

    @staticmethod
    def _resample_curve(source_depths: list[float | None], source_values: list, target_depths: list) -> list[float | None]:
        pairs = sorted((depth, value) for depth, value in zip(source_depths, source_values) if depth is not None and value is not None)
        if not pairs:
            return [None] * len(target_depths)
        xs = [pair[0] for pair in pairs]
        ys = [float(pair[1]) for pair in pairs]
        result = []
        for target in target_depths:
            if target is None:
                result.append(None)
                continue
            index = bisect_left(xs, float(target))
            if index == 0:
                result.append(ys[0])
            elif index >= len(xs):
                result.append(ys[-1])
            else:
                x1, x2, y1, y2 = xs[index - 1], xs[index], ys[index - 1], ys[index]
                ratio = (float(target) - x1) / max(x2 - x1, 1e-9)
                result.append(y1 + ratio * (y2 - y1))
        return result

    @staticmethod
    def _build_log_preview(log_data: dict | None) -> dict | None:
        if not log_data:
            return None
        depth_curve = log_data.get("depth_curve")
        curve_metas = [
            curve
            for curve in log_data.get("curves", [])
            if curve.get("mnemonic") and curve.get("mnemonic") != depth_curve and curve.get("visible", True)
        ]
        depths = list(log_data.get("depths", []) or [])
        tracks = []
        for curve_meta in curve_metas:
            curve_name = str(curve_meta["mnemonic"])
            values = list((log_data.get("values", {}) or {}).get(curve_name, []) or [])
            valid = [float(value) for value in values if value is not None]
            if not valid:
                continue
            lower, upper = min(valid), max(valid)
            span = max(upper - lower, 1e-9)
            step = max(1, len(depths) // 700)
            points = [
                {"depth": float(depths[index]), "normalized": (float(value) - lower) / span}
                for index, value in enumerate(values)
                if index < len(depths) and index % step == 0 and value is not None and depths[index] is not None
            ]
            tracks.append({"curve": curve_meta.get("display_name") or curve_name, "points": points, "min": lower, "max": upper})
        valid_depths = [float(depth) for depth in depths if depth is not None]
        preview = {"well_name": log_data.get("well_name"), "tracks": tracks, "depth_unit": log_data.get("depth_unit") or "m"}
        if len(valid_depths) >= 2:
            preview["depth_range"] = (min(valid_depths), max(valid_depths))
        return preview

    def _build_well_log_preview(self, well_name: str, kind: str) -> dict | None:
        preview = self._build_log_preview(self._well_log(well_name, kind))
        depth_settings = self._well_depth_settings.get(well_name, {})
        if preview is None and well_name in self._well_depth_ranges:
            preview = {"well_name": well_name, "tracks": [], "depth_unit": depth_settings.get("unit") or "m"}
        if preview is not None and well_name in self._well_depth_ranges:
            preview = {
                **preview,
                "depth_range": self._well_depth_ranges[well_name],
                "depth_reference": self._well_depth_references.get(well_name, "MD"),
                "depth_unit": depth_settings.get("unit") or preview.get("depth_unit") or "m",
                "depth_datum": depth_settings.get("datum") or "",
            }
        return preview

    def _ordered_detections(self, well_name: str | None = None):
        return [
            detection
            for record in (self._records_for_well(well_name) if well_name else self._records)
            for detection in StackColumnItem._sort_by_reading_order(record.detections)
        ]

    def _ask_for_well_depth_setup(self, log_data: dict, well_name: str) -> None:
        depths = [float(value) for value in log_data.get("depths", []) if value is not None]
        if len(depths) < 2:
            return
        references = [str(curve.get("mnemonic") or "") for curve in log_data.get("curves", []) if str(curve.get("mnemonic") or "")]
        selected = str(log_data.get("depth_curve") or "MD")
        dialog = WellDepthDialog(min(depths), max(depths), references, selected, self, self._well_depth_settings.get(well_name))
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            top, base, reference, settings = dialog.values()
        except ValueError as exc:
            QMessageBox.warning(self, "\u0413\u043b\u0443\u0431\u0438\u043d\u0430 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u044b", str(exc))
            return
        self._well_depth_ranges[well_name] = (top, base)
        self._well_depth_references[well_name] = reference
        self._well_depth_settings[well_name] = settings
        self._apply_depth_curve(log_data, settings.get("source_curve") or "")

    def configure_well_depth(self) -> None:
        well_name = self._choose_well("\u0413\u043b\u0443\u0431\u0438\u043d\u0430 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u044b")
        if well_name is None:
            return
        log_data = self._well_log(well_name, "gis") or self._well_log(well_name, "rigis")
        depths = [float(value) for value in (log_data or {}).get("depths", []) if value is not None]
        top, base = self._well_depth_ranges.get(well_name, (min(depths), max(depths)) if len(depths) >= 2 else (0.0, 1000.0))
        references = [str(curve.get("mnemonic") or "") for curve in (log_data or {}).get("curves", []) if str(curve.get("mnemonic") or "")]
        selected = self._well_depth_references.get(well_name, str((log_data or {}).get("depth_curve") or "MD"))
        dialog = WellDepthDialog(top, base, references, selected, self, self._well_depth_settings.get(well_name))
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            top, base, reference, settings = dialog.values()
        except ValueError as exc:
            QMessageBox.warning(self, "\u0413\u043b\u0443\u0431\u0438\u043d\u0430 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u044b", str(exc))
            return
        self._well_depth_ranges[well_name] = (top, base)
        self._well_depth_references[well_name] = reference
        self._well_depth_settings[well_name] = settings
        source_curve = settings.get("source_curve") or ""
        for kind in ("gis", "rigis"):
            candidate = self._well_log(well_name, kind)
            if candidate:
                self._apply_depth_curve(candidate, source_curve)
        self._show_stack()

    @staticmethod
    def _apply_depth_curve(log_data: dict, curve_name: str) -> None:
        """Use an explicitly selected LAS depth curve; never infer an MD/TVD conversion."""
        curve_name = str(curve_name or "").strip()
        values = list((log_data.get("values") or {}).get(curve_name, []) or [])
        valid = [value for value in values if value is not None]
        if len(valid) < 2:
            return
        log_data["depths"] = values
        log_data["depth_curve"] = curve_name

    def configure_core_interval(self) -> None:
        well_name = self._choose_well("\u0418\u043d\u0442\u0435\u0440\u0432\u0430\u043b \u043a\u0435\u0440\u043d\u0430")
        if well_name is None:
            return
        fallback = self._well_depth_ranges.get(well_name, (0.0, 1.0))
        top, base = self._well_intervals.get(well_name, fallback)
        dialog = DepthRangeDialog(top, base, self)
        dialog.setWindowTitle("\u0418\u043d\u0442\u0435\u0440\u0432\u0430\u043b \u043a\u0435\u0440\u043d\u0430 \u0432 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u0435")
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            top, base = dialog.values()
            if top is None or base is None or top >= base:
                raise ValueError("\u041d\u0438\u0437 \u043a\u0435\u0440\u043d\u0430 \u0434\u043e\u043b\u0436\u0435\u043d \u0431\u044b\u0442\u044c \u0431\u043e\u043b\u044c\u0448\u0435 \u0432\u0435\u0440\u0445\u0430")
        except ValueError as exc:
            QMessageBox.warning(self, "\u0418\u043d\u0442\u0435\u0440\u0432\u0430\u043b \u043a\u0435\u0440\u043d\u0430", str(exc))
            return
        self._apply_core_interval(top, base, well_name)
        self._show_stack()

    def _current_core_interval(self, log_data: dict, well_name: str | None = None) -> tuple[float, float]:
        if well_name and well_name in self._well_intervals:
            return self._well_intervals[well_name]
        depths = [
            value
            for detection in self._ordered_detections(well_name)
            for value in (detection.depth_from, detection.depth_to)
            if value is not None
        ]
        if len(depths) >= 2:
            return min(depths), max(depths)
        log_depths = [float(value) for value in log_data.get("depths", []) if value is not None]
        return (min(log_depths), max(log_depths)) if log_depths else (0.0, 1.0)

    def _ask_for_core_interval(self, title: str, log_data: dict, well_name: str | None = None) -> None:
        existing_depths = [
            value
            for detection in self._ordered_detections(well_name)
            for value in (detection.depth_from, detection.depth_to)
            if value is not None
        ]
        if well_name not in self._well_intervals and len(existing_depths) < 2:
            top, base = None, None
        else:
            top, base = self._current_core_interval(log_data, well_name)
        dialog = DepthRangeDialog(top, base, self)
        dialog.setWindowTitle(f"Интервал {'керна' if self._ordered_detections(well_name) else 'скважины'} для {title}")
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            top, base = dialog.values()
            if top is None or base is None or top >= base:
                raise ValueError("Низ интервала должен быть больше верха")
        except ValueError as exc:
            QMessageBox.warning(self, "Проверьте интервал", str(exc))
            return
        self._apply_core_interval(top, base, well_name)

    def _apply_core_interval(self, top: float, base: float, well_name: str | None = None) -> None:
        ordered = self._ordered_detections(well_name)
        if not ordered:
            if well_name:
                self._well_intervals[well_name] = (float(top), float(base))
            return
        heights = []
        for detection in ordered:
            ys = [point.y() for point in detection.polygon]
            heights.append(max(1.0, max(ys) - min(ys)) if ys else 1.0)
        total = sum(heights)
        cursor = float(top)
        for index, (detection, height) in enumerate(zip(ordered, heights)):
            next_depth = float(base) if index == len(ordered) - 1 else cursor + (base - top) * height / total
            detection.depth_from = cursor
            detection.depth_to = next_depth
            cursor = next_depth

    def _show_stack(self) -> None:
        wells = {well_name: self._records_for_well(well_name) for well_name in self._well_names()}
        if not wells:
            return
        well_logs = {
            well_name: (
                self._build_well_log_preview(well_name, "gis"),
                self._build_well_log_preview(well_name, "rigis"),
            )
            for well_name in wells
        }
        self.workspace.show_correlation(
            wells,
            well_logs,
            well_interpretations=self._well_interpretations,
            spacing=180.0,
            vertical_scale=1.0,
            connect_layers=False,
            description_only=True,
            place_source_photos=True,
        )

    def show_correlation_profile(self) -> None:
        if len(self._well_names()) < 2:
            QMessageBox.information(self, "Корреляция", "Для корреляции создайте или загрузите минимум две скважины.")
            return
        self._show_stack()
        self.sheets.setCurrentWidget(self.correlation_workspace)
        self.correlation_workspace.fit_content()
        self.statusBar().showMessage("Корреляция построена · пунктиром соединены совпадающие литологические интервалы")

    def change_well_order(self) -> None:
        wells = self._well_names()
        if len(wells) < 2:
            QMessageBox.information(self, "\u041f\u043e\u0440\u044f\u0434\u043e\u043a \u0441\u043a\u0432\u0430\u0436\u0438\u043d", "\u0414\u043e\u0431\u0430\u0432\u044c\u0442\u0435 \u0445\u043e\u0442\u044f \u0431\u044b \u0434\u0432\u0435 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u044b.")
            return
        well_name, accepted = QInputDialog.getItem(self, "\u041f\u043e\u0440\u044f\u0434\u043e\u043a \u0441\u043a\u0432\u0430\u0436\u0438\u043d", "\u041f\u0435\u0440\u0435\u043c\u0435\u0441\u0442\u0438\u0442\u044c \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u0443:", wells, 0, False)
        if not accepted or not well_name:
            return
        targets = ["\u0412 \u043d\u0430\u0447\u0430\u043b\u043e", *[name for name in wells if name != well_name]]
        target, accepted = QInputDialog.getItem(self, "\u041f\u043e\u0440\u044f\u0434\u043e\u043a \u0441\u043a\u0432\u0430\u0436\u0438\u043d", "\u0420\u0430\u0441\u043f\u043e\u043b\u043e\u0436\u0438\u0442\u044c:", targets, 0, False)
        if not accepted:
            return
        new_order = [name for name in wells if name != well_name]
        insert_at = 0 if target == targets[0] else new_order.index(target) + 1
        new_order.insert(insert_at, str(well_name))
        self._well_order = new_order
        self._show_stack()
        self.statusBar().showMessage("\u041f\u043e\u0440\u044f\u0434\u043e\u043a \u0441\u043a\u0432\u0430\u0436\u0438\u043d \u043e\u0431\u043d\u043e\u0432\u043b\u0451\u043d")

    def _set_well_order_from_canvas(self, ordered_wells) -> None:
        requested = [str(name) for name in ordered_wells if str(name)]
        available = self._well_names()
        if not requested or set(requested) - set(available):
            return
        self._well_order = [*requested, *(name for name in available if name not in requested)]
        self._show_stack()
        self.statusBar().showMessage("\u041f\u043e\u0440\u044f\u0434\u043e\u043a \u0441\u043a\u0432\u0430\u0436\u0438\u043d \u0438\u0437\u043c\u0435\u043d\u0451\u043d \u043f\u0435\u0440\u0435\u0442\u0430\u0441\u043a\u0438\u0432\u0430\u043d\u0438\u0435\u043c")

    def add_correlation_line_from_layer(self) -> None:
        choices: list[tuple[str, str, float, str]] = []
        for well_name in self._well_names():
            for index, detection in enumerate(self._ordered_detections(well_name), start=1):
                depths = [value for value in (detection.depth_from, detection.depth_to) if value is not None]
                if not depths:
                    continue
                depth = sum(float(value) for value in depths) / len(depths)
                label = str(detection.label or "\u0421\u043b\u043e\u0439")
                choices.append((f"{well_name} \u00b7 {label} {index} \u00b7 {depth:g} \u043c", well_name, depth, label))
        if not choices:
            QMessageBox.information(self, "\u041b\u0438\u043d\u0438\u044f \u043f\u043e \u0441\u043b\u043e\u044e", "\u0421\u043d\u0430\u0447\u0430\u043b\u0430 \u0437\u0430\u0434\u0430\u0439\u0442\u0435 \u0433\u043b\u0443\u0431\u0438\u043d\u044b \u0441\u043b\u043e\u0451\u0432.")
            return
        titles = [choice[0] for choice in choices]
        selected, accepted = QInputDialog.getItem(self, "\u041b\u0438\u043d\u0438\u044f \u043f\u043e \u0441\u043b\u043e\u044e", "\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u043b\u043e\u0439-\u043e\u0440\u0438\u0435\u043d\u0442\u0438\u0440:", titles, 0, False)
        if not accepted or not selected:
            return
        _, well_name, depth, label = choices[titles.index(selected)]
        name, accepted = QInputDialog.getText(self, "\u041b\u0438\u043d\u0438\u044f \u043f\u043e \u0441\u043b\u043e\u044e", "\u041d\u0430\u0437\u0432\u0430\u043d\u0438\u0435 \u043b\u0438\u043d\u0438\u0438:", text=label)
        if not accepted or not name.strip():
            return
        self._show_stack()
        if self.correlation_workspace.add_straight_layer_correlation_curve(name.strip(), well_name, depth):
            self.sheets.setCurrentWidget(self.correlation_workspace)
            self.statusBar().showMessage("\u041b\u0438\u043d\u0438\u044f \u043f\u0440\u0438\u0432\u044f\u0437\u0430\u043d\u0430 \u043a \u0441\u043b\u043e\u044e \u0438 \u043f\u043e\u0441\u0442\u0440\u043e\u0435\u043d\u0430 \u043e\u0434\u043d\u043e\u0439 \u043f\u0440\u044f\u043c\u043e\u0439 \u0447\u0435\u0440\u0435\u0437 \u0432\u0441\u0435 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u044b")

    def add_manual_correlation_curve(self) -> None:
        if len(self._well_names()) < 2:
            QMessageBox.information(self, "Корреляция", "Для линии нужны минимум две скважины.")
            return
        self._show_stack()
        name, accepted = QInputDialog.getText(self, "Новая линия корреляции", "Название линии:", text="Новый пласт")
        if not accepted or not name.strip():
            return
        if self.correlation_workspace.add_manual_correlation_curve(name.strip()):
            self.statusBar().showMessage(f"Добавлена линия корреляции: {name.strip()} · перетащите её на нужной скважине")

    def configure_correlation(self) -> None:
        if len(self._well_names()) < 2:
            QMessageBox.information(self, "Корреляция", "Сначала добавьте минимум две скважины.")
            return
        spacing, accepted = QInputDialog.getDouble(
            self,
            "Корреляция: расстояние",
            "Расстояние между скважинами, px:",
            self._correlation_spacing,
            20.0,
            5000.0,
            0,
        )
        if not accepted:
            return
        scale, accepted = QInputDialog.getDouble(
            self,
            "Корреляция: шаг по глубине",
            "Вертикальный масштаб (1 = исходный):",
            self._correlation_vertical_scale,
            0.2,
            8.0,
            2,
        )
        if not accepted:
            return
        self._correlation_spacing = spacing
        self._correlation_vertical_scale = scale
        self._show_stack()
        self.correlation_workspace.fit_content()

    def _open_correlation_curve_menu(self, curve: dict, well_name: str) -> None:
        menu = QMenu(self)
        rename = menu.addAction("Изменить название…")
        recolor = menu.addAction("Изменить цвет…")
        thickness = menu.addAction("Толщина линии…")
        style_menu = menu.addMenu("Тип линии")
        styles = {
            "Сплошная": "solid",
            "Пунктир": "dash",
            "Точки": "dot",
            "Штрих-точка": "dash_dot",
        }
        for title, value in styles.items():
            action = style_menu.addAction(title)
            action.setCheckable(True)
            action.setChecked(curve.get("style") == value)
            action.triggered.connect(lambda _=False, selected=value: self._set_curve_value(curve, "style", selected))
        menu.addSeparator()
        up = menu.addAction(f"Сдвинуть выше на скважине «{well_name}»")
        down = menu.addAction(f"Сдвинуть ниже на скважине «{well_name}»")
        reset = menu.addAction(f"Сбросить сдвиг на «{well_name}»")
        break_line = menu.addAction(f"Разорвать на скважине «{well_name}»")
        restore_menu = menu.addMenu("Восстановить на скважине")
        for broken_well in curve.get("breaks", []):
            action = restore_menu.addAction(str(broken_well))
            action.triggered.connect(lambda _=False, selected=str(broken_well): self._set_curve_break(curve, selected, False))

        rename.triggered.connect(lambda: self._rename_curve(curve))
        recolor.triggered.connect(lambda: self._recolor_curve(curve))
        thickness.triggered.connect(lambda: self._change_curve_thickness(curve))
        up.triggered.connect(lambda: self._shift_curve_in_well(curve, well_name, -10.0))
        down.triggered.connect(lambda: self._shift_curve_in_well(curve, well_name, 10.0))
        reset.triggered.connect(lambda: self._set_curve_well_offset(curve, well_name, 0.0))
        break_line.triggered.connect(lambda: self._set_curve_break(curve, well_name, True))
        menu.exec(QCursor.pos())

    def _set_curve_value(self, curve: dict, key: str, value) -> None:
        curve[key] = value
        self._show_stack()

    def _rename_curve(self, curve: dict) -> None:
        name, accepted = QInputDialog.getText(self, "Название корреляции", "Название:", text=str(curve.get("name") or ""))
        if accepted and name.strip():
            self._set_curve_value(curve, "name", name.strip())

    def _recolor_curve(self, curve: dict) -> None:
        color = QColorDialog.getColor(QColor(str(curve.get("color") or "#734cbe")), self, "Цвет корреляционной линии")
        if color.isValid():
            self._set_curve_value(curve, "color", color.name(QColor.NameFormat.HexRgb))

    def _change_curve_thickness(self, curve: dict) -> None:
        width, accepted = QInputDialog.getDouble(
            self,
            "Толщина корреляционной линии",
            "Толщина, px:",
            float(curve.get("width", 1.8)),
            0.5,
            20.0,
            1,
        )
        if accepted:
            self._set_curve_value(curve, "width", width)

    def _shift_curve_in_well(self, curve: dict, well_name: str, delta: float) -> None:
        offsets = curve.setdefault("offsets", {})
        self._set_curve_well_offset(curve, well_name, float(offsets.get(well_name, 0.0)) + delta)

    def _set_curve_well_offset(self, curve: dict, well_name: str, offset: float) -> None:
        curve.setdefault("offsets", {})[well_name] = offset
        self._show_stack()

    def _set_curve_break(self, curve: dict, well_name: str, broken: bool) -> None:
        breaks = [str(name) for name in curve.get("breaks", [])]
        if broken and well_name not in breaks:
            breaks.append(well_name)
        if not broken:
            breaks = [name for name in breaks if name != well_name]
        curve["breaks"] = breaks
        self._show_stack()

    def _paint_stack_column(self, record: PhotoRecord, detection, column: str) -> None:
        if column == "lithology":
            field, title, fallback = "Цвет литологии", "Кисть литологии", "#d8b45b"
        else:
            field, title, fallback = "Цвет насыщения", "Кисть насыщения", "#e0b653"
        initial = QColor(str(detection.attributes.get(field) or fallback))
        color = QColorDialog.getColor(initial, self, title)
        if not color.isValid():
            return
        value = color.name(QColor.NameFormat.HexRgb)
        detection.attributes[field] = value
        if column == "saturation":
            saturation = str(detection.attributes.get("Флюидонасыщение") or "")
            if saturation:
                self._settings.setValue(f"saturation_colors/{saturation}", value)
        else:
            self._settings.setValue("last/lithology_color", value)
        self._refresh_facies_views(record)
        self.statusBar().showMessage(f"{title}: цвет интервала изменён")

    def _create_interpretation_interval(self, well_name: str, kind: str, depth_from: float, depth_to: float) -> None:
        title = "Литология" if kind == "lithology" else "Насыщение"
        default_name = "Новый литологический интервал" if kind == "lithology" else "Новый интервал насыщения"
        name, accepted = QInputDialog.getText(self, title, "Название интервала:", text=default_name)
        if not accepted or not name.strip():
            return
        fallback = "#d8b45b" if kind == "lithology" else "#e0b653"
        color = QColorDialog.getColor(QColor(fallback), self, f"Цвет: {title}")
        if not color.isValid():
            return
        self._well_interpretations.setdefault(well_name, []).append(
            {
                "kind": kind,
                "name": name.strip(),
                "color": color.name(QColor.NameFormat.HexRgb),
                "depth_from": depth_from,
                "depth_to": depth_to,
            }
        )
        self._show_stack()
        self.statusBar().showMessage(f"Добавлен интервал «{name.strip()}»: {depth_from:g}–{depth_to:g} м")

    def _resize_interpretation_interval(self, well_name: str, index: int, depth_from: float, depth_to: float) -> None:
        intervals = self._well_interpretations.get(well_name, [])
        if not 0 <= index < len(intervals):
            return
        intervals[index]["depth_from"] = depth_from
        intervals[index]["depth_to"] = depth_to
        self._show_stack()
        self.statusBar().showMessage(f"Интервал изменён: {depth_from:g}–{depth_to:g} м")

    def _open_interpretation_interval_menu(self, well_name: str, index: int) -> None:
        intervals = self._well_interpretations.get(well_name, [])
        if not 0 <= index < len(intervals):
            return
        interval = intervals[index]
        menu = QMenu(self)
        rename = menu.addAction("Изменить название…")
        recolor = menu.addAction("Изменить цвет…")
        delete = menu.addAction("Удалить интервал")
        rename.triggered.connect(lambda: self._rename_interpretation_interval(well_name, index))
        recolor.triggered.connect(lambda: self._recolor_interpretation_interval(well_name, index))
        delete.triggered.connect(lambda: self._delete_interpretation_interval(well_name, index))
        menu.exec(QCursor.pos())

    def _rename_interpretation_interval(self, well_name: str, index: int) -> None:
        interval = self._well_interpretations[well_name][index]
        name, accepted = QInputDialog.getText(self, "Название интервала", "Название:", text=str(interval.get("name") or ""))
        if accepted and name.strip():
            interval["name"] = name.strip()
            self._show_stack()

    def _recolor_interpretation_interval(self, well_name: str, index: int) -> None:
        interval = self._well_interpretations[well_name][index]
        color = QColorDialog.getColor(QColor(str(interval.get("color") or "#8f71d2")), self, "Цвет интервала")
        if color.isValid():
            interval["color"] = color.name(QColor.NameFormat.HexRgb)
            self._show_stack()

    def _delete_interpretation_interval(self, well_name: str, index: int) -> None:
        intervals = self._well_interpretations.get(well_name, [])
        if not 0 <= index < len(intervals):
            return
        del intervals[index]
        if not intervals:
            self._well_interpretations.pop(well_name, None)
        self._show_stack()
        self.statusBar().showMessage("Интерпретационный интервал удалён")

    def export_presentation_png(self) -> None:
        default_name = f"{self._project_title or 'kern_analyzer'}_presentation.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт для презентации",
            str((self._project_folder or self._projects_root()) / default_name),
            "PNG высокого качества (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            canvas = self.sheets.currentWidget()
            canvas.export_png(path, long_side=6000)
        except Exception as exc:
            QMessageBox.critical(self, "Экспорт PNG", str(exc))
            return
        self.statusBar().showMessage(f"Экспортировано для презентации: {Path(path).name} · до 6000 px по длинной стороне")

    def _core_description_report_wells(self) -> list[dict]:
        """Capture the exact visual and editable data of the core-description sheet."""
        self._show_stack()
        columns = {item.well_name: item for item in self.workspace._correlation_items if item.well_name}
        wells: list[dict] = []
        for well_name in self._well_names():
            column = columns.get(well_name)
            if column is None:
                continue
            region = column.sceneBoundingRect()
            for record in self._records_for_well(well_name):
                photo_item = self.workspace._items_by_identifier.get(record.identifier)
                if photo_item is not None:
                    region = region.united(photo_item.sceneBoundingRect())
            image = self.workspace.render_scene_region(region.adjusted(-12, -12, 12, 12), long_side=7200)
            layers = []
            for record in self._records_for_well(well_name):
                for detection in StackColumnItem._sort_by_reading_order(record.detections):
                    layers.append(
                        {
                            "label": detection.label,
                            "depth_from": detection.depth_from,
                            "depth_to": detection.depth_to,
                            "attributes": dict(detection.attributes or {}),
                        }
                    )
            wells.append(
                {
                    "name": well_name,
                    "image": image,
                    "depth_range": self._well_depth_ranges.get(well_name),
                    "depth_reference": self._well_depth_references.get(well_name, "MD"),
                    "depth_settings": dict(self._well_depth_settings.get(well_name, {})),
                    "core_interval": self._well_intervals.get(well_name),
                    "layers": layers,
                }
            )
        return wells

    def export_core_description_report(self) -> None:
        if not self._well_names():
            QMessageBox.information(self, "Отчёт", "Сначала добавьте хотя бы одну скважину или фотографии керна.")
            return
        default = (self._project_folder or self._projects_root()) / f"{self._project_title or 'kern_analyzer'}_послойное_описание.xlsx"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить послойное седиментологическое описание",
            str(default),
            "Книга Excel (*.xlsx);;Документ Word (*.docx);;PDF (*.pdf)",
        )
        if not path:
            return
        suffix = Path(path).suffix.casefold()
        if suffix == ".pdf" or "PDF" in selected_filter:
            report_format = "pdf"
        elif suffix == ".xlsx" or "Excel" in selected_filter:
            report_format = "xlsx"
        else:
            report_format = "docx"
        if not suffix:
            path += f".{report_format}"
        try:
            wells = self._core_description_report_wells()
            export_core_description_report(path, self._project_title, wells, report_format)
        except Exception as exc:
            QMessageBox.critical(self, "Экспорт отчёта", str(exc))
            return
        self.statusBar().showMessage(f"Отчёт сохранён: {Path(path).name} · скважин: {len(wells)}")

    def edit_gis(self) -> None:
        self._edit_log("gis")

    def edit_rigis(self) -> None:
        self._edit_log("rigis")

    def _edit_log(self, kind: str) -> None:
        title = "ГИС" if kind == "gis" else "РИГИС"
        well_name = self._choose_well(f"{title}: выбор скважины")
        if well_name is None:
            return
        log_data = self._well_log(well_name, kind)
        if not log_data:
            QMessageBox.information(self, title, f"Сначала загрузите LAS-файл для {title} этой скважины.")
            return
        top, base = self._current_core_interval(log_data, well_name)
        dialog = LogEditorDialog(log_data, top, base, f"Настройка {title} · {well_name}", self, settings_key=kind)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            updated_data, top, base = dialog.values()
        except ValueError as exc:
            QMessageBox.warning(self, "Проверьте данные", str(exc))
            return
        self._set_well_log(well_name, kind, updated_data)
        self._apply_core_interval(top, base, well_name)
        self._show_stack()
        self.statusBar().showMessage(f"{title} обновлён и привязан к интервалу {top:g}–{base:g} м")

    def create_demo_las(self) -> None:
        """Create editable synthetic LAS curves for immediately testing the stack layout."""
        well_name = self._choose_well("Демо-ГИС: выбор скважины")
        if well_name is None:
            return
        ordered = self._ordered_detections(well_name)
        if not ordered:
            QMessageBox.information(self, "Демо-ГИС", "Сначала загрузите фото и дождитесь сегментации.")
            return
        start_depth = 1000.0
        for index, detection in enumerate(ordered):
            if detection.depth_from is None:
                detection.depth_from = start_depth + index
            if detection.depth_to is None:
                detection.depth_to = start_depth + index + 1
        end_depth = max(detection.depth_to or start_depth for detection in ordered)
        step = 0.02
        depths = [round(start_depth + index * step, 4) for index in range(int((end_depth - start_depth) / step) + 1)]
        values = {
            "GR": [65 + 30 * math.sin((depth - start_depth) * 2.6) for depth in depths],
            "RT": [20 + 15 * math.cos((depth - start_depth) * 1.7) + 5 * math.sin((depth - start_depth) * 6) for depth in depths],
            "RHOB": [2.35 + 0.18 * math.sin((depth - start_depth) * 3.1) for depth in depths],
            "NPHI": [0.24 + 0.09 * math.cos((depth - start_depth) * 2.2) for depth in depths],
        }
        self._set_well_log(well_name, "gis", {
            "file_name": "demo_las.las",
            "well_name": "Демо-скважина",
            "depth_curve": "DEPT",
            "depth_unit": "M",
            "curves": [
                {"mnemonic": "DEPT", "unit": "M"},
                {"mnemonic": "GR", "unit": "API"},
                {"mnemonic": "RT", "unit": "OHMM"},
                {"mnemonic": "RHOB", "unit": "G/C3"},
                {"mnemonic": "NPHI", "unit": "V/V"},
            ],
            "depths": depths,
            "values": {"DEPT": depths, **values},
        })
        for record in self._records_for_well(well_name):
            self.workspace.update_photo_detections(record)
        self._show_stack()
        self.statusBar().showMessage("Созданы демо-данные LAS · двойной клик по «Привязке» редактирует интервалы")

    def _show_segmentation_progress(self, current: int, total: int, file_name: str) -> None:
        label = f"Сегментация {current}/{total}: {file_name}"
        self.statusBar().showMessage(label)
        self._update_activity(label, current, total)

    def _apply_image_detections(self, image_path: str, detections) -> None:
        record = next((item for item in self._records if item.path == image_path), None)
        if record is None:
            return
        record.detections = postprocess_detections(record, list(detections))
        self.workspace.update_photo_detections(record)
        self._segmentation_updated_count += 1
        if self._segmentation_updated_count % 10 == 0:
            self._refresh_project_tree()

    def _segmentation_error(self, message: str) -> None:
        self._segmentation_failed = True
        QMessageBox.critical(self, "Ошибка сегментации", message)

    def _on_segmentation_finished(self) -> None:
        self._segmentation_thread = None
        self._segmentation_worker = None
        self._refresh_project_tree()
        self._refresh_training_bar()
        if self._queued_records:
            queued_records, self._queued_records = self._queued_records, []
            self.run_segmentation(queued_records)
            return
        if self._pending_photo_imports and not self._segmentation_failed:
            self._load_next_photo_batch()
            return
        self._close_activity()
        if not self._segmentation_failed:
            self._show_stack()
            if self._project_folder is not None:
                self._save_current_project(self._project_folder)
            count = sum(len(record.detections) for record in self._records)
            self.statusBar().showMessage(f"Сегментация завершена · фаций: {count} · общий столбик справа от фотографий создан")

    def _open_facies_editor(self, record: PhotoRecord, detection) -> None:
        # Correlation editing is deliberately disabled while Kern Analyzer focuses
        # on core description and creation of a training dataset.
        if False:
            menu = QMenu(self)
            align = menu.addAction("\u041d\u043e\u0432\u0430\u044f \u043f\u0440\u044f\u043c\u0430\u044f \u043f\u043e \u0441\u043b\u043e\u044e")
            attach_menu = menu.addMenu("\u041f\u0440\u0438\u0432\u044f\u0437\u0430\u0442\u044c \u043a \u0441\u043e\u0437\u0434\u0430\u043d\u043d\u043e\u0439 \u043b\u0438\u043d\u0438\u0438")
            curve_actions = {}
            for curve in self.correlation_workspace.correlation_curves():
                action = attach_menu.addAction(str(curve.get("name") or curve.get("label") or "\u041b\u0438\u043d\u0438\u044f"))
                curve_actions[action] = curve
            attach_menu.setEnabled(bool(curve_actions))
            edit = menu.addAction("\u0420\u0435\u0434\u0430\u043a\u0442\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u043b\u0438\u0442\u043e\u043b\u043e\u0433\u0438\u044e")
            selected = menu.exec(QCursor.pos())
            if selected is align:
                self._align_straight_to_layer(record, detection)
                return
            if selected in curve_actions:
                self._align_layer_to_existing_curve(record, detection, curve_actions[selected])
                return
            if selected is not edit:
                return
        for current_dialog in list(self._facies_dialogs):
            current_dialog.close()
        self.workspace.end_polygon_edit()
        dialog = FaciesDialog(
            detection.label,
            detection.confidence,
            detection.attributes,
            detection.depth_from,
            detection.depth_to,
            self,
        )
        dialog.setModal(False)
        dialog.accepted.connect(lambda: self._save_lithology(record, detection, dialog))
        dialog.delete_requested.connect(lambda: self._delete_facies(record, detection))
        dialog.finished.connect(lambda _: self._forget_facies_dialog(dialog))
        self._facies_dialogs.append(dialog)
        dialog.show()
        self.statusBar().showMessage("Выберите фацию и независимые параметры описания")

    def _align_layer_to_existing_curve(self, record: PhotoRecord, detection, curve: dict) -> None:
        depths = [float(value) for value in (detection.depth_from, detection.depth_to) if value is not None]
        if not depths:
            QMessageBox.information(self, "\u041f\u0440\u0438\u0432\u044f\u0437\u043a\u0430 \u043a \u043b\u0438\u043d\u0438\u0438", "\u0421\u043d\u0430\u0447\u0430\u043b\u0430 \u0437\u0430\u0434\u0430\u0439\u0442\u0435 \u0433\u043b\u0443\u0431\u0438\u043d\u0443 \u0441\u043b\u043e\u044f.")
            return
        if self.correlation_workspace.align_curve_to_layer(curve, record.well_name, sum(depths) / len(depths)):
            self.statusBar().showMessage("\u0421\u043b\u043e\u0439 \u043f\u0440\u0438\u0432\u044f\u0437\u0430\u043d \u043a \u0432\u044b\u0431\u0440\u0430\u043d\u043d\u043e\u0439 \u043a\u043e\u0440\u0440\u0435\u043b\u044f\u0446\u0438\u043e\u043d\u043d\u043e\u0439 \u043b\u0438\u043d\u0438\u0438")

    def _align_straight_to_layer(self, record: PhotoRecord, detection) -> None:
        if len(self._well_names()) < 2:
            QMessageBox.information(self, "\u0412\u044b\u0440\u0430\u0432\u043d\u0438\u0432\u0430\u043d\u0438\u0435", "\u0414\u043b\u044f \u043b\u0438\u043d\u0438\u0438 \u043d\u0443\u0436\u043d\u044b \u043c\u0438\u043d\u0438\u043c\u0443\u043c \u0434\u0432\u0435 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u044b.")
            return
        depths = [float(value) for value in (detection.depth_from, detection.depth_to) if value is not None]
        if not depths:
            QMessageBox.information(self, "\u0412\u044b\u0440\u0430\u0432\u043d\u0438\u0432\u0430\u043d\u0438\u0435", "\u0421\u043d\u0430\u0447\u0430\u043b\u0430 \u0437\u0430\u0434\u0430\u0439\u0442\u0435 \u0433\u043b\u0443\u0431\u0438\u043d\u0443 \u0441\u043b\u043e\u044f.")
            return
        depth = sum(depths) / len(depths)
        self._show_stack()
        name = f"{detection.label} \u00b7 {depth:g} \u043c"
        if self.correlation_workspace.add_straight_layer_correlation_curve(name, record.well_name, depth):
            self.statusBar().showMessage("\u0421\u043b\u043e\u0439 \u0432\u044b\u0440\u043e\u0432\u043d\u0435\u043d \u043e\u0434\u043d\u043e\u0439 \u043f\u0440\u044f\u043c\u043e\u0439 \u043f\u043e \u0432\u0441\u0435\u043c \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u0430\u043c")

    def _begin_polygon_edit(self, record: PhotoRecord, detection) -> None:
        for current_dialog in list(self._facies_dialogs):
            current_dialog.close()
        if self.workspace.begin_polygon_edit(record, detection):
            self.statusBar().showMessage(
                "Контур: перетаскивайте точки · «＋ Точка» или Shift+клик добавляет · «− Точка»/Delete удаляет · ✓ Готово завершает"
            )

    def _prepare_polygon_vertex_insert(self) -> None:
        if self.workspace.prepare_polygon_vertex_insert():
            self.statusBar().showMessage("Добавление вершины: щёлкните в нужном месте на контуре · Esc — отменить")
            return
        QMessageBox.information(self, "Добавить точку", "Сначала щёлкните по контуру, который нужно редактировать.")

    def _delete_selected_polygon_vertex(self) -> None:
        if self.workspace.delete_selected_polygon_vertex():
            self.statusBar().showMessage("Вершина удалена")
            return
        QMessageBox.information(
            self,
            "Удалить точку",
            "Сначала щёлкните по вершине контура. У полигона всегда должно остаться минимум три точки.",
        )

    def _finish_polygon_edit(self) -> None:
        self.workspace.end_polygon_edit()
        self.statusBar().showMessage("Редактирование контура завершено")

    def _set_new_contour_mode(self, active: bool) -> None:
        if active and not self._records:
            self._new_contour_action.setChecked(False)
            self.statusBar().showMessage("Сначала загрузите фотографию")
            return
        self.workspace.set_new_contour_mode(active)
        if active:
            self.statusBar().showMessage(
                "Новый контур: ставьте точки левой кнопкой на фото · Enter или двойной клик — завершить · Esc — отменить"
            )

    def _add_manual_contour(self, record: PhotoRecord, detection) -> None:
        # It was drawn by the user, so it should be visible in the counter
        # immediately.  It becomes a trainable example once a facies code is
        # selected; an unnamed YOLO class would be invalid training data.
        detection.training_ready = True
        record.detections.append(detection)
        self.workspace.set_new_contour_mode(False)
        self._new_contour_action.setChecked(False)
        self._refresh_facies_views(record)
        self.workspace.begin_polygon_edit(record, detection)
        self.statusBar().showMessage("Новый контур добавлен в счётчик «новые» · назначьте фацию правой кнопкой мыши")

    def _save_lithology(self, record: PhotoRecord, detection, dialog: FaciesDialog) -> None:
        try:
            depth_from, depth_to = dialog.selected_depth_range()
        except ValueError as exc:
            QMessageBox.warning(self, "Проверьте глубины", str(exc))
            return
        detection.label = dialog.selected_facies()
        if not detection.label:
            QMessageBox.warning(self, "Не указана фация", "Введите или выберите основную метку фации, например DWCh.")
            return
        detection.attributes = dialog.selected_attributes()
        detection.depth_from = depth_from
        detection.depth_to = depth_to
        geometry_updated = self._sync_polygon_to_depth_range(record, detection)
        # The user explicitly saved this layer, so it is safe to use as a
        # training example. Fresh automatic predictions never get this flag.
        detection.training_ready = True
        self._refresh_facies_views(record)
        suffix = " · подсветка контура обновлена" if geometry_updated else ""
        self.statusBar().showMessage(f"Параметры фации сохранены: {detection.label}{suffix}")

    def _open_depth_binding(self, record: PhotoRecord, detection) -> None:
        dialog = DepthRangeDialog(detection.depth_from, detection.depth_to, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            detection.depth_from, detection.depth_to = dialog.values()
        except ValueError as exc:
            QMessageBox.warning(self, "Проверьте глубины", str(exc))
            return
        geometry_updated = self._sync_polygon_to_depth_range(record, detection)
        self._refresh_facies_views(record)
        suffix = " · подсветка контура обновлена" if geometry_updated else ""
        self.statusBar().showMessage(f"Привязка слоя обновлена{suffix}")

    def _delete_facies(self, record: PhotoRecord, detection) -> None:
        record.detections = [item for item in record.detections if item is not detection]
        self._refresh_facies_views(record)
        self.statusBar().showMessage("Подсветка удалена")

    def _forget_facies_dialog(self, dialog: FaciesDialog) -> None:
        if dialog in self._facies_dialogs:
            self._facies_dialogs.remove(dialog)

    def _on_facies_geometry_changed(self, record: PhotoRecord, detection) -> None:
        depth_updated = self._sync_depth_range_from_polygon(record, detection)
        # Moving a vertex is an explicit user review.  A model suggestion only
        # becomes a training example after this action (or after saving it in
        # the facies dialog), never merely because the model drew it.
        if str(detection.label or "").strip() not in {"", "Новый контур"}:
            detection.training_ready = True
        self._refresh_facies_views(record)
        suffix = " · интервал глубин обновлён" if depth_updated else ""
        if detection.training_ready:
            suffix += " · разметка добавлена в обучение"
        self.statusBar().showMessage(f"Контур фации обновлён{suffix}")

    @staticmethod
    def _photo_interval_from_path(path: Path) -> tuple[float | None, float | None]:
        try:
            interval = photo_interval_from_filename(path)
        except ValueError:
            return None, None
        return interval.top, interval.base

    def _record_photo_interval(self, record: PhotoRecord) -> tuple[float | None, float | None]:
        if record.photo_depth_from is not None and record.photo_depth_to is not None:
            return record.photo_depth_from, record.photo_depth_to
        return self._photo_interval_from_path(Path(record.path))

    @staticmethod
    def _depth_segment_for_polygon(record: PhotoRecord, detection) -> dict[str, float] | None:
        """Return the calibration column that owns most of a polygon."""
        if not record.depth_segments:
            return None
        bounds = QPolygonF(detection.polygon).boundingRect()
        center_x = bounds.center().x()
        center_y = bounds.center().y()

        def score(segment: dict[str, float]) -> tuple[float, float]:
            left, top = segment["left"], segment["top"]
            right, bottom = segment["right"], segment["bottom"]
            overlap = max(0.0, min(bounds.right(), right) - max(bounds.left(), left)) * max(0.0, min(bounds.bottom(), bottom) - max(bounds.top(), top))
            contains_center = left <= center_x <= right and top <= center_y <= bottom
            return (1.0 if contains_center else 0.0, overlap)

        return max(record.depth_segments, key=score, default=None)

    def _depth_mapping_for_polygon(self, record: PhotoRecord, detection) -> dict[str, float] | None:
        segment = self._depth_segment_for_polygon(record, detection)
        if segment is not None:
            return segment
        photo_top, photo_base = self._record_photo_interval(record)
        if photo_top is None or photo_base is None:
            return None
        return {
            "left": 0.0, "top": 0.0, "right": float(record.pixmap.width()), "bottom": float(record.pixmap.height()),
            "depth_from": photo_top, "depth_to": photo_base,
        }

    def _sync_depth_range_from_polygon(self, record: PhotoRecord, detection) -> bool:
        """Map the edited vertical polygon span back to measured depth."""
        segment = self._depth_mapping_for_polygon(record, detection)
        if segment is None:
            return False
        bounds = QPolygonF(detection.polygon).boundingRect()
        pixel_top, pixel_base = segment["top"], segment["bottom"]
        depth_top, depth_base = segment["depth_from"], segment["depth_to"]
        top_y = max(pixel_top, min(pixel_base, bounds.top()))
        base_y = max(pixel_top, min(pixel_base, bounds.bottom()))
        if base_y - top_y <= 1e-6 or depth_base <= depth_top:
            return False
        depth_span = depth_base - depth_top
        pixel_span = pixel_base - pixel_top
        detection.depth_from = round(depth_top + (top_y - pixel_top) / pixel_span * depth_span, 3)
        detection.depth_to = round(depth_top + (base_y - pixel_top) / pixel_span * depth_span, 3)
        return True

    def _sync_polygon_to_depth_range(self, record: PhotoRecord, detection) -> bool:
        """Resize an existing polygon vertically to the edited depth range."""
        segment = self._depth_mapping_for_polygon(record, detection)
        depth_from, depth_to = detection.depth_from, detection.depth_to
        if (
            segment is None or depth_from is None or depth_to is None
            or depth_to <= depth_from or not detection.polygon
        ):
            return False
        bounds = QPolygonF(detection.polygon).boundingRect()
        old_top, old_base = bounds.top(), bounds.bottom()
        if old_base - old_top <= 1e-6:
            return False
        pixel_top, pixel_base = segment["top"], segment["bottom"]
        depth_top, depth_base = segment["depth_from"], segment["depth_to"]
        if depth_from < depth_top - 1e-6 or depth_to > depth_base + 1e-6 or depth_base <= depth_top:
            return False
        scale = (pixel_base - pixel_top) / (depth_base - depth_top)
        target_top = pixel_top + (depth_from - depth_top) * scale
        target_base = pixel_top + (depth_to - depth_top) * scale
        if target_base - target_top <= 1e-6:
            return False
        target_scale = (target_base - target_top) / (old_base - old_top)
        record_width = record.pixmap.width()
        detection.polygon = [
            QPointF(
                max(0.0, min(float(record_width), point.x())),
                max(0.0, min(float(record.pixmap.height()), target_top + (point.y() - old_top) * target_scale)),
            )
            for point in detection.polygon
        ]
        return True

    def _refresh_facies_views(self, record: PhotoRecord) -> None:
        self.workspace.update_photo_detections(record)
        if self._well_names():
            self._show_stack()
        self._refresh_project_tree()
        self._refresh_training_bar()
