from __future__ import annotations

from hashlib import md5

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPainterPathStroker, QPen, QPixmap, QPolygonF, QTransform
from PySide6.QtWidgets import QGraphicsObject, QGraphicsScene, QGraphicsView, QToolTip

from app.domain.models import FaciesDetection, PhotoRecord
from app.domain.lithology import LITHOLOGY_LEGEND


class WorkspaceCanvas(QGraphicsView):
    """Infinite Miro-like board with movable photos and editable facies masks."""

    facies_selected = Signal(object, object)
    facies_context_requested = Signal(object, object)
    facies_geometry_changed = Signal(object, object)
    facies_created = Signal(object, object)
    depth_binding_requested = Signal(object, object)
    column_paint_requested = Signal(object, object, str)
    interpretation_interval_created = Signal(str, str, float, float)
    interpretation_interval_resized = Signal(str, int, float, float)
    interpretation_interval_context_requested = Signal(str, int)
    correlation_curve_context_requested = Signal(object, str)
    well_order_changed = Signal(object)
    photo_deleted = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.scene().setSceneRect(-30_000, -30_000, 60_000, 60_000)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setBackgroundBrush(QColor("#fbfcfe"))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scale = 1.0
        self._pan_start = None
        self._pan_scroll = None
        self._items_by_identifier: dict[str, PhotoItem] = {}
        self._stack_item: StackColumnItem | None = None
        self._correlation_items: list[StackColumnItem] = []
        self._correlation_lines: list[CorrelationCurveItem] = []
        self._correlation_curves: list[dict] = []

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.fillRect(rect, QColor("#fbfcfe"))
        minor_step, major_step = 28, 140
        left = int(rect.left()) - int(rect.left()) % minor_step - minor_step
        top = int(rect.top()) - int(rect.top()) % minor_step - minor_step
        right, bottom = int(rect.right()) + minor_step, int(rect.bottom()) + minor_step
        painter.setPen(QPen(QColor("#e9edf4"), 1))
        for x in range(left, right, minor_step):
            painter.drawLine(x, top, x, bottom)
        for y in range(top, bottom, minor_step):
            painter.drawLine(left, y, right, y)
        painter.setPen(QPen(QColor("#dce2ec"), 1))
        for x in range(left - left % major_step, right, major_step):
            painter.drawLine(x, top, x, bottom)
        for y in range(top - top % major_step, bottom, major_step):
            painter.drawLine(left, y, right, y)

    def wheelEvent(self, event) -> None:
        factor = 1.14 if event.angleDelta().y() > 0 else 1 / 1.14
        new_scale = max(0.15, min(12.0, self._scale * factor))
        self.scale(new_scale / self._scale, new_scale / self._scale)
        self._scale = new_scale

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = event.pos()
            self._pan_scroll = (self.horizontalScrollBar().value(), self.verticalScrollBar().value())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_start is not None and self._pan_scroll is not None:
            delta = event.pos() - self._pan_start
            self.horizontalScrollBar().setValue(self._pan_scroll[0] - delta.x())
            self.verticalScrollBar().setValue(self._pan_scroll[1] - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_start is not None:
            self._pan_start = None
            self._pan_scroll = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected_photos()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self.set_new_contour_mode(False)
            self.end_polygon_edit()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.finish_new_contour():
                event.accept()
                return
        super().keyPressEvent(event)

    def add_photos(self, records: list[PhotoRecord]) -> None:
        for number, record in enumerate(records, start=len(self._items_by_identifier)):
            item = self._create_photo_item(record)
            col, row = number % 20, number // 20
            item.setPos(col * 228, row * 330)

    def restore_photo(self, record: PhotoRecord, position: QPointF) -> None:
        item = self._create_photo_item(record)
        item.setPos(position)

    def delete_selected_photos(self) -> None:
        selected = [item for item in self.scene().selectedItems() if isinstance(item, PhotoItem)]
        for item in selected:
            identifier = item.record.identifier
            position = QPointF(item.pos())
            self._items_by_identifier.pop(identifier, None)
            self.scene().removeItem(item)
            self.photo_deleted.emit(identifier, position)

    def photo_positions(self) -> dict[str, QPointF]:
        return {identifier: QPointF(item.pos()) for identifier, item in self._items_by_identifier.items()}

    def clear_workspace(self) -> None:
        self._items_by_identifier.clear()
        self.clear_generated_items()
        self._correlation_curves = []
        self._gis_preview = None
        self._rigis_preview = None
        self.scene().clear()

    def clear_generated_items(self) -> None:
        """Remove assembled-core and correlation views while retaining source photos."""
        if self._stack_item is not None:
            self.scene().removeItem(self._stack_item)
            self._stack_item = None
        self._clear_correlation_items()

    def fit_content(self) -> None:
        bounds = self.scene().itemsBoundingRect()
        if not bounds.isEmpty():
            self.fitInView(bounds.adjusted(-80, -80, 80, 80), Qt.AspectRatioMode.KeepAspectRatio)
            self._scale = self.transform().m11()

    def focus_photos(self) -> bool:
        """Return from logs/correlation to the original photo markup area."""
        photo_items = list(self._items_by_identifier.values())
        if not photo_items:
            return False
        bounds = photo_items[0].sceneBoundingRect()
        for item in photo_items[1:]:
            bounds = bounds.united(item.sceneBoundingRect())
        for selected in self.scene().selectedItems():
            selected.setSelected(False)
        self.fitInView(bounds.adjusted(-48, -48, 48, 48), Qt.AspectRatioMode.KeepAspectRatio)
        self._scale = self.transform().m11()
        self.setFocus()
        return True

    def update_photo_detections(self, record: PhotoRecord) -> None:
        item = self._items_by_identifier.get(record.identifier)
        if item:
            item.set_detections(record.detections)

    def show_stack(self, records: list[PhotoRecord], gis_preview: dict | None = None, rigis_preview: dict | None = None, well_name: str | None = None, interpretation_intervals: list[dict] | None = None, description_only: bool = False) -> None:
        if gis_preview is not None:
            self._gis_preview = gis_preview
        elif not hasattr(self, "_gis_preview"):
            self._gis_preview = None
        if rigis_preview is not None:
            self._rigis_preview = rigis_preview
        elif not hasattr(self, "_rigis_preview"):
            self._rigis_preview = None
        self._clear_correlation_items()
        if self._stack_item is not None:
            self.scene().removeItem(self._stack_item)
        self._stack_item = StackColumnItem(records, self._gis_preview, self._rigis_preview, well_name=well_name, interpretation_intervals=interpretation_intervals, description_only=description_only)
        self._stack_item.facies_selected.connect(self.facies_selected)
        self._stack_item.facies_context_requested.connect(self.facies_context_requested)
        self._stack_item.depth_binding_requested.connect(self.depth_binding_requested)
        self._stack_item.column_paint_requested.connect(self.column_paint_requested)
        self._stack_item.interpretation_interval_created.connect(self.interpretation_interval_created)
        self._stack_item.interpretation_interval_resized.connect(self.interpretation_interval_resized)
        self._stack_item.interpretation_interval_context_requested.connect(self.interpretation_interval_context_requested)
        self._stack_item.well_drag_finished.connect(self._emit_well_order)
        self.scene().addItem(self._stack_item)
        photo_right_edge = max(
            (item.pos().x() + item.boundingRect().right() for item in self._items_by_identifier.values()),
            default=0,
        )
        self._stack_item.setPos(photo_right_edge + 72, 0)

    def show_correlation(
        self,
        well_records: dict[str, list[PhotoRecord]],
        well_logs: dict[str, tuple[dict | None, dict | None]],
        well_interpretations: dict[str, list[dict]] | None = None,
        spacing: float = 180.0,
        vertical_scale: float = 1.0,
        connect_layers: bool = True,
        description_only: bool = False,
        place_source_photos: bool = False,
    ) -> None:
        """Lay out independently assembled wells as a Petrel-like correlation panel."""
        if self._stack_item is not None:
            self.scene().removeItem(self._stack_item)
            self._stack_item = None
        self._clear_correlation_items()
        cursor = 30.0 if place_source_photos else max(
            (item.pos().x() + item.boundingRect().right() for item in self._items_by_identifier.values()),
            default=-42,
        ) + 72
        for well_name, records in well_records.items():
            gis_preview, rigis_preview = well_logs.get(well_name, (None, None))
            if place_source_photos and records:
                photo_items = [self._items_by_identifier.get(record.identifier) for record in records]
                photo_items = [item for item in photo_items if item is not None]
                photo_width = max((item.boundingRect().width() for item in photo_items), default=0.0)
                photo_y = 0.0
                for photo_item in photo_items:
                    photo_item.setPos(cursor, photo_y)
                    photo_y += photo_item.boundingRect().height() + 12
                cursor += photo_width + 18
            item = StackColumnItem(records, gis_preview, rigis_preview, well_name=well_name, interpretation_intervals=(well_interpretations or {}).get(well_name, []), description_only=description_only)
            item.facies_selected.connect(self.facies_selected)
            item.facies_context_requested.connect(self.facies_context_requested)
            item.depth_binding_requested.connect(self.depth_binding_requested)
            item.column_paint_requested.connect(self.column_paint_requested)
            item.interpretation_interval_created.connect(self.interpretation_interval_created)
            item.interpretation_interval_resized.connect(self.interpretation_interval_resized)
            item.interpretation_interval_context_requested.connect(self.interpretation_interval_context_requested)
            item.well_drag_finished.connect(self._emit_well_order)
            item.setTransform(QTransform.fromScale(1.0, max(0.2, vertical_scale)))
            item.setPos(cursor, 0)
            self.scene().addItem(item)
            self._correlation_items.append(item)
            cursor += item.boundingRect().width() + max(20.0, spacing)
        if connect_layers:
            self._create_correlation_lines()

    def _clear_correlation_items(self) -> None:
        for line in self._correlation_lines:
            self.scene().removeItem(line)
        self._correlation_lines = []
        for item in self._correlation_items:
            self.scene().removeItem(item)
        self._correlation_items = []

    def _emit_well_order(self, _well_name: str) -> None:
        items = self._correlation_items or ([self._stack_item] if self._stack_item is not None else [])
        order = [item.well_name for item in sorted(items, key=lambda item: item.pos().x()) if item.well_name]
        if order:
            self.well_order_changed.emit(order)

    def _create_correlation_lines(self) -> None:
        """Create editable correlation horizons over the entire well panels."""
        markers_by_label: dict[str, dict[str, QPointF]] = {}
        for item in self._correlation_items:
            for label, point in item.correlation_markers():
                markers_by_label.setdefault(label, {}).setdefault(item.well_name, point)
        active_labels = {label for label, markers in markers_by_label.items() if len(markers) >= 2}
        existing = {str(curve.get("label")): curve for curve in self._correlation_curves}
        for label in sorted(active_labels, key=str.casefold):
            if label not in existing:
                self._correlation_curves.append(
                    {
                        "label": label,
                        "name": label,
                        "color": facies_color(label).darker(115).name(),
                        "width": 1.8,
                        "style": "dash",
                        "offsets": {},
                    }
                )
        for curve in self._correlation_curves:
            label = str(curve.get("label") or "")
            if curve.get("straight_y") is not None:
                segments = self._straight_curve_segments(float(curve["straight_y"]))
            elif curve.get("manual_ratio") is not None:
                segments = self._manual_curve_segments(float(curve.get("manual_ratio", 0.5)))
            elif label in active_labels:
                segments = self._curve_segments(markers_by_label[label])
            else:
                continue
            line = CorrelationCurveItem(curve, segments)
            line.context_requested.connect(self.correlation_curve_context_requested)
            line.changed.connect(self._on_correlation_curve_changed)
            line.setZValue(20)
            self.scene().addItem(line)
            self._correlation_lines.append(line)

    def _manual_curve_segments(self, ratio: float) -> list[dict]:
        ratio = max(0.0, min(1.0, ratio))
        return [
            {
                "well_name": item.well_name,
                "left": item.mapToScene(QPointF(0, item.boundingRect().height() * ratio)),
                "right": item.mapToScene(QPointF(item.boundingRect().width(), item.boundingRect().height() * ratio)),
            }
            for item in self._correlation_items
        ]

    def _straight_curve_segments(self, y: float) -> list[dict]:
        """One immutable horizontal horizon spanning every well panel."""
        return [
            {
                "well_name": item.well_name,
                "left": QPointF(item.mapToScene(QPointF(0, 0)).x(), y),
                "right": QPointF(item.mapToScene(QPointF(item.boundingRect().width(), 0)).x(), y),
            }
            for item in self._correlation_items
        ]

    def add_manual_correlation_curve(self, name: str) -> bool:
        if len(self._correlation_items) < 2:
            return False
        self._correlation_curves.append(
            {
                "label": f"manual-{len(self._correlation_curves) + 1}",
                "name": name,
                "color": "#734cbe",
                "width": 1.8,
                "style": "dash",
                "offsets": {},
                "breaks": [],
                "manual_ratio": 0.5,
            }
        )
        self.refresh_correlation_curves()
        return True

    def add_layer_correlation_curve(self, name: str, well_name: str, depth: float) -> bool:
        """Create an editable horizon initially aligned with a chosen layer."""
        item = next((candidate for candidate in self._correlation_items if candidate.well_name == well_name), None)
        if item is None:
            return False
        self._correlation_curves.append(
            {
                "label": f"layer-{len(self._correlation_curves) + 1}",
                "name": name,
                "color": "#e05d9c",
                "width": 2.2,
                "style": "dash",
                "offsets": {},
                "breaks": [],
                "manual_ratio": item.depth_ratio(depth),
                "anchor_well": well_name,
                "anchor_depth": float(depth),
            }
        )
        self.refresh_correlation_curves()
        return True

    def add_straight_layer_correlation_curve(self, name: str, well_name: str, depth: float) -> bool:
        item = next((candidate for candidate in self._correlation_items if candidate.well_name == well_name), None)
        if item is None or len(self._correlation_items) < 2:
            return False
        anchors = item._depth_anchors()
        if len(anchors) < 2:
            return False
        y = item.mapToScene(QPointF(0, item._depth_to_stack_y(float(depth), anchors))).y()
        self._correlation_curves.append(
            {
                "label": f"straight-{len(self._correlation_curves) + 1}",
                "name": name,
                "color": "#e05d9c",
                "width": 2.2,
                "style": "solid",
                "offsets": {},
                "breaks": [],
                "straight_y": y,
                "locked_straight": True,
                "anchor_well": well_name,
                "anchor_depth": float(depth),
            }
        )
        self.refresh_correlation_curves()
        return True

    def correlation_curves(self) -> list[dict]:
        return list(self._correlation_curves)

    def align_curve_to_layer(self, curve: dict, well_name: str, depth: float) -> bool:
        item = next((candidate for candidate in self._correlation_items if candidate.well_name == well_name), None)
        if item is None:
            return False
        anchors = item._depth_anchors()
        if len(anchors) < 2:
            return False
        target_y = item.mapToScene(QPointF(0, item._depth_to_stack_y(float(depth), anchors))).y()
        if curve.get("locked_straight"):
            curve["straight_y"] = target_y
            curve["anchor_well"] = well_name
            curve["anchor_depth"] = float(depth)
        else:
            line = next((candidate for candidate in self._correlation_lines if candidate.curve is curve), None)
            segment = next((segment for segment in (line.segments if line else []) if segment["well_name"] == well_name), None)
            if segment is None:
                return False
            curve.setdefault("offsets", {})[well_name] = target_y - float(segment["left"].y())
        self.refresh_correlation_curves()
        return True

    def refresh_correlation_curves(self) -> None:
        for line in self._correlation_lines:
            self.scene().removeItem(line)
        self._correlation_lines = []
        if self._correlation_items:
            self._create_correlation_lines()

    def _curve_segments(self, marker_by_well: dict[str, QPointF]) -> list[dict]:
        known_ratios = [
            marker_by_well[item.well_name].y() / max(1.0, item.boundingRect().height())
            for item in self._correlation_items
            if item.well_name in marker_by_well
        ]
        fallback_ratio = sum(known_ratios) / len(known_ratios) if known_ratios else 0.5
        segments = []
        for item in self._correlation_items:
            point = marker_by_well.get(item.well_name)
            local_y = point.y() if point is not None else item.boundingRect().height() * fallback_ratio
            left = item.mapToScene(QPointF(0, local_y))
            right = item.mapToScene(QPointF(item.boundingRect().width(), local_y))
            segments.append({"well_name": item.well_name, "left": left, "right": right})
        return segments

    def _on_correlation_curve_changed(self) -> None:
        self.scene().update()

    def export_png(self, file_path: str, long_side: int = 6000) -> None:
        """Render the complete scene at presentation resolution, independent of monitor DPI."""
        source = self.scene().itemsBoundingRect().adjusted(-36, -36, 36, 36)
        if source.isEmpty():
            raise ValueError("На рабочей области пока нечего экспортировать")
        scale = max(1.0, float(long_side) / max(source.width(), source.height()))
        image = QImage(
            max(1, round(source.width() * scale)),
            max(1, round(source.height() * scale)),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(QColor("#fbfcfe"))
        painter = QPainter(image)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.scene().render(painter, painter.viewport(), source)
        painter.end()
        if not image.save(file_path, "PNG", 100):
            raise OSError("Не удалось сохранить PNG")

    def render_scene_region(self, source: QRectF, long_side: int = 6000) -> QImage:
        """Return a high-resolution image of one logical part of the board."""
        if source.isEmpty():
            raise ValueError("На рабочей области пока нечего экспортировать")
        scale = max(1.0, float(long_side) / max(source.width(), source.height()))
        image = QImage(
            max(1, round(source.width() * scale)),
            max(1, round(source.height() * scale)),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(QColor("#fbfcfe"))
        painter = QPainter(image)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.scene().render(painter, painter.viewport(), source)
        painter.end()
        return image

    def has_stack(self) -> bool:
        return self._stack_item is not None or bool(self._correlation_items)

    def begin_polygon_edit(self, record: PhotoRecord, detection: FaciesDetection) -> bool:
        item = self._items_by_identifier.get(record.identifier)
        if item is None:
            return False
        self.end_polygon_edit()
        item.setSelected(True)
        return item.begin_polygon_edit(detection)

    def focus_photo(self, identifier: str) -> bool:
        item = self._items_by_identifier.get(identifier)
        if item is None:
            return False
        for selected in self.scene().selectedItems():
            selected.setSelected(False)
        item.setSelected(True)
        self.centerOn(item)
        return True

    def focus_facies(self, record: PhotoRecord, detection: FaciesDetection) -> bool:
        if not self.focus_photo(record.identifier):
            return False
        return self.begin_polygon_edit(record, detection)

    def end_polygon_edit(self) -> None:
        for item in self._items_by_identifier.values():
            item.end_polygon_edit()

    def prepare_polygon_vertex_insert(self) -> bool:
        """Arm the selected polygon for one click-to-insert vertex action."""
        return any(item.prepare_polygon_vertex_insert() for item in self._items_by_identifier.values())

    def delete_selected_polygon_vertex(self) -> bool:
        """Remove the highlighted vertex of the polygon being edited."""
        return any(item.delete_selected_polygon_vertex() for item in self._items_by_identifier.values())

    def set_new_contour_mode(self, active: bool) -> None:
        if active:
            self.end_polygon_edit()
        for item in self._items_by_identifier.values():
            item.set_new_contour_mode(active)

    def finish_new_contour(self) -> bool:
        created = False
        for item in self._items_by_identifier.values():
            created = item.finish_new_contour() or created
        return created

    def _create_photo_item(self, record: PhotoRecord) -> "PhotoItem":
        item = PhotoItem(record)
        item.facies_selected.connect(self.facies_selected)
        item.facies_context_requested.connect(self.facies_context_requested)
        item.facies_geometry_changed.connect(self.facies_geometry_changed)
        item.facies_created.connect(self.facies_created)
        self.scene().addItem(item)
        self._items_by_identifier[record.identifier] = item
        return item


class FaciesOverlayItem(QGraphicsObject):
    """Renders mutually exclusive masks and lets the active polygon vertices move or be added."""

    facies_clicked = Signal(object)
    facies_context_requested = Signal(object)
    facies_geometry_changed = Signal(object)
    facies_created = Signal(object)

    def __init__(self, source_size, target_rect: QRectF, detections: list[FaciesDetection], parent=None):
        super().__init__(parent)
        self.source_width, self.source_height = max(1, source_size.width()), max(1, source_size.height())
        self.target_rect = target_rect
        self.detections = detections
        self._hovered = -1
        self._editing_index = -1
        self._drag_vertex = -1
        self._selected_vertex = -1
        self._insert_vertex_on_click = False
        self._drawing_new_contour = False
        self._new_points: list[QPointF] = []
        self._visible_paths: list[QPainterPath] = []
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsObject.GraphicsItemFlag.ItemIsFocusable, True)
        self._rebuild_visible_paths()

    def boundingRect(self) -> QRectF:
        return self.target_rect

    def set_detections(self, detections: list[FaciesDetection]) -> None:
        self.prepareGeometryChange()
        self.detections = detections
        self._editing_index = -1
        self._drag_vertex = -1
        self._selected_vertex = -1
        self._insert_vertex_on_click = False
        self._drawing_new_contour = False
        self._new_points = []
        self._rebuild_visible_paths()
        self.update()

    def begin_polygon_edit(self, detection: FaciesDetection) -> bool:
        try:
            self._editing_index = next(index for index, item in enumerate(self.detections) if item is detection)
        except StopIteration:
            return False
        self._drag_vertex = -1
        self._selected_vertex = -1
        self._insert_vertex_on_click = False
        self.setFocus()
        self.update()
        return True

    def end_polygon_edit(self) -> None:
        if self._editing_index >= 0:
            self._editing_index = -1
            self._drag_vertex = -1
            self._selected_vertex = -1
            self._insert_vertex_on_click = False
            self.update()

    def prepare_vertex_insert(self) -> bool:
        if self._editing_index < 0:
            return False
        self._insert_vertex_on_click = True
        self.setCursor(Qt.CursorShape.CrossCursor)
        return True

    def delete_selected_vertex(self) -> bool:
        return self._delete_selected_vertex()

    def set_new_contour_mode(self, active: bool) -> None:
        self._drawing_new_contour = active
        self._new_points = []
        self.setCursor(Qt.CursorShape.CrossCursor if active else Qt.CursorShape.ArrowCursor)
        self.update()

    def finish_new_contour(self) -> bool:
        if not self._drawing_new_contour or len(self._new_points) < 3:
            return False
        detection = FaciesDetection(label="Новый контур", confidence=1.0, polygon=list(self._new_points))
        self._drawing_new_contour = False
        self._new_points = []
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        self.facies_created.emit(detection)
        return True

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for index, detection in enumerate(self.detections):
            path = self._visible_paths[index] if index < len(self._visible_paths) else QPainterPath()
            if path.isEmpty():
                continue
            color = facies_color(detection.label)
            fill = QColor(color)
            fill.setAlpha(105 if index != self._hovered else 165)
            painter.setBrush(fill)
            painter.setPen(QPen(color.darker(125), 2 if index == self._hovered else 1.25))
            painter.drawPath(path)

        if 0 <= self._editing_index < len(self.detections):
            painter.setBrush(QColor("#ffffff"))
            painter.setPen(QPen(QColor("#333a9d"), 1.5))
            for index, point in enumerate(self.detections[self._editing_index].polygon):
                painter.setBrush(QColor("#ffd866") if index == self._selected_vertex else QColor("#ffffff"))
                painter.drawEllipse(self._to_target(point), 4.5, 4.5)

        if self._drawing_new_contour:
            painter.setPen(QPen(QColor("#5149ca"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QColor("#ffffff"))
            target_points = [self._to_target(point) for point in self._new_points]
            if len(target_points) >= 2:
                painter.drawPolyline(QPolygonF(target_points))
            for point in target_points:
                painter.drawEllipse(point, 4.5, 4.5)

    def hoverMoveEvent(self, event) -> None:
        previous = self._hovered
        self._hovered = self._detection_at(event.pos())
        self.setCursor(Qt.CursorShape.PointingHandCursor if self._hovered >= 0 else Qt.CursorShape.ArrowCursor)
        if self._hovered != previous:
            if self._hovered >= 0:
                detection = self.detections[self._hovered]
                lines = [f"{detection.label}", f"Уверенность модели: {detection.confidence:.0%}"]
                if detection.alternatives:
                    lines.append("Альтернативные маски:")
                    lines.extend(f"• {label}: {confidence:.0%}" for label, confidence in sorted(detection.alternatives.items(), key=lambda item: item[1], reverse=True))
                QToolTip.showText(event.screenPos(), "\n".join(lines))
            else:
                QToolTip.hideText()
        self.update()

    def hoverLeaveEvent(self, event) -> None:
        self._hovered = -1
        QToolTip.hideText()
        self.update()

    def mousePressEvent(self, event) -> None:
        if self._drawing_new_contour:
            if event.button() == Qt.MouseButton.LeftButton:
                self._new_points.append(self._to_source(event.pos()))
                self.update()
                event.accept()
                return
            if event.button() == Qt.MouseButton.RightButton:
                self._new_points = []
                self.update()
                event.accept()
                return
        index = self._detection_at(event.pos())
        if event.button() == Qt.MouseButton.RightButton:
            if index >= 0:
                self.facies_context_requested.emit(self.detections[index])
                event.accept()
                return
            event.ignore()
            return
        if 0 <= self._editing_index < len(self.detections):
            if event.button() == Qt.MouseButton.LeftButton and (
                self._insert_vertex_on_click
                or event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            ):
                edge, _ = self._closest_edge(event.pos())
                if edge >= 0:
                    detection = self.detections[self._editing_index]
                    self._selected_vertex = edge + 1
                    detection.polygon.insert(self._selected_vertex, self._to_source(event.pos()))
                    self._insert_vertex_on_click = False
                    self._rebuild_visible_paths()
                    self.update()
                    self.facies_geometry_changed.emit(detection)
                    event.accept()
                    return
            vertex = self._vertex_at(event.pos())
            if vertex >= 0:
                self._drag_vertex = vertex
                self._selected_vertex = vertex
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
            edge = self._edge_at(event.pos())
            if edge >= 0:
                detection = self.detections[self._editing_index]
                # A click on an existing contour edge creates a new editable node.
                detection.polygon.insert(edge + 1, self._to_source(event.pos()))
                self._drag_vertex = edge + 1
                self._selected_vertex = edge + 1
                self._rebuild_visible_paths()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.update()
                event.accept()
                return
        if index >= 0:
            self.facies_clicked.emit(self.detections[index])
            event.accept()
            return
        event.ignore()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_vertex >= 0 and 0 <= self._editing_index < len(self.detections):
            detection = self.detections[self._editing_index]
            detection.polygon[self._drag_vertex] = self._to_source(event.pos())
            self._rebuild_visible_paths()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_vertex >= 0 and 0 <= self._editing_index < len(self.detections):
            detection = self.detections[self._editing_index]
            self._drag_vertex = -1
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.facies_geometry_changed.emit(detection)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._drawing_new_contour and event.button() == Qt.MouseButton.LeftButton:
            point = self._to_source(event.pos())
            if not self._new_points or (self._new_points[-1] - point).manhattanLength() > 1:
                self._new_points.append(point)
            self.finish_new_contour()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and 0 <= self._editing_index < len(self.detections):
            detection = self.detections[self._editing_index]
            active_path = QPainterPath()
            active_path.addPolygon(self._polygon(detection))
            active_path.closeSubpath()
            if active_path.contains(event.pos()):
                edge = self._closest_edge(event.pos())
                if edge >= 0:
                    # Double-click lets the user place a new vertex precisely
                    # where it is needed, even away from the current edge.
                    detection.polygon.insert(edge + 1, self._to_source(event.pos()))
                    self._rebuild_visible_paths()
                    self.update()
                    self.facies_geometry_changed.emit(detection)
                    event.accept()
                    return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self._editing_index >= 0:
            self._delete_selected_vertex()
            event.accept()
            return
        super().keyPressEvent(event)

    def _delete_selected_vertex(self) -> bool:
        if not (0 <= self._editing_index < len(self.detections)):
            return False
        detection = self.detections[self._editing_index]
        if len(detection.polygon) <= 3 or not 0 <= self._selected_vertex < len(detection.polygon):
            return False
        detection.polygon.pop(self._selected_vertex)
        self._selected_vertex = min(self._selected_vertex, len(detection.polygon) - 1)
        self._rebuild_visible_paths()
        self.update()
        self.facies_geometry_changed.emit(detection)
        return True

    def _detection_at(self, point: QPointF) -> int:
        for index in reversed(range(len(self.detections))):
            if index < len(self._visible_paths) and self._visible_paths[index].contains(point):
                return index
        return -1

    def _vertex_at(self, point: QPointF) -> int:
        detection = self.detections[self._editing_index]
        for index, source_point in enumerate(detection.polygon):
            if (self._to_target(source_point) - point).manhattanLength() <= 10:
                return index
        return -1

    def _edge_at(self, point: QPointF) -> int:
        """Return the edge preceding a point when the pointer is close to the contour."""
        index, distance = self._closest_edge(point)
        return index if distance <= 7.0 else -1

    def _closest_edge(self, point: QPointF) -> tuple[int, float]:
        polygon = self.detections[self._editing_index].polygon
        if len(polygon) < 2:
            return -1, float("inf")
        target_polygon = [self._to_target(source_point) for source_point in polygon]
        closest_index, closest_distance = -1, float("inf")
        for index, start in enumerate(target_polygon):
            end = target_polygon[(index + 1) % len(target_polygon)]
            distance = self._distance_to_segment(point, start, end)
            if distance < closest_distance:
                closest_index, closest_distance = index, distance
        return closest_index, closest_distance

    @staticmethod
    def _distance_to_segment(point: QPointF, start: QPointF, end: QPointF) -> float:
        dx, dy = end.x() - start.x(), end.y() - start.y()
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            return (point - start).manhattanLength()
        ratio = ((point.x() - start.x()) * dx + (point.y() - start.y()) * dy) / length_squared
        ratio = max(0.0, min(1.0, ratio))
        nearest = QPointF(start.x() + ratio * dx, start.y() + ratio * dy)
        dx, dy = point.x() - nearest.x(), point.y() - nearest.y()
        return (dx * dx + dy * dy) ** 0.5

    def _to_target(self, point: QPointF) -> QPointF:
        return QPointF(
            self.target_rect.left() + point.x() / self.source_width * self.target_rect.width(),
            self.target_rect.top() + point.y() / self.source_height * self.target_rect.height(),
        )

    def _to_source(self, point: QPointF) -> QPointF:
        x = (point.x() - self.target_rect.left()) / max(1.0, self.target_rect.width()) * self.source_width
        y = (point.y() - self.target_rect.top()) / max(1.0, self.target_rect.height()) * self.source_height
        return QPointF(max(0.0, min(self.source_width, x)), max(0.0, min(self.source_height, y)))

    def _polygon(self, detection: FaciesDetection) -> QPolygonF:
        return QPolygonF([self._to_target(point) for point in detection.polygon])

    def _rebuild_visible_paths(self) -> None:
        self._visible_paths = [QPainterPath() for _ in self.detections]
        claimed = QPainterPath()
        # Higher-confidence masks own every intersection, so highlights cannot overlap.
        for index in sorted(range(len(self.detections)), key=lambda item: self.detections[item].confidence, reverse=True):
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.WindingFill)
            path.addPolygon(self._polygon(self.detections[index]))
            # addPolygon does not close its subpath.  Boolean operations on an
            # open contour leave visual intersections, so always close first.
            path.closeSubpath()
            path = path.simplified()
            self._visible_paths[index] = path.subtracted(claimed).simplified()
            claimed = claimed.united(path).simplified()


class PhotoItem(QGraphicsObject):
    CARD_WIDTH = 210
    HEADER_HEIGHT = 44
    facies_selected = Signal(object, object)
    facies_context_requested = Signal(object, object)
    facies_geometry_changed = Signal(object, object)
    facies_created = Signal(object, object)

    def __init__(self, record: PhotoRecord):
        super().__init__()
        self.record = record
        self.preview_size = record.pixmap.size().scaled(QSize(self.CARD_WIDTH - 12, 260), Qt.AspectRatioMode.KeepAspectRatio)
        self.preview_rect = QRectF(6, self.HEADER_HEIGHT, self.preview_size.width(), self.preview_size.height())
        self._rect = QRectF(0, 0, self.CARD_WIDTH, self.HEADER_HEIGHT + self.preview_size.height() + 12)
        self.setFlags(QGraphicsObject.GraphicsItemFlag.ItemIsMovable | QGraphicsObject.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(2)
        self.overlay = FaciesOverlayItem(record.pixmap.size(), self.preview_rect, record.detections, self)
        self.overlay.facies_clicked.connect(self._on_facies_clicked)
        self.overlay.facies_context_requested.connect(self._on_facies_context_requested)
        self.overlay.facies_geometry_changed.connect(self._on_facies_geometry_changed)
        self.overlay.facies_created.connect(self._on_facies_created)

    def boundingRect(self) -> QRectF:
        return self._rect.adjusted(-2, -2, 2, 2)

    def set_detections(self, detections: list[FaciesDetection]) -> None:
        self.overlay.set_detections(detections)

    def begin_polygon_edit(self, detection: FaciesDetection) -> bool:
        return self.overlay.begin_polygon_edit(detection)

    def end_polygon_edit(self) -> None:
        self.overlay.end_polygon_edit()

    def prepare_polygon_vertex_insert(self) -> bool:
        return self.overlay.prepare_vertex_insert()

    def delete_selected_polygon_vertex(self) -> bool:
        return self.overlay.delete_selected_vertex()

    def set_new_contour_mode(self, active: bool) -> None:
        self.overlay.set_new_contour_mode(active)

    def finish_new_contour(self) -> bool:
        return self.overlay.finish_new_contour()

    def _on_facies_clicked(self, detection: FaciesDetection) -> None:
        self.facies_selected.emit(self.record, detection)

    def _on_facies_context_requested(self, detection: FaciesDetection) -> None:
        self.facies_context_requested.emit(self.record, detection)

    def _on_facies_geometry_changed(self, detection: FaciesDetection) -> None:
        self.facies_geometry_changed.emit(self.record, detection)

    def _on_facies_created(self, detection: FaciesDetection) -> None:
        self.facies_created.emit(self.record, detection)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(QPen(QColor("#7169df") if self.isSelected() else QColor("#dfe3ec"), 2 if self.isSelected() else 1))
        painter.drawRoundedRect(self._rect, 8, 8)
        painter.setPen(QColor("#5149ca"))
        well_name = self.record.well_name or "\u0421\u043a\u0432\u0430\u0436\u0438\u043d\u0430 1"
        painter.drawText(QRectF(8, 3, self.CARD_WIDTH - 16, 17), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "\u0421\u041a\u0412. " + well_name)
        painter.setPen(QColor("#34394c"))
        painter.drawText(QRectF(8, 20, self.CARD_WIDTH - 16, 20), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.record.path.split("/")[-1].split("\\")[-1])
        painter.drawPixmap(self.preview_rect.toRect(), self.record.pixmap)


class CorePhotoStripItem(QGraphicsObject):
    """Read-only source core-photo strip placed to the left of a well panel."""

    WIDTH = 174
    HEADER = 38
    GAP = 12

    def __init__(self, records: list[PhotoRecord], well_name: str):
        super().__init__()
        self.records = list(records)
        self.well_name = str(well_name or "")
        self._previews: list[tuple[PhotoRecord, QPixmap, QRectF]] = []
        cursor = self.HEADER + 8
        for record in self.records:
            # The main core sheet may hold a device-backed pixmap that is not
            # paintable in the second graphics scene.  Reload from its source
            # path first, then fall back to the in-memory original.
            source = record.pixmap
            preview = source.scaled(
                QSize(self.WIDTH - 16, 220),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            rect = QRectF((self.WIDTH - preview.width()) / 2, cursor + 18, preview.width(), preview.height())
            self._previews.append((record, preview, rect))
            cursor = rect.bottom() + self.GAP
        self._height = max(self.HEADER + 70, cursor + 4)
        self.setZValue(1)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.WIDTH, self._height)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(QPen(QColor("#b8c2d8"), 1.2))
        painter.drawRoundedRect(self.boundingRect(), 9, 9)
        painter.setPen(QColor("#353a50"))
        painter.drawText(
            QRectF(6, 5, self.WIDTH - 12, 25),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            f"\u0424\u041e\u0422\u041e \u041a\u0415\u0420\u041d\u0410 \u00b7 {self.well_name}",
        )
        painter.setPen(QColor("#69728a"))
        for record, preview, rect in self._previews:
            file_name = record.path.split("/")[-1].split("\\")[-1]
            painter.drawText(QRectF(7, rect.top() - 18, self.WIDTH - 14, 16), Qt.AlignmentFlag.AlignCenter, file_name)
            # Paint an image rather than a device-backed pixmap: it remains
            # reliable when this independent sheet is exported or redrawn.
            painter.drawImage(rect.toRect(), preview.toImage())
            painter.setPen(QPen(QColor("#d9deeb"), 1))
            painter.drawRect(rect)


class CorrelationCurveItem(QGraphicsObject):
    """Editable horizon drawn over every track in a correlation panel."""

    context_requested = Signal(object, str)
    changed = Signal()

    def __init__(self, curve: dict, segments: list[dict]):
        super().__init__()
        self.curve = curve
        self.segments = segments
        self._path = QPainterPath()
        self._drag_well: str | None = None
        self._drag_start_y = 0.0
        self._drag_original_offset = 0.0
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        self.setAcceptHoverEvents(True)
        self._rebuild_path()

    def boundingRect(self) -> QRectF:
        return self._path.boundingRect().adjusted(-10, -12, 10, 12)

    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(max(12.0, float(self.curve.get("width", 1.8)) + 8.0))
        return stroker.createStroke(self._path)

    def _offset_for(self, well_name: str) -> float:
        if self.curve.get("locked_straight"):
            return 0.0
        return float(dict(self.curve.get("offsets") or {}).get(well_name, 0.0))

    def _rebuild_path(self) -> None:
        self.prepareGeometryChange()
        path = QPainterPath()
        drawing = False
        breaks = {str(name) for name in self.curve.get("breaks", [])}
        for index, segment in enumerate(self.segments):
            if segment["well_name"] in breaks:
                drawing = False
                continue
            offset = self._offset_for(segment["well_name"])
            left = QPointF(segment["left"].x(), segment["left"].y() + offset)
            right = QPointF(segment["right"].x(), segment["right"].y() + offset)
            if not drawing:
                path.moveTo(left)
            else:
                path.lineTo(left)
            path.lineTo(right)
            drawing = True
        self._path = path
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        styles = {
            "solid": Qt.PenStyle.SolidLine,
            "dash": Qt.PenStyle.DashLine,
            "dot": Qt.PenStyle.DotLine,
            "dash_dot": Qt.PenStyle.DashDotLine,
        }
        color = QColor(str(self.curve.get("color") or "#734cbe"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(color, float(self.curve.get("width", 1.8)), styles.get(str(self.curve.get("style")), Qt.PenStyle.DashLine)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self._path)
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(QPen(color, 1.3))
        breaks = {str(name) for name in self.curve.get("breaks", [])}
        for segment in self.segments:
            if segment["well_name"] in breaks:
                continue
            offset = self._offset_for(segment["well_name"])
            center = QPointF((segment["left"].x() + segment["right"].x()) / 2, segment["left"].y() + offset)
            painter.drawEllipse(center, 4.2, 4.2)
        painter.setPen(QPen(color, 1))
        for segment in self.segments:
            if segment["well_name"] in breaks:
                continue
            offset = self._offset_for(segment["well_name"])
            painter.drawText(
                QRectF(segment["left"].x() + 6, segment["left"].y() + offset - 19, 190, 18),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                str(self.curve.get("name") or self.curve.get("label") or "Корреляция"),
            )

    def _well_at(self, point: QPointF) -> str | None:
        breaks = {str(name) for name in self.curve.get("breaks", [])}
        for segment in self.segments:
            if segment["well_name"] in breaks:
                continue
            offset = self._offset_for(segment["well_name"])
            left_x, right_x = segment["left"].x(), segment["right"].x()
            y = segment["left"].y() + offset
            if left_x - 10 <= point.x() <= right_x + 10 and abs(point.y() - y) <= 14:
                return str(segment["well_name"])
        return None

    def hoverMoveEvent(self, event) -> None:
        self.setCursor(Qt.CursorShape.OpenHandCursor if self._well_at(event.pos()) else Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event) -> None:
        well_name = self._well_at(event.pos())
        if well_name is None:
            event.ignore()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.context_requested.emit(self.curve, well_name)
            event.accept()
            return
        if self.curve.get("locked_straight"):
            event.accept()
            return
        self._drag_well = well_name
        self._drag_start_y = event.pos().y()
        self._drag_original_offset = self._offset_for(well_name)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_well is None:
            event.ignore()
            return
        offsets = self.curve.setdefault("offsets", {})
        offsets[self._drag_well] = self._drag_original_offset + event.pos().y() - self._drag_start_y
        self._rebuild_path()
        self.changed.emit()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_well is not None:
            self._drag_well = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.changed.emit()
            event.accept()
            return
        event.ignore()


class StackColumnItem(QGraphicsObject):
    HEADER = 52
    GAP = 8
    # One metre uses the same vertical scale in every tablet.  Therefore
    # extending TD adds real space below the well instead of compressing core.
    DEPTH_PIXELS_PER_METRE = 1.45
    MIN_DEPTH_BODY_HEIGHT = 620.0
    DEPTH_WIDTH = 120
    LITHOLOGY_WIDTH = 110
    SATURATION_WIDTH = 126
    GIS_TRACK_WIDTH = 92
    _legend_tiles: dict[tuple[str, str], QPixmap] = {}
    facies_selected = Signal(object, object)
    facies_context_requested = Signal(object, object)
    depth_binding_requested = Signal(object, object)
    column_paint_requested = Signal(object, object, str)
    interpretation_interval_created = Signal(str, str, float, float)
    interpretation_interval_resized = Signal(str, int, float, float)
    interpretation_interval_context_requested = Signal(str, int)
    well_drag_finished = Signal(str)

    def __init__(
        self,
        records: list[PhotoRecord],
        gis_preview: dict | None = None,
        rigis_preview: dict | None = None,
        well_name: str | None = None,
        interpretation_intervals: list[dict] | None = None,
        description_only: bool = False,
    ):
        super().__init__()
        self.records = list(records)
        self.well_name = str(well_name or "").strip()
        self.interpretation_intervals = [dict(item) for item in (interpretation_intervals or [])]
        self.gis_preview = dict(gis_preview or {})
        self.rigis_preview = dict(rigis_preview or {})
        self.description_only = bool(description_only)
        self.depth_reference = next((str(item.get("depth_reference") or "") for item in (self.gis_preview, self.rigis_preview) if item.get("depth_reference")), "")
        self.depth_unit = next((str(item.get("depth_unit") or "") for item in (self.gis_preview, self.rigis_preview) if item.get("depth_unit")), "m")
        self.depth_datum = next((str(item.get("depth_datum") or "") for item in (self.gis_preview, self.rigis_preview) if item.get("depth_datum")), "")
        self.depth_range = next(
            (
                tuple(item["depth_range"])
                for item in (self.gis_preview, self.rigis_preview)
                if isinstance(item.get("depth_range"), (tuple, list)) and len(item["depth_range"]) == 2
            ),
            None,
        )
        self._placements: list[tuple[QPixmap, PhotoRecord, FaciesDetection, QRectF, FaciesOverlayItem]] = []
        self._hovered_column: tuple[PhotoRecord, FaciesDetection, str] | None = None
        self._column_draw_start: tuple[str, float] | None = None
        self._column_draw_current_y: float | None = None
        self._interval_resize: tuple[int, str] | None = None
        self._well_drag_start: QPointF | None = None
        self._well_drag_origin: QPointF | None = None
        layers = self._build_layers()
        max_crop_width = max((pixmap.width() for pixmap, _, _, _ in layers), default=140)
        self.column_width = max(160, max_crop_width + 16)
        # Petrel-like layout: depth | assembled core | GIS | saturation.
        # The lithology track is intentionally hidden for now: lithological
        # observations remain editable in a layer card, without duplicating a
        # coloured column on the tablet.
        # GIS therefore remains immediately to the right of the core column.
        self.depth_x = 8
        self.core_x = self.depth_x + self.DEPTH_WIDTH + 10
        self.gis_x = self.core_x + self.column_width + 10
        self.gis_tracks = [] if self.description_only else list(self.gis_preview.get("tracks", []) or [])
        self.gis_width = 0 if self.description_only else max(150, 12 + len(self.gis_tracks) * self.GIS_TRACK_WIDTH)
        self.lithology_x = -1  # compatibility for old saved interval records
        self.saturation_x = self.gis_x if self.description_only else self.gis_x + self.gis_width + 10
        self.rigis_x = self.saturation_x + self.SATURATION_WIDTH + 10
        self.rigis_tracks = [] if self.description_only else list(self.rigis_preview.get("tracks", []) or [])
        self.rigis_width = 0 if self.description_only else max(150, 12 + len(self.rigis_tracks) * self.GIS_TRACK_WIDTH)
        self.width = self.core_x + self.column_width + 10 if self.description_only else self.rigis_x + self.rigis_width + 10
        self._height = self.HEADER + 10
        full_well_layout = bool(self.depth_range) and bool(layers) and all(
            source_detection.depth_from is not None and source_detection.depth_to is not None
            for _, _, source_detection, _ in layers
        )
        if full_well_layout:
            top, base = (float(value) for value in self.depth_range)
            self._height = self.HEADER + 10 + self._body_height_for_depth_range()
            body_top, body_bottom = self.HEADER + 10, self._height - 8
            for pixmap, record, source_detection, display_detection in layers:
                depth_from, depth_to = float(source_detection.depth_from), float(source_detection.depth_to)
                y1 = body_top + (depth_from - top) / max(base - top, 1e-9) * (body_bottom - body_top)
                y2 = body_top + (depth_to - top) / max(base - top, 1e-9) * (body_bottom - body_top)
                interval_height = max(2.0, abs(y2 - y1))
                scale = min((self.column_width - 16) / max(1, pixmap.width()), interval_height / max(1, pixmap.height()))
                draw_width = max(2.0, pixmap.width() * scale)
                draw_height = max(2.0, pixmap.height() * scale)
                rect = QRectF(
                    self.core_x + (self.column_width - draw_width) / 2,
                    min(y1, y2) + (interval_height - draw_height) / 2,
                    draw_width,
                    draw_height,
                )
                overlay = FaciesOverlayItem(pixmap.size(), rect, [display_detection], self)
                overlay.facies_clicked.connect(lambda _, rec=record, det=source_detection: self.facies_selected.emit(rec, det))
                overlay.facies_context_requested.connect(lambda _, rec=record, det=source_detection: self.facies_context_requested.emit(rec, det))
                self._placements.append((pixmap, record, source_detection, rect, overlay))
        else:
            for pixmap, record, source_detection, display_detection in layers:
                x = self.core_x + (self.column_width - pixmap.width()) / 2
                rect = QRectF(x, self._height, pixmap.width(), pixmap.height())
                overlay = FaciesOverlayItem(pixmap.size(), rect, [display_detection], self)
                overlay.facies_clicked.connect(lambda _, rec=record, det=source_detection: self.facies_selected.emit(rec, det))
                overlay.facies_context_requested.connect(lambda _, rec=record, det=source_detection: self.facies_context_requested.emit(rec, det))
                self._placements.append((pixmap, record, source_detection, rect, overlay))
                self._height += pixmap.height() + self.GAP
        if not self._placements and self.depth_range is not None:
            self._height = max(self._height, self.HEADER + 10 + self._body_height_for_depth_range())
        self._height += 4
        self.setZValue(1)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        self.setAcceptHoverEvents(True)

    def _body_height_for_depth_range(self) -> float:
        if self.depth_range is None:
            return self.MIN_DEPTH_BODY_HEIGHT
        top, base = (float(value) for value in self.depth_range)
        span = abs(base - top)
        # LAS depth values in feet need the same on-screen scale as metres.
        metres = span * (0.3048 if self.depth_unit.casefold() == "ft" else 1.0)
        return max(self.MIN_DEPTH_BODY_HEIGHT, metres * self.DEPTH_PIXELS_PER_METRE)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self._height)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(QPen(QColor("#7169df"), 1.5))
        painter.drawRoundedRect(self.boundingRect(), 10, 10)
        painter.setPen(QColor("#353a50"))
        headers = [
            (QRectF(self.depth_x, 7, self.DEPTH_WIDTH, 24), "ПРИВЯЗКА, м"),
            (QRectF(self.core_x + 8, 7, self.column_width - 16, 24), f"КЕРН · {self.well_name}" if self.well_name else "СОБРАННЫЙ КЕРН"),
            (QRectF(self.gis_x, 7, self.gis_width, 24), "ГИС"),
            (QRectF(self.saturation_x, 7, self.SATURATION_WIDTH, 24), "НАСЫЩЕНИЕ"),
            (QRectF(self.rigis_x, 7, self.rigis_width, 24), "РИГИС"),
        ]
        if self.depth_reference:
            datum_suffix = f" · {self.depth_datum}" if self.depth_datum else ""
            headers[0] = (headers[0][0], f"{self.depth_reference}, {self.depth_unit}{datum_suffix}")
        if self.well_name:
            painter.setPen(QColor("#5149ca"))
            painter.drawText(QRectF(self.core_x + 6, 3, self.column_width - 12, 15), Qt.AlignmentFlag.AlignCenter, self.well_name)
            painter.setPen(QColor("#353a50"))
            headers[1] = (QRectF(self.core_x + 8, 20, self.column_width - 16, 24), "\u041a\u0415\u0420\u041d")
        if self.description_only:
            headers = [headers[0], headers[1]]
        for rect, title in headers:
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, title)
        painter.setPen(QPen(QColor("#e2e6ef"), 1))
        separators = (self.core_x - 5,) if self.description_only else (self.core_x - 5, self.gis_x - 5, self.saturation_x - 5, self.rigis_x - 5)
        for x in separators:
            painter.drawLine(x, self.HEADER, x, self._height - 6)
        for pixmap, record, source_detection, rect, overlay in self._placements:
            painter.drawPixmap(rect.toRect(), pixmap)
            if not self.description_only:
                self._draw_saturation_badge(painter, rect, source_detection)
        self._draw_interpretation_intervals(painter)
        if self._hovered_column is not None:
            hovered_record, hovered_detection, column = self._hovered_column
            x, width = self.saturation_x, self.SATURATION_WIDTH
            painter.setPen(QPen(QColor("#5149ca"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for _, record, detection, rect, _ in self._placements:
                if record is hovered_record and detection is hovered_detection:
                    painter.drawRect(QRectF(x + 4, rect.top(), width - 8, rect.height()))
                    break
        if not self._placements:
            self._draw_empty_well_depths(painter)
        if not self.description_only:
            self._draw_gis_tracks(painter)
            self._draw_rigis_tracks(painter)

    def _draw_depth_binding(self, painter: QPainter, rect: QRectF, detection: FaciesDetection) -> None:
        painter.setPen(QPen(QColor("#cfd5e2"), 1))
        painter.drawLine(self.depth_x, rect.top(), self.depth_x + self.DEPTH_WIDTH, rect.top())
        painter.drawLine(self.depth_x, rect.bottom(), self.depth_x + self.DEPTH_WIDTH, rect.bottom())
        painter.setPen(QColor("#394158"))
        text = self._format_depth_pair(detection.depth_from, detection.depth_to)
        painter.drawText(
            QRectF(self.depth_x + 5, rect.top() + 2, self.DEPTH_WIDTH - 10, max(18, rect.height() - 4)),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            text,
        )

    def _draw_empty_well_depths(self, painter: QPainter) -> None:
        if self.depth_range is None:
            return
        top, base = (float(value) for value in self.depth_range)
        painter.setPen(QPen(QColor("#cfd5e2"), 1))
        painter.drawLine(self.depth_x, self.HEADER + 10, self.depth_x + self.DEPTH_WIDTH, self.HEADER + 10)
        painter.drawLine(self.depth_x, self._height - 8, self.depth_x + self.DEPTH_WIDTH, self._height - 8)
        painter.setPen(QColor("#394158"))
        painter.drawText(QRectF(self.depth_x + 6, self.HEADER + 14, self.DEPTH_WIDTH - 12, 20), Qt.AlignmentFlag.AlignCenter, f"{top:g}")
        painter.drawText(QRectF(self.depth_x + 6, self._height - 32, self.DEPTH_WIDTH - 12, 20), Qt.AlignmentFlag.AlignCenter, f"{base:g}")

    def _draw_lithology_badge(self, painter: QPainter, rect: QRectF, detection: FaciesDetection) -> None:
        info = self._lithology_info(detection.label)
        color = QColor((info or {}).get("color") or facies_color(detection.label))
        custom_color = QColor(str(detection.attributes.get("Цвет литологии") or ""))
        if custom_color.isValid():
            color = custom_color
        badge = QRectF(self.lithology_x + 8, rect.top() + 1, self.LITHOLOGY_WIDTH - 16, max(2, rect.height() - 2))
        # The legend is rendered as reusable raster tiles, so the track uses
        # pictograms rather than a flat programmatic fill.
        tile = self._legend_tile(color, (info or {}).get("pattern", "solid"))
        painter.fillRect(badge, color)
        painter.drawPixmap(badge.toRect(), tile)
        painter.setPen(QPen(color.darker(135), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(badge)
        symbol = (info or {}).get("symbol") or detection.label[:5]
        if badge.height() >= 20:
            painter.setPen(QColor("#ffffff") if color.lightness() < 115 else QColor("#273044"))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, symbol)

    def _draw_saturation_badge(self, painter: QPainter, rect: QRectF, detection: FaciesDetection) -> None:
        value = str(detection.attributes.get("Флюидонасыщение") or "").strip()
        styles = {
            "Сильно нефтенасыщенный": ("#99512e", "НФ сильн."),
            "Слабо нефтенасыщенный": ("#e0b653", "НФ слаб."),
            "С признаками УВ": ("#be6c56", "признаки УВ"),
        }
        color_name, short_label = styles.get(value, ("#f6f7fa", "—"))
        custom_color = QColor(str(detection.attributes.get("Цвет насыщения") or ""))
        if custom_color.isValid():
            color_name = custom_color.name(QColor.NameFormat.HexRgb)
        badge = QRectF(self.saturation_x + 8, rect.top() + 1, self.SATURATION_WIDTH - 16, max(2, rect.height() - 2))
        color = QColor(color_name)
        painter.fillRect(badge, color)
        painter.setPen(QPen(color.darker(125) if value else QColor("#cfd5e2"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(badge)
        if badge.height() >= 20:
            painter.setPen(QColor("#ffffff") if color.lightness() < 125 else QColor("#485167"))
            painter.drawText(
                badge.adjusted(3, 2, -3, -2),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                short_label if badge.height() < 46 else (value or short_label),
            )

    def _draw_interpretation_intervals(self, painter: QPainter) -> None:
        anchors = self._depth_anchors()
        if len(anchors) < 2:
            return
        for index, interval in enumerate(self.interpretation_intervals):
            try:
                top, base = float(interval["depth_from"]), float(interval["depth_to"])
            except (KeyError, TypeError, ValueError):
                continue
            kind = str(interval.get("kind") or "")
            if kind != "saturation":
                continue
            if self.description_only:
                continue
            y1, y2 = self._depth_to_stack_y(top, anchors), self._depth_to_stack_y(base, anchors)
            x, width = self.saturation_x, self.SATURATION_WIDTH
            color = QColor(str(interval.get("color") or "#8f71d2"))
            color.setAlpha(225)
            badge = QRectF(x + 8, min(y1, y2), width - 16, max(3.0, abs(y2 - y1)))
            painter.fillRect(badge, color)
            painter.setPen(QPen(color.darker(140), 1.4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(badge)
            painter.setBrush(QColor("#ffffff"))
            painter.setPen(QPen(color.darker(150), 1))
            for y in (badge.top(), badge.bottom()):
                painter.drawRect(QRectF(badge.center().x() - 7, y - 2.5, 14, 5))
            if badge.height() >= 20:
                painter.setPen(QColor("#ffffff") if color.lightness() < 125 else QColor("#273044"))
                painter.drawText(badge.adjusted(3, 2, -3, -2), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, str(interval.get("name") or "Интервал"))
        if self._column_draw_start is not None and self._column_draw_current_y is not None:
            _, start_y = self._column_draw_start
            x, width = self.saturation_x, self.SATURATION_WIDTH
            preview = QRectF(x + 8, min(start_y, self._column_draw_current_y), width - 16, abs(self._column_draw_current_y - start_y))
            color = QColor("#5149ca")
            color.setAlpha(70)
            painter.setBrush(color)
            painter.setPen(QPen(QColor("#5149ca"), 1.5, Qt.PenStyle.DashLine))
            painter.drawRect(preview)
            depth_a = self._stack_y_to_depth(start_y, anchors)
            depth_b = self._stack_y_to_depth(self._column_draw_current_y, anchors)
            if depth_a is not None and depth_b is not None:
                painter.setPen(QColor("#37307d"))
                painter.drawText(preview.adjusted(2, 2, -2, -2), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, f"{min(depth_a, depth_b):g}–{max(depth_a, depth_b):g} м")

    @staticmethod
    def _lithology_info(label: str) -> dict | None:
        label = str(label or "").casefold()
        aliases = {
            "sandstone": "Песчаник",
            "siltstone": "Алевролит",
            "mudstone": "Аргиллит",
            "madstone": "Аргиллит",
            "shale": "Аргиллит",
            "coal": "Уголь",
            "gravel": "Гравелит",
        }
        name = aliases.get(label)
        if name is None:
            name = next((item["name"] for item in LITHOLOGY_LEGEND if item["name"].casefold() == label), None)
        return next((item for item in LITHOLOGY_LEGEND if item["name"] == name), None)

    @classmethod
    def _legend_tile(cls, color: QColor, pattern: str) -> QPixmap:
        key = (color.name(QColor.NameFormat.HexRgb), str(pattern))
        if key in cls._legend_tiles:
            return cls._legend_tiles[key]
        tile = QPixmap(48, 48)
        tile.fill(color)
        tile_painter = QPainter(tile)
        cls._draw_lithology_pattern(tile_painter, QRectF(0, 0, 48, 48), str(pattern), color)
        tile_painter.end()
        cls._legend_tiles[key] = tile
        return tile

    @staticmethod
    def _draw_lithology_pattern(painter: QPainter, rect: QRectF, pattern: str, color: QColor) -> None:
        ink = QColor("#ffffff") if color.lightness() < 115 else QColor("#323846")
        ink.setAlpha(125)
        painter.setPen(QPen(ink, 1))
        if pattern in {"horizontal", "thin_horizontal", "dark_lines", "coal_lines", "alternating"}:
            step = 5 if pattern == "thin_horizontal" else 8
            y = rect.top() + step
            while y < rect.bottom():
                painter.drawLine(rect.left() + 3, y, rect.right() - 3, y)
                y += step
        if pattern in {"dots", "pebbles"}:
            radius = 1.4 if pattern == "dots" else 2.6
            y = rect.top() + 7
            while y < rect.bottom():
                x = rect.left() + 10
                while x < rect.right() - 5:
                    painter.drawEllipse(QPointF(x, y), radius, radius)
                    x += 17
                y += 13
        if pattern in {"cross", "alternating"}:
            offset = -rect.height()
            while offset < rect.width() + rect.height():
                painter.drawLine(rect.left() + offset, rect.top(), rect.left() + offset + rect.height(), rect.bottom())
                painter.drawLine(rect.left() + offset + rect.height(), rect.top(), rect.left() + offset, rect.bottom())
                offset += 16

    def _draw_gis_tracks(self, painter: QPainter) -> None:
        self._draw_log_tracks(painter, self.gis_x, self.gis_width, self.gis_tracks, "LAS не загружен")

    def _draw_rigis_tracks(self, painter: QPainter) -> None:
        self._draw_log_tracks(painter, self.rigis_x, self.rigis_width, self.rigis_tracks, "РИГИС не загружен")

    def _draw_log_tracks(self, painter: QPainter, x_origin: float, track_width: float, tracks: list[dict], missing_label: str) -> None:
        if not tracks:
            painter.setPen(QColor("#8c94a8"))
            painter.drawText(QRectF(x_origin + 8, self.HEADER + 8, track_width - 16, 36), Qt.AlignmentFlag.AlignCenter, missing_label)
            return
        depth_anchors = self._depth_anchors()
        if len(depth_anchors) < 2 or depth_anchors[0][0] == depth_anchors[-1][0]:
            painter.setPen(QColor("#8c94a8"))
            painter.drawText(QRectF(x_origin + 8, self.HEADER + 8, track_width - 16, 48), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, "Задайте глубины слоёв")
            return
        minimum, maximum = depth_anchors[0][0], depth_anchors[-1][0]
        plot_top, plot_bottom = depth_anchors[0][1], depth_anchors[-1][1]
        colors = [QColor("#d85f5f"), QColor("#3279c5"), QColor("#499b73"), QColor("#a768c7")]
        for index, track in enumerate(tracks):
            track_rect = QRectF(x_origin + 7 + index * self.GIS_TRACK_WIDTH, plot_top, self.GIS_TRACK_WIDTH - 10, plot_bottom - plot_top)
            painter.setPen(QPen(QColor("#d8dde8"), 1))
            painter.drawRect(track_rect)
            painter.setPen(QColor("#4b5368"))
            painter.drawText(QRectF(track_rect.left(), 40, track_rect.width(), 15), Qt.AlignmentFlag.AlignCenter, str(track.get("curve", "ГИС")))
            path = QPainterPath()
            has_point = False
            for point in track.get("points", []) or []:
                try:
                    depth, normalized = float(point["depth"]), float(point["normalized"])
                except (KeyError, TypeError, ValueError):
                    continue
                if depth < minimum or depth > maximum:
                    continue
                x = track_rect.left() + max(0.0, min(1.0, normalized)) * track_rect.width()
                y = self._depth_to_stack_y(depth, depth_anchors)
                if not has_point:
                    path.moveTo(x, y)
                    has_point = True
                else:
                    path.lineTo(x, y)
            painter.setClipRect(track_rect)
            painter.setPen(QPen(colors[index % len(colors)], 1.4))
            painter.drawPath(path)
            painter.setClipping(False)

    def _depth_anchors(self) -> list[tuple[float, float]]:
        """Map measured depths to the actual layer boundaries on the assembled core."""
        grouped: dict[float, list[float]] = {}
        if self.depth_range is not None:
            top, base = (float(value) for value in self.depth_range)
            grouped.setdefault(top, []).append(self.HEADER + 10)
            grouped.setdefault(base, []).append(self._height - 8)
        for _, _, detection, rect, _ in self._placements:
            if detection.depth_from is not None:
                grouped.setdefault(float(detection.depth_from), []).append(rect.top())
            if detection.depth_to is not None:
                grouped.setdefault(float(detection.depth_to), []).append(rect.bottom())
        anchors = [(depth, sum(ys) / len(ys)) for depth, ys in sorted(grouped.items())]
        if anchors or self.depth_range is None:
            return anchors
        top, base = (float(value) for value in self.depth_range)
        return [(top, self.HEADER + 10), (base, self._height - 8)] if top < base else []

    def correlation_markers(self) -> list[tuple[str, QPointF]]:
        """One marker per layer, used by the correlation connector on the canvas."""
        return [
            (detection.label, QPointF(self.core_x + self.column_width, rect.center().y()))
            for _, _, detection, rect, _ in self._placements
            if detection.label and detection.label != "Новый контур"
        ]

    def depth_ratio(self, depth: float) -> float:
        """Return a stable 0..1 position for a depth in this well tablet."""
        anchors = self._depth_anchors()
        if len(anchors) < 2:
            return 0.5
        y = self._depth_to_stack_y(float(depth), anchors)
        top, base = anchors[0][1], anchors[-1][1]
        return max(0.0, min(1.0, (y - top) / max(base - top, 1e-9)))

    @staticmethod
    def _depth_to_stack_y(depth: float, anchors: list[tuple[float, float]]) -> float:
        for (start_depth, start_y), (end_depth, end_y) in zip(anchors, anchors[1:]):
            if start_depth <= depth <= end_depth:
                ratio = (depth - start_depth) / max(end_depth - start_depth, 1e-9)
                return start_y + ratio * (end_y - start_y)
        return anchors[0][1] if depth <= anchors[0][0] else anchors[-1][1]

    @staticmethod
    def _stack_y_to_depth(y: float, anchors: list[tuple[float, float]]) -> float | None:
        for (start_depth, start_y), (end_depth, end_y) in zip(anchors, anchors[1:]):
            if min(start_y, end_y) <= y <= max(start_y, end_y):
                ratio = (y - start_y) / max(end_y - start_y, 1e-9)
                return start_depth + ratio * (end_depth - start_depth)
        return None

    def mouseDoubleClickEvent(self, event) -> None:
        super().mouseDoubleClickEvent(event)

    def hoverMoveEvent(self, event) -> None:
        self._hovered_column = self._column_at(event.pos())
        if event.pos().y() <= self.HEADER:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self._interval_handle_at(event.pos()) is not None:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif self._track_kind_at(event.pos()):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def hoverLeaveEvent(self, event) -> None:
        self._hovered_column = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.pos().y() <= self.HEADER:
            self._well_drag_start = event.scenePos()
            self._well_drag_origin = QPointF(self.pos())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        interval_index = self._interval_at(event.pos())
        if interval_index is not None and event.button() == Qt.MouseButton.RightButton:
            self.interpretation_interval_context_requested.emit(self.well_name, interval_index)
            event.accept()
            return
        handle = self._interval_handle_at(event.pos())
        if handle is not None and event.button() == Qt.MouseButton.LeftButton:
            self._interval_resize = handle
            self.setCursor(Qt.CursorShape.SizeVerCursor)
            event.accept()
            return
        kind = self._track_kind_at(event.pos())
        if kind is not None and event.button() == Qt.MouseButton.LeftButton:
            self._column_draw_start = (kind, event.pos().y())
            self._column_draw_current_y = event.pos().y()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._well_drag_start is not None and self._well_drag_origin is not None:
            delta = event.scenePos() - self._well_drag_start
            self.setPos(self._well_drag_origin.x() + delta.x(), self._well_drag_origin.y())
            event.accept()
            return
        if self._interval_resize is not None:
            self.update()
            event.accept()
            return
        if self._column_draw_start is not None:
            self._column_draw_current_y = event.pos().y()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._well_drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            self._well_drag_start = None
            self._well_drag_origin = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.well_drag_finished.emit(self.well_name)
            event.accept()
            return
        if self._interval_resize is not None and event.button() == Qt.MouseButton.LeftButton:
            index, edge = self._interval_resize
            self._interval_resize = None
            depth = self._stack_y_to_depth(event.pos().y(), self._depth_anchors())
            if depth is not None and 0 <= index < len(self.interpretation_intervals):
                interval = self.interpretation_intervals[index]
                try:
                    top, base = float(interval["depth_from"]), float(interval["depth_to"])
                    top, base = (depth, base) if edge == "top" else (top, depth)
                    if abs(base - top) > 1e-8:
                        self.interpretation_interval_resized.emit(self.well_name, index, min(top, base), max(top, base))
                except (KeyError, TypeError, ValueError):
                    pass
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
            event.accept()
            return
        if self._column_draw_start is None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        kind, start_y = self._column_draw_start
        self._column_draw_start = None
        self._column_draw_current_y = None
        end_y = event.pos().y()
        if abs(end_y - start_y) < 5:
            column = self._column_at(event.pos())
            if column is not None:
                record, detection, column_kind = column
                self.column_paint_requested.emit(record, detection, column_kind)
        else:
            anchors = self._depth_anchors()
            depth_from = self._stack_y_to_depth(start_y, anchors)
            depth_to = self._stack_y_to_depth(end_y, anchors)
            if depth_from is not None and depth_to is not None and abs(depth_to - depth_from) > 1e-8:
                self.interpretation_interval_created.emit(self.well_name, kind, min(depth_from, depth_to), max(depth_from, depth_to))
        self.update()
        event.accept()

    def _column_at(self, point: QPointF) -> tuple[PhotoRecord, FaciesDetection, str] | None:
        for _, record, detection, rect, _ in self._placements:
            if not rect.top() <= point.y() <= rect.bottom():
                continue
            if not self.description_only and QRectF(self.saturation_x, rect.top(), self.SATURATION_WIDTH, rect.height()).contains(point):
                return record, detection, "saturation"
        return None

    def _track_kind_at(self, point: QPointF) -> str | None:
        if not self.HEADER <= point.y() <= self._height - 6:
            return None
        if not self.description_only and self.saturation_x <= point.x() <= self.saturation_x + self.SATURATION_WIDTH:
            return "saturation"
        return None

    def _interval_handle_at(self, point: QPointF) -> tuple[int, str] | None:
        anchors = self._depth_anchors()
        if len(anchors) < 2:
            return None
        for index, interval in enumerate(self.interpretation_intervals):
            try:
                kind = str(interval.get("kind"))
                top_y = self._depth_to_stack_y(float(interval["depth_from"]), anchors)
                base_y = self._depth_to_stack_y(float(interval["depth_to"]), anchors)
            except (KeyError, TypeError, ValueError):
                continue
            if self.description_only or kind != "saturation":
                continue
            x, width = self.saturation_x, self.SATURATION_WIDTH
            if not (x <= point.x() <= x + width):
                continue
            if abs(point.y() - top_y) <= 7:
                return index, "top"
            if abs(point.y() - base_y) <= 7:
                return index, "base"
        return None

    def _interval_at(self, point: QPointF) -> int | None:
        anchors = self._depth_anchors()
        if len(anchors) < 2:
            return None
        for index, interval in enumerate(self.interpretation_intervals):
            try:
                kind = str(interval.get("kind"))
                y1 = self._depth_to_stack_y(float(interval["depth_from"]), anchors)
                y2 = self._depth_to_stack_y(float(interval["depth_to"]), anchors)
            except (KeyError, TypeError, ValueError):
                continue
            if self.description_only or kind != "saturation":
                continue
            x, width = self.saturation_x, self.SATURATION_WIDTH
            if x <= point.x() <= x + width and min(y1, y2) <= point.y() <= max(y1, y2):
                return index
        return None

    @staticmethod
    def _format_depth_pair(depth_from: float | None, depth_to: float | None) -> str:
        if depth_from is None and depth_to is None:
            return "—"
        top = "?" if depth_from is None else f"{depth_from:g}"
        base = "?" if depth_to is None else f"{depth_to:g}"
        return f"{top}\n—\n{base}"

    def _build_layers(self) -> list[tuple[QPixmap, PhotoRecord, FaciesDetection, FaciesDetection]]:
        layers: list[tuple[QPixmap, PhotoRecord, FaciesDetection, FaciesDetection]] = []
        for record in self.records:
            # A core photo can contain several physical columns.  Preserve the
            # DeepCore reading order: top-to-bottom inside the left column,
            # then move to the next column on the right.
            detections = self._sort_by_reading_order(record.detections)
            for detection in detections:
                crop = self._crop_detection(record, detection)
                if crop is not None:
                    pixmap, translated = crop
                    layers.append((pixmap, record, detection, translated))
        return layers

    @staticmethod
    def _sort_by_reading_order(detections: list[FaciesDetection]) -> list[FaciesDetection]:
        indexed = list(enumerate(detections))
        if len(indexed) <= 1:
            return list(detections)

        measured = []
        for original_index, detection in indexed:
            bounds = QPolygonF(detection.polygon).boundingRect()
            if bounds.isEmpty():
                continue
            measured.append(
                {
                    "detection": detection,
                    "original_index": original_index,
                    "left": bounds.left(),
                    "top": bounds.top(),
                    "center_x": bounds.center().x(),
                    "width": max(1.0, bounds.width()),
                }
            )
        if not measured:
            return list(detections)

        widths = sorted(row["width"] for row in measured)
        column_tolerance = max(12.0, widths[len(widths) // 2] * 0.65)
        columns: list[dict] = []
        for row in sorted(measured, key=lambda item: (item["center_x"], item["top"], item["original_index"])):
            target = min(
                (column for column in columns if abs(row["center_x"] - column["center_x"]) <= column_tolerance),
                key=lambda column: abs(row["center_x"] - column["center_x"]),
                default=None,
            )
            if target is None:
                target = {"center_x": row["center_x"], "items": []}
                columns.append(target)
            target["items"].append(row)
            target["center_x"] = sum(item["center_x"] for item in target["items"]) / len(target["items"])

        ordered = []
        for column in sorted(columns, key=lambda item: item["center_x"]):
            ordered.extend(
                row["detection"]
                for row in sorted(column["items"], key=lambda item: (item["top"], item["left"], item["original_index"]))
            )
        placed_ids = {id(detection) for detection in ordered}
        ordered.extend(detection for detection in detections if id(detection) not in placed_ids)
        return ordered

    @staticmethod
    def _crop_detection(record: PhotoRecord, detection: FaciesDetection) -> tuple[QPixmap, FaciesDetection] | None:
        bounds = QPolygonF(detection.polygon).boundingRect().toAlignedRect()
        bounds = bounds.intersected(QRect(0, 0, record.pixmap.width(), record.pixmap.height()))
        if bounds.width() <= 0 or bounds.height() <= 0:
            return None
        # Keep only the exact edited polygon, not a rectangular bounding-box crop.
        pixmap = QPixmap(bounds.size())
        pixmap.fill(Qt.GlobalColor.transparent)
        translated_polygon = QPolygonF(
            [QPointF(point.x() - bounds.left(), point.y() - bounds.top()) for point in detection.polygon]
        )
        painter = QPainter(pixmap)
        clip_path = QPainterPath()
        clip_path.setFillRule(Qt.FillRule.WindingFill)
        clip_path.addPolygon(translated_polygon)
        clip_path.closeSubpath()
        painter.setClipPath(clip_path)
        painter.drawPixmap(QRect(0, 0, bounds.width(), bounds.height()), record.pixmap, bounds)
        painter.end()
        if pixmap.isNull():
            return None
        translated = FaciesDetection(
            label=detection.label,
            confidence=detection.confidence,
            polygon=list(translated_polygon),
            attributes=dict(detection.attributes),
            alternatives=dict(detection.alternatives),
        )
        return pixmap, translated


def facies_color(label: str) -> QColor:
    palette = ["#7169df", "#e47a52", "#43a78f", "#d5a73d", "#be5a9c", "#528ad9"]
    index = int(md5(label.encode("utf-8")).hexdigest(), 16) % len(palette)
    return QColor(palette[index])
