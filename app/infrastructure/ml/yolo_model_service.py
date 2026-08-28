from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, Qt, Signal, Slot
from PySide6.QtGui import QPainterPath, QPolygonF

from app.domain.models import FaciesDetection
from app.infrastructure.ml.rule_based_facies import RuleBasedFaciesDetector


class YoloModelService:
    """Kern Analyzer segmentation adapter based on the original YOLO service."""

    # Ultralytics defaults to 0.25, which is useful for exploration but makes
    # weak guesses look like accepted facies on a working core description.
    DEFAULT_CONFIDENCE_THRESHOLD = 0.50

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float | None = None,
        image_size: int = 640,
        max_detections: int = 1000,
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.device, self.device_label = self._best_device()
        self._facies_catalog = self._load_facies_catalog(model_path)
        self.confidence_threshold = self._normalize_confidence(confidence_threshold)
        self.image_size = self._normalize_image_size(image_size)
        self.max_detections = max(50, min(3000, int(max_detections)))

    def predict(
        self,
        image_path: str,
        target_size: tuple[int, int] | None = None,
        core_columns_override: list[dict[str, float]] | None = None,
    ) -> list[FaciesDetection]:
        detections: list[FaciesDetection] = []
        # One image at a time avoids loading a large incoming folder into RAM.
        # FP16 is enabled only for CUDA and substantially reduces VRAM use.
        results = self.model(
            image_path,
            verbose=False,
            device=self.device,
            half=self.device != "cpu",
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            # A core photo can contain hundreds of small packages. Keep them
            # available for review instead of silently stopping at YOLO's
            # default per-image detection cap.
            max_det=self.max_detections,
        )

        core_columns = self._core_columns(image_path)
        for result in results:
            source_height, source_width = getattr(result, "orig_shape", (1, 1))
            x_scale = 1.0 if target_size is None else target_size[0] / max(1, source_width)
            y_scale = 1.0 if target_size is None else target_size[1] / max(1, source_height)
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue

            mask_polygons = []
            if getattr(result, "masks", None) is not None and getattr(result.masks, "xy", None) is not None:
                mask_polygons = list(result.masks.xy)

            for index, box in enumerate(boxes):
                class_id = int(box.cls.item())
                label = str(names[class_id])
                # ``shlak`` is a technical/background class, not a facies.
                # It must neither appear as a mask nor enter a report/training
                # candidate, irrespective of letter case or Russian spelling.
                if self._is_excluded_label(label):
                    continue
                confidence = float(box.conf.item())
                polygon = self._mask_polygon(mask_polygons, index)
                if not polygon:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    polygon = [QPointF(x1, y1), QPointF(x2, y1), QPointF(x2, y2), QPointF(x1, y2)]
                if x_scale != 1.0 or y_scale != 1.0:
                    polygon = [QPointF(point.x() * x_scale, point.y() * y_scale) for point in polygon]
                if core_columns_override:
                    polygon = self._clip_polygon_to_rectangles(polygon, core_columns_override)
                else:
                    polygon = self._clip_polygon_to_core_columns(polygon, core_columns, x_scale, y_scale)
                if len(polygon) < 3:
                    continue

                detections.append(
                    FaciesDetection(
                        label=label,
                        confidence=confidence,
                        polygon=polygon,
                        attributes=dict(self._facies_catalog.get(str(names[class_id]), {})),
                    )
                )
        return self._remove_overlapping_detections(detections)

    @staticmethod
    def _is_excluded_label(label: str) -> bool:
        normalized = str(label or "").strip().casefold().replace("ё", "е")
        return normalized in {"shlak", "slag", "шлак"}

    @staticmethod
    def _core_columns(image_path: str) -> list[tuple[int, int, int, int]]:
        """Find physical core columns; everything around them is ignored."""
        try:
            import cv2
            import numpy as np

            image = cv2.imdecode(np.frombuffer(Path(image_path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return []
            return RuleBasedFaciesDetector._find_core_columns(image)
        except (OSError, ImportError):
            # Never discard a valid prediction merely because a source image
            # cannot be re-read at this point.
            return []

    @staticmethod
    def _clip_polygon_to_core_columns(
        polygon: list[QPointF],
        columns: list[tuple[int, int, int, int]],
        x_scale: float,
        y_scale: float,
    ) -> list[QPointF]:
        if not columns or not polygon:
            return polygon
        bounds = QPolygonF(polygon).boundingRect()
        # The detection must substantially belong to one core column.  A small
        # tolerance preserves contacts that touch the tray edge, but labels,
        # rulers and background outside the columns are removed.
        target = None
        largest_overlap = 0.0
        for left, top, right, bottom in columns:
            column = QPolygonF([
                QPointF(left * x_scale, top * y_scale), QPointF(right * x_scale, top * y_scale),
                QPointF(right * x_scale, bottom * y_scale), QPointF(left * x_scale, bottom * y_scale),
            ]).boundingRect()
            overlap = bounds.intersected(column)
            area = overlap.width() * overlap.height()
            if area > largest_overlap:
                largest_overlap, target = area, column
        if target is None or largest_overlap < bounds.width() * bounds.height() * 0.55:
            return []
        return YoloModelService._clip_polygon_to_rect(polygon, target)

    @staticmethod
    def _clip_polygon_to_rectangles(points: list[QPointF], columns: list[dict[str, float]]) -> list[QPointF]:
        """Use explicitly corrected display-space column bounds when present."""
        if not points:
            return points
        bounds = QPolygonF(points).boundingRect()
        target = None
        largest_overlap = 0.0
        for values in columns:
            try:
                left, top = float(values["left"]), float(values["top"])
                right, bottom = float(values["right"]), float(values["bottom"])
            except (KeyError, TypeError, ValueError):
                continue
            rect = QPolygonF([QPointF(left, top), QPointF(right, top), QPointF(right, bottom), QPointF(left, bottom)]).boundingRect()
            overlap = bounds.intersected(rect)
            area = overlap.width() * overlap.height()
            if area > largest_overlap:
                largest_overlap, target = area, rect
        if target is None or largest_overlap < bounds.width() * bounds.height() * 0.55:
            return []
        return YoloModelService._clip_polygon_to_rect(points, target)

    @staticmethod
    def _clip_polygon_to_rect(points: list[QPointF], rect) -> list[QPointF]:
        """Clip a mask to the physical core column (Sutherland–Hodgman)."""
        clipped = list(points)
        edges = (
            (lambda point: point.x() >= rect.left(), lambda a, b: QPointF(rect.left(), a.y() + (b.y() - a.y()) * (rect.left() - a.x()) / (b.x() - a.x()))),
            (lambda point: point.x() <= rect.right(), lambda a, b: QPointF(rect.right(), a.y() + (b.y() - a.y()) * (rect.right() - a.x()) / (b.x() - a.x()))),
            (lambda point: point.y() >= rect.top(), lambda a, b: QPointF(a.x() + (b.x() - a.x()) * (rect.top() - a.y()) / (b.y() - a.y()), rect.top())),
            (lambda point: point.y() <= rect.bottom(), lambda a, b: QPointF(a.x() + (b.x() - a.x()) * (rect.bottom() - a.y()) / (b.y() - a.y()), rect.bottom())),
        )
        for inside, intersection in edges:
            if not clipped:
                break
            output: list[QPointF] = []
            previous = clipped[-1]
            previous_inside = inside(previous)
            for current in clipped:
                current_inside = inside(current)
                if current_inside != previous_inside:
                    # Parallel edges cannot cross this clipping edge.
                    delta = (current.x() - previous.x()) if abs(current.x() - previous.x()) > 1e-9 else (current.y() - previous.y())
                    if abs(delta) > 1e-9:
                        output.append(intersection(previous, current))
                if current_inside:
                    output.append(current)
                previous, previous_inside = current, current_inside
            clipped = output
        return clipped

    @classmethod
    def _normalize_confidence(cls, value: float | None) -> float:
        """Keep a sensible user-selected threshold for YOLO inference."""
        try:
            threshold = float(cls.DEFAULT_CONFIDENCE_THRESHOLD if value is None else value)
        except (TypeError, ValueError):
            threshold = cls.DEFAULT_CONFIDENCE_THRESHOLD
        return max(0.01, min(0.99, threshold))

    @staticmethod
    def _normalize_image_size(value: int) -> int:
        try:
            size = int(value)
        except (TypeError, ValueError):
            size = 640
        return max(320, min(1536, size))

    @staticmethod
    def _best_device() -> tuple[int | str, str]:
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                return 0, f"GPU: {name}"
        except (ImportError, RuntimeError):
            pass
        return "cpu", "CPU (CUDA не найдена)"

    @staticmethod
    def _load_facies_catalog(model_path: Path) -> dict[str, dict[str, str]]:
        """Read the Excel facies dictionary bundled with a fine-tuned model."""
        catalog_paths = [model_path.parent / "facies_catalog.json"]
        # Compatibility with a training run that was interrupted immediately
        # after weights were written, before its catalog could be copied.
        # Normal completed runs always use the first path above.
        try:
            run_stamp = model_path.parents[2].name
            training_folder = model_path.parents[4]
            catalog_paths.append(training_folder / "datasets" / run_stamp / "facies_catalog.json")
        except IndexError:
            pass
        for catalog_path in catalog_paths:
            try:
                payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            return {
                str(label): {str(key): str(value) for key, value in values.items() if value is not None}
                for label, values in payload.items()
                if isinstance(values, dict)
            }
        return {}

    @staticmethod
    def _mask_polygon(mask_polygons, index: int) -> list[QPointF]:
        if index >= len(mask_polygons):
            return []
        points = [QPointF(float(point[0]), float(point[1])) for point in mask_polygons[index] if len(point) >= 2]
        if len(points) <= 500:
            return points
        step = max(1, round(len(points) / 500))
        return points[::step][:500]

    @staticmethod
    def _remove_overlapping_detections(detections: list[FaciesDetection]) -> list[FaciesDetection]:
        """Suppress a lower-confidence mask when it substantially duplicates another one."""
        accepted: list[FaciesDetection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            candidate = YoloModelService._polygon_path(detection.polygon)
            candidate_area = YoloModelService._path_area(candidate)
            if candidate_area <= 0:
                continue
            duplicates_existing = False
            for existing in accepted:
                overlap = candidate.intersected(YoloModelService._polygon_path(existing.polygon))
                overlap_area = YoloModelService._path_area(overlap)
                # Tiny shared edges are fine; a real shared area indicates that
                # the model predicted two masks for the same fragment.
                if overlap_area >= min(candidate_area, YoloModelService._path_area(YoloModelService._polygon_path(existing.polygon))) * 0.12:
                    # YOLO exposes the confidence of the selected class only.
                    # A competing overlapping mask of another class is still
                    # useful to an interpreter, so retain it as an alternative
                    # even though only one visual mask is rendered.
                    if detection.label != existing.label:
                        existing.alternatives[detection.label] = max(
                            existing.alternatives.get(detection.label, 0.0),
                            detection.confidence,
                        )
                    duplicates_existing = True
                    break
            if not duplicates_existing:
                accepted.append(detection)
        return accepted

    @staticmethod
    def _polygon_path(points: list[QPointF]) -> QPainterPath:
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.WindingFill)
        path.addPolygon(QPolygonF(points))
        path.closeSubpath()
        return path.simplified()

    @staticmethod
    def _path_area(path: QPainterPath) -> float:
        total = 0.0
        for polygon in path.toFillPolygons():
            points = list(polygon)
            if len(points) < 3:
                continue
            total += abs(
                sum(
                    point.x() * points[(index + 1) % len(points)].y()
                    - points[(index + 1) % len(points)].x() * point.y()
                    for index, point in enumerate(points)
                )
            ) / 2.0
        return total


class SegmentationWorker(QObject):
    progress_changed = Signal(int, int, str)
    image_ready = Signal(str, object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        model_path: Path,
        image_paths: list[tuple[str, int, int, list[dict[str, float]] | None]],
        confidence_threshold: float | None = None,
        image_size: int = 640,
        max_detections: int = 1000,
    ):
        super().__init__()
        self.model_path = model_path
        self.image_paths = list(image_paths)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.max_detections = max_detections

    @Slot()
    def run(self) -> None:
        try:
            service = YoloModelService(self.model_path, self.confidence_threshold, self.image_size, self.max_detections)
            total = len(self.image_paths)
            self.progress_changed.emit(0, total, f"{service.device_label} · {service.image_size}px · порог {service.confidence_threshold:.0%}")
            for index, (image_path, width, height, core_columns) in enumerate(self.image_paths, start=1):
                self.progress_changed.emit(index, total, Path(image_path).name)
                self.image_ready.emit(image_path, service.predict(image_path, (width, height), core_columns))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()
