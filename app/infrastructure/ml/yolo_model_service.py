from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, Qt, Signal, Slot
from PySide6.QtGui import QPainterPath, QPolygonF

from app.domain.models import FaciesDetection


class YoloModelService:
    """DeepCore segmentation adapter based on the original YOLO service."""

    # Ultralytics defaults to 0.25, which is useful for exploration but makes
    # weak guesses look like accepted facies on a working core description.
    CONFIDENCE_THRESHOLD = 0.50

    def __init__(self, model_path: Path):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.device, self.device_label = self._best_device()
        self._facies_catalog = self._load_facies_catalog(model_path)

    def predict(self, image_path: str, target_size: tuple[int, int] | None = None) -> list[FaciesDetection]:
        detections: list[FaciesDetection] = []
        # One image at a time avoids loading a large incoming folder into RAM.
        # FP16 is enabled only for CUDA and substantially reduces VRAM use.
        results = self.model(
            image_path,
            verbose=False,
            device=self.device,
            half=self.device != "cpu",
            imgsz=640,
            conf=self.CONFIDENCE_THRESHOLD,
        )

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
                confidence = float(box.conf.item())
                polygon = self._mask_polygon(mask_polygons, index)
                if not polygon:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    polygon = [QPointF(x1, y1), QPointF(x2, y1), QPointF(x2, y2), QPointF(x1, y2)]
                if x_scale != 1.0 or y_scale != 1.0:
                    polygon = [QPointF(point.x() * x_scale, point.y() * y_scale) for point in polygon]

                detections.append(
                    FaciesDetection(
                        label=str(names[class_id]),
                        confidence=confidence,
                        polygon=polygon,
                        attributes=dict(self._facies_catalog.get(str(names[class_id]), {})),
                    )
                )
        return self._remove_overlapping_detections(detections)

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

    def __init__(self, model_path: Path, image_paths: list[tuple[str, int, int]]):
        super().__init__()
        self.model_path = model_path
        self.image_paths = list(image_paths)

    @Slot()
    def run(self) -> None:
        try:
            service = YoloModelService(self.model_path)
            total = len(self.image_paths)
            self.progress_changed.emit(0, total, service.device_label)
            for index, (image_path, width, height) in enumerate(self.image_paths, start=1):
                self.progress_changed.emit(index, total, Path(image_path).name)
                self.image_ready.emit(image_path, service.predict(image_path, (width, height)))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()
