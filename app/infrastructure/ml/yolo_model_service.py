from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, Qt, Signal, Slot
from PySide6.QtGui import QPainterPath, QPolygonF

from app.domain.facies_catalog import facies_identity, resolve_facies_class
from app.domain.models import FaciesDetection
from app.infrastructure.facies_postprocess import UNRECOGNIZED_FACIES, complete_core_column_coverage
from app.infrastructure.ml.core_column_service import CoreColumnRecognizer, normalize_columns
from app.infrastructure.ml.rule_based_facies import RuleBasedFaciesDetector, TextureInterval


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
        self._description_generator = None
        self._description_facies: set[str] = set()
        checkpoint = getattr(self.model, "ckpt", None)
        embedded = checkpoint.get("core_description_checkpoint") if isinstance(checkpoint, dict) else None
        standalone = Path(model_path).with_name("description_best.pt")
        if embedded is not None or standalone.is_file():
            from excel_photo_model_studio.description_model import DescriptionGenerator

            self._description_generator = DescriptionGenerator(embedded if embedded is not None else standalone)
            self._description_facies = set(self._description_generator.checkpoint.get("facies_names", []))
        self.device, self.device_label = self._best_device()
        self._facies_catalog = self._load_facies_catalog(model_path)
        self.fallback_facies_label = UNRECOGNIZED_FACIES
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

        core_columns = self._core_columns(image_path) if core_columns_override is None else []
        source_height = target_size[1] if target_size is not None else 1
        source_width = target_size[0] if target_size is not None else 1
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
                if core_columns_override is not None:
                    polygon = self._clip_polygon_to_rectangles(polygon, core_columns_override)
                else:
                    polygon = self._clip_polygon_to_core_columns(polygon, core_columns, x_scale, y_scale)
                if len(polygon) < 3:
                    continue

                attributes = self._facies_attributes(label)
                detections.append(
                    FaciesDetection(
                        label=attributes.get("Код фации", label),
                        confidence=confidence,
                        polygon=polygon,
                        attributes=attributes,
                    )
                )
        detections = self._remove_overlapping_detections(detections)
        coverage_columns = core_columns_override
        if coverage_columns is None:
            coverage_columns = [
                {
                    "left": left * (1.0 if target_size is None else target_size[0] / max(1, source_width)),
                    "top": top * (1.0 if target_size is None else target_size[1] / max(1, source_height)),
                    "right": right * (1.0 if target_size is None else target_size[0] / max(1, source_width)),
                    "bottom": bottom * (1.0 if target_size is None else target_size[1] / max(1, source_height)),
                }
                for left, top, right, bottom in core_columns
            ]
        # A full-photo prediction can miss contacts after a long narrow core
        # column is reduced to ``imgsz``.  Detect persistent visual packages in
        # every stage-1 column and classify those crops separately at full model
        # resolution.  The full-photo masks remain useful context and fallback.
        interval_detections = self._predict_structure_intervals(
            image_path,
            target_size,
            coverage_columns,
            detections,
        )
        detections.extend(interval_detections)
        completed = complete_core_column_coverage(detections, coverage_columns, self.fallback_facies_label)
        self._attach_generated_descriptions(image_path, target_size, coverage_columns, completed)
        return completed

    def _attach_generated_descriptions(
        self,
        image_path: str,
        target_size: tuple[int, int] | None,
        columns: list[dict[str, float]],
        detections: list[FaciesDetection],
    ) -> None:
        """Use the embedded text head only on final, classified physical-core bands."""
        generator = getattr(self, "_description_generator", None)
        if generator is None or not columns:
            return
        source = cv2.imdecode(np.frombuffer(Path(image_path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if source is None or not source.size:
            raise ValueError(f"Не удалось прочитать керн для краткого описания: {image_path}")
        source_height, source_width = source.shape[:2]
        display_width, display_height = target_size or (source_width, source_height)
        x_scale = source_width / max(1, display_width)
        y_scale = source_height / max(1, display_height)
        for detection in detections:
            if detection.label == UNRECOGNIZED_FACIES or self._is_excluded_label(detection.label):
                continue
            # The catalogue may canonicalize Tcr -> Tcr@19. The text head was
            # trained on the exact dataset class, so retain that original ID.
            facies = detection.attributes.get("__source_model_class", detection.attributes.get("Класс модели", detection.label))
            if self._description_facies and facies not in self._description_facies:
                detection.attributes["Статус описания"] = "нет обученного текстового класса для этой фации"
                continue
            polygon = self._clip_polygon_to_rectangles(detection.polygon, columns)
            if len(polygon) < 3:
                continue
            bounds = QPolygonF(polygon).boundingRect()
            left, right = max(0, int(np.floor(bounds.left() * x_scale))), min(source_width, int(np.ceil(bounds.right() * x_scale)))
            top, bottom = max(0, int(np.floor(bounds.top() * y_scale))), min(source_height, int(np.ceil(bounds.bottom() * y_scale)))
            crop = source[top:bottom, left:right]
            if not crop.size:
                continue
            description = generator.generate(crop, facies=facies).strip()
            if not description:
                detection.attributes["Статус описания"] = "модель выдала пустой текст, требуется проверка"
                continue
            detection.attributes["Краткое описание"] = description
            detection.attributes["Источник описания"] = "нейросеть best.pt: фото интервала и прогноз фации"
            detection.attributes["Статус описания"] = "сгенерированный черновик, требуется проверка геолога"

    def _predict_structure_intervals(
        self,
        image_path: str,
        target_size: tuple[int, int] | None,
        columns: list[dict[str, float]],
        context: list[FaciesDetection],
    ) -> list[FaciesDetection]:
        """Find and independently classify visual intervals inside core columns."""
        if not columns:
            return []
        try:
            encoded = np.frombuffer(Path(image_path).read_bytes(), dtype=np.uint8)
            source = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except OSError:
            return []
        if source is None or source.size == 0:
            return []
        source_height, source_width = source.shape[:2]
        target_width, target_height = target_size or (source_width, source_height)
        x_scale = target_width / max(1, source_width)
        y_scale = target_height / max(1, source_height)

        source_columns: list[tuple[int, int, int, int]] = []
        for values in columns:
            try:
                source_columns.append((
                    round(float(values["left"]) / x_scale),
                    round(float(values["top"]) / y_scale),
                    round(float(values["right"]) / x_scale),
                    round(float(values["bottom"]) / y_scale),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        intervals = RuleBasedFaciesDetector().split_columns(source, source_columns)
        if not intervals:
            return []

        # Limit the pathological case of an exceptionally noisy photograph and
        # process small batches to keep inference memory bounded.
        intervals = intervals[: min(160, self.max_detections)]
        crops: list[np.ndarray] = []
        usable: list[TextureInterval] = []
        for interval in intervals:
            crop = source[interval.top:interval.bottom, interval.left:interval.right]
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                continue
            crops.append(crop.copy())
            usable.append(interval)
        if not crops:
            return []

        crop_results = []
        try:
            for start in range(0, len(crops), 16):
                crop_results.extend(self.model(
                    crops[start:start + 16],
                    verbose=False,
                    device=self.device,
                    half=self.device != "cpu",
                    imgsz=self.image_size,
                    # Keep weak candidates visible for review, but never make
                    # an arbitrary class assignment when no evidence exists.
                    conf=0.01,
                    max_det=min(50, self.max_detections),
                ))
        except Exception:
            # Structural contacts are still useful with labels inherited from
            # the full-photo prediction, so a failed crop batch is non-fatal.
            crop_results = []

        polygons = [
            [QPointF(x * x_scale, y * y_scale) for x, y in interval.polygon]
            for interval in usable
        ]
        classifications: list[tuple[str, float, dict[str, float]] | None] = []
        for index, polygon in enumerate(polygons):
            result = crop_results[index] if index < len(crop_results) else None
            classified = self._interval_classification(result)
            if classified is None:
                classified = self._inherit_interval_class(polygon, context)
            classifications.append(classified)

        output: list[FaciesDetection] = []
        for index, (interval, polygon) in enumerate(zip(usable, polygons)):
            classified = classifications[index]
            if classified is None:
                label, confidence, alternatives = UNRECOGNIZED_FACIES, 0.0, {}
            else:
                label, confidence, alternatives = classified
            attributes = self._facies_attributes(label)
            attributes.update({
                "Источник распознавания": "интервальный анализ столбика",
                "Граница интервала": self._translated_texture_evidence(interval.evidence),
                # Internal stable group: gap filling may subdivide an interval,
                # but equal labels across a real structural contact must not be
                # merged back into one long block.
                "__structural_interval": f"{interval.column_index}:{index}",
            })
            output.append(FaciesDetection(
                label=attributes.get("Код фации", label),
                confidence=max(0.0, min(1.0, confidence)),
                polygon=polygon,
                attributes=attributes,
                alternatives=alternatives,
            ))
        return output

    def _interval_classification(self, result) -> tuple[str, float, dict[str, float]] | None:
        """Choose the class explaining the largest confident part of a crop."""
        if result is None or getattr(result, "boxes", None) is None:
            return None
        names = result.names
        height, width = getattr(result, "orig_shape", (1, 1))
        crop_area = max(1.0, float(height * width))
        candidates: list[tuple[float, float, str]] = []
        alternatives: dict[str, float] = {}
        for box in result.boxes:
            class_id = int(box.cls.item())
            label = str(names[class_id])
            if self._is_excluded_label(label):
                continue
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area_fraction = max(0.0, (x2 - x1) * (y2 - y1)) / crop_area
            score = confidence * (0.55 + 0.45 * min(1.0, area_fraction) ** 0.5)
            candidates.append((score, confidence, label))
            alternatives[label] = max(alternatives.get(label, 0.0), confidence)
        if not candidates:
            return None
        _, confidence, label = max(candidates, key=lambda item: (item[0], item[1]))
        alternatives.pop(label, None)
        return label, confidence, alternatives

    @staticmethod
    def _inherit_interval_class(
        polygon: list[QPointF],
        context: list[FaciesDetection],
    ) -> tuple[str, float, dict[str, float]] | None:
        target = QPolygonF(polygon).boundingRect()
        target_area = max(1.0, target.width() * target.height())
        ranked: list[tuple[float, FaciesDetection]] = []
        for detection in context:
            overlap = target.intersected(QPolygonF(detection.polygon).boundingRect())
            fraction = max(0.0, overlap.width() * overlap.height()) / target_area
            if fraction > 0:
                ranked.append((fraction * max(0.05, detection.confidence), detection))
        if not ranked:
            return None
        _, chosen = max(ranked, key=lambda item: item[0])
        # A displayed code can be ambiguous; retain the precise model class
        # (Dch@47 vs Dch@92) when inheriting an overlapping prediction.
        return chosen.attributes.get("Класс модели", chosen.label), chosen.confidence * 0.80, dict(chosen.alternatives)

    @staticmethod
    def _translated_texture_evidence(evidence: str) -> str:
        return {
            "high lamination density": "изменение слоистости",
            "pale comparatively uniform texture": "переход к светлой однородной структуре",
            "dark comparatively uniform texture": "переход к тёмной однородной структуре",
            "persistent change of visual texture": "устойчивое изменение текстуры",
        }.get(str(evidence), str(evidence))

    @staticmethod
    def _is_excluded_label(label: str) -> bool:
        normalized = str(label or "").strip().casefold().replace("ё", "е")
        return normalized in {"shlak", "slag", "шлак"}

    @staticmethod
    def _core_columns(image_path: str) -> list[tuple[int, int, int, int]]:
        """Run the independent stage-1 recognizer before facies inference."""
        try:
            columns = CoreColumnRecognizer().recognize_path(image_path)
            return [
                (round(item["left"]), round(item["top"]), round(item["right"]), round(item["bottom"]))
                for item in columns
            ]
        except (OSError, ImportError, ValueError):
            # Never discard a valid prediction merely because a source image
            # cannot be re-read at this point.
            return []

    def _facies_attributes(self, label: str) -> dict[str, str]:
        """Only attach reference context, never fabricate observed lithology.

        Historical sidecars held a majority lithology vector per class. Those
        parameters are class priors, not observations of this core interval.
        """
        trained = self._facies_catalog.get(label, {})
        # A canonical model label is authoritative over stale sidecar indices.
        resolved = resolve_facies_class(label) if "@" in label else resolve_facies_class(label, trained.get("Индекс фации"))
        values = dict(resolved["metadata"]) if resolved else {}
        values["Класс модели"] = resolved["model_label"] if resolved else label
        values["__source_model_class"] = label
        values["Статус фации"] = "прогноз, требуется проверка" if resolved else "не сопоставлена со справочником"
        values["Источник описания"] = "справочник фаций; гипотеза по прогнозу модели, не наблюдение"
        return values

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
                    if facies_identity(detection.label, detection.attributes) != facies_identity(existing.label, existing.attributes):
                        alternative_label = detection.attributes.get("Класс модели", detection.label)
                        existing.alternatives[alternative_label] = max(
                            existing.alternatives.get(alternative_label, 0.0),
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
    # path, gap-free facies bands, stage-1 core-column rectangles
    image_ready = Signal(str, object, object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        model_path: Path,
        image_paths: list[tuple[str, int, int, list[dict[str, float]] | None, bool]],
        confidence_threshold: float | None = None,
        image_size: int = 640,
        max_detections: int = 1000,
        column_model_path: Path | None = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.image_paths = list(image_paths)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.max_detections = max_detections
        self.column_model_path = column_model_path

    @Slot()
    def run(self) -> None:
        try:
            service = YoloModelService(self.model_path, self.confidence_threshold, self.image_size, self.max_detections)
            column_recognizer = CoreColumnRecognizer(self.column_model_path, image_size=self.image_size)
            total = len(self.image_paths)
            self.progress_changed.emit(0, total, f"{service.device_label} · {service.image_size}px · порог {service.confidence_threshold:.0%}")
            for index, values in enumerate(self.image_paths, start=1):
                image_path, width, height, core_columns = values[:4]
                columns_verified = bool(values[4]) if len(values) >= 5 else bool(core_columns)
                self.progress_changed.emit(index, total, f"Этап 1/2 · столбики · {Path(image_path).name}")
                if columns_verified and core_columns:
                    recognized_columns = normalize_columns(core_columns, (width, height))
                else:
                    recognized_columns = column_recognizer.recognize_path(image_path, (width, height))
                self.progress_changed.emit(index, total, f"Этап 2/2 · интервалы и фации · {Path(image_path).name}")
                detections = service.predict(image_path, (width, height), recognized_columns)
                self.image_ready.emit(image_path, detections, recognized_columns)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()
