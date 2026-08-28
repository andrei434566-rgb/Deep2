"""Quality gates and active-learning ordering for local fine-tuning."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from PySide6.QtGui import QPolygonF

from app.domain.models import FaciesDetection, PhotoRecord


@dataclass(frozen=True)
class TrainingQuality:
    verified: int
    by_facies: dict[str, int]
    underrepresented: dict[str, int]
    unnamed_manual: int
    missing_depth: int
    duplicate_masks: int
    too_small_masks: int
    too_large_masks: int
    severe_imbalance: bool

    @property
    def blocking_reasons(self) -> list[str]:
        """Problems serious enough to make a training run misleading."""
        reasons: list[str] = []
        if self.severe_imbalance:
            counts = ", ".join(f"{label}: {count}" for label, count in self.by_facies.items())
            reasons.append(f"сильный дисбаланс классов ({counts})")
        return reasons

    @property
    def summary(self) -> str:
        parts = [f"проверено: {self.verified}"]
        if self.underrepresented:
            values = ", ".join(f"{label}: {count}" for label, count in self.underrepresented.items())
            parts.append(f"мало примеров — {values}")
        if self.unnamed_manual:
            parts.append(f"без фации: {self.unnamed_manual}")
        if self.missing_depth:
            parts.append(f"без глубин: {self.missing_depth}")
        if self.duplicate_masks:
            parts.append(f"дубликаты: {self.duplicate_masks}")
        if self.too_small_masks:
            parts.append(f"слишком малые: {self.too_small_masks}")
        if self.too_large_masks:
            parts.append(f"слишком большие: {self.too_large_masks}")
        if self.severe_imbalance:
            parts.append("сильный дисбаланс — обучение заблокировано")
        return " · ".join(parts)


def training_quality(records: list[PhotoRecord], min_per_facies: int = 5) -> TrainingQuality:
    """Summarise only examples explicitly approved by the interpreter."""
    labels: Counter[str] = Counter()
    unnamed_manual = 0
    missing_depth = 0
    duplicate_masks = 0
    too_small_masks = 0
    too_large_masks = 0
    for record in records:
        approved: list[FaciesDetection] = []
        for detection in record.detections:
            if not detection.training_ready:
                continue
            label = str(detection.label or "").strip()
            if not label or label == "Новый контур":
                unnamed_manual += 1
                continue
            labels[label] += 1
            approved.append(detection)
            if detection.depth_from is None or detection.depth_to is None:
                missing_depth += 1
            size_kind = _mask_size_kind(record, detection)
            too_small_masks += size_kind == "small"
            too_large_masks += size_kind == "large"
        duplicate_masks += _duplicate_count(approved)
    ordered = dict(sorted(labels.items(), key=lambda item: (item[1], item[0].casefold())))
    nonzero_counts = list(ordered.values())
    # A class represented four times less than the dominant class causes a
    # segmentation model to prefer the dominant facies.  Very small datasets
    # are handled by the ordinary minimum-example warning instead.
    severe_imbalance = (
        len(nonzero_counts) > 1
        and sum(nonzero_counts) >= 12
        and max(nonzero_counts) / max(1, min(nonzero_counts)) >= 4
    )
    return TrainingQuality(
        verified=sum(labels.values()),
        by_facies=ordered,
        underrepresented={label: count for label, count in ordered.items() if count < min_per_facies},
        unnamed_manual=unnamed_manual,
        missing_depth=missing_depth,
        duplicate_masks=duplicate_masks,
        too_small_masks=too_small_masks,
        too_large_masks=too_large_masks,
        severe_imbalance=severe_imbalance,
    )


def _mask_size_kind(record: PhotoRecord, detection: FaciesDetection) -> str | None:
    """Flag accidental dots and nearly-full-photo masks, without rejecting them."""
    if record.pixmap.isNull() or len(detection.polygon) < 3:
        return None
    rect = QPolygonF(detection.polygon).boundingRect()
    image_area = max(1.0, float(record.pixmap.width() * record.pixmap.height()))
    fraction = max(0.0, rect.width() * rect.height()) / image_area
    if rect.width() < 5 or rect.height() < 5 or fraction < 0.00025:
        return "small"
    if fraction > 0.90:
        return "large"
    return None


def _duplicate_count(detections: list[FaciesDetection]) -> int:
    """Count near-identical approved masks using bounding-box IoU.

    This intentionally remains a warning: two layers can legitimately share
    much of a rectangular bounding box, while exact geometric duplicates are
    still caught reliably enough for a pre-training checklist.
    """
    duplicates = 0
    rects = [(detection.label, QPolygonF(detection.polygon).boundingRect()) for detection in detections]
    for index, (label, rect) in enumerate(rects):
        area = rect.width() * rect.height()
        if area <= 0:
            continue
        for other_label, other in rects[:index]:
            if other_label != label:
                continue
            intersection = rect.intersected(other)
            union = area + other.width() * other.height() - intersection.width() * intersection.height()
            if union > 0 and intersection.width() * intersection.height() / union >= 0.90:
                duplicates += 1
                break
    return duplicates


def review_queue(records: list[PhotoRecord], limit: int = 100) -> list[tuple[PhotoRecord, FaciesDetection]]:
    """Prioritise automatic masks with low confidence or close alternatives."""
    candidates: list[tuple[float, PhotoRecord, FaciesDetection]] = []
    for record in records:
        for detection in record.detections:
            if detection.training_ready or not str(detection.label or "").strip():
                continue
            alternatives = [float(value) for value in detection.alternatives.values()]
            rival = max(alternatives, default=0.0)
            # Lower confidence and a close competitor both move a mask upward.
            ambiguity = max(0.0, rival - float(detection.confidence) + 0.15)
            size_kind = _mask_size_kind(record, detection)
            geometry_penalty = 0.25 if size_kind is not None else 0.0
            priority = (1.0 - float(detection.confidence)) + ambiguity + geometry_penalty
            candidates.append((priority, record, detection))
    candidates.sort(key=lambda item: (-item[0], item[1].well_name.casefold(), item[2].confidence))
    return [(record, detection) for _, record, detection in candidates[:max(1, limit)]]
