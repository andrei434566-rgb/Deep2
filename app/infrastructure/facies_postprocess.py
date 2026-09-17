"""Clean-up and gap-free projection of facies inside physical core columns."""

from __future__ import annotations

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPolygonF

from app.domain.facies_catalog import facies_identity
from app.domain.models import FaciesDetection, PhotoRecord

UNRECOGNIZED_FACIES = "Не определена"


def postprocess_detections(
    record: PhotoRecord,
    detections: list[FaciesDetection],
    fallback_label: str | None = None,
) -> list[FaciesDetection]:
    """Remove tiny fragments and make each detected core column gap-free.

    Reviewed contours are never rewritten.  All decisions are constrained to a
    single physical core column, preventing boundaries from being smoothed
    through a ruler, tray or a neighbouring column.  Fresh automatic results
    are finally converted to a vertical partition: every selected core pixel
    belongs to exactly one facies band.
    """
    if record.pixmap.isNull():
        return detections
    min_height = max(5.0, record.pixmap.height() * 0.002)
    columns = record.core_columns or [{"left": 0, "top": 0, "right": record.pixmap.width(), "bottom": record.pixmap.height()}]
    automatic: list[tuple[int, FaciesDetection]] = []
    retained: list[FaciesDetection] = []
    for detection in detections:
        rect = QPolygonF(detection.polygon).boundingRect()
        if detection.training_ready or rect.height() >= min_height:
            automatic.append((_column_for(rect, columns), detection)) if not detection.training_ready else retained.append(detection)
    automatic.sort(key=lambda item: (item[0], QPolygonF(item[1].polygon).boundingRect().top()))
    for column_index, detection in automatic:
        previous = next((item for item in reversed(retained) if not item.training_ready and _column_for(QPolygonF(item.polygon).boundingRect(), columns) == column_index), None)
        if previous is not None and _can_merge(previous, detection):
            retained[-1] = _merged(previous, detection)
        else:
            retained.append(detection)
    if not columns:
        return retained
    reviewed = [item for item in retained if item.training_ready]
    automatic = [item for item in retained if not item.training_ready]
    covered = complete_core_column_coverage(
        automatic,
        columns,
        fallback_label=fallback_label,
        depth_segments=record.depth_segments,
    )
    return reviewed + covered


def complete_core_column_coverage(
    detections: list[FaciesDetection],
    columns: list[dict[str, float]],
    fallback_label: str | None = None,
    depth_segments: list[dict[str, float]] | None = None,
) -> list[FaciesDetection]:
    """Turn arbitrary model masks into one gap-free partition per core column.

    Existing predictions define all candidate contacts. Uncovered intervals
    remain geometrically filled but explicitly unrecognized. Adjacency or an
    arbitrary first model class is not evidence of geological identity.
    ``fallback_label`` is retained for call compatibility, never used to guess.
    """
    valid_columns = [_valid_column(item) for item in columns]
    valid_columns = [item for item in valid_columns if item is not None]
    if not valid_columns:
        return list(detections)
    valid_detections = [
        item for item in detections
        if str(item.label or "").strip() and len(item.polygon) >= 3
    ]
    result: list[FaciesDetection] = []
    for column_index, column in enumerate(valid_columns):
        candidates: list[tuple[float, float, float, FaciesDetection]] = []
        for detection in valid_detections:
            rect = QPolygonF(detection.polygon).boundingRect()
            horizontal_overlap = max(0.0, min(rect.right(), column["right"]) - max(rect.left(), column["left"]))
            if horizontal_overlap <= 0:
                continue
            top = max(column["top"], rect.top())
            bottom = min(column["bottom"], rect.bottom())
            if bottom - top <= 0.25:
                continue
            width_fraction = horizontal_overlap / max(1.0, min(rect.width(), column["right"] - column["left"]))
            if width_fraction >= 0.15:
                candidates.append((top, bottom, width_fraction, detection))

        boundaries = {column["top"], column["bottom"]}
        for top, bottom, _, _ in candidates:
            boundaries.update((top, bottom))
        edges = sorted(boundaries)
        bands: list[FaciesDetection] = []
        for top, bottom in zip(edges[:-1], edges[1:]):
            if bottom - top <= 1e-6:
                continue
            midpoint = (top + bottom) / 2
            active = [item for item in candidates if item[0] <= midpoint < item[1]]
            gap_filled = not active
            if active:
                # Interval crops are classified at higher effective resolution
                # than the compressed full photograph, so they own the label
                # inside their structural package when available.
                chosen = max(
                    active,
                    key=lambda item: (
                        bool(item[3].attributes.get("__structural_interval")),
                        item[3].confidence * item[2],
                        item[2],
                        item[3].confidence,
                    ),
                )[3]
            else:
                chosen = FaciesDetection(UNRECOGNIZED_FACIES, 0.0, [], attributes={
                    "Статус фации": "нет прогноза, требуется разметка",
                })
            attributes = dict(chosen.attributes)
            attributes["Покрытие керна"] = (
                "интервал без достоверного прогноза" if gap_filled else "прогноз модели"
            )
            depth_from, depth_to = _band_depths(column, top, bottom, depth_segments or [])
            confidence = max(0.0, min(1.0, chosen.confidence * (0.5 if gap_filled else 1.0)))
            band = FaciesDetection(
                label=chosen.label,
                confidence=confidence,
                polygon=[
                    QPointF(column["left"], top), QPointF(column["right"], top),
                    QPointF(column["right"], bottom), QPointF(column["left"], bottom),
                ],
                attributes=attributes,
                depth_from=depth_from,
                depth_to=depth_to,
                alternatives=dict(chosen.alternatives),
            )
            if bands and _same_band(bands[-1], band):
                bands[-1] = _merged(bands[-1], band)
            else:
                bands.append(band)
        result.extend(bands)
    return result


def _valid_column(values: dict[str, float]) -> dict[str, float] | None:
    try:
        column = {key: float(values[key]) for key in ("left", "top", "right", "bottom")}
    except (KeyError, TypeError, ValueError):
        return None
    return column if column["right"] > column["left"] and column["bottom"] > column["top"] else None


def _same_band(first: FaciesDetection, second: FaciesDetection) -> bool:
    first_interval = first.attributes.get("__structural_interval")
    second_interval = second.attributes.get("__structural_interval")
    same_structure = not (first_interval or second_interval) or first_interval == second_interval
    same_identity = facies_identity(first.label, first.attributes) == facies_identity(second.label, second.attributes)
    same_status = first.attributes.get("Статус фации") == second.attributes.get("Статус фации")
    return same_identity and same_status and same_structure and abs(QPolygonF(first.polygon).boundingRect().bottom() - QPolygonF(second.polygon).boundingRect().top()) <= 1e-6


def _band_depths(
    column: dict[str, float],
    top: float,
    bottom: float,
    depth_segments: list[dict[str, float]],
) -> tuple[float | None, float | None]:
    center_x = (column["left"] + column["right"]) / 2
    segment = next(
        (
            item for item in depth_segments
            if all(key in item for key in ("left", "top", "right", "bottom", "depth_from", "depth_to"))
            and float(item["left"]) <= center_x <= float(item["right"])
        ),
        None,
    )
    if segment is None:
        return None, None
    pixel_span = float(segment["bottom"]) - float(segment["top"])
    if pixel_span <= 0:
        return None, None
    depth_span = float(segment["depth_to"]) - float(segment["depth_from"])
    start = float(segment["depth_from"]) + (top - float(segment["top"])) / pixel_span * depth_span
    end = float(segment["depth_from"]) + (bottom - float(segment["top"])) / pixel_span * depth_span
    return round(min(start, end), 4), round(max(start, end), 4)


def _column_for(rect, columns: list[dict[str, float]]) -> int:
    center_x = rect.center().x()
    for index, column in enumerate(columns):
        if float(column.get("left", 0)) <= center_x <= float(column.get("right", 0)):
            return index
    return -1


def _can_merge(first: FaciesDetection, second: FaciesDetection) -> bool:
    if facies_identity(first.label, first.attributes) != facies_identity(second.label, second.attributes):
        return False
    if first.attributes.get("Статус фации") != second.attributes.get("Статус фации"):
        return False
    first_interval = first.attributes.get("__structural_interval")
    second_interval = second.attributes.get("__structural_interval")
    if (first_interval or second_interval) and first_interval != second_interval:
        return False
    one, two = QPolygonF(first.polygon).boundingRect(), QPolygonF(second.polygon).boundingRect()
    vertical_gap = two.top() - one.bottom()
    horizontal_overlap = max(0.0, min(one.right(), two.right()) - max(one.left(), two.left()))
    return -2.0 <= vertical_gap <= 8.0 and horizontal_overlap >= min(one.width(), two.width()) * 0.65


def _merged(first: FaciesDetection, second: FaciesDetection) -> FaciesDetection:
    one, two = QPolygonF(first.polygon).boundingRect(), QPolygonF(second.polygon).boundingRect()
    rect = one.united(two)
    return FaciesDetection(
        label=first.label,
        # Do not hide the least certain part of a merged band from review.
        confidence=min(first.confidence, second.confidence),
        polygon=[QPointF(rect.left(), rect.top()), QPointF(rect.right(), rect.top()), QPointF(rect.right(), rect.bottom()), QPointF(rect.left(), rect.bottom())],
        attributes=dict(first.attributes),
        depth_from=min(value for value in (first.depth_from, second.depth_from) if value is not None) if first.depth_from is not None or second.depth_from is not None else None,
        depth_to=max(value for value in (first.depth_to, second.depth_to) if value is not None) if first.depth_to is not None or second.depth_to is not None else None,
        alternatives={**first.alternatives, **second.alternatives},
    )
