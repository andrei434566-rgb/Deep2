"""Conservative clean-up of automatic facies masks inside core columns."""

from __future__ import annotations

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPolygonF

from app.domain.models import FaciesDetection, PhotoRecord


def postprocess_detections(record: PhotoRecord, detections: list[FaciesDetection]) -> list[FaciesDetection]:
    """Remove tiny automatic fragments and join clearly contiguous same-facies zones.

    Reviewed contours are never rewritten.  All decisions are constrained to a
    single physical core column, preventing boundaries from being smoothed
    through a ruler, tray or a neighbouring column.
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
    return retained


def _column_for(rect, columns: list[dict[str, float]]) -> int:
    center_x = rect.center().x()
    for index, column in enumerate(columns):
        if float(column.get("left", 0)) <= center_x <= float(column.get("right", 0)):
            return index
    return -1


def _can_merge(first: FaciesDetection, second: FaciesDetection) -> bool:
    if first.label != second.label:
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
        confidence=max(first.confidence, second.confidence),
        polygon=[QPointF(rect.left(), rect.top()), QPointF(rect.right(), rect.top()), QPointF(rect.right(), rect.bottom()), QPointF(rect.left(), rect.bottom())],
        attributes=dict(first.attributes),
        depth_from=min(value for value in (first.depth_from, second.depth_from) if value is not None) if first.depth_from is not None or second.depth_from is not None else None,
        depth_to=max(value for value in (first.depth_to, second.depth_to) if value is not None) if first.depth_to is not None or second.depth_to is not None else None,
        alternatives={**first.alternatives, **second.alternatives},
    )
