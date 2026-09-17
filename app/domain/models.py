from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPixmap


@dataclass
class FaciesDetection:
    label: str
    confidence: float
    polygon: list[QPointF]
    attributes: dict[str, str] = field(default_factory=dict)
    depth_from: float | None = None
    depth_to: float | None = None
    training_ready: bool = False
    alternatives: dict[str, float] = field(default_factory=dict)
    # Interpreter-defined vertical order in the continuous assembled core.
    # ``None`` keeps the natural left-column/top-to-bottom photo reading order.
    stack_order: int | None = None


@dataclass
class PhotoRecord:
    identifier: str
    path: str
    pixmap: QPixmap
    detections: list[FaciesDetection] = field(default_factory=list)
    well_name: str = "Скважина 1"
    # Measured interval encoded in the source photo name.  It is retained
    # separately from ``path`` because imported Excel photos are copied to a
    # project folder with a numeric prefix.
    photo_depth_from: float | None = None
    photo_depth_to: float | None = None
    # Per-column calibration is required when one photograph contains several
    # core columns read from left to right. Coordinates are in pixmap pixels.
    depth_segments: list[dict[str, float]] = field(default_factory=list)
    # Optional interpreter-corrected physical core bounds.  They are kept
    # separately from depth calibration so segmentation can ignore rulers,
    # labels and slag even before depth intervals are assigned.
    core_columns: list[dict[str, float]] = field(default_factory=list)
    # Only interpreter-confirmed rectangles may be used to train the separate
    # stage-1 column recognizer.  Automatically detected rectangles are kept
    # for inference and display, but never silently become training truth.
    core_columns_verified: bool = False
