"""Folder-based project persistence for Kern Analyzer."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPixmap

from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.image_loading import load_working_pixmap, source_image_size

MANIFEST_NAME = "project.json"


def save_project(
    folder: Path,
    title: str,
    records: list[PhotoRecord],
    positions: dict[str, QPointF],
    wells: list[dict] | None = None,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    images_folder = folder / "images"
    images_folder.mkdir(exist_ok=True)
    for record in records:
        source = Path(record.path)
        if not source.is_file():
            raise FileNotFoundError(f"Не найдено фото проекта: {source}")
        target = images_folder / f"{record.identifier}_{source.name}"
        try:
            same_file = source.resolve() == target.resolve()
        except OSError:
            same_file = source == target
        if not same_file:
            shutil.copy2(source, target)
            record.path = str(target)
    payload = {
        "format": "kern_analyzer-project",
        "title": title,
        "wells": list(wells or []),
        "photos": [
            {
                "id": record.identifier,
                "path": record.path,
                "well_name": record.well_name,
                "photo_depth_from": record.photo_depth_from,
                "photo_depth_to": record.photo_depth_to,
                "depth_segments": record.depth_segments,
                "core_columns": record.core_columns,
                "display_size": [record.pixmap.width(), record.pixmap.height()],
                "position": [positions.get(record.identifier, QPointF()).x(), positions.get(record.identifier, QPointF()).y()],
                "detections": [
                    {
                        "label": detection.label,
                        "confidence": detection.confidence,
                        "polygon": [[point.x(), point.y()] for point in detection.polygon],
                        "attributes": detection.attributes,
                        "depth_from": detection.depth_from,
                        "depth_to": detection.depth_to,
                        "training_ready": detection.training_ready,
                        "alternatives": detection.alternatives,
                    }
                    for detection in record.detections
                ],
            }
            for record in records
        ],
    }
    manifest = folder / MANIFEST_NAME
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_project(folder: Path) -> tuple[str, list[PhotoRecord], dict[str, QPointF], list[str], list[dict]]:
    manifest = folder / MANIFEST_NAME
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("format") != "kern_analyzer-project":
        raise ValueError("Выбранная папка не является проектом Kern Analyzer")

    records: list[PhotoRecord] = []
    positions: dict[str, QPointF] = {}
    missing: list[str] = []
    for item in payload.get("photos", []):
        path = str(item.get("path") or "")
        source_size = source_image_size(path)
        pixmap = load_working_pixmap(path)
        if pixmap.isNull():
            missing.append(path)
            continue
        saved_size = item.get("display_size")
        fallback_width = source_size.width() if source_size.width() > 0 else pixmap.width()
        fallback_height = source_size.height() if source_size.height() > 0 else pixmap.height()
        previous_size = saved_size if isinstance(saved_size, (list, tuple)) else [fallback_width, fallback_height]
        previous_width = float(previous_size[0]) if len(previous_size) >= 2 and previous_size[0] else float(pixmap.width())
        previous_height = float(previous_size[1]) if len(previous_size) >= 2 and previous_size[1] else float(pixmap.height())
        x_scale, y_scale = pixmap.width() / max(1.0, previous_width), pixmap.height() / max(1.0, previous_height)
        detections = [
            FaciesDetection(
                label=str(detection.get("label") or "Новый контур"),
                confidence=float(detection.get("confidence") or 0.0),
                polygon=[QPointF(float(point[0]) * x_scale, float(point[1]) * y_scale) for point in detection.get("polygon", []) if len(point) >= 2],
                attributes=dict(detection.get("attributes") or {}),
                depth_from=_float_or_none(detection.get("depth_from")),
                depth_to=_float_or_none(detection.get("depth_to")),
                training_ready=bool(detection.get("training_ready", False)),
                alternatives={str(name): float(score) for name, score in dict(detection.get("alternatives") or {}).items()},
            )
            for detection in item.get("detections", [])
        ]
        identifier = str(item.get("id") or path)
        records.append(
            PhotoRecord(
                identifier=identifier,
                path=path,
                pixmap=pixmap,
                detections=detections,
                well_name=str(item.get("well_name") or "Скважина 1"),
                photo_depth_from=_float_or_none(item.get("photo_depth_from")),
                photo_depth_to=_float_or_none(item.get("photo_depth_to")),
                depth_segments=[
                    {
                        str(key): float(value) * (x_scale if str(key) in {"left", "right"} else y_scale if str(key) in {"top", "bottom"} else 1.0)
                        for key, value in segment.items()
                    }
                    for segment in item.get("depth_segments", [])
                    if isinstance(segment, dict)
                ],
                core_columns=[
                    {
                        str(key): float(value) * (x_scale if str(key) in {"left", "right"} else y_scale if str(key) in {"top", "bottom"} else 1.0)
                        for key, value in column.items()
                    }
                    for column in item.get("core_columns", [])
                    if isinstance(column, dict)
                ],
            )
        )
        position = item.get("position") or [0, 0]
        positions[identifier] = QPointF(float(position[0]), float(position[1]))
    wells = [dict(item) for item in payload.get("wells", []) if isinstance(item, dict) and item.get("name")]
    return str(payload.get("title") or folder.name), records, positions, missing, wells


def _float_or_none(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
