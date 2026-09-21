from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .models import (
    Annotation, COLUMN_ORDER_AUTO, COLUMN_ORDER_LEFT_TO_RIGHT,
    COLUMN_ORDER_RIGHT_TO_LEFT, Match, normalize_column_order,
)


def read_image(path: Path) -> np.ndarray:
    try:
        data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Не удалось прочитать изображение: {Path(path).name}") from exc
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось открыть изображение: {Path(path).name}")
    return image


def detect_core_columns(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Transparent bootstrap detector for elongated, low-saturation core columns."""
    if image is None or image.size == 0 or image.ndim != 3:
        return []
    height, width = image.shape[:2]
    if max(height, width) > 2400:
        scale = 2400.0 / max(height, width)
        reduced = cv2.resize(
            image, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        reduced_boxes = detect_core_columns(reduced)
        inverse = 1.0 / scale
        return [
            (
                max(0, round(left * inverse)), max(0, round(top * inverse)),
                min(width, round(right * inverse)), min(height, round(bottom * inverse)),
            )
            for left, top, right, bottom in reduced_boxes
        ]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidate = ((hsv[:, :, 1] < 75) & (hsv[:, :, 2] < 240)).astype(np.uint8) * 255
    kernel_height = max(9, min(31, round(height * 0.008)))
    connected = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, kernel_height))
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for component in range(1, count):
        left, top, box_width, box_height, area = (int(value) for value in stats[component])
        aspect = box_height / max(1, box_width)
        fill = area / max(1, box_width * box_height)
        center_x = left + box_width / 2
        if (
            box_width >= max(10, int(width * 0.015))
            and box_width <= width * 0.38
            and box_height >= max(45, int(height * 0.08))
            and 0.7 <= aspect <= 45
            and fill >= 0.12
            and width * 0.08 < center_x < width * 0.92
            and top < height * 0.85
        ):
            boxes.append((left, top, left + box_width, top + box_height))
    boxes = _deduplicate(boxes)
    boxes = _filter_width_outliers(boxes)
    if boxes:
        return boxes

    # Conservative fallback: detect persistent vertical occupancy.
    binary = candidate > 0
    column_score = _smooth(binary.mean(axis=0).astype(np.float32), 9)
    for left, right in _runs(column_score >= 0.40):
        if right - left < max(10, int(width * 0.015)) or right - left > width * 0.38:
            continue
        row_score = _smooth(binary[:, left:right].mean(axis=1).astype(np.float32), 17)
        row_runs = _runs(row_score >= 0.20)
        if row_runs:
            top, bottom = max(row_runs, key=lambda item: item[1] - item[0])
            if bottom - top >= height * 0.18:
                boxes.append((left, top, right, bottom))
    return _filter_width_outliers(_deduplicate(boxes))


def project_matches(matches: list[Match]) -> tuple[
    list[Annotation],
    dict[Path, list[tuple[int, int, int, int]]],
    dict[Path, str],
]:
    """Project depth overlaps onto detected columns; every result remains unapproved."""
    grouped: dict[Path, list[Match]] = {}
    for match in matches:
        grouped.setdefault(match.photo.path, []).append(match)
    annotations: list[Annotation] = []
    columns_by_photo: dict[Path, list[tuple[int, int, int, int]]] = {}
    orders_by_photo: dict[Path, str] = {}
    for photo_path, photo_matches in grouped.items():
        image = read_image(photo_path)
        height, width = image.shape[:2]
        columns = detect_core_columns(image)
        if not columns:
            columns = [(0, 0, width, max(1, int(height * 0.88)))]
        columns = sorted(columns, key=lambda box: (box[0], box[1]))
        requested_order = normalize_column_order(photo_matches[0].photo.column_order)
        effective_order = detect_column_order(image, columns) if requested_order == COLUMN_ORDER_AUTO else requested_order
        if effective_order == COLUMN_ORDER_RIGHT_TO_LEFT:
            columns.reverse()
        columns_by_photo[photo_path] = columns
        orders_by_photo[photo_path] = effective_order
        photo = photo_matches[0].photo
        if not photo.has_interval:
            continue
        visual_length = sum(max(1, bottom - top) for _, top, _, bottom in columns)
        depth_cursor = float(photo.top)
        for column_index, (left, pixel_top, right, pixel_bottom) in enumerate(columns):
            depth_span = (float(photo.base) - float(photo.top)) * (pixel_bottom - pixel_top) / visual_length
            column_top, column_base = depth_cursor, depth_cursor + depth_span
            depth_cursor = column_base
            for match in photo_matches:
                overlap_top = max(match.description.top, column_top)
                overlap_base = min(match.description.base, column_base)
                if overlap_base - overlap_top <= 1e-9:
                    continue
                y0 = pixel_top + (overlap_top - column_top) / (column_base - column_top) * (pixel_bottom - pixel_top)
                y1 = pixel_top + (overlap_base - column_top) / (column_base - column_top) * (pixel_bottom - pixel_top)
                y0, y1 = _minimum_vertical_span(y0, y1, pixel_top, pixel_bottom)
                polygon = (
                    (float(left), float(y0)), (float(max(left + 1, right - 1)), float(y0)),
                    (float(max(left + 1, right - 1)), float(y1)), (float(left), float(y1)),
                )
                identity = "|".join((
                    str(photo_path.resolve()), match.description.source_id, str(column_index),
                    f"{overlap_top:.6f}", f"{overlap_base:.6f}", match.description.label,
                ))
                annotation_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                annotations.append(Annotation(
                    annotation_id=annotation_id, photo_path=photo_path, well=photo.well,
                    photo_top=float(photo.top), photo_base=float(photo.base),
                    depth_top=overlap_top, depth_base=overlap_base, label=match.description.label,
                    polygon=polygon, image_width=width, image_height=height,
                    source_sheet=match.description.sheet, source_row=match.description.row,
                    source_file=match.description.source_file,
                    target_text=match.description.target_text,
                    association=match.description.association,
                    environment=match.description.environment,
                    field_name=match.description.field_name,
                ))
    return annotations, columns_by_photo, orders_by_photo


def detect_column_order(image: np.ndarray, columns: list[tuple[int, int, int, int]]) -> str:
    """Infer the horizontal reading order from top/bottom text or depth markers."""
    columns = sorted(columns, key=lambda box: (box[0], box[1]))
    if len(columns) < 2 or image is None or image.size == 0:
        return COLUMN_ORDER_LEFT_TO_RIGHT
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    widths = [right - left for left, _, right, _ in columns]
    pad = max(4, round(float(np.median(widths)) * 0.25))
    marker_span = max(50, round(height * 0.065))
    top_scores: list[float] = []
    bottom_scores: list[float] = []
    for left, top, right, bottom in columns:
        x0, x1 = max(0, left - pad), min(width, right + pad)
        top_region = gray[max(0, top - marker_span):max(1, top - 8), x0:x1]
        bottom_region = gray[min(height - 1, bottom + 8):min(height, bottom + marker_span), x0:x1]
        top_scores.append(_dark_marker_score(top_region))
        bottom_scores.append(_dark_marker_score(bottom_region))

    top_marker = _dominant_marker(top_scores)
    bottom_marker = _dominant_marker(bottom_scores)
    last = len(columns) - 1
    if top_marker == last and bottom_marker == 0:
        return COLUMN_ORDER_RIGHT_TO_LEFT
    if top_marker == 0 and bottom_marker == last:
        return COLUMN_ORDER_LEFT_TO_RIGHT
    if top_marker is not None:
        return COLUMN_ORDER_RIGHT_TO_LEFT if top_marker > last / 2 else COLUMN_ORDER_LEFT_TO_RIGHT
    if bottom_marker is not None:
        return COLUMN_ORDER_RIGHT_TO_LEFT if bottom_marker < last / 2 else COLUMN_ORDER_LEFT_TO_RIGHT
    return COLUMN_ORDER_LEFT_TO_RIGHT


def _dark_marker_score(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    return float((region < 110).mean())


def _dominant_marker(scores: list[float]) -> int | None:
    if not scores:
        return None
    ordered = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    best_index, best = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    if best < 0.0025 or (second > 0 and best < second * 1.6):
        return None
    return best_index


def _minimum_vertical_span(
    y0: float, y1: float, column_top: int, column_bottom: int, minimum_pixels: float = 2.0,
) -> tuple[float, float]:
    """Keep sub-pixel depth intervals trainable while preserving their exact depth metadata."""
    available = max(0.0, float(column_bottom - column_top))
    required = min(minimum_pixels, available)
    if y1 - y0 >= required:
        return y0, y1
    center = (y0 + y1) / 2.0
    expanded_top = max(float(column_top), center - required / 2.0)
    expanded_bottom = min(float(column_bottom), expanded_top + required)
    expanded_top = max(float(column_top), expanded_bottom - required)
    return expanded_top, expanded_bottom


def render_previews(annotations: list[Annotation], destination: Path) -> dict[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "preview_manifest.json"
    try:
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old_manifest = {}
    grouped: dict[Path, list[Annotation]] = {}
    for annotation in annotations:
        grouped.setdefault(annotation.photo_path, []).append(annotation)
    output: dict[Path, Path] = {}
    new_manifest = {}
    for photo_path, items in sorted(grouped.items(), key=lambda pair: pair[0].name.casefold()):
        signature = _preview_signature(photo_path, items)
        target_name = f"{hashlib.sha256(str(photo_path.resolve()).encode('utf-8')).hexdigest()[:12]}_{photo_path.stem}.jpg"
        target = destination / target_name
        cached = old_manifest.get(str(photo_path), {})
        if cached.get("signature") == signature and target.is_file():
            output[photo_path] = target
            new_manifest[str(photo_path)] = {"signature": signature, "target": target_name}
            continue
        image = read_image(photo_path)
        original_height, original_width = image.shape[:2]
        scale = min(1.0, 1800.0 / max(original_height, original_width))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        overlay = image.copy()
        for item in items:
            color = _label_color(item.label)
            points = _preview_points(item.polygon, scale, image.shape[0])
            cv2.fillPoly(overlay, [points], color)
        image = cv2.addWeighted(overlay, 0.62, image, 0.38, 0)
        for item in items:
            color = _label_color(item.label)
            points = _preview_points(item.polygon, scale, image.shape[0])
            stroke = max(5, round(image.shape[1] / 300))
            cv2.polylines(image, [points], True, (255, 255, 255), stroke + 5, cv2.LINE_AA)
            cv2.polylines(image, [points], True, color, stroke, cv2.LINE_AA)
            interval = f"MASK {item.depth_top:.2f}-{item.depth_base:.2f}m"
            font_scale = max(0.7, min(1.25, image.shape[1] / 2200))
            font_thickness = max(2, round(image.shape[1] / 1000))
            (text_width, text_height), baseline = cv2.getTextSize(
                interval, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            text_x = max(5, min(int(points[0][0]) + 8, image.shape[1] - text_width - 16))
            text_y = max(text_height + 12, int(points[0][1]) + text_height + 14)
            cv2.rectangle(
                image,
                (text_x - 7, text_y - text_height - 8),
                (text_x + text_width + 7, text_y + baseline + 7),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                image, interval, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA,
            )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            target.write_bytes(encoded.tobytes())
            output[photo_path] = target
            new_manifest[str(photo_path)] = {"signature": signature, "target": target_name}
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(new_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return output


def _preview_signature(photo_path: Path, items: list[Annotation]) -> str:
    stat = photo_path.stat()
    payload = {
        "photo_size": stat.st_size,
        "photo_mtime_ns": stat.st_mtime_ns,
        "annotations": [
            {
                "id": item.annotation_id,
                "label": item.label,
                "depth_top": item.depth_top,
                "depth_base": item.depth_base,
                "polygon": item.polygon,
            }
            for item in sorted(items, key=lambda value: value.annotation_id)
        ],
        "renderer": 2,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _preview_points(
    polygon: tuple[tuple[float, float], ...], scale: float, image_height: int,
) -> np.ndarray:
    points = np.array(polygon, dtype=np.float32) * float(scale)
    y0, y1 = float(points[:, 1].min()), float(points[:, 1].max())
    if y1 - y0 < 4.0:
        center = (y0 + y1) / 2.0
        expanded_top = max(0.0, center - 2.0)
        expanded_bottom = min(float(max(0, image_height - 1)), expanded_top + 4.0)
        expanded_top = max(0.0, expanded_bottom - 4.0)
        points[np.isclose(points[:, 1], y0), 1] = expanded_top
        points[np.isclose(points[:, 1], y1), 1] = expanded_bottom
    return np.rint(points).astype(np.int32)


def _label_color(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    palette = (
        (0, 0, 255),      # red
        (255, 0, 255),    # magenta
        (0, 215, 255),    # yellow
        (255, 255, 0),    # cyan
        (0, 140, 255),    # orange
        (255, 80, 0),     # bright blue
    )
    return palette[digest[0] % len(palette)]


def _deduplicate(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    result: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: (item[0], item[1], -(item[3] - item[1]))):
        if any(_iou(box, existing) >= 0.68 for existing in result):
            continue
        result.append(box)
    return sorted(result, key=lambda item: (item[0], item[1]))


def _filter_width_outliers(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Reject rulers/arrows when at least three core columns share one width."""
    if len(boxes) < 3:
        return boxes
    widths = sorted(box[2] - box[0] for box in boxes)
    median = float(np.median(widths))
    filtered = [box for box in boxes if median * 0.55 <= box[2] - box[0] <= median * 1.8]
    return filtered if len(filtered) >= 2 else boxes


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / max(1, first_area + second_area - intersection)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1:
        return values
    return np.convolve(values, np.ones(window, dtype=np.float32) / window, mode="same")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return list(zip(starts.tolist(), ends.tolist()))
