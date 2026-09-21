from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .models import Annotation, Match


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


def project_matches(matches: list[Match]) -> tuple[list[Annotation], dict[Path, list[tuple[int, int, int, int]]]]:
    """Project depth overlaps onto detected columns; every result remains unapproved."""
    grouped: dict[Path, list[Match]] = {}
    for match in matches:
        grouped.setdefault(match.photo.path, []).append(match)
    annotations: list[Annotation] = []
    columns_by_photo: dict[Path, list[tuple[int, int, int, int]]] = {}
    for photo_path, photo_matches in grouped.items():
        image = read_image(photo_path)
        height, width = image.shape[:2]
        columns = detect_core_columns(image)
        if not columns:
            columns = [(0, 0, width, max(1, int(height * 0.88)))]
        columns = sorted(columns, key=lambda box: (box[0], box[1]))
        columns_by_photo[photo_path] = columns
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
                if overlap_base - overlap_top <= 1e-5:
                    continue
                y0 = pixel_top + (overlap_top - column_top) / (column_base - column_top) * (pixel_bottom - pixel_top)
                y1 = pixel_top + (overlap_base - column_top) / (column_base - column_top) * (pixel_bottom - pixel_top)
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
                ))
    return annotations, columns_by_photo


def render_previews(annotations: list[Annotation], destination: Path) -> dict[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    grouped: dict[Path, list[Annotation]] = {}
    for annotation in annotations:
        grouped.setdefault(annotation.photo_path, []).append(annotation)
    output: dict[Path, Path] = {}
    for index, (photo_path, items) in enumerate(sorted(grouped.items(), key=lambda pair: pair[0].name.casefold()), start=1):
        image = read_image(photo_path)
        overlay = image.copy()
        for item in items:
            color = _label_color(item.label)
            points = np.array(item.polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [points], color)
            cv2.polylines(image, [points], True, color, max(2, round(image.shape[1] / 700)))
        image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
        for item in items:
            color = _label_color(item.label)
            points = np.array(item.polygon, dtype=np.int32)
            anchor = (max(5, int(points[0][0]) + 4), max(22, int(points[0][1]) + 20))
            interval = f"{item.depth_top:.2f}-{item.depth_base:.2f}m"
            cv2.putText(image, interval, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
        target = destination / f"{index:05d}_{photo_path.stem}.jpg"
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            target.write_bytes(encoded.tobytes())
            output[photo_path] = target
    return output


def _label_color(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return tuple(int(70 + value % 170) for value in digest[:3])


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
