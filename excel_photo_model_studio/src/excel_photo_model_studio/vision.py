from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .depth import centimeters_to_meters, meters_to_centimeters
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


def detect_core_columns_from_path(path: Path) -> list[tuple[int, int, int, int]]:
    """Detect columns once per unchanged source file during project creation."""
    resolved = Path(path).expanduser().resolve(strict=True)
    stat = resolved.stat()
    return list(_cached_core_columns(str(resolved), stat.st_size, stat.st_mtime_ns))


@lru_cache(maxsize=512)
def _cached_core_columns(
    path: str, _size: int, _mtime_ns: int,
) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(detect_core_columns(read_image(Path(path))))


def detect_core_columns(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect physical core columns with the Kern Analyzer stage-1 method.

    This mirrors the main application's deterministic ``CoreColumnRecognizer``
    fallback: compact connected components are attempted first, followed by a
    conservative vertical-occupancy projection. Facies are never inferred here.
    """
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
    candidate = _core_candidate_mask(image)

    # Projection must be attempted before connected components. Horizontal
    # depth lines often connect every real core column into one very wide
    # component, while the narrow ruler remains separate and used to be
    # returned as the only "core" object.
    projection_boxes = _projection_core_boxes(candidate)
    component_boxes = _core_component_boxes(candidate)
    combined_boxes = list(projection_boxes)
    for component_box in component_boxes:
        if any(_same_horizontal_lane(component_box, projected) for projected in projection_boxes):
            continue
        combined_boxes.append(component_box)
    selected = _select_core_boxes(combined_boxes, candidate)
    if selected:
        return selected

    fallback_boxes = _fallback_component_boxes(candidate.astype(np.uint8) * 255)
    return _select_core_boxes(fallback_boxes, candidate)


def _core_candidate_mask(image: np.ndarray) -> np.ndarray:
    """Separate core material from a pale document page or a coloured tray."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    pale_page_fraction = float(((saturation < 45) & (value > 235)).mean())
    if pale_page_fraction >= 0.20:
        # Scanned reports can contain almost-white sandstone (V=240..250).
        # A fixed V<235 threshold discarded it and left only the black ruler.
        page_level = float(np.quantile(value, 0.90))
        value_limit = int(np.clip(round(page_level - 3.0), 232, 252))
        return value < value_limit
    return (saturation < 95) & (value < 238)


def _projection_core_boxes(candidate: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find filled vertical lanes, including one short or partial core lane."""
    height, width = candidate.shape
    vertical_start = max(0, int(height * 0.035))
    vertical_end = min(height, max(vertical_start + 1, int(height * 0.92)))
    work = candidate[vertical_start:vertical_end]
    column_score = _smooth(work.mean(axis=0).astype(np.float32), 7)
    background = float(np.quantile(column_score, 0.35))
    high = float(np.quantile(column_score, 0.95))
    threshold = float(np.clip(background + (high - background) * 0.34, 0.16, 0.46))
    active_columns = column_score >= threshold
    min_width = max(12, int(width * 0.026))
    boxes: list[tuple[int, int, int, int]] = []
    for left, right in _runs(active_columns):
        # A cropped photograph may contain only one core lane occupying much of
        # the frame; page-style reports still produce separate narrow x-runs.
        if right - left < min_width or right - left > width * 0.72:
            continue
        row_score = _smooth(candidate[:, left:right].mean(axis=1).astype(np.float32), 15)
        active_rows = (row_score >= 0.20).astype(np.uint8)
        # Join modest blank breaks inside a physical column without stretching
        # the result over captions above and below the photographed core.
        # Keep fractures inside the photographed core connected, but never
        # bridge the whitespace between the core and printed depth numbers.
        # The previous 2.5% kernel could pull a header such as "4130.00" into
        # the column box, making the depth count start above the actual core.
        join_gap = max(3, int(height * 0.008))
        active_rows = cv2.morphologyEx(
            active_rows.reshape(-1, 1), cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, join_gap)),
        ).reshape(-1).astype(bool)
        row_runs = _runs(active_rows)
        if not row_runs:
            continue
        top, bottom = max(row_runs, key=lambda run: run[1] - run[0])
        if bottom - top < max(35, height * 0.10):
            continue
        boxes.append((
            max(0, left - 1), max(0, top - 1),
            min(width, right + 1), min(height, bottom + 1),
        ))
    return boxes


def _core_component_boxes(candidate: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find complete core objects before analysing full-height projections."""
    height, width = candidate.shape
    kernel_height = max(9, min(25, round(height * 0.006)))
    connected = cv2.morphologyEx(
        candidate.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, kernel_height)),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for component in range(1, count):
        left, top, box_width, box_height, area = (int(value) for value in stats[component])
        aspect = box_height / max(box_width, 1)
        center_x = left + box_width / 2
        filled_fraction = area / max(1, box_width * box_height)
        if (
            box_width >= max(12, int(width * 0.026))
            and box_width <= width * 0.30
            and box_height >= max(40, int(height * 0.08))
            and 1.00 <= aspect <= 35.0
            and filled_fraction >= 0.15
            and width * 0.06 < center_x < width * 0.94
            and top < height * 0.75
        ):
            boxes.append((left, top, left + box_width, top + box_height))
    return sorted(boxes, key=lambda item: item[0])


def _fallback_component_boxes(candidate: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Fallback for photographs where the tray and core have similar colour."""
    height, width = candidate.shape
    connected = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, max(17, int(height * 0.075)))),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for component in range(1, count):
        left, top, box_width, box_height, area = (int(value) for value in stats[component])
        aspect = box_height / max(box_width, 1)
        if (
            box_width >= max(12, int(width * 0.026))
            and box_width <= width * 0.30
            and box_height >= height * 0.10
            and 1.0 <= aspect <= 35.0
            and area >= box_width * box_height * 0.20
        ):
            boxes.append((left, top, left + box_width, top + box_height))
    return sorted(boxes, key=lambda item: item[0])


def calibrate_core_columns(
    columns: list[tuple[int, int, int, int]], photo_top: float, photo_base: float,
) -> list[tuple[tuple[int, int, int, int], float, float]]:
    """Assign consecutive centimetres to ordered one-metre physical columns."""
    if not columns or photo_base <= photo_top:
        return []
    photo_top_cm = meters_to_centimeters(photo_top)
    photo_base_cm = meters_to_centimeters(photo_base)
    total_cm = photo_base_cm - photo_top_cm
    capacities = _core_column_capacities_cm(columns)
    capacity_sum = sum(capacities)
    cursor_cm = photo_top_cm
    calibrated: list[tuple[tuple[int, int, int, int], float, float]] = []
    if total_cm <= capacity_sum + 2:
        # Normal report: each complete lane is exactly one metre; the last
        # lane may contain only the remaining centimetres.
        for index, (box, capacity_cm) in enumerate(zip(columns, capacities)):
            remaining_cm = photo_base_cm - cursor_cm
            if remaining_cm <= 0:
                break
            if index == len(columns) - 1:
                column_base_cm = photo_base_cm
            else:
                column_base_cm = cursor_cm + min(capacity_cm, remaining_cm)
            calibrated.append((
                box,
                centimeters_to_meters(cursor_cm),
                centimeters_to_meters(column_base_cm),
            ))
            cursor_cm = column_base_cm
        return calibrated

    # Never stretch detected core to cover a longer photo interval. Each full
    # physical column holds at most 1 m (a partial last column less); stretching
    # a mistaken/drilling interval over the image shifts every facies mask.
    # Any excess remains uncovered and is reported by project validation.
    for box, capacity_cm in zip(columns, capacities):
        remaining_cm = photo_base_cm - cursor_cm
        if remaining_cm <= 0:
            break
        column_base_cm = min(photo_base_cm, cursor_cm + capacity_cm)
        calibrated.append((
            box,
            centimeters_to_meters(cursor_cm),
            centimeters_to_meters(column_base_cm),
        ))
        cursor_cm = column_base_cm
    return calibrated


def core_photo_capacity_centimeters(columns: list[tuple[int, int, int, int]]) -> int:
    return sum(_core_column_capacities_cm(columns))


def _core_column_capacities_cm(columns: list[tuple[int, int, int, int]]) -> list[int]:
    if not columns:
        return []
    heights = [max(1, bottom - top) for _, top, _, bottom in columns]
    # The upper median keeps a half-height final lane at 50 cm even when a
    # photograph contains only one full and one partial lane.
    reference_height = max(1, sorted(heights)[len(heights) // 2])
    return [max(1, min(100, round(height * 100 / reference_height))) for height in heights]


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
        photo = photo_matches[0].photo
        columns = (
            detect_core_columns_from_path(photo_path)
            if photo.source.startswith("ocr_")
            else detect_core_columns(image)
        )
        if not columns:
            columns_by_photo[photo_path] = []
            requested_order = normalize_column_order(photo_matches[0].photo.column_order)
            orders_by_photo[photo_path] = (
                requested_order if requested_order != COLUMN_ORDER_AUTO else COLUMN_ORDER_LEFT_TO_RIGHT
            )
            continue
        columns = sorted(columns, key=lambda box: (box[0], box[1]))
        requested_order = normalize_column_order(photo_matches[0].photo.column_order)
        effective_order = detect_column_order(image, columns) if requested_order == COLUMN_ORDER_AUTO else requested_order
        if effective_order == COLUMN_ORDER_RIGHT_TO_LEFT:
            columns.reverse()
        columns_by_photo[photo_path] = columns
        orders_by_photo[photo_path] = effective_order
        if not photo.has_interval:
            continue
        calibrated = calibrate_core_columns(columns, float(photo.top), float(photo.base))
        for column_index, (box, column_top, column_base) in enumerate(calibrated):
            left, pixel_top, right, pixel_bottom = box
            for match in photo_matches:
                overlap_top_cm = max(
                    meters_to_centimeters(match.description.top), meters_to_centimeters(column_top),
                )
                overlap_base_cm = min(
                    meters_to_centimeters(match.description.base), meters_to_centimeters(column_base),
                )
                if overlap_base_cm <= overlap_top_cm:
                    continue
                overlap_top = centimeters_to_meters(overlap_top_cm)
                overlap_base = centimeters_to_meters(overlap_base_cm)
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
                    facies_top=match.description.top, facies_base=match.description.base,
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
        image = cv2.addWeighted(overlay, 0.72, image, 0.28, 0)
        for item in items:
            color = _label_color(item.label)
            points = _preview_points(item.polygon, scale, image.shape[0])
            stroke = max(5, round(image.shape[1] / 300))
            cv2.polylines(image, [points], True, (255, 255, 255), stroke + 5, cv2.LINE_AA)
            cv2.polylines(image, [points], True, color, stroke, cv2.LINE_AA)
            facies_top = item.facies_top if item.facies_top is not None else item.depth_top
            facies_base = item.facies_base if item.facies_base is not None else item.depth_base
            interval = f"FACIES {item.label} | {facies_top:.2f}-{facies_base:.2f}m"
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
                "facies_top": item.facies_top,
                "facies_base": item.facies_base,
                "polygon": item.polygon,
            }
            for item in sorted(items, key=lambda value: value.annotation_id)
        ],
        "renderer": 3,
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


def _same_horizontal_lane(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int],
) -> bool:
    overlap = min(first[2], second[2]) - max(first[0], second[0])
    return overlap >= min(first[2] - first[0], second[2] - second[0]) * 0.58


def _select_core_boxes(
    boxes: list[tuple[int, int, int, int]], candidate: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Reject rulers and captions without requiring several core columns."""
    height, width = candidate.shape
    plausible: list[tuple[int, int, int, int]] = []
    for box in _deduplicate(boxes):
        left, top, right, bottom = box
        box_width, box_height = right - left, bottom - top
        if box_width < max(12, int(width * 0.026)) or box_height < max(35, int(height * 0.10)):
            continue
        center_x = (left + right) / 2.0
        if not width * 0.04 < center_x < width * 0.96:
            continue
        region = candidate[max(0, top):min(height, bottom), max(0, left):min(width, right)]
        if region.size == 0:
            continue
        fill = float(region.mean())
        dense_rows = float((region.mean(axis=1) >= 0.35).mean())
        dense_columns = float((region.mean(axis=0) >= 0.22).mean())
        if fill < 0.12 or dense_rows < 0.16 or dense_columns < 0.22:
            continue

        narrow = box_width < width * 0.075
        near_page_edge = center_x < width * 0.20 or center_x > width * 0.93
        sparse_scale = fill < 0.48 or dense_rows < 0.45 or dense_columns < 0.55
        if narrow and near_page_edge and sparse_scale:
            continue
        if box_width < width * 0.045 and fill < 0.45:
            continue
        plausible.append(box)
    return _filter_width_outliers(plausible)


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
