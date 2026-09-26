from __future__ import annotations

import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .depth import centimeters_to_meters, meters_to_centimeters, parse_decimal_value
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
    image: np.ndarray | None = None,
    column_depths: tuple[tuple[float, float, float], ...] = (),
) -> list[tuple[tuple[int, int, int, int], float, float]]:
    """Project the photo interval onto ordered columns using the printed ruler when available."""
    if not columns or photo_base <= photo_top:
        return []
    photo_top_cm = meters_to_centimeters(photo_top)
    photo_base_cm = meters_to_centimeters(photo_base)
    total_cm = photo_base_cm - photo_top_cm
    explicit = _column_depths_by_box(
        columns, column_depths, image.shape[1] if image is not None else None,
    )
    if len(explicit) == len(columns):
        return [
            (box, explicit[index][0], explicit[index][1])
            for index, box in enumerate(columns)
        ]
    grid_scale = _depth_grid_pixels_per_centimeter(image) if image is not None else None
    capacities = (
        _scaled_column_capacities_cm(columns, grid_scale)
        if grid_scale is not None else _core_column_capacities_cm(columns)
    )
    capacity_sum = sum(capacities)
    if column_depths:
        # A partial/stale OCR map must not block mask creation forever. Fall
        # back only when independently measured capacity agrees with the
        # photo interval and every label that did parse agrees with its exact
        # physical lane. Conflicting OCR remains a hard blocker.
        if abs(total_cm - capacity_sum) > 2:
            return []
        cursor_cm = photo_top_cm
        calibrated = []
        for index, (box, capacity_cm) in enumerate(zip(columns, capacities)):
            column_base_cm = (
                photo_base_cm if index == len(columns) - 1 else cursor_cm + capacity_cm
            )
            calibrated.append((
                box, centimeters_to_meters(cursor_cm), centimeters_to_meters(column_base_cm),
            ))
            cursor_cm = column_base_cm
        partial = _partial_column_depths_by_box(
            columns, column_depths, image.shape[1] if image is not None else None,
        )
        if not partial or any(
            abs(meters_to_centimeters(partial[index][0]) - meters_to_centimeters(top)) > 2
            or abs(meters_to_centimeters(partial[index][1]) - meters_to_centimeters(base)) > 2
            for index, (_box, top, base) in enumerate(calibrated)
            if index in partial
        ):
            return []
        return calibrated
    if grid_scale is not None and abs(total_cm - capacity_sum) > 2:
        # The printed centimetre ruler is independent evidence of physical
        # length. Do not squeeze five full 1 m columns into, e.g., a 4.1 m
        # caption or stretch a 20 cm fragment into a metre. The project must
        # resolve this contradiction before producing training masks.
        return []
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


def core_photo_capacity_centimeters(
    columns: list[tuple[int, int, int, int]], image: np.ndarray | None = None,
    column_depths: tuple[tuple[float, float, float], ...] = (),
) -> int:
    explicit = _column_depths_by_box(columns, column_depths, image.shape[1] if image is not None else None)
    if len(explicit) == len(columns):
        return sum(max(0, meters_to_centimeters(base) - meters_to_centimeters(top)) for top, base in explicit.values())
    return sum(_core_column_capacities_cm(columns, image))


def extract_core_column_depths(
    image: np.ndarray,
    columns: list[tuple[int, int, int, int]],
    *,
    reference_interval: tuple[float, float] | None = None,
    expected_intervals: tuple[tuple[float, float], ...] = (),
    metadata: dict[str, str] | None = None,
    preferred_basis: str = "unknown",
) -> tuple[tuple[float, float, float], ...]:
    """Read the printed upper/lower depths beside each core column.

    Report captions sometimes describe the whole sampled interval while one
    page shows only a subset. The per-column ruler labels are more precise for
    masks, especially for short partial columns. OCR is best-effort: if a
    coherent set of labels cannot be recovered, callers retain the existing
    page-interval/ruler calibration.
    """
    if image is None or image.size == 0 or not columns or image.ndim != 3:
        return ()
    try:
        import pytesseract
        from pytesseract import Output

        from .photos import _configure_tesseract

        if not _configure_tesseract(pytesseract):
            return ()
        height, width = image.shape[:2]
        top_edge = min(box[1] for box in columns)
        bottom_edge = max(box[3] for box in columns)
        header_y0 = max(0, top_edge - max(40, round(height * 0.16)))
        header_y1 = min(height, top_edge + max(2, round(height * 0.015)))
        # A short physical fragment can end halfway down the page while its
        # bottom-depth labels remain in the standard footer below the 0–100 cm
        # ruler. Search the page footer, not merely a few pixels below the rock.
        footer_y0 = max(0, min(
            bottom_edge - max(2, round(height * 0.015)), round(height * 0.70),
        ))
        footer_y1 = min(height, max(round(height * 0.98), bottom_edge + round(height * 0.16)))
        if header_y1 <= header_y0 or footer_y1 <= footer_y0:
            return ()

        token_rows = []
        label_tokens = []
        for y0, y1 in ((header_y0, header_y1), (footer_y0, footer_y1)):
            region = cv2.cvtColor(image[y0:y1], cv2.COLOR_BGR2GRAY)
            if region.size == 0:
                continue
            scale = min(2.0, max(1.0, 1000.0 / max(1, region.shape[1])))
            if scale > 1.0:
                region = cv2.resize(
                    region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
                )
            try:
                data = pytesseract.image_to_data(
                    region, lang="rus+eng", config="--psm 11", output_type=Output.DICT,
                )
            except Exception:
                data = pytesseract.image_to_data(
                    region, lang="eng", config="--psm 11", output_type=Output.DICT,
                )
            for index, raw in enumerate(data.get("text", [])):
                value = _depth_label_value(raw)
                try:
                    confidence = float(data["conf"][index])
                    x = (float(data["left"][index]) + float(data["width"][index]) / 2) / scale
                    y = y0 + (float(data["top"][index]) + float(data["height"][index]) / 2) / scale
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
                if value is not None and confidence >= 5:
                    token_rows.append((value, x, y, confidence))
                elif confidence >= 5 and str(raw).strip():
                    label_tokens.append((str(raw), x, y))
        candidate_bases = {}
        candidate_ranges = _pair_column_depth_rows(
            token_rows, columns, width, height,
            label_tokens=label_tokens, candidate_bases=candidate_bases,
        )
        if not candidate_ranges:
            return ()

        valid = []
        for candidate_index, ranges in enumerate(candidate_ranges):
            if len(ranges) != len(columns):
                continue
            ordered = sorted((top, base) for top, base in ranges.values())
            # Overlapping column ranges would duplicate centimetres of core.
            if any(meters_to_centimeters(next_top) < meters_to_centimeters(previous_base)
                   for (_, previous_base), (next_top, _) in zip(ordered, ordered[1:])):
                continue
            spatial = [ranges[index][0] for index in sorted(ranges)]
            if len(spatial) > 1 and not (
                all(a < b for a, b in zip(spatial, spatial[1:]))
                or all(a > b for a, b in zip(spatial, spatial[1:]))
            ):
                continue
            gaps = [abs(next_top - previous_base) for (_, previous_base), (next_top, _) in zip(ordered, ordered[1:])]
            if gaps and max(gaps) > 0.50:
                continue
            page_top, page_base = ordered[0][0], ordered[-1][1]
            if expected_intervals and not any(
                page_top >= expected_top - 0.5 and page_base <= expected_base + 0.5
                and page_base > expected_top and page_top < expected_base
                for expected_top, expected_base in expected_intervals
            ):
                continue
            reference_error = (
                abs(page_top - reference_interval[0]) + abs(page_base - reference_interval[1])
                if reference_interval is not None else 0.0
            )
            basis = candidate_bases.get(candidate_index, "unknown")
            if preferred_basis in {"drilling", "gis"} and basis not in {preferred_basis, "unknown"}:
                continue
            # Labelled drilling values are the default for facies-by-drilling.
            # A caption/reference alone must not silently change coordinates.
            basis_preference = 0 if basis == "drilling" else (1 if basis == "gis" else 2)
            # When the user has entered a page interval, its nearest complete
            # set of physical labels is the best evidence of the coordinate
            # system. Use the drilling-first preference only to break ties.
            valid.append((reference_error, basis_preference, sum(gaps), ranges, basis))
        if not valid:
            return ()
        selected_entry = min(valid, key=lambda item: item[:3])
        selected = selected_entry[3]
        if metadata is not None:
            metadata["depth_basis"] = selected_entry[4]
        return tuple(
            (center / width, selected[index][0], selected[index][1])
            for index, center in sorted(
                enumerate((box[0] + box[2]) / 2 for box in columns), key=lambda item: item[1],
            )
            if index in selected
        )
    except Exception:
        # OCR is an optional enhancement; detection and ruler-only depth
        # calibration must still work when Tesseract is missing or fails.
        return ()


def _depth_label_value(raw: str) -> float | None:
    text = str(raw or "").strip()
    text = text.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"}))
    # Values below 1000 m are real depths too. Only accept a complete token;
    # parse_decimal_value handles Excel/OCR comma and dot decimal marks alike.
    token = text.strip("[](){}|:;").strip()
    token = re.sub(r"\s*[mм]\s*$", "", token, flags=re.IGNORECASE).strip()
    value = parse_decimal_value(token) if re.fullmatch(
        r"(?:\d[\d.,\s\u00a0\u202f'’]*|[.,]\d+)", token,
    ) else None
    if value is not None and 0 <= value < 100000:
        return float(value)
    compact = re.sub(r"\D", "", token)
    if len(compact) == 6 and compact.isdigit():
        value = float(f"{compact[:4]}.{compact[4:]}")
        return value if value < 100000 else None
    return None


def _pair_column_depth_rows(
    token_rows, columns, width: int, height: int, *, label_tokens=(), candidate_bases=None,
):
    centers = [(box[0] + box[2]) / 2 for box in columns]
    minimum_count = max(1, math.ceil(len(columns) * 0.6))
    tolerance = max(3.0, height * 0.006)
    grouped: list[list[tuple[float, float, float, float]]] = []
    for token in sorted(token_rows, key=lambda item: item[2]):
        if not grouped or token[2] - float(np.median([item[2] for item in grouped[-1]])) > tolerance:
            grouped.append([token])
        else:
            grouped[-1].append(token)

    header_limit = min(box[1] for box in columns) + height * 0.015
    footer_limit = max(max(box[3] for box in columns) - height * 0.015, height * 0.70)
    header_rows = []
    footer_rows = []
    for group in grouped:
        row_center = float(np.median([item[2] for item in group]))
        assigned: dict[int, tuple[float, float]] = {}
        for value, x, _y, confidence in group:
            index = min(range(len(centers)), key=lambda item: abs(x - centers[item]))
            lane = columns[index][2] - columns[index][0]
            nearest_gap = min((abs(centers[index] - center) for other, center in enumerate(centers) if other != index), default=lane)
            if abs(x - centers[index]) > max(lane * 1.25, nearest_gap * 0.48):
                continue
            old = assigned.get(index)
            if old is None or confidence > old[1]:
                assigned[index] = (value, confidence)
        vector = {index: item[0] for index, item in assigned.items()}
        if len(vector) < minimum_count:
            continue
        label = " ".join(
            text for text, x, y in sorted(label_tokens, key=lambda item: item[1])
            if abs(y - row_center) <= tolerance * 1.5 and x < min(centers)
        )
        basis = _depth_row_basis(label)
        row = (row_center, vector, basis)
        if row_center <= header_limit:
            header_rows.append(row)
        if row_center >= footer_limit:
            footer_rows.append(row)

    ranges = []
    for top_index, (_top_y, tops, top_basis) in enumerate(header_rows):
        for bottom_index, (_bottom_y, bases, bottom_basis) in enumerate(footer_rows):
            # Report tables print raw and adjusted depths on separate rows.
            # Never take the roof from one coordinate system and the sole
            # from the other (the old Cartesian product did exactly that).
            if top_basis != "unknown" and bottom_basis != "unknown":
                if top_basis != bottom_basis:
                    continue
            elif len(header_rows) != len(footer_rows) or top_index != bottom_index:
                continue
            matched = {
                index: (tops[index], bases[index])
                for index in tops.keys() & bases.keys()
                if 1 <= meters_to_centimeters(bases[index]) - meters_to_centimeters(tops[index]) <= 150
            }
            if len(matched) >= minimum_count:
                if candidate_bases is not None:
                    candidate_bases[len(ranges)] = top_basis if top_basis != "unknown" else bottom_basis
                ranges.append(matched)
    return ranges


def _depth_row_basis(label: str) -> str:
    label = str(label).casefold().replace("ё", "е")
    if re.search(r"увяз|гис|adjust|tied|log depth", label):
        return "gis"
    if re.search(r"по\s+керну|бурен|core depth|depth.*core|drill", label):
        return "drilling"
    return "unknown"


def _column_depths_by_box(columns, column_depths, image_width: int | None):
    if not column_depths or not columns or not image_width or len(column_depths) != len(columns):
        return {}
    result = {}
    if len({item[0] for item in column_depths}) != len(column_depths):
        return {}
    # A one-to-one spatial assignment prevents a single OCR label from being
    # reused for two neighbouring core boxes after a detector change.
    ordered_boxes = sorted(enumerate(columns), key=lambda item: (item[1][0] + item[1][2]) / 2)
    for (index, (left, _top, right, _bottom)), (x_fraction, top, base) in zip(ordered_boxes, sorted(column_depths)):
        center = (left + right) / 2 / image_width
        if abs(x_fraction - center) <= max(0.03, (right - left) / image_width):
            span = meters_to_centimeters(base) - meters_to_centimeters(top)
            if 1 <= span <= 150:
                result[index] = (top, base)
    if len(result) != len(columns):
        return {}
    ranges = sorted(result.values())
    if any(meters_to_centimeters(top) < meters_to_centimeters(previous_base)
           for (_, previous_base), (top, _) in zip(ranges, ranges[1:])):
        return {}
    spatial = [result[index][0] for index, _box in ordered_boxes]
    if len(spatial) > 1 and not (
        all(a < b for a, b in zip(spatial, spatial[1:]))
        or all(a > b for a, b in zip(spatial, spatial[1:]))
    ):
        return {}
    return result


def _partial_column_depths_by_box(columns, column_depths, image_width: int | None):
    """Map any coherent subset of OCR depth labels to unique physical lanes."""
    if not column_depths or not columns or not image_width:
        return {}
    ordered_boxes = sorted(enumerate(columns), key=lambda item: (item[1][0] + item[1][2]) / 2)
    result = {}
    for x_fraction, top, base in column_depths:
        if not 1 <= meters_to_centimeters(base) - meters_to_centimeters(top) <= 150:
            return {}
        center_x = float(x_fraction) * image_width
        index, box = min(
            ordered_boxes,
            key=lambda item: abs(center_x - (item[1][0] + item[1][2]) / 2),
        )
        left, _top, right, _bottom = box
        lane = right - left
        nearest_gap = min((
            abs((left + right) / 2 - (other[0] + other[2]) / 2)
            for other_index, other in ordered_boxes if other_index != index
        ), default=lane)
        if abs(center_x - (left + right) / 2) > max(lane * 1.25, nearest_gap * 0.48):
            return {}
        if index in result:
            return {}
        result[index] = (top, base)
    return result


def _core_column_capacities_cm(
    columns: list[tuple[int, int, int, int]], image: np.ndarray | None = None,
) -> list[int]:
    if not columns:
        return []
    heights = [max(1, bottom - top) for _, top, _, bottom in columns]
    pixels_per_cm = _depth_grid_pixels_per_centimeter(image) if image is not None else None
    if pixels_per_cm is not None:
        measured = _scaled_column_capacities_cm(columns, pixels_per_cm)
        # The printed 0–100 cm ruler is the only reliable reference when the
        # page contains one short or partial core column. Without it, normalizing
        # that lone fragment to itself incorrectly assigns it a full metre.
        if measured and all(1 <= value <= 100 for value in measured):
            return measured
    # The upper median keeps a half-height final lane at 50 cm even when a
    # photograph contains only one full and one partial lane.
    reference_height = max(1, sorted(heights)[len(heights) // 2])
    return [max(1, min(100, round(height * 100 / reference_height))) for height in heights]


def _scaled_column_capacities_cm(columns, pixels_per_cm: float) -> list[int]:
    return [
        max(1, min(100, round(max(1, bottom - top) / pixels_per_cm)))
        for _left, top, _right, bottom in columns
    ]


def _depth_grid_pixels_per_centimeter(image: np.ndarray | None) -> float | None:
    """Estimate centimetres from the repeated 10 cm horizontal ruler lines."""
    if image is None or image.size == 0 or image.ndim != 3:
        return None
    height, width = image.shape[:2]
    if height < 100 or width < 100:
        return None
    scale = min(1.0, 2200.0 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 145)
    minimum_length = max(40, round(width * 0.12))
    segments = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=max(20, round(minimum_length * 0.22)),
        minLineLength=minimum_length,
        maxLineGap=max(8, round(width * 0.025)),
    )
    if segments is None:
        return None
    y_positions: list[float] = []
    for line in segments.reshape(-1, 4):
        x0, y0, x1, y1 = (float(value) for value in line)
        if x1 < x0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        dx, dy = x1 - x0, y1 - y0
        length = float(np.hypot(dx, dy))
        if length < minimum_length or dx <= 0 or abs(dy / dx) > 0.18:
            continue
        center_x = width / 2
        if x0 <= center_x <= x1:
            y = y0 + (center_x - x0) * dy / dx
        else:
            y = (y0 + y1) / 2
        if height * 0.05 <= y <= height * 0.95:
            y_positions.append(y)
    if len(y_positions) < 6:
        return None

    # Hough returns several segments per printed line. Collapse nearby y
    # coordinates before measuring the regular depth-grid spacing.
    y_positions.sort()
    tolerance = max(3.0, height * 0.004)
    clusters: list[list[float]] = []
    for y in y_positions:
        if not clusters or y - clusters[-1][-1] > tolerance:
            clusters.append([y])
        else:
            clusters[-1].append(y)
    centers = [float(np.median(cluster)) for cluster in clusters]
    if len(centers) < 6 or centers[-1] - centers[0] < height * 0.40:
        return None

    gaps = np.diff(centers)
    # Main report grids mark each ten centimetres. Exclude duplicate Hough
    # edges and page-sized gaps before taking the robust repeated spacing.
    plausible = gaps[(gaps >= height * 0.018) & (gaps <= height * 0.18)]
    if len(plausible) < 5:
        return None
    pixels_per_cm = float(np.median(plausible)) / 10.0 / scale
    return pixels_per_cm if 1.0 <= pixels_per_cm <= height * 0.15 else None


def project_matches(
    matches: list[Match],
    *,
    capacities_by_photo: dict[Path, int] | None = None,
    depth_ranges_by_photo: dict[Path, list[tuple[tuple[int, int, int, int], float, float]]] | None = None,
) -> tuple[
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
            if capacities_by_photo is not None:
                capacities_by_photo[photo_path] = 0
            if depth_ranges_by_photo is not None:
                depth_ranges_by_photo[photo_path] = []
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
        if capacities_by_photo is not None:
            capacities_by_photo[photo_path] = core_photo_capacity_centimeters(
                columns, image, photo.column_depths,
            )
        if not photo.has_interval:
            if depth_ranges_by_photo is not None:
                depth_ranges_by_photo[photo_path] = []
            continue
        calibrated = calibrate_core_columns(
            columns, float(photo.top), float(photo.base), image=image,
            column_depths=photo.column_depths,
        )
        if depth_ranges_by_photo is not None:
            depth_ranges_by_photo[photo_path] = calibrated
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
        box = _trim_column_caption_rows(box, candidate)
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


def _trim_column_caption_rows(box, candidate):
    """Trim sparse printed numbers joined to the filled rock by morphology.

    Use unsmoothed occupancy, not the dilated component/projection used for
    discovery. Even a 1-pixel header gap must not move the depth origin above
    the physical core. Only trim a short edge region and only when the bulk
    of the lane is dense, so naturally sparse/fragmented rock is preserved.
    """
    left, top, right, bottom = box
    height, width = candidate.shape
    region = candidate[max(0, top):min(height, bottom), max(0, left):min(width, right)]
    if region.size == 0:
        return box
    occupancy = region.mean(axis=1)
    if float(np.median(occupancy)) < 0.65:
        return box
    dense = occupancy >= 0.68
    minimum_run = max(6, min(20, round(len(occupancy) * 0.025)))
    solid = [(start, end) for start, end in _runs(dense) if end - start >= minimum_run]
    if not solid:
        return box
    first, last = solid[0][0], solid[-1][1]
    maximum_trim = max(12, min(round(height * 0.05), round(len(occupancy) * 0.15)))
    new_top = top + first if first <= maximum_trim else top
    new_bottom = top + last if len(occupancy) - last <= maximum_trim else bottom
    return left, new_top, right, new_bottom


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
