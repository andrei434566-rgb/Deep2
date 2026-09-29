from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .depth import centimeters_to_meters, meters_to_centimeters
from .matching import sort_photo_records
from .models import COLUMN_ORDER_RIGHT_TO_LEFT
from .photos import discover_photos, enrich_core_column_depths
from .standard_excel import export_standardized_workbook
from .vision import calibrate_core_columns, detect_column_order, detect_core_columns, read_image


def analyze_photos_to_excel(
    model_path: Path,
    photos_dir: Path,
    destination: Path,
    *,
    confidence: float = 0.25,
) -> dict:
    """Apply a YOLO segmentation model and map class metadata into the Excel output."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Для анализа нужны PyTorch и пакет ultralytics.") from exc
    model_path = Path(model_path).expanduser().resolve(strict=True)
    photos_dir = Path(photos_dir).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().absolute()
    if model_path.suffix.lower() != ".pt":
        raise ValueError("Выберите созданный этой системой файл best.pt.")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("Порог уверенности должен быть между 0 и 1.")

    contract_path = model_path.with_name("model_contract.json")
    metadata_path = model_path.with_name("class_metadata.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.is_file() else {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    classes = metadata.get("classes") or contract.get("classes") or []
    facies_reference = {
        int(item["class_id"]): item for item in classes
        if isinstance(item, dict) and str(item.get("class_id", "")).isdigit()
    }
    description_generator = None
    description_model_warning = ""
    description_model_path = model_path.with_name("description_best.pt")
    if description_model_path.is_file():
        try:
            from .description_model import DescriptionGenerator
            description_generator = DescriptionGenerator(description_model_path)
        except Exception as exc:
            description_model_warning = str(exc)

    records = discover_photos(photos_dir, use_ocr=True)
    records = enrich_core_column_depths(records)
    if not records:
        raise ValueError("В выбранной папке не найдены фотографии.")
    unresolved = [record.path.name for record in records if not record.has_interval]
    if unresolved:
        raise ValueError(
            "Нельзя экспортировать неполную скважину: не определена глубина фото: "
            + "; ".join(unresolved)
        )
    records = sort_photo_records(records)
    visual_model = YOLO(str(model_path))
    output_rows: list[dict] = []
    skipped_photos: list[str] = []
    problems: list[str] = []
    for record in records:
            image = read_image(record.path)
            height, width = image.shape[:2]
            columns = sorted(detect_core_columns(image), key=lambda box: (box[0], box[1]))
            if not columns:
                skipped_photos.append(record.path.name)
                problems.append(f"{record.path.name}: не найден керн")
                continue
            order = detect_column_order(image, columns)
            if order == COLUMN_ORDER_RIGHT_TO_LEFT:
                columns.reverse()
            calibrated = calibrate_core_columns(
                columns, float(record.top), float(record.base), image=image,
                column_depths=record.column_depths,
            )
            if len(calibrated) != len(columns):
                problems.append(f"{record.path.name}: не все столбики получили глубину")
                continue
            calibration_gaps = _uncovered_intervals(
                float(record.top), float(record.base), [(top, base) for _, top, base in calibrated],
            )
            if calibration_gaps:
                problems.append(f"{record.path.name}: глубина колонок не покрывает интервал фото {calibration_gaps}")
                continue
            results = visual_model.predict(
                source=str(record.path), conf=float(confidence), retina_masks=True,
                save=False, verbose=False, max_det=3000,
            )
            result = results[0] if results else None
            if result is None or result.masks is None or result.boxes is None:
                skipped_photos.append(record.path.name)
                problems.append(f"{record.path.name}: модель не распознала фации")
                continue
            polygons = list(result.masks.xy)
            class_ids = [int(value) for value in result.boxes.cls.detach().cpu().tolist()]
            confidences = [float(value) for value in result.boxes.conf.detach().cpu().tolist()]
            photo_rows = []
            for prediction_index, (polygon, class_id, score) in enumerate(
                zip(polygons, class_ids, confidences), start=1,
            ):
                points = np.asarray(polygon, dtype=np.float32)
                if points.ndim != 2 or points.shape[0] < 3:
                    continue
                class_name = _class_name(visual_model.names, class_id)
                reference = facies_reference.get(class_id, {})
                facies_index = str(reference.get("facies_index") or class_name)
                facies_name = str(reference.get("facies_name") or class_name)
                # A prediction can cross column lanes. Clip and map each physical
                # piece separately; never assign a ruler/background mask to the
                # nearest column merely because it has a similar y coordinate.
                for column_index, (box, depth_top, depth_base) in enumerate(calibrated):
                    clipped = _clip_polygon_to_box(points, box)
                    if clipped.shape[0] < 3 or abs(cv2.contourArea(clipped)) < 1:
                        continue
                    depth_interval = polygon_depth_interval(
                        clipped, [box], depth_top, depth_base,
                        calibrated_columns=[(box, depth_top, depth_base)],
                    )
                    if depth_interval is None:
                        continue
                    photo_rows.append({
                    "field_name": reference.get("field_name", ""),
                    "well": record.well,
                    "core_top": float(record.top),
                    "core_base": float(record.base),
                    "facies_top": depth_interval[0],
                    "facies_base": depth_interval[1],
                    "facies_index": facies_index,
                    "facies_name": facies_name,
                    "association": reference.get("association", ""),
                    "environment": reference.get("environment", ""),
                    "description": reference.get("default_description", ""),
                    "confidence": score,
                    "source_photo": str(record.path),
                    "column_index": column_index,
                    "prediction_index": prediction_index,
                    "depth_basis": getattr(record, "depth_basis", "unknown"),
                    })
            resolved_rows = []
            for column_index, (box, depth_top, depth_base) in enumerate(calibrated):
                candidates = [row for row in photo_rows if row["column_index"] == column_index]
                gaps = _uncovered_intervals(depth_top, depth_base, [
                    (row["facies_top"], row["facies_base"]) for row in candidates
                ])
                if gaps:
                    problems.append(f"{record.path.name}, столбик {column_index + 1}: фации не покрывают {gaps}")
                    continue
                for row in _resolve_prediction_overlaps(candidates):
                    left, top, right, bottom = box
                    scale = (bottom - top) / (depth_base - depth_top)
                    y0 = max(0, int(np.floor(top + (row["facies_top"] - depth_top) * scale)))
                    y1 = min(height, int(np.ceil(top + (row["facies_base"] - depth_top) * scale)))
                    crop = image[y0:y1, max(0, left):min(width, right)]
                    if crop.size == 0:
                        problems.append(f"{record.path.name}: пустая вырезка интервала")
                        continue
                    row["_interval_crop"] = _compact_description_crop(crop)
                    if not row["description"]:
                        row["description"] = row["facies_name"]
                    resolved_rows.append(row)
            photo_rows = sorted(resolved_rows, key=lambda row: (row["facies_top"], row["facies_base"]))
            output_rows.extend(photo_rows)
    if problems:
        raise ValueError("Неполный результат не экспортирован. " + "; ".join(problems))
    if not output_rows:
        raise ValueError("Модель не нашла ни одного интервала фаций на выбранных фотографиях.")
    output_rows.sort(key=lambda row: (str(row.get("well", "")).casefold(), row["facies_top"], row["facies_base"]))
    generated_description_count, fallback_description_count = _apply_interval_descriptions(
        output_rows, description_generator,
    )
    layer_counts: dict[str, int] = {}
    for row in output_rows:
        well = row["well"]
        layer_counts[well] = layer_counts.get(well, 0) + 1
        row["layer_no"] = layer_counts[well]
        if row.get("depth_basis") == "gis":
            # A GIS-labelled photograph does not provide drilling depths. Do
            # not put adjusted values under the drilling header in the report.
            row["thickness"] = round(row["facies_base"] - row["facies_top"], 2)
            for key in ("facies_top", "facies_base", "core_top", "core_base"):
                row["gis_" + key] = row.pop(key)
    export_standardized_workbook(output_rows, destination)
    return {
        "schema": "kern-standard-excel-inference-v1",
        "model": str(model_path),
        "photos": len(records),
        "rows": len(output_rows),
        "skipped_photos": skipped_photos,
        "class_metadata_available": bool(facies_reference),
        "description_model_available": description_generator is not None,
        "description_model_warning": description_model_warning,
        "generated_descriptions": generated_description_count,
        "fallback_descriptions": fallback_description_count,
        "description_policy": (
            "Краткое описание генерируется отдельной символьной моделью из вырезки интервала, "
            "индекса фации и длины полного непрерывного интервала. Части, продолжающиеся на соседнем фото, "
            "получают один и тот же текст; при низкой уверенности используется подтверждённый пример класса."
            if description_generator is not None
            else "Текстовая модель отсутствует; используется подтверждённое описание класса из метаданных."
        ),
        "output_excel": str(destination),
        "target_headers": {
            "facies_index": "Индекс фации",
            "facies_name": "Название фации",
            "target_text": "Краткое описание",
        },
    }


def _apply_interval_descriptions(rows: list[dict], generator) -> tuple[int, int]:
    """Generate once for a full same-facies interval, including page/column continuations."""
    groups: list[list[dict]] = []
    for row in rows:
        if not groups:
            groups.append([row])
            continue
        previous = groups[-1][-1]
        same_photo = str(previous.get("source_photo", "")) == str(row.get("source_photo", ""))
        same_prediction = previous.get("prediction_index") == row.get("prediction_index")
        continues = (
            str(previous.get("well", "")).casefold() == str(row.get("well", "")).casefold()
            and str(previous.get("facies_index", "")).casefold() == str(row.get("facies_index", "")).casefold()
            and meters_to_centimeters(previous["facies_base"]) == meters_to_centimeters(row["facies_top"])
            and (not same_photo or same_prediction)
        )
        if continues:
            groups[-1].append(row)
        else:
            groups.append([row])

    generated_rows = 0
    for group in groups:
        fallback = next((str(row.get("description", "")).strip() for row in group if str(row.get("description", "")).strip()), "")
        if not fallback:
            fallback = str(group[0].get("facies_name", "")).strip()
        description = fallback
        if generator is not None:
            # The generator sees the clearest reviewed/predicted interval crop,
            # while its numeric condition is the total depth span of all pages.
            crop = max(
                (row.get("_interval_crop") for row in group if row.get("_interval_crop") is not None),
                key=lambda image: image.shape[0] * image.shape[1], default=None,
            )
            full_thickness = max(0.0, float(group[-1]["facies_base"]) - float(group[0]["facies_top"]))
            if crop is not None:
                candidate, confidence = generator.generate_with_confidence(
                    crop, facies=group[0].get("facies_index", ""),
                    facies_name=group[0].get("facies_name", ""), interval_m=full_thickness,
                )
                if candidate and confidence >= 0.10:
                    description = candidate
                    generated_rows += len(group)
        for row in group:
            row["description"] = description
            row.pop("_interval_crop", None)
    return generated_rows, len(rows) - generated_rows


def _compact_description_crop(image: np.ndarray, max_side: int = 256) -> np.ndarray:
    """Bound memory while retaining the crop's original aspect ratio and texture."""
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(1, width, height))
    if scale >= 1.0:
        return image.copy()
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def polygon_depth_interval(
    polygon: np.ndarray,
    columns: list[tuple[int, int, int, int]],
    photo_top: float,
    photo_base: float,
    image: np.ndarray | None = None,
    calibrated_columns: list[tuple[tuple[int, int, int, int], float, float]] | None = None,
) -> tuple[float, float] | None:
    """Convert one predicted mask to depth using the ordered core columns."""
    polygon = np.asarray(polygon, dtype=np.float32)
    if (polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2
            or not np.isfinite(polygon).all() or not columns or photo_base <= photo_top):
        return None
    calibrated = calibrated_columns if calibrated_columns is not None else calibrate_core_columns(
        columns, photo_top, photo_base, image=image,
    )
    intersections = [_clip_polygon_to_box(polygon, box) for box, _, _ in calibrated]
    areas = [abs(cv2.contourArea(points)) if len(points) >= 3 else 0.0 for points in intersections]
    if not areas or max(areas) <= 0:
        return None
    column_index = int(np.argmax(areas))
    polygon = intersections[column_index]
    (left, column_top, right, column_bottom), column_depth_top, column_depth_base = calibrated[column_index]
    del left, right
    y0 = max(float(column_top), float(np.min(polygon[:, 1])))
    y1 = min(float(column_bottom), float(np.max(polygon[:, 1])))
    if y1 <= y0:
        return None
    column_span_cm = meters_to_centimeters(column_depth_base) - meters_to_centimeters(column_depth_top)
    pixel_span = max(1.0, float(column_bottom - column_top))
    depth_top_cm = meters_to_centimeters(column_depth_top) + round(
        (y0 - column_top) * column_span_cm / pixel_span
    )
    depth_base_cm = meters_to_centimeters(column_depth_top) + round(
        (y1 - column_top) * column_span_cm / pixel_span
    )
    if depth_base_cm <= depth_top_cm:
        depth_base_cm = min(meters_to_centimeters(column_depth_base), depth_top_cm + 1)
    return centimeters_to_meters(depth_top_cm), centimeters_to_meters(depth_base_cm)


def _clip_polygon_to_box(polygon: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Sutherland–Hodgman clipping; valid also for concave segmentation outlines."""
    points = [np.asarray(point, dtype=np.float32) for point in polygon]
    for axis, boundary, keep_greater in ((0, box[0], True), (0, box[2], False),
                                         (1, box[1], True), (1, box[3], False)):
        if not points:
            break
        clipped = []
        previous = points[-1]
        previous_inside = previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
        for current in points:
            current_inside = current[axis] >= boundary if keep_greater else current[axis] <= boundary
            if current_inside != previous_inside:
                ratio = (boundary - previous[axis]) / (current[axis] - previous[axis])
                clipped.append(previous + ratio * (current - previous))
            if current_inside:
                clipped.append(current)
            previous, previous_inside = current, current_inside
        points = clipped
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def _uncovered_intervals(top: float, base: float, intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cursor, stop = meters_to_centimeters(top), meters_to_centimeters(base)
    gaps = []
    for first, last in sorted(intervals):
        first, last = min(stop, max(cursor, meters_to_centimeters(first))), min(stop, meters_to_centimeters(last))
        if last <= cursor:
            continue
        if first > cursor:
            gaps.append((centimeters_to_meters(cursor), centimeters_to_meters(first)))
        cursor = max(cursor, last)
    if cursor < stop:
        gaps.append((centimeters_to_meters(cursor), centimeters_to_meters(stop)))
    return gaps


def _resolve_prediction_overlaps(rows: list[dict]) -> list[dict]:
    """Highest-confidence prediction owns each centimetre; never duplicate depth."""
    boundaries = sorted({meters_to_centimeters(row[key]) for row in rows for key in ("facies_top", "facies_base")})
    result = []
    for top, base in zip(boundaries, boundaries[1:]):
        candidates = [row for row in rows if meters_to_centimeters(row["facies_top"]) <= top
                      and meters_to_centimeters(row["facies_base"]) >= base]
        if not candidates:
            continue
        winner = max(candidates, key=lambda row: (row["confidence"], -row["prediction_index"]))
        top_m, base_m = centimeters_to_meters(top), centimeters_to_meters(base)
        if result and result[-1]["prediction_index"] == winner["prediction_index"] and result[-1]["facies_base"] == top_m:
            result[-1]["facies_base"] = base_m
        else:
            result.append({**winner, "facies_top": top_m, "facies_base": base_m})
    return result


def _class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)
