from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .description_model import DescriptionGenerator
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
    description_model: Path | None = None,
    confidence: float = 0.25,
) -> dict:
    """Apply the unified model and export the stable 22-column well description."""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Для анализа нужны PyTorch и ultralytics.") from exc
    model_path = Path(model_path).expanduser().resolve(strict=True)
    photos_dir = Path(photos_dir).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().absolute()
    if model_path.suffix.lower() != ".pt":
        raise ValueError("Выберите созданный этой системой файл best.pt.")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("Порог уверенности должен быть между 0 и 1.")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    contract = checkpoint.get("core_model_contract", {}) if isinstance(checkpoint, dict) else {}
    if not contract:
        contract_path = model_path.with_name("model_contract.json")
        if contract_path.is_file():
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
    embedded_description = checkpoint.get("core_description_checkpoint") if isinstance(checkpoint, dict) else None
    standalone = Path(description_model).expanduser().resolve(strict=True) if description_model else None
    sibling = model_path.with_name("description_best.pt")
    if standalone is None and embedded_description is None and sibling.is_file():
        standalone = sibling
    if standalone is None and embedded_description is None:
        raise ValueError("В best.pt нет модели столбца 22 и рядом не найден description_best.pt.")

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
    text_model = DescriptionGenerator(standalone if standalone is not None else embedded_description)
    facies_reference = contract.get("facies_reference", {}) if isinstance(contract, dict) else {}
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
                facies = _class_name(visual_model.names, class_id)
                reference = facies_reference.get(facies, {}) if isinstance(facies_reference, dict) else {}
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
                    "facies_name": facies,
                    "association": reference.get("association", ""),
                    "environment": reference.get("environment", ""),
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
                    row["description"] = text_model.generate(crop, facies=row["facies_name"])
                    if not row["description"]:
                        problems.append(f"{record.path.name}: модель выдала пустое краткое описание")
                        continue
                    resolved_rows.append(row)
            photo_rows = sorted(resolved_rows, key=lambda row: (row["facies_top"], row["facies_base"]))
            output_rows.extend(photo_rows)
    if problems:
        raise ValueError("Неполный результат не экспортирован. " + "; ".join(problems))
    if not output_rows:
        raise ValueError("Модель не нашла ни одного интервала фаций на выбранных фотографиях.")
    output_rows.sort(key=lambda row: (str(row.get("well", "")).casefold(), row["facies_top"], row["facies_base"]))
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
        "output_excel": str(destination),
        "target_columns": 22,
        "facies_column": 19,
        "description_column": 22,
    }


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
