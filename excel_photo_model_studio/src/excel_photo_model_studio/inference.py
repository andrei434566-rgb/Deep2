from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .description_model import generate_description
from .models import COLUMN_ORDER_RIGHT_TO_LEFT
from .photos import discover_photos
from .standard_excel import export_standardized_workbook
from .vision import detect_column_order, detect_core_columns, read_image


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

    records = [record for record in discover_photos(photos_dir) if record.has_interval]
    if not records:
        raise ValueError("Не найдены фотографии с интервалами в именах файлов.")
    visual_model = YOLO(str(model_path))
    facies_reference = contract.get("facies_reference", {}) if isinstance(contract, dict) else {}
    output_rows: list[dict] = []
    skipped_photos: list[str] = []
    with tempfile.TemporaryDirectory(prefix="excel_photo_inference_") as temporary_directory:
        temporary = Path(temporary_directory)
        if standalone is None:
            standalone = temporary / "embedded_description.pt"
            torch.save(embedded_description, standalone)
        for record in records:
            image = read_image(record.path)
            height, width = image.shape[:2]
            columns = sorted(detect_core_columns(image), key=lambda box: (box[0], box[1]))
            if not columns:
                skipped_photos.append(record.path.name)
                continue
            order = detect_column_order(image, columns)
            if order == COLUMN_ORDER_RIGHT_TO_LEFT:
                columns.reverse()
            results = visual_model.predict(
                source=str(record.path), conf=float(confidence), retina_masks=True,
                save=False, verbose=False,
            )
            result = results[0] if results else None
            if result is None or result.masks is None or result.boxes is None:
                skipped_photos.append(record.path.name)
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
                depth_interval = polygon_depth_interval(
                    points, columns, float(record.top), float(record.base),
                )
                if depth_interval is None:
                    continue
                facies = _class_name(visual_model.names, class_id)
                x0 = max(0, int(np.floor(points[:, 0].min())))
                x1 = min(width, int(np.ceil(points[:, 0].max())) + 1)
                y0 = max(0, int(np.floor(points[:, 1].min())))
                y1 = min(height, int(np.ceil(points[:, 1].max())) + 1)
                crop = image[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                crop_path = temporary / f"{record.path.stem}_{prediction_index:04d}.jpg"
                ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 96])
                if not ok:
                    continue
                crop_path.write_bytes(encoded.tobytes())
                description = generate_description(standalone, crop_path, facies=facies)
                reference = facies_reference.get(facies, {}) if isinstance(facies_reference, dict) else {}
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
                    "description": description,
                    "confidence": score,
                    "source_photo": str(record.path),
                })
            photo_rows.sort(key=lambda row: (row["facies_top"], row["facies_base"]))
            for layer_number, row in enumerate(photo_rows, start=1):
                row["layer_no"] = layer_number
            output_rows.extend(photo_rows)
    if not output_rows:
        raise ValueError("Модель не нашла ни одного интервала фаций на выбранных фотографиях.")
    output_rows.sort(key=lambda row: (str(row.get("well", "")).casefold(), row["facies_top"], row["facies_base"]))
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
) -> tuple[float, float] | None:
    """Convert one predicted mask to depth using the ordered core columns."""
    if polygon.size == 0 or not columns or photo_base <= photo_top:
        return None
    x_center = float(np.mean(polygon[:, 0]))
    column_index = min(
        range(len(columns)),
        key=lambda index: _horizontal_distance(x_center, columns[index][0], columns[index][2]),
    )
    left, column_top, right, column_bottom = columns[column_index]
    del left, right
    y0 = max(float(column_top), float(np.min(polygon[:, 1])))
    y1 = min(float(column_bottom), float(np.max(polygon[:, 1])))
    if y1 <= y0:
        return None
    visual_length = sum(max(1, bottom - top) for _, top, _, bottom in columns)
    depth_per_pixel = (photo_base - photo_top) / visual_length
    previous_pixels = sum(max(1, bottom - top) for _, top, _, bottom in columns[:column_index])
    depth_top = photo_top + (previous_pixels + y0 - column_top) * depth_per_pixel
    depth_base = photo_top + (previous_pixels + y1 - column_top) * depth_per_pixel
    return round(depth_top, 4), round(depth_base, 4)


def _horizontal_distance(x: float, left: int, right: int) -> float:
    if left <= x <= right:
        return 0.0
    return min(abs(x - left), abs(x - right))


def _class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)
