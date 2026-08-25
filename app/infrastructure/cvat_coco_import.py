"""Import a CVAT COCO ZIP export into a DeepCore project.

The importer intentionally accepts only the common COCO export produced by
CVAT: an instances JSON annotation file plus the source images. Imported
polygons are treated as human-verified outlines; the 16 descriptive fields
remain empty until a geologist completes them in DeepCore.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPixmap

from app.domain.models import FaciesDetection, PhotoRecord


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def import_cvat_coco_zip(archive_path: Path, images_destination: Path) -> tuple[list[PhotoRecord], dict[str, int]]:
    """Extract images and translate CVAT's COCO polygons to project records."""
    if archive_path.suffix.casefold() != ".zip":
        raise ValueError("Для импорта CVAT выберите ZIP-архив в формате COCO.")

    with zipfile.ZipFile(archive_path) as archive:
        annotation_member = _find_annotations_member(archive)
        payload = json.loads(archive.read(annotation_member).decode("utf-8-sig"))
        categories = {
            int(item["id"]): str(item.get("name") or f"Класс {item['id']}")
            for item in payload.get("categories", [])
            if item.get("id") is not None
        }
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            image_id = annotation.get("image_id")
            if image_id is not None:
                annotations_by_image[int(image_id)].append(annotation)

        images_destination.mkdir(parents=True, exist_ok=True)
        records: list[PhotoRecord] = []
        imported_contours = 0
        for position, image_info in enumerate(payload.get("images", []), start=1):
            image_id = int(image_info.get("id", position))
            source_name = str(image_info.get("file_name") or "")
            image_member = _find_image_member(archive, source_name)
            if image_member is None:
                continue
            file_name = _safe_output_name(position, source_name)
            output_path = images_destination / file_name
            output_path.write_bytes(archive.read(image_member))
            pixmap = QPixmap(str(output_path))
            if pixmap.isNull():
                output_path.unlink(missing_ok=True)
                continue

            detections: list[FaciesDetection] = []
            for annotation in annotations_by_image.get(image_id, []):
                polygon = _annotation_polygon(annotation)
                category = categories.get(int(annotation.get("category_id", -1)), "Без класса")
                if len(polygon) < 3:
                    continue
                detections.append(
                    FaciesDetection(
                        label=category,
                        confidence=1.0,
                        polygon=polygon,
                        training_ready=True,
                    )
                )
                imported_contours += 1
            records.append(
                PhotoRecord(
                    identifier=f"cvat_{image_id}_{output_path.stem}",
                    path=str(output_path),
                    pixmap=pixmap,
                    detections=detections,
                )
            )
    if not records:
        raise ValueError("В архиве не найдены пригодные изображения COCO.")
    return records, {
        "images": len(records),
        "contours": imported_contours,
        "classes": len(categories),
    }


def _find_annotations_member(archive: zipfile.ZipFile) -> str:
    candidates = [
        member
        for member in archive.namelist()
        if PurePosixPath(member).name.casefold().endswith(".json")
        and ("instances" in member.casefold() or "annotation" in member.casefold())
    ]
    if not candidates:
        raise ValueError("В ZIP не найден файл COCO-аннотаций instances*.json.")
    return sorted(candidates, key=lambda value: (0 if "instances" in value.casefold() else 1, len(value)))[0]


def _find_image_member(archive: zipfile.ZipFile, file_name: str) -> str | None:
    normalized = file_name.replace("\\", "/").lstrip("/")
    exact = [member for member in archive.namelist() if member.replace("\\", "/").endswith(normalized)]
    if exact:
        return min(exact, key=len)
    base_name = PurePosixPath(normalized).name.casefold()
    matches = [
        member
        for member in archive.namelist()
        if PurePosixPath(member).suffix.casefold() in IMAGE_SUFFIXES
        and PurePosixPath(member).name.casefold() == base_name
    ]
    return min(matches, key=len) if matches else None


def _safe_output_name(position: int, source_name: str) -> str:
    candidate = PurePosixPath(source_name).name or f"image_{position:05d}.jpg"
    stem = re.sub(r"[^\w.-]+", "_", Path(candidate).stem, flags=re.UNICODE).strip("._") or "image"
    suffix = Path(candidate).suffix.casefold()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".jpg"
    return f"{position:05d}_{stem}{suffix}"


def _annotation_polygon(annotation: dict) -> list[QPointF]:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        candidates = [part for part in segmentation if isinstance(part, list) and len(part) >= 6]
        if candidates:
            # A FaciesDetection stores one editable polygon. CVAT polygons are
            # normally single-part; for a multi-part annotation keep the
            # largest part for safe manual review.
            values = max(candidates, key=lambda part: len(part))
            return [QPointF(float(values[index]), float(values[index + 1])) for index in range(0, len(values) - 1, 2)]
    bbox = annotation.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        x, y, width, height = (float(value) for value in bbox[:4])
        return [QPointF(x, y), QPointF(x + width, y), QPointF(x + width, y + height), QPointF(x, y + height)]
    return []
