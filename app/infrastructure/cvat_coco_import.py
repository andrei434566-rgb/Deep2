"""Import CVAT COCO exports into a Kern Analyzer project.

The importer accepts ordinary polygon/bbox annotations and CVAT mask exports
(COCO RLE). Images may be embedded in the ZIP or supplied separately: the
latter is common when a CVAT server exports annotations only.
"""

from __future__ import annotations

import json
import re
import zipfile
import hashlib
from collections import defaultdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PySide6.QtCore import QPointF

from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.image_loading import load_working_pixmap, source_image_size


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_BATCH_IMAGES = 2500
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 8 * 1024**3


class CvatImagesMissingError(ValueError):
    """Raised when an annotation-only export needs the original image folder."""


def import_cvat_coco_zip(
    archive_path: Path,
    images_destination: Path,
    external_images_dir: Path | None = None,
) -> tuple[list[PhotoRecord], dict[str, int]]:
    """Extract a COCO export and translate its masks to editable detections."""
    if archive_path.suffix.casefold() != ".zip":
        raise ValueError("Для импорта CVAT выберите ZIP-архив в формате COCO.")
    if external_images_dir is not None and not external_images_dir.is_dir():
        raise ValueError("Указанная папка с исходными изображениями недоступна.")

    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.namelist() if not member.endswith("/")]
        annotation_member, payload = _load_coco_payload(archive, members)
        categories = {
            int(item["id"]): str(item.get("name") or f"Класс {item['id']}")
            for item in payload.get("categories", [])
            if item.get("id") is not None
        }
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            try:
                image_id = int(annotation.get("image_id"))
            except (TypeError, ValueError):
                continue
            annotations_by_image[image_id].append(annotation)

        disk_images = _index_external_images(external_images_dir) if external_images_dir else {}
        images_destination.mkdir(parents=True, exist_ok=True)
        records: list[PhotoRecord] = []
        imported_contours = skipped_annotations = 0
        missing_images: list[str] = []
        unreadable_images: list[str] = []
        for position, image_info in enumerate(payload.get("images", []), start=1):
            try:
                image_id = int(image_info.get("id", position))
            except (TypeError, ValueError):
                image_id = position
            source_name = str(image_info.get("file_name") or "")
            image_bytes = _read_image_bytes(archive, members, source_name, external_images_dir, disk_images)
            if image_bytes is None:
                missing_images.append(source_name or f"image id {image_id}")
                continue
            output_path = images_destination / _safe_output_name(position, source_name)
            output_path.write_bytes(image_bytes)
            pixmap = load_working_pixmap(output_path)
            source_size = source_image_size(output_path)
            if pixmap.isNull() or not source_size.isValid():
                output_path.unlink(missing_ok=True)
                unreadable_images.append(source_name or output_path.name)
                continue
            x_scale = pixmap.width() / max(1, source_size.width())
            y_scale = pixmap.height() / max(1, source_size.height())

            detections: list[FaciesDetection] = []
            for annotation in annotations_by_image.get(image_id, []):
                try:
                    category_id = int(annotation.get("category_id", -1))
                except (TypeError, ValueError):
                    category_id = -1
                polygons = _annotation_polygons(annotation, source_size.height(), source_size.width())
                if not polygons:
                    skipped_annotations += 1
                    continue
                for polygon in polygons:
                    detections.append(FaciesDetection(
                        label=categories.get(category_id, "Без класса"),
                        confidence=1.0,
                        polygon=[QPointF(x * x_scale, y * y_scale) for x, y in polygon],
                        training_ready=True,
                    ))
                    imported_contours += 1
            records.append(PhotoRecord(
                identifier=f"cvat_{image_id}_{output_path.stem}",
                path=str(output_path), pixmap=pixmap, detections=detections,
            ))

    if not records:
        if missing_images and external_images_dir is None:
            raise CvatImagesMissingError(
                "В ZIP есть COCO-аннотации, но нет исходных изображений. "
                "Выберите папку с JPG/PNG исходной задачи CVAT."
            )
        details = []
        if missing_images:
            details.append(f"не найдены изображения: {len(missing_images)}")
        if unreadable_images:
            details.append(f"не удалось открыть: {len(unreadable_images)}")
        raise ValueError("В архиве не найдены пригодные изображения COCO" + (f" ({'; '.join(details)})." if details else "."))
    return records, {
        "images": len(records), "contours": imported_contours, "classes": len(categories),
        "missing_images": len(missing_images), "unreadable_images": len(unreadable_images),
        "skipped_annotations": skipped_annotations, "annotation_file": annotation_member,
    }


def import_cvat_coco_zips(
    archive_paths: list[Path],
    images_destination: Path,
    external_images_dir: Path | None = None,
    progress=None,
) -> tuple[list[PhotoRecord], dict[str, int]]:
    """Import several CVAT jobs sequentially without unpacking them all at once.

    Each archive gets its own on-disk folder, avoiding filename collisions.
    Exact duplicate source images are skipped by content hash.  The explicit
    limits protect the desktop UI from a corrupt ZIP or an impractical number
    of QPixmaps; users can split a larger export into projects.
    """
    paths = [Path(path) for path in archive_paths]
    if not paths:
        raise ValueError("Не выбраны ZIP-архивы CVAT.")
    total_size = 0
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Не найден архив: {path}")
        with zipfile.ZipFile(path) as archive:
            unpacked = sum(item.file_size for item in archive.infolist())
        if unpacked > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ValueError(f"{path.name}: распакованный объём больше 8 ГБ. Разделите экспорт на меньшие части.")
        total_size += unpacked

    all_records: list[PhotoRecord] = []
    aggregate = defaultdict(int)
    seen_hashes: set[str] = set()
    for index, path in enumerate(paths, start=1):
        if progress:
            progress(index - 1, len(paths), f"CVAT {index}/{len(paths)}: {path.name}")
        archive_folder = images_destination / f"job_{index:03d}_{_safe_folder_name(path.stem)}"
        records, summary = import_cvat_coco_zip(path, archive_folder, external_images_dir)
        for key, value in summary.items():
            if isinstance(value, int):
                aggregate[key] += value
        retained: list[PhotoRecord] = []
        for record in records:
            digest = _file_sha256(Path(record.path))
            if digest in seen_hashes:
                Path(record.path).unlink(missing_ok=True)
                aggregate["duplicate_images"] += 1
                continue
            seen_hashes.add(digest)
            record.identifier = f"cvat_job{index}_{record.identifier}"
            retained.append(record)
        all_records.extend(retained)
        if len(all_records) > MAX_BATCH_IMAGES:
            raise ValueError(
                f"Импортировано больше {MAX_BATCH_IMAGES} уникальных фото. "
                "Чтобы не переполнить память приложения, разделите ZIP на несколько проектов."
            )
        if progress:
            progress(index, len(paths), f"CVAT {index}/{len(paths)}: готово · фото {len(all_records)}")
    if not all_records:
        raise ValueError("После удаления дубликатов не осталось пригодных фото.")
    aggregate["images"] = len(all_records)
    aggregate["archives"] = len(paths)
    return all_records, dict(aggregate)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_folder_name(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._") or "archive"


def _load_coco_payload(archive: zipfile.ZipFile, members: list[str]) -> tuple[str, dict]:
    """Locate COCO JSON by structure, not merely by a particular CVAT name."""
    for member in sorted((item for item in members if item.casefold().endswith(".json")), key=len):
        try:
            payload = json.loads(archive.read(member).decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and {"images", "annotations", "categories"}.issubset(payload):
            return member, payload
    raise ValueError("В ZIP не найден COCO JSON. В CVAT выберите экспорт «COCO 1.0».")


def _read_image_bytes(archive, members, source_name, external_images_dir, disk_images) -> bytes | None:
    member = _find_image_member(members, source_name)
    if member is not None:
        return archive.read(member)
    if external_images_dir is None:
        return None
    normalized = source_name.replace("\\", "/").lstrip("/")
    direct_path = external_images_dir / Path(normalized)
    if direct_path.is_file():
        return direct_path.read_bytes()
    matches = disk_images.get(PurePosixPath(normalized).name.casefold(), [])
    return matches[0].read_bytes() if len(matches) == 1 else None


def _index_external_images(directory: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES:
            index[path.name.casefold()].append(path)
    return index


def _find_image_member(members: list[str], file_name: str) -> str | None:
    normalized = file_name.replace("\\", "/").lstrip("/")
    normalized_members = {member.replace("\\", "/"): member for member in members}
    for candidate in (normalized, f"images/{normalized}", f"images/default/{normalized}"):
        if candidate in normalized_members:
            return normalized_members[candidate]
    exact = [member for member in members if member.replace("\\", "/").endswith(normalized)]
    if exact:
        return min(exact, key=len)
    base_name = PurePosixPath(normalized).name.casefold()
    matches = [
        member for member in members
        if PurePosixPath(member).suffix.casefold() in IMAGE_SUFFIXES
        and PurePosixPath(member).name.casefold() == base_name
    ]
    return matches[0] if len(matches) == 1 else None


def _safe_output_name(position: int, source_name: str) -> str:
    candidate = PurePosixPath(source_name).name or f"image_{position:05d}.jpg"
    stem = re.sub(r"[^\w.-]+", "_", Path(candidate).stem, flags=re.UNICODE).strip("._") or "image"
    suffix = Path(candidate).suffix.casefold()
    return f"{position:05d}_{stem}{suffix if suffix in IMAGE_SUFFIXES else '.jpg'}"


def _annotation_polygons(annotation: dict, image_height: int, image_width: int) -> list[list[tuple[float, float]]]:
    """Convert polygon, bbox, and RLE COCO shapes to editable polygons."""
    segmentation = annotation.get("segmentation")
    polygons: list[list[tuple[float, float]]] = []
    if isinstance(segmentation, list):
        for part in segmentation:
            if not isinstance(part, list) or len(part) < 6 or len(part) % 2:
                continue
            try:
                polygons.append([(float(part[index]), float(part[index + 1])) for index in range(0, len(part), 2)])
            except (TypeError, ValueError):
                continue
    elif isinstance(segmentation, dict):
        mask = _decode_coco_rle(segmentation, image_height, image_width)
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons.extend(
                [(float(point[0][0]), float(point[0][1])) for point in contour]
                for contour in contours
                if len(contour) >= 3 and cv2.contourArea(contour) >= 2
            )
    if polygons:
        return polygons
    bbox = annotation.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            x, y, width, height = (float(value) for value in bbox[:4])
        except (TypeError, ValueError):
            return []
        if width > 0 and height > 0:
            return [[(x, y), (x + width, y), (x + width, y + height), (x, y + height)]]
    return []


def _decode_coco_rle(rle: dict, image_height: int, image_width: int) -> np.ndarray | None:
    size, counts = rle.get("size"), rle.get("counts")
    if not isinstance(size, list) or len(size) != 2:
        return None
    try:
        height, width = int(size[0]), int(size[1])
    except (TypeError, ValueError):
        return None
    if height <= 0 or width <= 0:
        return None
    runs = _decode_rle_counts(counts)
    if runs is None:
        return None
    flat = np.zeros(height * width, dtype=np.uint8)
    position, value = 0, 0
    for run in runs:
        if run < 0:
            return None
        end = min(position + run, flat.size)
        if value:
            flat[position:end] = 255
        position, value = end, 1 - value
        if position >= flat.size:
            break
    if position < flat.size:
        return None
    mask = flat.reshape((height, width), order="F")
    return mask if mask.shape == (image_height, image_width) else cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)


def _decode_rle_counts(counts: object) -> list[int] | None:
    if isinstance(counts, list):
        try:
            return [int(value) for value in counts]
        except (TypeError, ValueError):
            return None
    if isinstance(counts, bytes):
        try:
            counts = counts.decode("ascii")
        except UnicodeDecodeError:
            return None
    if not isinstance(counts, str):
        return None
    output: list[int] = []
    position = 0
    while position < len(counts):
        value, shift, more = 0, 0, True
        while more:
            if position >= len(counts):
                return None
            code = ord(counts[position]) - 48
            position += 1
            value |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            shift += 1
            if shift > 13:
                return None
        if code & 0x10:
            value |= -1 << (5 * shift)
        if len(output) > 2:
            value += output[-2]
        output.append(value)
    return output
