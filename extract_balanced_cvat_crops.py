"""Create a balanced crop dataset from a CVAT COCO export.

Usage:
    python extract_balanced_cvat_crops.py
or:
    python extract_balanced_cvat_crops.py path/to/cvat_coco_export.zip

The archive must be exported from CVAT as COCO 1.0 with images included.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np


@dataclass(frozen=True)
class AnnotationRecord:
    annotation_id: int
    image_id: int
    category_id: int
    segmentation: object
    bbox: tuple[float, float, float, float] | None


class CocoZipDataset:
    """Read the images and polygon annotations from a CVAT COCO ZIP export."""

    def __init__(self, archive_path: Path) -> None:
        self.archive_path = archive_path
        self.archive = zipfile.ZipFile(archive_path)
        self.names = [name for name in self.archive.namelist() if not name.endswith("/")]
        self.annotation_name, self.payload = self._load_coco_payload()
        self.images = {int(item["id"]): item for item in self.payload.get("images", [])}
        self.categories = {int(item["id"]): str(item["name"]) for item in self.payload.get("categories", [])}
        self._image_names_by_basename = self._index_images()
        self._external_images_by_basename: dict[str, list[Path]] = {}
        self._external_images_root: Path | None = None

    def close(self) -> None:
        self.archive.close()

    def _load_coco_payload(self) -> tuple[str, dict]:
        candidates = [name for name in self.names if name.lower().endswith(".json")]
        for name in candidates:
            try:
                payload = json.loads(self.archive.read(name).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if {"images", "annotations", "categories"}.issubset(payload):
                return name, payload
        raise ValueError("COCO JSON не найден. Экспортируйте задачу CVAT в формате COCO 1.0.")

    def _index_images(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for name in self.names:
            if PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                result[PurePosixPath(name).name].append(name)
        return result

    @property
    def has_embedded_images(self) -> bool:
        return bool(self._image_names_by_basename)

    def set_external_images_dir(self, directory: Path) -> None:
        if not directory.is_dir():
            raise ValueError(f"Папка с исходными изображениями не найдена: {directory}")
        index: dict[str, list[Path]] = defaultdict(list)
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                index[path.name].append(path)
        if not index:
            raise ValueError("В указанной папке не найдены изображения.")
        self._external_images_root = directory
        self._external_images_by_basename = index

    def read_image(self, file_name: str) -> np.ndarray | None:
        normalised = file_name.replace("\\", "/").lstrip("/")
        candidates = [normalised, f"images/default/{normalised}", f"images/{normalised}"]
        for candidate in candidates:
            if candidate in self.names:
                return self._decode_image(self.archive.read(candidate))

        matches = self._image_names_by_basename.get(PurePosixPath(normalised).name, [])
        if len(matches) == 1:
            return self._decode_image(self.archive.read(matches[0]))
        if self._external_images_root is not None:
            exact = self._external_images_root / Path(normalised)
            if exact.is_file():
                return self._decode_image(exact.read_bytes())
            disk_matches = self._external_images_by_basename.get(PurePosixPath(normalised).name, [])
            if len(disk_matches) == 1:
                return self._decode_image(disk_matches[0].read_bytes())
        return None

    @staticmethod
    def _decode_image(data: bytes) -> np.ndarray | None:
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def polygon_lists(segmentation: object) -> list[list[float]]:
    """Keep valid COCO polygon rings."""
    if not isinstance(segmentation, list):
        return []
    rings: list[list[float]] = []
    for ring in segmentation:
        if not isinstance(ring, list) or len(ring) < 6 or len(ring) % 2:
            continue
        try:
            values = [float(value) for value in ring]
        except (TypeError, ValueError):
            continue
        rings.append(values)
    return rings


def valid_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x, y, width, height = (float(part) for part in value)
    except (TypeError, ValueError):
        return None
    return (x, y, width, height) if width > 1 and height > 1 else None


def supported_segmentation(segmentation: object, bbox: tuple[float, float, float, float] | None) -> bool:
    """COCO can represent a shape as polygon, RLE mask, or only a bbox."""
    if polygon_lists(segmentation):
        return True
    if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
        return True
    return bbox is not None


def eligible_annotations(dataset: CocoZipDataset) -> dict[int, list[AnnotationRecord]]:
    grouped: dict[int, list[AnnotationRecord]] = defaultdict(list)
    for item in dataset.payload.get("annotations", []):
        category_id = int(item.get("category_id", -1))
        image_id = int(item.get("image_id", -1))
        if category_id not in dataset.categories or image_id not in dataset.images:
            continue
        segmentation = item.get("segmentation")
        bbox = valid_bbox(item.get("bbox"))
        if not supported_segmentation(segmentation, bbox):
            continue
        grouped[category_id].append(
            AnnotationRecord(
                annotation_id=int(item.get("id", -1)),
                image_id=image_id,
                category_id=category_id,
                segmentation=segmentation,
                bbox=bbox,
            )
        )
    return grouped


def raw_annotation_counts(dataset: CocoZipDataset) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for item in dataset.payload.get("annotations", []):
        category_id = int(item.get("category_id", -1))
        if category_id in dataset.categories:
            counts[category_id] += 1
    return counts


def choose_categories(
    categories: dict[int, str],
    grouped: dict[int, list[AnnotationRecord]],
    raw_counts: dict[int, int],
) -> list[int]:
    ordered = sorted(categories, key=lambda category_id: categories[category_id].casefold())
    print("\nДоступные классы с полигонами:")
    for number, category_id in enumerate(ordered, start=1):
        usable = len(grouped.get(category_id, []))
        raw = raw_counts.get(category_id, 0)
        print(f"  [{number}] {categories[category_id]} — пригодно: {usable}; всего аннотаций: {raw}")

    while True:
        answer = input("\nВведите номера нужных классов через запятую или all: ").strip().lower()
        if answer == "all":
            return [category_id for category_id in ordered if grouped.get(category_id)]
        try:
            positions = [int(value.strip()) for value in answer.split(",") if value.strip()]
            selected = list(dict.fromkeys(ordered[position - 1] for position in positions))
        except (ValueError, IndexError):
            print("Не удалось прочитать выбор. Пример: 1,3,5 или all.")
            continue
        selected = [category_id for category_id in selected if grouped.get(category_id)]
        if selected:
            return selected
        print("У выбранных классов нет пригодных полигонов.")


def read_positive_int(prompt: str, default: int) -> int:
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()
        if not answer:
            return default
        try:
            value = int(answer)
            if value > 0:
                return value
        except ValueError:
            pass
        print("Введите положительное целое число.")


def read_format() -> str:
    while True:
        answer = input("Формат фрагментов png/jpg [png]: ").strip().lower() or "png"
        if answer in {"png", "jpg", "jpeg"}:
            return "jpg" if answer == "jpeg" else answer
        print("Допустимые варианты: png или jpg.")


def safe_folder_name(name: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(". ")
    return clean or "unnamed"


def diversified_sample(records: list[AnnotationRecord], limit: int, rng: random.Random) -> list[AnnotationRecord]:
    """Round-robin sample, avoiding a dataset made from only one source photo."""
    by_image: dict[int, list[AnnotationRecord]] = defaultdict(list)
    for record in records:
        by_image[record.image_id].append(record)
    buckets = list(by_image.values())
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)

    selected: list[AnnotationRecord] = []
    while buckets and len(selected) < limit:
        next_round: list[list[AnnotationRecord]] = []
        for bucket in buckets:
            if len(selected) >= limit:
                break
            selected.append(bucket.pop())
            if bucket:
                next_round.append(bucket)
        buckets = next_round
    return selected


def annotation_mask(record: AnnotationRecord, image_shape: tuple[int, int, int]) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    contours = []
    for ring in polygon_lists(record.segmentation):
        points = np.asarray(ring, dtype=np.float32).reshape(-1, 2)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        contours.append(np.round(points).astype(np.int32))
    if contours:
        cv2.fillPoly(mask, contours, 255)
        return mask
    if isinstance(record.segmentation, dict):
        decoded = decode_coco_rle(record.segmentation, height, width)
        if decoded is not None:
            return decoded
    if record.bbox is not None:
        x, y, box_width, box_height = record.bbox
        x1 = int(np.clip(round(x), 0, width - 1))
        y1 = int(np.clip(round(y), 0, height - 1))
        x2 = int(np.clip(round(x + box_width), x1 + 1, width))
        y2 = int(np.clip(round(y + box_height), y1 + 1, height))
        cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
    return mask


def decode_coco_rle(rle: dict, image_height: int, image_width: int) -> np.ndarray | None:
    """Decode COCO RLE without requiring pycocotools.

    CVAT uses RLE when an annotation is a mask, or when its `is_crowd` option is
    enabled. COCO stores mask pixels in Fortran column-major order.
    """
    size = rle.get("size")
    counts = rle.get("counts")
    if not isinstance(size, list) or len(size) != 2:
        return None
    try:
        height, width = int(size[0]), int(size[1])
    except (TypeError, ValueError):
        return None
    if height <= 0 or width <= 0:
        return None
    runs = decode_rle_counts(counts)
    if runs is None:
        return None
    flat = np.zeros(height * width, dtype=np.uint8)
    position = 0
    value = 0
    for run in runs:
        if run < 0:
            return None
        end = min(position + run, flat.size)
        if value:
            flat[position:end] = 255
        position = end
        value = 1 - value
        if position >= flat.size:
            break
    if position < flat.size:
        return None
    decoded = flat.reshape((height, width), order="F")
    if decoded.shape != (image_height, image_width):
        decoded = cv2.resize(decoded, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    return decoded


def decode_rle_counts(counts: object) -> list[int] | None:
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

    # COCO compressed-RLE decoder, compatible with pycocotools encoding.
    output: list[int] = []
    position = 0
    while position < len(counts):
        value = 0
        shift = 0
        more = True
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


def crop_one_annotation(
    image: np.ndarray,
    record: AnnotationRecord,
    margin_percent: float,
    mask_outside: bool,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    mask = annotation_mask(record, image.shape)
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    if width < 4 or height < 4:
        return None

    margin = round(max(width, height) * margin_percent / 100.0)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + width + margin)
    y2 = min(image.shape[0], y + height + margin)
    crop = image[y1:y2, x1:x2].copy()

    if mask_outside:
        crop_mask = mask[y1:y2, x1:x2]
        # Neutral background prevents pixels from adjacent facies entering the class crop.
        crop[crop_mask == 0] = (127, 127, 127)
    return crop, (x1, y1, x2 - x1, y2 - y1)


def write_crop(path: Path, image: np.ndarray, image_format: str) -> bool:
    extension = ".png" if image_format == "png" else ".jpg"
    options = [cv2.IMWRITE_PNG_COMPRESSION, 3] if image_format == "png" else [cv2.IMWRITE_JPEG_QUALITY, 95]
    ok, encoded = cv2.imencode(extension, image, options)
    if ok:
        encoded.tofile(str(path))
    return bool(ok)


def main() -> int:
    print("=" * 64)
    print("CVAT COCO → сбалансированные фрагменты по классам")
    print("=" * 64)
    raw_path = sys.argv[1] if len(sys.argv) > 1 else input("Путь к ZIP-экспорту CVAT COCO: ").strip().strip('"')
    archive_path = Path(raw_path)
    if not archive_path.is_file():
        print("Файл не найден.")
        return 1

    try:
        dataset = CocoZipDataset(archive_path)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Не удалось прочитать экспорт: {error}")
        return 1

    try:
        grouped = eligible_annotations(dataset)
        raw_counts = raw_annotation_counts(dataset)
        if not grouped:
            print("В экспорте нет полигонов. Нужна разметка Polygon или Mask, экспортированная в COCO 1.0.")
            return 1
        if not dataset.has_embedded_images:
            print("\nВ ZIP нет исходных фотографий.")
            raw_images_dir = input("Укажите папку, где лежат исходные изображения CVAT: ").strip().strip('"')
            try:
                dataset.set_external_images_dir(Path(raw_images_dir))
            except ValueError as error:
                print(error)
                return 1
        selected_categories = choose_categories(dataset.categories, grouped, raw_counts)
        target = read_positive_int("Сколько фрагментов нужно на каждый выбранный класс", 50)
        image_format = read_format()
        margin_percent = read_positive_int("Поле вокруг полигона в процентах (0 = только контур)", 0)
        mask_answer = input("Закрасить всё вне полигона нейтральным фоном? y/n [y]: ").strip().lower()
        mask_outside = mask_answer not in {"n", "no", "нет"}
        output_default = archive_path.with_name(f"{archive_path.stem}_balanced_crops")
        output_path = Path(input(f"Папка результата [{output_default}]: ").strip().strip('"') or output_default)
        if output_path.exists():
            print(f"Папка уже существует: {output_path}\nВыберите новую пустую папку, чтобы ничего не перезаписать.")
            return 1

        output_path.mkdir(parents=True)
        rng = random.Random(20260809)
        report_rows: list[dict[str, object]] = []
        metadata_rows: list[dict[str, object]] = []
        image_cache: dict[int, np.ndarray | None] = {}

        for folder_index, category_id in enumerate(selected_categories, start=1):
            category_name = dataset.categories[category_id]
            available = grouped[category_id]
            planned = diversified_sample(available, target, rng)
            folder = output_path / f"{folder_index:02d}_{safe_folder_name(category_name)}"
            folder.mkdir()
            written = 0
            skipped = 0

            for order, record in enumerate(planned, start=1):
                if record.image_id not in image_cache:
                    image_data = dataset.images[record.image_id]
                    image_cache[record.image_id] = dataset.read_image(str(image_data.get("file_name", "")))
                image = image_cache[record.image_id]
                if image is None:
                    skipped += 1
                    continue
                prepared = crop_one_annotation(image, record, margin_percent, mask_outside)
                if prepared is None:
                    skipped += 1
                    continue
                crop, (x, y, width, height) = prepared
                source_name = safe_folder_name(Path(str(dataset.images[record.image_id].get("file_name", "image"))).stem)
                crop_path = folder / f"{safe_folder_name(category_name)}_{order:04d}_{source_name}_ann{record.annotation_id}.{image_format}"
                if not write_crop(crop_path, crop, image_format):
                    skipped += 1
                    continue
                written += 1
                metadata_rows.append(
                    {
                        "crop_file": crop_path.relative_to(output_path).as_posix(),
                        "class_name": category_name,
                        "class_id": category_id,
                        "source_image": dataset.images[record.image_id].get("file_name", ""),
                        "annotation_id": record.annotation_id,
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                    }
                )
            report_rows.append(
                {
                    "folder": folder.name,
                    "class_name": category_name,
                    "available_polygons": len(available),
                    "requested": target,
                    "written": written,
                    "shortage": max(0, target - written),
                    "skipped": skipped,
                }
            )
            print(f"{category_name}: записано {written} из {target}; доступно полигонов: {len(available)}")

        with (output_path / "selection_report.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(report_rows[0]))
            writer.writeheader()
            writer.writerows(report_rows)
        with (output_path / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as file:
            headers = ["crop_file", "class_name", "class_id", "source_image", "annotation_id", "x", "y", "width", "height"]
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(metadata_rows)

        print("\nГотово.")
        print(f"Фрагменты: {output_path}")
        print("selection_report.csv показывает, каких классов не хватило до заданного числа.")
        print("metadata.csv хранит связь каждого фрагмента с исходным фото и полигоном CVAT.")
        return 0
    finally:
        dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
