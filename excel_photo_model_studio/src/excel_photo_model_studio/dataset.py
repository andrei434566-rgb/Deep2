from __future__ import annotations

import hashlib
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

def build_dataset(project_dir: Path | Iterable[Path], destination: Path) -> dict:
    raw_projects = [project_dir] if isinstance(project_dir, (str, Path)) else list(project_dir)
    project_dirs = [Path(value).expanduser().resolve(strict=True) for value in raw_projects]
    if not project_dirs:
        raise ValueError("В обучающем каталоге пока нет обработанных скважин.")
    destination = Path(destination).expanduser().absolute()
    if destination.exists():
        raise FileExistsError(f"Папка датасета уже существует: {destination}")
    rows = []
    seen_annotations = set()
    for current_project in project_dirs:
        report_path = current_project / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            blockers = _report_blockers(report)
            if blockers:
                raise ValueError(
                    f"{current_project.name}: датасет заблокирован: " + "; ".join(blockers)
                    + ". Исправьте сопоставление и пересчитайте проект."
                )
        with (current_project / "annotations.csv").open("r", encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source, delimiter=";"):
                if row.get("approved") != "1":
                    continue
                key = (row.get("photo", ""), row.get("annotation_id", ""))
                if key in seen_annotations:
                    continue
                seen_annotations.add(key)
                rows.append(row)
    if not rows:
        raise ValueError("Нет подтверждённых масок. Проверьте previews и подтвердите строки в приложении.")
    by_photo: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        _validate_annotation(row)
        by_photo[row["photo"]].append(row)
    if len(by_photo) < 2:
        raise ValueError("Нужно минимум два разных фото: одно фото нельзя одновременно использовать для train и val.")
    labels = sorted({row["label"] for row in rows}, key=str.casefold)
    class_ids = {label: index for index, label in enumerate(labels)}
    split_by_photo, strategy = _split_sources(by_photo)
    if "val" not in split_by_photo.values():
        raise ValueError("Не удалось выделить независимый val без удаления класса из train. Добавьте фото тех же классов.")
    for split in ("train", "val"):
        (destination / "images" / split).mkdir(parents=True, exist_ok=False)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=False)
        (destination / "crops" / split).mkdir(parents=True, exist_ok=False)
    samples = []
    caption_samples = []
    for photo_index, (photo_name, annotations) in enumerate(sorted(by_photo.items()), start=1):
        photo = Path(photo_name).resolve(strict=True)
        split = split_by_photo[photo_name]
        photo_bytes = photo.read_bytes()
        digest = hashlib.sha256(photo_bytes).hexdigest()
        stem = f"sample_{photo_index:06d}_{digest[:10]}"
        target_image = destination / "images" / split / f"{stem}{photo.suffix.lower()}"
        target_image.write_bytes(photo_bytes)
        lines = []
        source_image = cv2.imdecode(np.frombuffer(photo_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if source_image is None:
            raise ValueError(f"Не удалось декодировать подтверждённое фото: {photo}")
        for annotation_index, row in enumerate(annotations, start=1):
            polygon = json.loads(row["polygon_json"])
            width, height = float(row["image_width"]), float(row["image_height"])
            coords = " ".join(
                f"{max(0.0, min(1.0, float(x) / width)):.6f} {max(0.0, min(1.0, float(y) / height)):.6f}"
                for x, y in polygon
            )
            lines.append(f"{class_ids[row['label']]} {coords}")
            target_text = row.get("target_text", "").strip()
            if target_text:
                xs = [float(point[0]) for point in polygon]
                ys = [float(point[1]) for point in polygon]
                x0, x1 = max(0, int(min(xs))), min(source_image.shape[1], int(max(xs)) + 1)
                y0, y1 = max(0, int(min(ys))), min(source_image.shape[0], int(max(ys)) + 1)
                crop = source_image[y0:y1, x0:x1]
                if crop.size == 0:
                    raise ValueError(f"Пустая подтверждённая вырезка: {row.get('annotation_id', '')}")
                crop_path = destination / "crops" / split / f"{stem}_{annotation_index:03d}.jpg"
                ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 96])
                if not ok:
                    raise ValueError(f"Не удалось сохранить вырезку: {row.get('annotation_id', '')}")
                crop_path.write_bytes(encoded.tobytes())
                caption_samples.append({
                    "annotation_id": row.get("annotation_id", ""), "split": split,
                    "crop": str(crop_path.relative_to(destination).as_posix()),
                    "source_photo": str(photo), "source_sha256": digest,
                    "well": row.get("well", ""), "depth_top": float(row["depth_top"]),
                    "depth_base": float(row["depth_base"]), "facies": row["label"],
                    "association": row.get("association", ""), "environment": row.get("environment", ""),
                    "field_name": row.get("field_name", ""),
                    "target_text": target_text, "source_file": row.get("source_file", ""),
                    "source_sheet": row.get("source_sheet", ""), "source_row": row.get("source_row", ""),
                })
        label_path = destination / "labels" / split / f"{stem}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        samples.append({
            "source_photo": str(photo), "source_sha256": digest, "split": split,
            "image": str(target_image.relative_to(destination)),
            "label": str(label_path.relative_to(destination)),
            "well": annotations[0].get("well", ""), "annotation_count": len(annotations),
        })
    caption_path = destination / "caption_dataset.jsonl"
    caption_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in caption_samples), encoding="utf-8"
    )
    yaml_path = destination / "data.yaml"
    yaml_path.write_text("\n".join((
        f"path: {json.dumps(destination.as_posix(), ensure_ascii=False)}",
        "train: images/train", "val: images/val", f"nc: {len(labels)}", "names:",
        *(f"  {index}: {json.dumps(label, ensure_ascii=False)}" for index, label in enumerate(labels)), "",
    )), encoding="utf-8")
    manifest = {
        "schema": "excel-photo-yolo-seg-v1", "created_at": datetime.now().isoformat(timespec="seconds"),
        "project": str(project_dirs[0]) if len(project_dirs) == 1 else "",
        "projects": [str(path) for path in project_dirs], "project_count": len(project_dirs),
        "data_yaml": str(yaml_path), "class_names": labels,
        "photo_count": len(by_photo), "annotation_count": len(rows), "split_strategy": strategy,
        "caption_count": len(caption_samples),
        "train_caption_count": sum(item["split"] == "train" for item in caption_samples),
        "val_caption_count": sum(item["split"] == "val" for item in caption_samples),
        "caption_dataset": str(caption_path),
        "train_photo_count": sum(value == "train" for value in split_by_photo.values()),
        "val_photo_count": sum(value == "val" for value in split_by_photo.values()),
        "class_counts": dict(Counter(row["label"] for row in rows)), "samples": samples,
    }
    (destination / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "output_dir": str(destination)}


def _validate_annotation(row: dict[str, str]) -> None:
    try:
        polygon = json.loads(row["polygon_json"])
        width, height = int(row["image_width"]), int(row["image_height"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Повреждена маска {row.get('annotation_id', '')}.") from exc
    if not row.get("label", "").strip() or len(polygon) < 3 or width < 2 or height < 2:
        raise ValueError(f"Некорректная подтверждённая маска {row.get('annotation_id', '')}.")
    if not row.get("target_text", "").strip():
        raise ValueError(
            f"У подтверждённой маски {row.get('annotation_id', '')} нет обязательного краткого описания."
        )
    if any(len(point) != 2 or not (0 <= float(point[0]) <= width and 0 <= float(point[1]) <= height) for point in polygon):
        raise ValueError(f"Маска выходит за границы фото: {row.get('annotation_id', '')}.")


def _report_blockers(report: dict) -> list[str]:
    values = (
        (
            "photos_without_intervals",
            int(report.get("photos_without_intervals", report.get("unconfirmed_photos", 0)) or 0),
            "фото без обязательного интервала",
        ),
        (
            "photos_without_core_columns",
            int(report.get("photos_without_core_columns", 0) or 0),
            "фото без распознанного керна",
        ),
        (
            "photos_without_masks",
            int(report.get("photos_without_masks", 0) or 0),
            "фото без масок в датасете",
        ),
        (
            "uncovered_facies_intervals",
            int(report.get("uncovered_facies_intervals", 0) or 0),
            "участков керна без фации",
        ),
        (
            "uncovered_description_intervals",
            int(report.get("uncovered_description_intervals", 0) or 0),
            "участков керна без краткого описания",
        ),
        (
            "uncovered_excel_core_intervals",
            int(report.get("uncovered_excel_core_intervals", 0) or 0),
            "интервалов керна Excel без фотографии",
        ),
        (
            "facies_rows_without_description",
            int(report.get("facies_rows_without_description", 0) or 0),
            "строк фаций без краткого описания",
        ),
        (
            "invalid_thickness_rows",
            int(report.get("invalid_thickness_rows", 0) or 0),
            "строк с ошибкой толщины фации",
        ),
    )
    blockers = [f"{label}: {count}" for _key, count, label in values if count]
    annotations = int(report.get("annotations", 0) or 0)
    approved = int(report.get("approved_annotations", 0) or 0)
    if annotations and approved < annotations:
        blockers.append(f"неподтверждённых масок: {annotations - approved}")
    blocking_errors = int(report.get("blocking_errors", 0) or 0)
    if blocking_errors and not blockers:
        blockers.append(f"других критических ошибок проекта: {blocking_errors}")
    return blockers


def _split_sources(by_photo: dict[str, list[dict[str, str]]]) -> tuple[dict[str, str], str]:
    photo_labels = {photo: Counter(row["label"] for row in rows) for photo, rows in by_photo.items()}
    totals = sum(photo_labels.values(), Counter())
    by_well: dict[str, list[str]] = defaultdict(list)
    for photo, rows in by_photo.items():
        well = rows[0].get("well", "").strip().casefold()
        if well:
            by_well[well].append(photo)
    groups = by_well if len(by_well) >= 2 and sum(len(items) for items in by_well.values()) == len(by_photo) else {photo: [photo] for photo in by_photo}
    strategy = "well" if groups is by_well else "photo"
    target = max(1, round(len(by_photo) * 0.2))
    candidates = []
    for group_name, photos in groups.items():
        counts = sum((photo_labels[photo] for photo in photos), Counter())
        if any(totals[label] <= count for label, count in counts.items()):
            continue
        score = (abs(len(photos) - target), len(photos), group_name)
        candidates.append((score, photos))
    if not candidates:
        return ({photo: "train" for photo in by_photo}, strategy)
    _, validation_photos = min(candidates, key=lambda item: item[0])
    selected = set(validation_photos)
    return ({photo: ("val" if photo in selected else "train") for photo in by_photo}, strategy)
