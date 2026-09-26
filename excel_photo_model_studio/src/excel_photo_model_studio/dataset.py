from __future__ import annotations

import hashlib
import csv
import json
import math
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
            _verify_validation_snapshot(current_project, report)
        elif (current_project / "project.json").is_file():
            raise ValueError(f"{current_project.name}: нет отчёта проверки; пересчитайте проект.")
        with (current_project / "annotations.csv").open("r", encoding="utf-8-sig", newline="") as source:
            project_rows = list(csv.DictReader(source, delimiter=";"))
            if any(row.get("approved") != "1" for row in project_rows):
                raise ValueError(f"{current_project.name}: есть неподтверждённые маски. Нельзя обучать на части скважины.")
            if report_path.is_file() and len(project_rows) != int(report.get("annotations", len(project_rows))):
                raise ValueError(f"{current_project.name}: число масок не совпадает с проверенным отчётом; пересчитайте проект.")
            for row in project_rows:
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
    content_ids = {}
    for photo_name in by_photo:
        with Path(photo_name).open("rb") as source:
            content_ids[photo_name] = hashlib.file_digest(source, "sha256").hexdigest()
    split_by_photo, strategy = _split_sources(by_photo, content_ids)
    if "val" not in split_by_photo.values():
        raise ValueError("Не удалось выделить независимый val без удаления класса из train. Добавьте разные фото тех же классов; копии одного фото не являются независимой проверкой.")
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
        if digest != content_ids[photo_name]:
            raise ValueError(f"Фото изменилось во время сборки датасета: {photo}")
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
            if (int(height), int(width)) != source_image.shape[:2]:
                raise ValueError(f"Размер фото изменился после создания масок: {photo}. Пересчитайте проект.")
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
    points = np.asarray(polygon, dtype=np.float32)
    if not np.isfinite(points).all() or abs(cv2.contourArea(points)) < 0.5:
        raise ValueError(f"Пустая или некорректная площадь маски: {row.get('annotation_id', '')}.")
    top, base = float(row["depth_top"]), float(row["depth_base"])
    if not math.isfinite(top) or not math.isfinite(base) or base <= top:
        raise ValueError(f"Некорректный метраж маски: {row.get('annotation_id', '')}.")


def _verify_validation_snapshot(project: Path, report: dict) -> None:
    snapshot = report.get("validation_snapshot")
    if not snapshot:
        if (project / "project.json").is_file():
            raise ValueError(f"{project.name}: проект проверен старой версией. Пересчитайте перед обучением.")
        return
    for name, expected in snapshot.get("files", {}).items():
        path = Path(name)
        if not path.is_file() or [path.stat().st_size, path.stat().st_mtime_ns] != expected:
            raise ValueError(f"После проверки изменился файл {path.name}. Пересчитайте проект перед обучением.")
    from .photos import IMAGE_EXTENSIONS
    current_photos = sorted(
        str(path.resolve()) for path in Path(snapshot["photos_dir"]).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if current_photos != snapshot.get("photos"):
        raise ValueError("Состав папки фото изменился после проверки. Пересчитайте проект: нельзя пропускать новые фотографии.")


def _report_blockers(report: dict) -> list[str]:
    values = (
        ("projection_errors", int(report.get("projection_errors", 0) or 0), "ошибок наложения масок на керн"),
        ("facies_rows_with_incomplete_masks", int(report.get("facies_rows_with_incomplete_masks", 0) or 0), "фаций не полностью покрытых масками"),
        ("excel_facies_rows_without_photo_match", int(report.get("excel_facies_rows_without_photo_match", 0) or 0), "фаций Excel без фото"),
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


def _split_sources(by_photo: dict[str, list[dict[str, str]]], content_ids: dict[str, str] | None = None) -> tuple[dict[str, str], str]:
    photo_labels = {photo: Counter(row["label"] for row in rows) for photo, rows in by_photo.items()}
    totals = sum(photo_labels.values(), Counter())
    by_well: dict[str, list[str]] = defaultdict(list)
    for photo, rows in by_photo.items():
        well = rows[0].get("well", "").strip().casefold()
        if well:
            by_well[well].append(photo)
    target = max(1, round(len(by_photo) * 0.2))
    can_group_by_well = len(by_well) >= 2 and sum(len(items) for items in by_well.values()) == len(by_photo)
    groups = by_well if can_group_by_well else {photo: [photo] for photo in by_photo}
    strategy = "well" if can_group_by_well else "photo"

    def validation_candidates(grouped_photos: dict[str, list[str]]) -> list[tuple[tuple, list[str]]]:
        # Connected components of well membership and identical-content links.
        # Compute once, rather than rescanning thousands of photos per candidate.
        parent = {photo: photo for photo in by_photo}

        def find(photo):
            while parent[photo] != photo:
                parent[photo] = parent[parent[photo]]
                photo = parent[photo]
            return photo

        def unite(left, right):
            parent[find(right)] = find(left)

        for photos in grouped_photos.values():
            for photo in photos[1:]:
                unite(photos[0], photo)
        first_by_digest = {}
        for photo, digest in (content_ids or {}).items():
            if digest in first_by_digest:
                unite(first_by_digest[digest], photo)
            else:
                first_by_digest[digest] = photo
        components = defaultdict(list)
        for photo in by_photo:
            components[find(photo)].append(photo)
        candidates = []
        for group_name, photos in components.items():
            # Hold copies of an image on the same side even if filenames or
            # project/well names differ. Otherwise validation memorizes train.
            photos = sorted(photos)
            counts = sum((photo_labels[photo] for photo in photos), Counter())
            if any(totals[label] <= count for label, count in counts.items()):
                continue
            score = (abs(len(photos) - target), len(photos), group_name)
            candidates.append((score, photos))
        return candidates

    candidates = validation_candidates(groups)
    # Prefer holding out whole wells. If that leaves no class-complete validation
    # split, fall back to whole photos while still keeping every class in training.
    if not candidates and can_group_by_well:
        strategy = "photo_fallback"
        candidates = validation_candidates({photo: [photo] for photo in by_photo})
    if not candidates:
        return ({photo: "train" for photo in by_photo}, strategy)
    _, validation_photos = min(candidates, key=lambda item: item[0])
    selected = set(validation_photos)
    return ({photo: ("val" if photo in selected else "train") for photo in by_photo}, strategy)
