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

from .paths import resolve_existing_path

def build_dataset(project_dir: Path | Iterable[Path], destination: Path) -> dict:
    raw_projects = [project_dir] if isinstance(project_dir, (str, Path)) else list(project_dir)
    project_dirs = [resolve_existing_path(value) for value in raw_projects]
    if not project_dirs:
        raise ValueError("В обучающем каталоге пока нет обработанных скважин.")
    destination = Path(destination).expanduser().absolute()
    if destination.exists():
        raise FileExistsError(f"Папка датасета уже существует: {destination}")
    rows = []
    seen_annotations = set()
    source_target_headers = []
    for current_project in project_dirs:
        cache_manifest = {}
        cache_manifest_path = current_project / "cache_manifest.json"
        if cache_manifest_path.is_file():
            try:
                cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cache_manifest = {}
        is_approved_snapshot = (
            cache_manifest.get("schema") == "confirmed-well-cache-v2"
            and cache_manifest.get("confirmation_scope") == "approved_annotations"
        )
        report_path = current_project / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            blockers = [] if is_approved_snapshot else _report_blockers(report)
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
            if is_approved_snapshot:
                if len(project_rows) != int(cache_manifest.get("annotation_count", -1)):
                    raise ValueError(f"{current_project.name}: число подтверждённых масок не совпадает с кэшем.")
                cached_photos = {str(Path(row["photo"]).expanduser().resolve()) for row in project_rows}
                if len(cached_photos) != int(cache_manifest.get("photo_count", -1)):
                    raise ValueError(f"{current_project.name}: список фото не совпадает с кэшем подтверждённых масок.")
            if report_path.is_file() and len(project_rows) != int(report.get("annotations", len(project_rows))):
                raise ValueError(f"{current_project.name}: число масок не совпадает с проверенным отчётом; пересчитайте проект.")
            if (current_project / "project.json").is_file():
                source_target_headers.extend(
                    _restore_facies_targets_from_excel(current_project, project_rows)
                )
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
    facies_statistics = _fresh_facies_statistics(rows)
    for facies in facies_statistics:
        matching_rows = [
            row for row in rows
            if row.get("facies_index", "").casefold() == str(facies["facies_index"]).casefold()
        ]
        descriptions = Counter(
            " ".join(str(row.get("target_text", "")).split())
            for row in matching_rows if str(row.get("target_text", "")).strip()
        )
        facies["description_examples"] = [value for value, _count in descriptions.most_common(5)]
        facies["default_description"] = descriptions.most_common(1)[0][0] if descriptions else ""
        for field in ("association", "environment"):
            values = Counter(
                " ".join(str(row.get(field, "")).split())
                for row in matching_rows if str(row.get(field, "")).strip()
            )
            facies[field] = values.most_common(1)[0][0] if values else ""
    by_photo: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        _validate_annotation(row)
        by_photo[row["photo"]].append(row)
    labels = [item["facies_index"] for item in facies_statistics]
    class_ids = {label: index for index, label in enumerate(labels)}
    content_ids = {}
    for photo_name in by_photo:
        with Path(photo_name).open("rb") as source:
            content_ids[photo_name] = hashlib.file_digest(source, "sha256").hexdigest()
    split_by_photo, strategy = _split_sources(by_photo, content_ids)
    # Preserve every reviewed interval in the export even before an independent
    # validation set exists. train_model deliberately refuses to start until a
    # real, non-leaking val split is available.
    for split in ("train", "val"):
        (destination / "images" / split).mkdir(parents=True, exist_ok=False)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=False)
        (destination / "crops" / split).mkdir(parents=True, exist_ok=False)
    samples = []
    caption_samples = []
    for photo_index, (photo_name, annotations) in enumerate(sorted(by_photo.items()), start=1):
        photo = resolve_existing_path(photo_name)
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
            lines.append(f"{class_ids[row['facies_index']]} {coords}")
            target_text = " ".join(str(row.get("target_text", "")).split())
            if target_text:
                points = np.asarray(polygon, dtype=np.float32)
                xs, ys = points[:, 0], points[:, 1]
                x0, x1 = max(0, int(np.floor(xs.min()))), min(source_image.shape[1], int(np.ceil(xs.max())) + 1)
                y0, y1 = max(0, int(np.floor(ys.min()))), min(source_image.shape[0], int(np.ceil(ys.max())) + 1)
                crop = source_image[y0:y1, x0:x1].copy()
                if crop.size == 0:
                    raise ValueError(f"Пустая подтверждённая вырезка: {row.get('annotation_id', '')}")
                # Keep only pixels inside the reviewed interval mask. This prevents
                # ruler lines/arrows/background from becoming text-model shortcuts.
                local_polygon = np.rint(points - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
                interval_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                cv2.fillPoly(interval_mask, [local_polygon], 255)
                crop[interval_mask == 0] = 255
                crop_path = destination / "crops" / split / f"{stem}_{annotation_index:03d}.jpg"
                ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 96])
                if not ok:
                    raise ValueError(f"Не удалось сохранить вырезку: {row.get('annotation_id', '')}")
                crop_path.write_bytes(encoded.tobytes())
                facies_top = _number_or(row.get("facies_top"), float(row["depth_top"]))
                facies_base = _number_or(row.get("facies_base"), float(row["depth_base"]))
                caption_samples.append({
                    "annotation_id": row.get("annotation_id", ""),
                    "split": split,
                    "crop": crop_path.relative_to(destination).as_posix(),
                    "source_photo": str(photo),
                    "source_sha256": digest,
                    "well": row.get("well", ""),
                    "depth_top": float(row["depth_top"]),
                    "depth_base": float(row["depth_base"]),
                    "interval_top": facies_top,
                    "interval_base": facies_base,
                    "interval_m": max(0.0, facies_base - facies_top),
                    "facies": row["facies_index"],
                    "facies_index": row["facies_index"],
                    "facies_name": row["facies_name"],
                    "association": row.get("association", ""),
                    "environment": row.get("environment", ""),
                    "field_name": row.get("field_name", ""),
                    "target_text": target_text,
                    "source_file": row.get("source_file", ""),
                    "source_sheet": row.get("source_sheet", ""),
                    "source_row": row.get("source_row", ""),
                })
        label_path = destination / "labels" / split / f"{stem}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        samples.append({
            "source_photo": str(photo), "source_sha256": digest, "split": split,
            "image": str(target_image.relative_to(destination)),
            "label": str(label_path.relative_to(destination)),
            "well": annotations[0].get("well", ""), "annotation_count": len(annotations),
        })
    split_lists = {}
    for split in ("train", "val"):
        relative_images = [
            item["image"] for item in samples if item["split"] == split
        ]
        list_path = destination / f"{split}.txt"
        list_path.write_text("".join(f"{value}\n" for value in relative_images), encoding="utf-8")
        split_lists[split] = str(list_path)
    class_metadata = {
        "schema": "excel-photo-facies-metadata-v1",
        "target_headers": {
            "facies_index": "Индекс фации",
            "facies_name": "Название фации",
            "description": "Краткое описание",
        },
        "classes": [
            {"class_id": class_id, **facies_statistics[class_id]}
            for class_id in range(len(labels))
        ],
        "description_policy": (
            "YOLO predicts the interval mask and facies class. A Qwen3-VL LoRA adapter can be continued "
            "from the reviewed interval crops and their confirmed Excel short-description targets; "
            "VLM generates text only and does not predict depth, boundaries, or facies."
        ),
    }
    class_metadata_path = destination / "class_metadata.json"
    class_metadata_path.write_text(
        json.dumps(class_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    caption_path = destination / "caption_dataset.jsonl"
    caption_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in caption_samples),
        encoding="utf-8",
    )
    yaml_path = destination / "data.yaml"
    yaml_path.write_text("\n".join((
        f"path: {json.dumps(destination.as_posix(), ensure_ascii=False)}",
        "train: images/train", "val: images/val", f"nc: {len(labels)}", "names:",
        *(f"  {index}: {json.dumps(label, ensure_ascii=False)}" for index, label in enumerate(labels)), "",
    )), encoding="utf-8")
    manifest = {
        "schema": "excel-photo-yolo-seg-v4", "created_at": datetime.now().isoformat(timespec="seconds"),
        "project": str(project_dirs[0]) if len(project_dirs) == 1 else "",
        "projects": [str(path) for path in project_dirs], "project_count": len(project_dirs),
        "data_yaml": str(yaml_path), "class_names": labels,
        "class_metadata": str(class_metadata_path), "split_lists": split_lists,
        "facies_statistics": facies_statistics,
        "facies_count": len(facies_statistics),
        "source_target_headers": source_target_headers,
        "photo_count": len(by_photo), "annotation_count": len(rows), "split_strategy": strategy,
        "caption_count": len(caption_samples),
        "train_caption_count": sum(item["split"] == "train" for item in caption_samples),
        "val_caption_count": sum(item["split"] == "val" for item in caption_samples),
        "caption_dataset": str(caption_path),
        "train_photo_count": sum(value == "train" for value in split_by_photo.values()),
        "val_photo_count": sum(value == "val" for value in split_by_photo.values()),
        "training_ready": "val" in split_by_photo.values(),
        "training_note": "" if "val" in split_by_photo.values() else (
            "Датасет сохранён, но для обучения добавьте независимые фото для val "
            "с представленными в train классами."
        ),
        "class_counts": {item["facies_index"]: item["mask_count"] for item in facies_statistics},
        "samples": samples,
    }
    (destination / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "output_dir": str(destination)}


def _fresh_facies_statistics(rows: list[dict[str, str]]) -> list[dict[str, str | int | list[str]]]:
    """Reset class ids for this dataset and derive the taxonomy only from its Excel annotations."""
    by_key: dict[str, dict] = {}
    for row in rows:
        index = (
            " ".join(str(row.get("facies_index", "")).split())
            or " ".join(str(row.get("label", "")).split())
            or " ".join(str(row.get("facies_name", "")).split())
        )
        name = " ".join(str(row.get("facies_name", "")).split()) or index
        if not index:
            raise ValueError(f"У маски {row.get('annotation_id', '')} нет индекса фации.")
        key = index.casefold()
        known = by_key.get(key)
        if known is not None and known["facies_name"].casefold() != name.casefold():
            raise ValueError(
                f"Индекс фации «{known['facies_index']}» связан с разными названиями: "
                f"«{known['facies_name']}» и «{name}». Исправьте строки Excel до обучения."
            )
        if known is None:
            known = {
                "facies_index": index, "facies_name": name,
                "mask_count": 0, "description_count": 0,
            }
            by_key[key] = known
        known["mask_count"] += 1
        known["description_count"] += bool(str(row.get("target_text", "")).strip())
        # The segmentation class is reset to this dataset's Excel facies index.
        row["facies_index"] = known["facies_index"]
        row["facies_name"] = known["facies_name"]
        row["label"] = known["facies_index"]
    return sorted(by_key.values(), key=lambda item: str(item["facies_index"]).casefold())


def _restore_facies_targets_from_excel(project: Path, annotations: list[dict[str, str]]) -> list[dict]:
    config_path = project / "project.json"
    if not config_path.is_file():
        return []
    config = json.loads(config_path.read_text(encoding="utf-8"))
    excel_paths = config.get("excel_paths") or ([config["excel_path"]] if config.get("excel_path") else [])
    if not excel_paths:
        return []
    from .tabular import read_many_tables
    excel_rows, mappings, issues, _ = read_many_tables(
        [Path(value) for value in excel_paths], project / "column_mapping.json",
    )

    def source_key(source_file, sheet, row_number):
        if not str(source_file or "").strip():
            return None
        try:
            source = str(resolve_existing_path(source_file)).casefold()
            row = str(int(row_number))
        except (OSError, TypeError, ValueError):
            return None
        return source, " ".join(str(sheet).casefold().split()), row

    by_source = {
        source_key(row.source_file, row.sheet, row.row): row
        for row in excel_rows if source_key(row.source_file, row.sheet, row.row) is not None
    }
    missing_before = sum(not row.get("facies_index") or not row.get("facies_name") for row in annotations)
    restored = 0
    for annotation in annotations:
        key = source_key(
            annotation.get("source_file", ""), annotation.get("source_sheet", ""),
            annotation.get("source_row", ""),
        )
        source_row = by_source.get(key)
        if source_row is None:
            continue
        index = source_row.facies_index or source_row.label
        name = source_row.facies_name or source_row.label or index
        if not annotation.get("facies_index") or not annotation.get("facies_name"):
            restored += 1
        annotation["facies_index"] = index
        annotation["facies_name"] = name
        annotation["label"] = index
    if restored < missing_before:
        raise ValueError(
            f"{project.name}: для {missing_before - restored} масок не удалось восстановить индекс/название фации "
            "из исходного Excel по номеру строки. Пересчитайте проект и проверьте привязку Excel."
        )
    return [
        {
            "project": project.name, "source_file": item.source_file, "sheet": item.sheet,
            "facies_index": item.source_headers.get("class_index")
            or item.source_headers.get("class_code") or item.source_headers.get("label", ""),
            "facies_name": item.source_headers.get("label")
            or item.source_headers.get("class_code") or item.source_headers.get("class_index", ""),
            "target_text": item.source_headers.get("target_text")
            or item.source_headers.get("description", ""),
        }
        for item in mappings if item.class_index or item.label or item.target_text
    ]


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


def _number_or(value, fallback: float) -> float:
    if value is None or not str(value).strip():
        return float(fallback)
    try:
        result = float(str(value).strip().replace(" ", "").replace(",", "."))
    except ValueError:
        return float(fallback)
    return result if math.isfinite(result) else float(fallback)


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
