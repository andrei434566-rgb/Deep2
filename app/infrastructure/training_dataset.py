"""Auditable reviewed-mask export with fixed classes and source-safe validation."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PySide6.QtCore import QRect
from PySide6.QtGui import QImage, QPolygonF

from app.domain.facies_catalog import (
    FACIES_MODEL_CLASSES, FACIES_MODEL_SCHEMA, FACIES_REFERENCE_SHA256,
    resolve_facies_class,
)
from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS
from app.domain.models import PhotoRecord


MIN_TRAINING_SAMPLES = 5


def _has_valid_facies_label(detection) -> bool:
    return str(getattr(detection, "label", "") or "").strip() not in {"", "Новый контур"}


def verified_samples_count(records: list[PhotoRecord]) -> int:
    return sum(bool(d.training_ready and _has_valid_facies_label(d) and len(d.polygon) >= 3)
               for record in records for d in record.detections)


def automatic_samples_count(records: list[PhotoRecord]) -> int:
    return sum(bool(not d.training_ready and _has_valid_facies_label(d) and len(d.polygon) >= 3)
               for record in records for d in record.detections)


def unlabeled_manual_samples_count(records: list[PhotoRecord]) -> int:
    return sum(bool(d.training_ready and not _has_valid_facies_label(d) and len(d.polygon) >= 3)
               for record in records for d in record.detections)


def _source_hash(record: PhotoRecord) -> str:
    """Hash decoded pixels, not filenames: copied project images remain one source."""
    image = record.pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    digest = hashlib.sha256(f"{image.width()}x{image.height()}:RGBA8888:".encode())
    digest.update(bytes(image.constBits()))
    return digest.hexdigest()


def _canonical_polygon(points) -> tuple:
    values = tuple((round(point.x(), 4), round(point.y(), 4)) for point in points)
    if len(values) > 1 and values[0] == values[-1]:
        values = values[:-1]
    if not values:
        return ()
    forward = min(values[i:] + values[:i] for i in range(len(values)))
    backward = tuple(reversed(values))
    backward = min(backward[i:] + backward[:i] for i in range(len(backward)))
    return min(forward, backward)


def _geometry_problem(record: PhotoRecord, detection) -> str | None:
    points = [(point.x(), point.y()) for point in detection.polygon]
    if points and points[0] == points[-1]:
        points.pop()
    if record.pixmap.isNull():
        return "Исходное изображение не загружено"
    if len(points) < 3 or any(not math.isfinite(value) for point in points for value in point):
        return "Контур содержит меньше трёх вершин или неконечные координаты"
    if len(set(points)) != len(points):
        return "Повторяющиеся вершины контура"
    width, height = record.pixmap.width(), record.pixmap.height()
    if any(x < 0 or y < 0 or x > width or y > height for x, y in points):
        return "Контур выходит за границы изображения"
    area = abs(sum(x * points[(i + 1) % len(points)][1] - y * points[(i + 1) % len(points)][0]
                   for i, (x, y) in enumerate(points))) / 2
    if area < 4 or max(x for x, _ in points) - min(x for x, _ in points) < 4 or max(y for _, y in points) - min(y for _, y in points) < 4:
        return "Нулевая площадь или слишком малый контур"
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    def on_segment(a, b, c):
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    for i, a in enumerate(points):
        b = points[(i + 1) % len(points)]
        for j in range(i + 2, len(points)):
            if i == 0 and j == len(points) - 1:
                continue
            c, d = points[j], points[(j + 1) % len(points)]
            p, q, r, s = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
            if (p * q < 0 and r * s < 0) or any((abs(v) < 1e-9 and on_segment(u, w, z)) for v, u, w, z in (
                    (p, a, b, c), (q, a, b, d), (r, c, d, a), (s, c, d, b))):
                return "Самопересечение контура"
    return None


def _holdout(groups: dict[str, list[dict]]) -> set[str]:
    """Choose whole groups, retaining at least one training example of each class."""
    total = Counter(sample["label"] for group in groups.values() for sample in group)
    remaining = total.copy()
    selected: set[str] = set()
    covered: set[str] = set()
    target = min(len(groups) - 1, max(1, round(len(groups) * 0.2)))
    while len(selected) < target:
        candidates = []
        for key in sorted(groups):
            if key in selected:
                continue
            counts = Counter(sample["label"] for sample in groups[key])
            if any(remaining[label] <= count for label, count in counts.items()):
                continue
            score = (len(set(counts) - covered), sum(count / total[label] for label, count in counts.items()),
                     -abs(sum(counts.values()) - sum(total.values()) * 0.2 / max(1, target)))
            candidates.append((score, key, counts))
        if not candidates:
            break
        _, key, counts = max(candidates, key=lambda item: (item[0], item[1]))
        selected.add(key)
        covered.update(counts)
        remaining.subtract(counts)
    return selected


def _assign_splits(samples: list[dict]) -> tuple[str, bool]:
    by_hash = defaultdict(list)
    for sample in samples:
        by_hash[sample["source_sha256"]].append(sample)
    wells = {digest: {sample["well"] for sample in group} for digest, group in by_hash.items()}
    if wells and all(len(names) == 1 and next(iter(names)) for names in wells.values()):
        by_well = defaultdict(list)
        for sample in samples:
            by_well[sample["well"]].append(sample)
        chosen = _holdout(by_well)
        if chosen:
            for sample in samples:
                sample["split"] = "val" if sample["well"] in chosen else "train"
            return "well", True
    chosen = _holdout(by_hash)
    for sample in samples:
        sample["split"] = "val" if sample["source_sha256"] in chosen else "train"
    return ("photo" if chosen else "unavailable"), bool(chosen)


def _prepare(records: list[PhotoRecord], strict_catalog: bool) -> tuple[dict, list[dict]]:
    samples, invalid, unknown = [], [], []
    duplicates = 0
    seen: dict[tuple, dict] = {}
    for record in records:
        if not any(d.training_ready for d in record.detections):
            continue
        source_hash = _source_hash(record)
        for contour_index, detection in enumerate(record.detections):
            if not detection.training_ready:
                continue
            row = {"photo": str(record.path), "record_id": str(record.identifier), "contour_index": contour_index,
                   "label": str(detection.label or "").strip()}
            problem = _geometry_problem(record, detection)
            if problem:
                invalid.append({**row, "reason": problem})
                continue
            resolved = resolve_facies_class(row["label"], (detection.attributes or {}).get("Индекс фации"))
            if not _has_valid_facies_label(detection) or (strict_catalog and resolved is None):
                unknown.append({**row, "reason": "Выберите однозначную фацию и индекс из финального справочника"})
                continue
            label = resolved["model_label"] if resolved else row["label"]
            polygon = _canonical_polygon(detection.polygon)
            key = (source_hash, polygon)
            if key in seen:
                if seen[key]["label"] != label:
                    invalid.append({**row, "reason": "Одному контуру на копиях фото назначены разные фации"})
                else:
                    duplicates += 1
                    if seen[key]["well"] != str(record.well_name or "").strip().casefold():
                        seen[key]["well"] = ""
                continue
            sample = {"record": record, "detection": detection, "label": label,
                      "source_sha256": source_hash, "polygon": polygon,
                      "well": str(record.well_name or "").strip().casefold()}
            seen[key] = sample
            samples.append(sample)
    samples.sort(key=lambda s: (s["source_sha256"], s["label"], s["polygon"]))
    strategy, independent = _assign_splits(samples)
    counts = Counter(sample["label"] for sample in samples)
    train_counts = Counter(sample["label"] for sample in samples if sample["split"] == "train")
    val_counts = Counter(sample["label"] for sample in samples if sample["split"] == "val")
    names = [item["model_label"] for item in FACIES_MODEL_CLASSES] if strict_catalog else sorted(counts)
    missing = [name for name in names if name not in counts]
    no_val = [name for name in names if name in train_counts and name not in val_counts]
    blocking, warnings = [], []
    if invalid:
        blocking.append(f"Исправьте некорректные/противоречивые контуры: {len(invalid)}.")
    if unknown:
        blocking.append(f"Уточните фации по справочнику (включая индекс): {len(unknown)}.")
    if len(samples) < MIN_TRAINING_SAMPLES:
        blocking.append(f"Нужно минимум {MIN_TRAINING_SAMPLES} уникальных вручную проверенных слоёв; доступно {len(samples)}.")
    if not independent:
        blocking.append("Невозможно выделить отдельные контрольные фото, сохранив классы в обучении. Добавьте проверенные фото тех же фаций; слои одного фото нельзя делить между train и val.")
    if counts and max(counts.values()) > min(counts.values()) * 5:
        warnings.append("Дисбаланс классов допустим; редкие train-примеры повторяются ограниченно, контроль val сохраняет естественные частоты.")
    if strategy == "photo":
        warnings.append("Контроль по отдельным фото, но не независимым скважинам: перенос на новую скважину ещё не проверен.")
    if duplicates:
        warnings.append(f"Исключены точные дубликаты проверенных слоёв: {duplicates}.")
    if missing:
        warnings.append(f"Нет обучающих примеров для {len(missing)} классов справочника: наличие класса в списке не означает, что модель его выучила.")
    if no_val:
        warnings.append(f"Нет контрольных примеров для {len(no_val)} представленных классов; их качество оценить нельзя.")
    warnings.append("Экспорт содержит вырезки проверенных интервалов. Точность границ и непрерывной колонки нужно отдельно проверять на целых фото; справочник не обучает 16 литологических признаков.")
    source_digest = hashlib.sha256(json.dumps([
        [s["source_sha256"], s["label"], s["polygon"], s["well"], s["split"]] for s in samples
    ], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    report = {
        "schema_version": 1, "class_schema": FACIES_MODEL_SCHEMA if strict_catalog else "custom",
        "reference_sha256": FACIES_REFERENCE_SHA256 if strict_catalog else None,
        "sample_count": len(samples), "class_names": names, "class_counts": dict(counts),
        "train_class_counts": dict(train_counts), "val_class_counts": dict(val_counts),
        "photo_count": len({s["source_sha256"] for s in samples}),
        "well_count": len({s["well"] for s in samples if s["well"]}),
        "train_photo_count": len({s["source_sha256"] for s in samples if s["split"] == "train"}),
        "val_photo_count": len({s["source_sha256"] for s in samples if s["split"] == "val"}),
        "unrepresented_classes": missing, "unsupported_validation_classes": no_val,
        "split_strategy": strategy, "source_digest": source_digest,
        "duplicates_removed": duplicates, "invalid_samples": invalid, "unknown_labels": unknown,
        "validation_independent": independent, "validation_grouped_by_photo": independent,
        "blocking_reasons": blocking, "warnings": warnings, "can_export": not blocking,
    }
    return report, samples


def inspect_training_dataset(records: list[PhotoRecord], strict_catalog: bool = True) -> dict:
    """Read-only preflight. Does not create directories or silently correct labels."""
    return _prepare(records, strict_catalog)[0]


def _photo_level_splits(samples: list[tuple[PhotoRecord, object]]) -> dict[int, str] | None:
    """Compatibility helper; duplicates are one source even with different paths."""
    grouped = defaultdict(list)
    for record, detection in samples:
        grouped[_source_hash(record)].append({"record": record, "label": detection.label})
    selected = _holdout(grouped)
    if not selected:
        return None
    return {id(sample["record"]): "val" if key in selected else "train"
            for key, group in grouped.items() for sample in group}


def export_training_dataset(records: list[PhotoRecord], destination: Path, *, strict_catalog: bool = True,
                            balance: bool = True) -> dict[str, object]:
    """Export fixed IDs, audit metadata and bounded train-only class repetition."""
    report, samples = _prepare(records, strict_catalog)
    if report["blocking_reasons"]:
        raise ValueError("\n".join(report["blocking_reasons"]))
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Папка датасета уже существует: {destination}")
    for split in ("train", "val"):
        (destination / "images" / split).mkdir(parents=True)
        (destination / "labels" / split).mkdir(parents=True)
    names = report["class_names"]
    indices = {name: index for index, name in enumerate(names)}
    # Reference context is not a claim that this image's lithology was observed.
    catalog = {item["model_label"]: {**item["metadata"], "Обучающих примеров": report["train_class_counts"].get(item["model_label"], 0)}
               for item in FACIES_MODEL_CLASSES} if strict_catalog else {}
    manifest_samples, attributes = [], []
    for index, sample in enumerate(samples):
        record, detection, split, label = sample["record"], sample["detection"], sample["split"], sample["label"]
        if not _export_sample(record, detection, destination, split, index, indices[label]):
            raise OSError(f"Не удалось сохранить проверенный слой {index + 1}. Неполный датасет: {destination}")
        stem = f"sample_{index + 1:05d}"
        row = {"sample_id": stem, "split": split, "image": f"images/{split}/{stem}.jpg",
               "label_file": f"labels/{split}/{stem}.txt", "facies": label,
               "class_id": indices[label], "source_sha256": sample["source_sha256"],
               "source_file": Path(record.path).name, "well": record.well_name,
               "source_polygon": sample["polygon"], "depth_from": detection.depth_from, "depth_to": detection.depth_to}
        manifest_samples.append(row)
        detection_attributes = detection.attributes or {}
        source_fields = (
            "Месторождение", "№ скважины", "Интервал керна", "Интервал фации",
            "Толщина фации", "Литологическое описание (16 параметров)",
            "Краткое описание", "Источник описания", "__annotation_source",
        )
        attributes.append({
            **row,
            "facies_metadata": catalog.get(label, {}),
            "attributes_origin": "human_reviewed_annotation",
            "attributes": {
                name: str(detection_attributes.get(name) or "")
                for name in LITHOLOGY_ATTRIBUTE_OPTIONS
            },
            "source_metadata": {
                name: str(detection_attributes.get(name) or "")
                for name in source_fields if detection_attributes.get(name)
            },
        })
    if balance:
        by_class = defaultdict(list)
        for sample in manifest_samples:
            if sample["split"] == "train":
                by_class[sample["facies"]].append(sample)
        maximum = max(map(len, by_class.values()), default=0)
        budget = sum(map(len, by_class.values()))
        for label in sorted(by_class, key=lambda name: (len(by_class[name]), name)):
            originals = by_class[label]
            target = min(3 * len(originals), math.ceil(math.sqrt(maximum * len(originals))))
            for index in range(min(budget, target - len(originals))):
                original = originals[index % len(originals)]
                stem = f"repeat_{len(manifest_samples) + 1:05d}"
                repeated = {**original, "sample_id": stem, "image": f"images/train/{stem}.jpg",
                            "label_file": f"labels/train/{stem}.txt", "repeated_from": original["sample_id"]}
                shutil.copy2(destination / original["image"], destination / repeated["image"])
                shutil.copy2(destination / original["label_file"], destination / repeated["label_file"])
                manifest_samples.append(repeated)
                budget -= 1
    yaml_path = destination / "data.yaml"
    yaml_path.write_text("\n".join([
        "# Portable path: the training launcher resolves '.' against this file's directory.",
        "path: .", "train: images/train", "val: images/val", f"nc: {len(names)}", "names:",
        *[f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(names)], "",
    ]), encoding="utf-8")
    attributes_path = destination / "attributes.jsonl"
    attributes_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in attributes), encoding="utf-8")
    catalog_path = destination / "facies_catalog.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    registry_path = destination / "class_registry.json"
    registry_path.write_text(json.dumps({"schema": report["class_schema"], "reference_sha256": report["reference_sha256"],
                                         "classes": list(FACIES_MODEL_CLASSES) if strict_catalog else names}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    effective_counts = Counter(s["facies"] for s in manifest_samples if s["split"] == "train")
    manifest = {**report, "samples": manifest_samples, "training_sample_count": sum(effective_counts.values()),
                "effective_train_class_counts": dict(effective_counts), "oversampling_added": len(manifest_samples) - len(samples),
                "balancing": "bounded_sqrt_train_only" if balance else "none"}
    manifest_path = destination / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "data_yaml": yaml_path, "attributes_jsonl": attributes_path,
            "facies_catalog": catalog_path, "dataset_manifest": manifest_path,
            "class_registry": registry_path, "output_dir": destination}


def _export_sample(record: PhotoRecord, detection, destination: Path, split: str, index: int, class_index: int) -> bool:
    bounds = QPolygonF(detection.polygon).boundingRect().toAlignedRect().intersected(
        QRect(0, 0, record.pixmap.width(), record.pixmap.height()))
    if bounds.width() < 4 or bounds.height() < 4:
        return False
    normalized = [((p.x() - bounds.x()) / bounds.width(), (p.y() - bounds.y()) / bounds.height()) for p in detection.polygon]
    file_stem = f"sample_{index + 1:05d}"
    if not record.pixmap.copy(bounds).save(str(destination / "images" / split / f"{file_stem}.jpg"), "JPG", 100):
        return False
    coordinates = " ".join(f"{value:.6f}" for point in normalized for value in point)
    (destination / "labels" / split / f"{file_stem}.txt").write_text(f"{class_index} {coordinates}\n", encoding="utf-8")
    return True
