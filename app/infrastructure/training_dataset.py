"""Export manually verified facies into a YOLO segmentation dataset."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from PySide6.QtCore import QRect
from PySide6.QtGui import QPolygonF

from app.domain.facies_catalog import facies_metadata
from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS
from app.domain.models import PhotoRecord


MIN_TRAINING_SAMPLES = 5


def _has_valid_facies_label(detection) -> bool:
    return str(getattr(detection, "label", "") or "").strip() not in {"", "Новый контур"}


def verified_samples_count(records: list[PhotoRecord]) -> int:
    """Count only human-verified contours that can be exported safely."""
    return sum(
        1
        for record in records
        for detection in record.detections
        if detection.training_ready and _has_valid_facies_label(detection) and len(detection.polygon) >= 3
    )


def automatic_samples_count(records: list[PhotoRecord]) -> int:
    """Count model suggestions separately from reviewed training examples."""
    return sum(
        1
        for record in records
        for detection in record.detections
        if not detection.training_ready and _has_valid_facies_label(detection) and len(detection.polygon) >= 3
    )


def unlabeled_manual_samples_count(records: list[PhotoRecord]) -> int:
    """Count newly drawn contours waiting for the user to choose a facies."""
    return sum(
        1
        for record in records
        for detection in record.detections
        if detection.training_ready
        and str(detection.label or "").strip() in {"", "Новый контур"}
        and len(detection.polygon) >= 3
    )


def export_training_dataset(records: list[PhotoRecord], destination: Path) -> dict[str, object]:
    """Write crops and YOLO polygon labels for the reviewed parts of core photos."""
    samples = [
        (record, detection)
        for record in records
        for detection in record.detections
        if detection.training_ready and _has_valid_facies_label(detection) and len(detection.polygon) >= 3
    ]
    if len(samples) < MIN_TRAINING_SAMPLES:
        raise ValueError(f"Для дообучения нужно минимум {MIN_TRAINING_SAMPLES} вручную проверенных слоёв.")

    if destination.exists():
        raise FileExistsError(f"Папка датасета уже существует: {destination}")
    for split in ("train", "val"):
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Preserve a validation fraction inside every facies, rather than putting
    # one whole facies into only train or only validation by loading order.
    by_facies: dict[str, list[tuple[PhotoRecord, object]]] = defaultdict(list)
    for sample in samples:
        by_facies[str(sample[1].label).strip()].append(sample)
    class_names = sorted(by_facies, key=str.casefold)
    class_index = {name: index for index, name in enumerate(class_names)}
    facies_catalog = _facies_catalog(by_facies)
    assigned_samples: list[tuple[PhotoRecord, object, str]] = []
    for facies in class_names:
        facies_samples = by_facies[facies]
        validation_count = max(1, round(len(facies_samples) * 0.2)) if len(facies_samples) > 1 else 0
        split_at = len(facies_samples) - validation_count
        for sample_index, (record, detection) in enumerate(facies_samples):
            assigned_samples.append((record, detection, "train" if sample_index < split_at else "val"))

    exported = 0
    attribute_records: list[dict[str, object]] = []
    for index, (record, detection, split) in enumerate(assigned_samples):
        if _export_sample(record, detection, destination, split, index, class_index[detection.label]):
            exported += 1
            # Individual rows stay in the dataset for audit and possible
            # future attribute-specific training.  For the app, the stable
            # per-facies parameter profile is bundled with the final model.
            attribute_records.append(
                {
                    "sample_id": f"sample_{index + 1:05d}",
                    "split": split,
                    "image": f"images/{split}/sample_{index + 1:05d}.jpg",
                    "facies": detection.label,
                    "facies_metadata": facies_catalog[str(detection.label).strip()],
                    "attributes": {
                        field_name: str((detection.attributes or {}).get(field_name) or "")
                        for field_name in LITHOLOGY_ATTRIBUTE_OPTIONS
                    },
                }
            )

    if exported < MIN_TRAINING_SAMPLES:
        raise ValueError("Не удалось подготовить достаточно корректных контуров для дообучения.")

    yaml_path = destination / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {json.dumps(destination.as_posix(), ensure_ascii=False)}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(class_names)}",
                "names:",
                *[f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(class_names)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    attributes_path = destination / "attributes.jsonl"
    attributes_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in attribute_records),
        encoding="utf-8",
    )
    catalog_path = destination / "facies_catalog.json"
    catalog_path.write_text(
        json.dumps(facies_catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "sample_count": exported,
        "data_yaml": yaml_path,
        "attributes_jsonl": attributes_path,
        "facies_catalog": catalog_path,
        "class_names": class_names,
        "class_counts": {name: len(by_facies[name]) for name in class_names},
        "output_dir": destination,
    }


def _facies_catalog(by_facies: dict[str, list[tuple[PhotoRecord, object]]]) -> dict[str, dict[str, str]]:
    """Build a facies profile used to restore all description fields.

    YOLO predicts a facies class, not 16 independent categorical attributes.
    Each predicted class therefore receives the most frequent non-empty value
    for each field from the reviewed examples used to train that class.
    """
    catalog: dict[str, dict[str, str]] = {}
    fields = ("Индекс фации", "Название фации", "Код фации")
    for label, samples in by_facies.items():
        variants = {
            tuple(str((detection.attributes or {}).get(field) or "").strip() for field in fields)
            for _, detection in samples
        }
        if len(variants) > 1:
            raise ValueError(
                f"Для основной метки «{label}» в Excel указаны разные индекс, название или код фации. "
                "Исправьте справочник и повторите дообучение."
            )
        index, name, code = variants.pop() if variants else ("", "", "")
        catalog[label] = facies_metadata(label)
        catalog[label].update({
            "Индекс фации": index,
            "Название фации": name,
            "Код фации": code,
        })
        for field_name in LITHOLOGY_ATTRIBUTE_OPTIONS:
            values = Counter(
                str((detection.attributes or {}).get(field_name) or "").strip()
                for _, detection in samples
            )
            values.pop("", None)
            if values:
                catalog[label][field_name] = min(
                    values.items(),
                    key=lambda item: (-item[1], item[0].casefold()),
                )[0]
    return catalog


def _export_sample(record: PhotoRecord, detection, destination: Path, split: str, index: int, class_index: int) -> bool:
    bounds = QPolygonF(detection.polygon).boundingRect().toAlignedRect()
    image_bounds = QRect(0, 0, record.pixmap.width(), record.pixmap.height())
    bounds = bounds.intersected(image_bounds)
    if bounds.width() < 4 or bounds.height() < 4:
        return False

    local_points = [(point.x() - bounds.x(), point.y() - bounds.y()) for point in detection.polygon]
    normalized = [
        (max(0.0, min(1.0, x / bounds.width())), max(0.0, min(1.0, y / bounds.height())))
        for x, y in local_points
    ]
    file_stem = f"sample_{index + 1:05d}"
    image_path = destination / "images" / split / f"{file_stem}.jpg"
    label_path = destination / "labels" / split / f"{file_stem}.txt"
    if not record.pixmap.copy(bounds).save(str(image_path), "JPG", 100):
        return False
    coordinates = " ".join(f"{value:.6f}" for point in normalized for value in point)
    label_path.write_text(f"{class_index} {coordinates}\n", encoding="utf-8")
    return True
