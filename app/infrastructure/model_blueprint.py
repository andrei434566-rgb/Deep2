"""Portable facies training specification; deliberately contains no model weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.domain.facies_catalog import (
    FACIES_MODEL_CLASSES,
    FACIES_MODEL_SCHEMA,
    FACIES_REFERENCE_SHA256,
    FACIES_REFERENCE_SOURCE,
)
from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS


BLUEPRINT_SCHEMA = "kern-facies-blueprint-v1"


def class_registry() -> dict:
    """Return JSON-safe copies with a fingerprint for the fixed class order."""
    classes = json.loads(json.dumps(FACIES_MODEL_CLASSES, ensure_ascii=False))
    identity = [(item["class_id"], item["model_label"]) for item in classes]
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "schema": FACIES_MODEL_SCHEMA,
        "source_workbook": FACIES_REFERENCE_SOURCE,
        "source_sha256": FACIES_REFERENCE_SHA256,
        "identity_sha256": digest,
        "class_count": len(classes),
        "class_names": [item["model_label"] for item in classes],
        "classes": classes,
    }


def training_recipe() -> dict:
    """Document the supported worker recipe without pretending training ran."""
    return {
        "schema": BLUEPRINT_SCHEMA,
        "status": "specification_only_no_weights",
        "task": "segment",
        "input_model": "A trusted local YOLO segmentation .pt on the training computer",
        "class_registry": "class_registry.json",
        "class_identity": "case-sensitive letter code + workbook index",
        "device": "NVIDIA CUDA; CPU training disabled in the current worker",
        "seed": 42,
        "image_size": 640,
        "batch": "automatic GPU batch",
        "epochs": "chosen in UI/CLI; upper limit, not an accuracy guarantee",
        "early_stopping_patience_default": 12,
        "augmentation": {
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
            "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
            "translate": 0.05, "scale": 0.15,
            "hsv_h": 0.0, "hsv_s": 0.1, "hsv_v": 0.15,
        },
        "imbalance": {
            "strategy": "bounded_sqrt_train_only",
            "maximum_per_class_multiplier": 3,
            "maximum_total_train_multiplier": 2,
            "validation_resampling": False,
            "warning": "Repeated examples change training exposure, not independent evidence or class coverage.",
        },
        "validation": {
            "split": "separate wells where feasible; otherwise independent source photographs",
            "duplicate_photo_leakage": "forbidden between train and val",
            "primary_metric": "segmentation mAP50-95",
            "per_class": ["precision", "recall", "f1", "mAP50", "mAP50-95"],
            "absent_classes": "untrained/not evaluated, never assigned synthetic quality metrics",
            "final_test": "reserve additional untouched wells before production acceptance",
        },
        "promotion": "candidate requires human review; training does not certify geological accuracy",
        "lithology": "16 attributes are separate observations; the facies reference supplies hypotheses, not measured facts",
        "implementation": "app/infrastructure/ml/fine_tune_worker.py",
    }


def export_model_blueprint(output_dir: Path) -> dict[str, str | int]:
    """Write a small, new folder safe to transfer to the computer with weights.

    Existing paths are never overwritten, including symlinks. No images,
    checkpoints, virtual environments or personal project files are copied.
    """
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Папка заготовки уже существует: {output_dir}")
    registry = class_registry()
    attribute_schema = {
        "schema": "kern-lithology-observations-v1",
        "field_count": len(LITHOLOGY_ATTRIBUTE_OPTIONS),
        "missing_value": None,
        "notes": [
            "Пустой параметр означает неизвестно, а не отсутствие признака.",
            "Справочная характеристика фации не является наблюдением по фото.",
            "Часть параметров требует измерений или экспертного подтверждения; их нельзя гарантированно восстановить по фотографии.",
        ],
        "fields": [{"name": name, "suggested_values": values, "allow_free_text": True}
                   for name, values in LITHOLOGY_ATTRIBUTE_OPTIONS.items()],
    }
    files = {
        "class_registry.json": registry,
        "training_recipe.json": training_recipe(),
        "lithology_attributes.json": attribute_schema,
        "blueprint.json": {
            "schema": BLUEPRINT_SCHEMA,
            "status": "specification_only_no_weights",
            "class_count": registry["class_count"],
            "class_identity_sha256": registry["identity_sha256"],
            "reference_sha256": FACIES_REFERENCE_SHA256,
        },
    }
    guide_source = Path(__file__).resolve().parents[2] / "MODEL_GUIDE.md"
    guide = guide_source.read_text(encoding="utf-8") if guide_source.is_file() else (
        "# Заготовка модели фаций\n\nЭто справочники и параметры, а не обученные веса. "
        "На другом ПК используйте ту же версию Kern Analyzer, проверенный датасет "
        "и вашу YOLO segmentation .pt модель. Дообучение требует CUDA.\n"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, payload in files.items():
        (output_dir / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "START_HERE.md").write_text(guide, encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "class_count": len(FACIES_MODEL_CLASSES),
        "class_registry": str(output_dir / "class_registry.json"),
        "training_recipe": str(output_dir / "training_recipe.json"),
        "attribute_schema": str(output_dir / "lithology_attributes.json"),
        "guide": str(output_dir / "START_HERE.md"),
    }


def check_model_blueprint(directory: Path) -> dict:
    """Validate imported metadata against this app, without loading any .pt."""
    directory = Path(directory)
    registry = json.loads((directory / "class_registry.json").read_text(encoding="utf-8"))
    blueprint = json.loads((directory / "blueprint.json").read_text(encoding="utf-8"))
    expected = class_registry()
    if blueprint.get("schema") != BLUEPRINT_SCHEMA:
        raise ValueError("Неизвестная версия заготовки модели.")
    for key in ("schema", "source_sha256", "identity_sha256", "class_count", "class_names", "classes"):
        if registry.get(key) != expected[key]:
            raise ValueError(f"Справочник заготовки отличается от текущей версии: {key}. Не смешивайте версии классов.")
    if blueprint.get("class_identity_sha256") != expected["identity_sha256"]:
        raise ValueError("Нарушена контрольная сумма порядка классов.")
    return {"status": "compatible", "class_count": expected["class_count"], "identity_sha256": expected["identity_sha256"]}
