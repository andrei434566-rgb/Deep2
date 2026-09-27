from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def train_model(
    dataset_dir: Path,
    output_dir: Path,
    *,
    architecture: str = "yolo11n-seg.yaml",
    epochs: int = 50,
    patience: int = 12,
    image_size: int = 640,
    device: str | int | None = None,
) -> dict:
    dataset_dir = Path(dataset_dir).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().absolute()
    architecture = str(architecture).strip()
    if not architecture or Path(architecture).suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Для обучения с нуля укажите YAML-архитектуру сегментационной модели, а не готовый .pt.")
    if output_dir.exists():
        raise FileExistsError(f"Папка результата уже существует: {output_dir}")
    if epochs < 1 or patience < 1:
        raise ValueError("epochs и patience должны быть положительными.")
    yaml_path = dataset_dir / "data.yaml"
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not yaml_path.is_file() or not manifest_path.is_file():
        raise ValueError("Папка не является подготовленным датасетом этой системы.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Установите ultralytics из requirements.txt.") from exc
    if device is None:
        try:
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    runs_dir = output_dir.with_name(output_dir.name + "_runs")
    if runs_dir.exists():
        raise FileExistsError(f"Папка журналов уже существует: {runs_dir}")
    runs_dir.mkdir(parents=True)
    model = YOLO(architecture)
    result = model.train(
        data=str(yaml_path), task="segment", epochs=int(epochs), patience=int(patience),
        pretrained=False,
        imgsz=int(image_size), device=device, batch=-1 if device != "cpu" else 4,
        # Ultralytics' AMP self-test can fetch pretrained weights even when
        # pretrained=False. Keep scratch/offline training genuinely offline.
        cache=False, amp=False, seed=42, deterministic=True, workers=0,
        project=str(runs_dir), name="training", exist_ok=False,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, flipud=0.0,
        degrees=0.0, perspective=0.0, translate=0.05, scale=0.15,
    )
    save_dir = Path(str(getattr(result, "save_dir", runs_dir / "training")))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("Обучение завершилось без best.pt.")
    output_dir.mkdir(parents=True)
    published = output_dir / "best.pt"
    shutil.copy2(best, published)
    shutil.copy2(yaml_path, output_dir / "data.yaml")
    shutil.copy2(manifest_path, output_dir / "dataset_manifest.json")
    info = {
        "schema": "excel-photo-trained-model-v2", "status": "candidate_requires_review",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "initialization": "random_weights", "pretrained": False, "architecture": architecture,
        "best_model": str(published), "dataset": str(dataset_dir), "epochs_limit": epochs,
        "patience": patience, "image_size": image_size, "device": str(device), "training_run": str(save_dir),
    }
    (output_dir / "training_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def train_bundle(
    dataset_dir: Path,
    output_dir: Path,
    *,
    architecture: str = "yolo11n-seg.yaml",
    epochs: int = 50,
    patience: int = 12,
    image_size: int = 640,
    device: str | int | None = None,
    description_epochs: int = 40,
    description_patience: int = 8,
) -> dict:
    """Train the compatible two-part package used by Kern Analyzer."""
    dataset_dir = Path(dataset_dir).expanduser().resolve(strict=True)
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("train_caption_count", 0) < 5 or manifest.get("val_caption_count", 0) < 1:
        raise ValueError("Недостаточно строк с кратким описанием для обучения полного комплекта модели.")
    visual = train_model(
        dataset_dir, output_dir, architecture=architecture, epochs=epochs,
        patience=patience, image_size=image_size, device=device,
    )
    from .description_model import train_description_model
    text = train_description_model(
        dataset_dir, output_dir, epochs=description_epochs, patience=description_patience, device=device,
    )
    facies_reference = _facies_reference(dataset_dir)
    contract = {
        "schema": "kern-unified-model-bundle-v4",
        "status": "candidate_requires_geologist_review",
        "initialization": "random_weights",
        "pretrained": False,
        "architecture": architecture,
        "visual_model": "best.pt",
        "description_embedded_in_best_pt": True,
        "description_checkpoint_key": "core_description_checkpoint",
        "standalone_description_model": "description_best.pt",
        "target_fields": ["facies_index", "facies_name", "target_text"],
        "target_headers": {
            "facies_index": "Индекс фации",
            "facies_name": "Название фации",
            "target_text": "Краткое описание",
        },
        "facies_classes": manifest.get("class_names", []),
        "facies_statistics": manifest.get("facies_statistics", []),
        "source_target_headers": manifest.get("source_target_headers", []),
        "facies_class_key": "facies_index",
        "facies_taxonomy_policy": "fresh_per_dataset_from_source_excel",
        "description_target_header": "Краткое описание",
        "description_conditioning": ["interval_image", "facies_class"],
        "facies_reference": facies_reference,
        "dataset_manifest": "dataset_manifest.json",
        "note": "Excel fields are resolved by semantic header names per workbook and sheet; model class index/name mappings are derived afresh from this dataset.",
    }
    embed_description_checkpoint(
        Path(output_dir) / "best.pt", Path(output_dir) / "description_best.pt", contract,
    )
    (Path(output_dir) / "model_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"visual": visual, "description": text, "contract": contract, "output_dir": str(Path(output_dir).resolve())}


def embed_description_checkpoint(visual_model: Path, description_model: Path, contract: dict) -> Path:
    """Embed description and facies-index/name metadata without breaking Ultralytics loading."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Для объединения модели установите PyTorch.") from exc
    visual_model = Path(visual_model).resolve(strict=True)
    description_model = Path(description_model).resolve(strict=True)
    visual_checkpoint = torch.load(visual_model, map_location="cpu", weights_only=False)
    description_checkpoint = torch.load(description_model, map_location="cpu", weights_only=False)
    if not isinstance(visual_checkpoint, dict):
        raise ValueError("best.pt не содержит ожидаемый checkpoint Ultralytics.")
    if not isinstance(description_checkpoint, dict):
        raise ValueError("description_best.pt не является checkpoint модели описания.")
    target_header = description_checkpoint.get("target_header") or description_checkpoint.get(
        "target_headers", {},
    ).get("target_text")
    legacy_column = description_checkpoint.get("target_column")
    if target_header != "Краткое описание" and legacy_column != 23:
        raise ValueError("description_best.pt не указывает целевое поле по заголовку «Краткое описание».")
    visual_checkpoint["core_model_schema"] = "kern-unified-best-v3"
    visual_checkpoint["core_description_checkpoint"] = description_checkpoint
    visual_checkpoint["core_model_contract"] = contract
    temporary = visual_model.with_name(visual_model.name + ".tmp")
    torch.save(visual_checkpoint, temporary)
    os.replace(temporary, visual_model)
    return visual_model


def _facies_reference(dataset_dir: Path) -> dict[str, dict[str, str]]:
    from collections import Counter, defaultdict

    path = Path(dataset_dir) / "caption_dataset.jsonl"
    if not path.is_file():
        return {}
    values: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        facies = str(row.get("facies_index") or row.get("facies", "")).strip()
        if not facies:
            continue
        name = str(row.get("facies_name", "")).strip()
        if name:
            values[facies]["facies_name"][name] += 1
        for key in ("association", "environment", "field_name"):
            value = str(row.get(key, "")).strip()
            if value:
                values[facies][key][value] += 1
    return {
        facies: {
            key: counter.most_common(1)[0][0]
            for key, counter in fields.items() if counter
        }
        for facies, fields in sorted(values.items())
    }
