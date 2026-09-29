from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


DEFAULT_WEIGHTS = "yolo11s-seg.pt"


def train_model(
    dataset_dir: Path,
    output_dir: Path,
    *,
    weights: str | Path = DEFAULT_WEIGHTS,
    epochs: int = 300,
    patience: int = 80,
    image_size: int = 1024,
    batch_size: int = 2,
    device: int | str = 0,
) -> dict:
    """Fine-tune a YOLO11 segmentation checkpoint on the reviewed CVAT dataset."""
    dataset_dir = Path(dataset_dir).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().absolute()
    weights = str(weights).strip()
    if not weights:
        raise ValueError("Укажите предобученные веса YOLO11-seg.")
    if str(device).strip().casefold() not in {"0", "cuda:0"}:
        raise ValueError("Обучение в этой версии запускается только на GPU CUDA 0.")
    if output_dir.exists():
        raise FileExistsError(f"Папка результата уже существует: {output_dir}")
    if epochs < 1 or patience < 1 or image_size < 32 or batch_size < 1:
        raise ValueError("Эпохи, patience, размер изображения и batch должны быть положительными.")

    yaml_path = dataset_dir / "data.yaml"
    manifest_path = dataset_dir / "dataset_manifest.json"
    metadata_path = dataset_dir / "class_metadata.json"
    train_list = dataset_dir / "train.txt"
    val_list = dataset_dir / "val.txt"
    if not yaml_path.is_file() or not manifest_path.is_file() or not metadata_path.is_file():
        raise ValueError("Выберите датасет приложения в формате CVAT / Ultralytics YOLO Segmentation.")
    if not train_list.is_file() or not train_list.read_text(encoding="utf-8").strip():
        raise ValueError("В датасете нет списка обучающих фото train.txt.")
    if not val_list.is_file() or not val_list.read_text(encoding="utf-8").strip():
        raise ValueError("В датасете нет независимой выборки проверки val.txt. Добавьте подтверждённые фото/скважины.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not manifest.get("class_names") or not metadata.get("classes"):
        raise ValueError("В датасете не найден список классов фаций.")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Для обучения нужны CUDA-сборка PyTorch и пакет ultralytics.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU недоступна: обучение остановлено без перехода на CPU. "
            "Проверьте NVIDIA-драйвер и CUDA-сборку PyTorch в этой версии приложения."
        )
    try:
        torch.cuda.set_device(0)
        gpu_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        raise RuntimeError(f"Не удалось инициализировать CUDA GPU 0: {exc}") from exc

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    runs_root = output_dir.with_name(output_dir.name + "_runs")
    runs_dir = runs_root / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    runs_dir.mkdir(parents=True, exist_ok=False)
    model = YOLO(weights)
    _require_yolo11_segmentation(model, weights)
    result = model.train(
        data=str(yaml_path), task="segment", epochs=int(epochs), patience=int(patience),
        imgsz=int(image_size), device=0, batch=int(batch_size), workers=0,
        optimizer="AdamW", lr0=0.0005, cos_lr=True, amp=True,
        pretrained=True, cache=False, rect=True,
        save_period=25, seed=42, deterministic=True,
        hsv_h=0.005, hsv_s=0.25, hsv_v=0.25,
        degrees=2.0, translate=0.05, scale=0.15, shear=0.0,
        flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
        project=str(runs_dir), name="yolo11_finetune", exist_ok=False,
    )
    save_dir = Path(str(getattr(result, "save_dir", runs_dir / "yolo11_finetune")))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("Обучение YOLO11 завершилось без weights/best.pt.")

    # Free the segmentation trainer's optimizer/model allocations before the
    # separate text model uses the same GPU.
    import gc
    del result, model
    gc.collect()
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

    description_dir = runs_dir / "description_model"
    try:
        from .description_model import train_description_model
        description_info = train_description_model(
            dataset_dir, description_dir, epochs=40, patience=8, image_size=128,
            hidden_size=128, batch_size=8, device="cuda:0", progress=print,
        )
    except Exception as exc:
        description_info = {
            "schema": "excel-photo-description-training-v2",
            "status": "not_trained",
            "reason": f"Обучение текста не завершено: {exc}",
        }
        print("Модель сегментации обучена; генерация описаний пропущена. " + description_info["reason"])
    description_checkpoint = description_dir / "description_best.pt"

    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}_pending_", dir=output_dir.parent) as staging:
        package_dir = Path(staging) / output_dir.name
        package_dir.mkdir()
        published = package_dir / "best.pt"
        shutil.copy2(best, published)
        shutil.copy2(yaml_path, package_dir / "data.yaml")
        shutil.copy2(manifest_path, package_dir / "dataset_manifest.json")
        shutil.copy2(metadata_path, package_dir / "class_metadata.json")
        if description_info.get("status") == "trained_candidate" and description_checkpoint.is_file():
            shutil.copy2(description_checkpoint, package_dir / "description_best.pt")
            shutil.copy2(
                description_dir / "description_training_info.json",
                package_dir / "description_training_info.json",
            )
        else:
            (package_dir / "description_training_info.json").write_text(
                json.dumps(description_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
        contract = {
            "schema": "excel-photo-yolo11-seg-v2",
            "task": "instance-segmentation",
            "architecture": "YOLO11-seg",
            "best_model": "best.pt",
            "class_metadata": "class_metadata.json",
            "target_headers": metadata.get("target_headers", {}),
            "classes": metadata["classes"],
            "description_model": "description_best.pt" if description_info.get("status") == "trained_candidate" else None,
            "description_policy": (
                "Separate interval-image + facies-index + interval-thickness character decoder; "
                "review generated text before geological use."
                if description_info.get("status") == "trained_candidate"
                else "Text model not trained: " + str(description_info.get("reason", "insufficient interval examples"))
            ),
        }
        (package_dir / "model_contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        info = {
            "schema": "excel-photo-model-training-v2",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "candidate_requires_review",
            "architecture": "YOLO11-seg",
            "pretrained_weights": weights,
            "device": "cuda:0",
            "gpu_name": str(gpu_name),
            "best_model": str(output_dir / "best.pt"),
            "dataset": str(dataset_dir),
            "training_run": str(save_dir),
            "epochs_limit": int(epochs),
            "patience": int(patience),
            "image_size": int(image_size),
            "batch_size": int(batch_size),
            "class_count": len(manifest["class_names"]),
            "photo_count": int(manifest.get("photo_count", 0)),
            "mask_count": int(manifest.get("annotation_count", 0)),
            "description_model_status": description_info.get("status", "not_trained"),
            "description_model_reason": description_info.get("reason", ""),
            "description_train_samples": int(description_info.get("train_samples", 0)),
            "description_val_samples": int(description_info.get("val_samples", 0)),
            "description_best_epoch": int(description_info.get("best_epoch", 0)),
            "description_best_val_loss": description_info.get("best_val_loss"),
        }
        (package_dir / "training_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        package_dir.rename(output_dir)
    return info


def _require_yolo11_segmentation(model, weights: str) -> None:
    task = str(getattr(model, "task", "")).casefold()
    if task and task not in {"segment", "segmentation"}:
        raise ValueError(f"Выбран checkpoint для задачи «{task}», нужен YOLO11 segmentation.")
    core = getattr(model, "model", None)
    config = getattr(core, "yaml", {}) if core is not None else {}
    yaml_file = str(config.get("yaml_file", "")) if isinstance(config, dict) else ""
    identity = (yaml_file + " " + str(weights)).casefold()
    if "yolo11" not in identity:
        raise ValueError(
            "Выбраны не YOLO11-веса. Старый best.pt/last.pt автоматически не подхватывается: "
            "укажите yolo11n-seg.pt или yolo11s-seg.pt."
        )
