from __future__ import annotations

import json
import importlib.util
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .paths import resolve_existing_path


DEFAULT_WEIGHTS = "yolo11s-seg.pt"
OFFICIAL_YOLO11_SEG_WEIGHTS = {
    "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",
}


def training_data_warnings(manifest: dict) -> list[str]:
    """Describe small or weakly independent datasets without hiding the choice to train."""
    warnings = []
    photo_count = int(manifest.get("photo_count", 0) or 0)
    annotation_count = int(manifest.get("annotation_count", 0) or 0)
    train_count = int(manifest.get("train_photo_count", 0) or 0)
    val_count = int(manifest.get("val_photo_count", 0) or 0)
    class_counts = manifest.get("class_counts", {})
    class_count = int(manifest.get("facies_count", len(class_counts)) or 0)

    if photo_count < 20:
        warnings.append(
            f"Мало независимых фото: {photo_count}. Это пробный набор; для устойчивой модели нужны "
            "разнообразные реальные фото из нескольких скважин."
        )
    if annotation_count < max(40, class_count * 10):
        warnings.append(
            f"Мало масок: {annotation_count} на {class_count} классов. Модель будет нестабильна, "
            "особенно на редких фациях."
        )
    if train_count < 10 or val_count < 3:
        warnings.append(
            f"Маленькое разбиение: train — {train_count} фото, val — {val_count}. "
            "Оценка качества на таком val ненадёжна."
        )
    sparse_classes = [
        f"{name} ({count})" for name, count in class_counts.items()
        if int(count or 0) < 10
    ]
    if sparse_classes:
        warnings.append("Меньше 10 масок для классов: " + ", ".join(sparse_classes) + ".")
    nonzero_counts = [int(count or 0) for count in class_counts.values() if int(count or 0) > 0]
    if len(nonzero_counts) > 1 and max(nonzero_counts) >= 5 * min(nonzero_counts):
        warnings.append("Сильный дисбаланс числа масок между фациями; редкие классы модель может пропускать.")
    if manifest.get("split_strategy") not in {"well"}:
        warnings.append(
            "Контрольная выборка разделена по фото, а не по скважинам; результат может переоценивать "
            "качество на новой скважине."
        )
    wells = {
        " ".join(str(sample.get("well", "")).casefold().split())
        for sample in manifest.get("samples", []) if isinstance(sample, dict) and sample.get("well")
    }
    if wells and all(name.startswith(("test-", "test_", "synthetic", "тест")) for name in wells):
        warnings.append(
            "Названия скважин похожи на синтетический тестовый набор: он проверяет конвейер, "
            "но не подходит для получения рабочей геологической модели."
        )
    return warnings


def _resolve_pretrained_weights(weights: str, settings=None) -> str:
    """Prefer an explicitly selected/local cached checkpoint before Ultralytics can download it."""
    requested = Path(weights).expanduser()
    if requested.is_file():
        return str(resolve_existing_path(requested))

    name = requested.name
    is_official_asset_name = name.casefold() in OFFICIAL_YOLO11_SEG_WEIGHTS
    has_path_component = requested.is_absolute() or requested.parent != Path(".")
    if has_path_component or (requested.suffix.casefold() == ".pt" and not is_official_asset_name):
        raise FileNotFoundError(
            f"Файл весов не найден: {requested}. Укажите существующий checkpoint .pt "
            "в поле «Локальный .pt», либо выберите официальную модель YOLO11 из списка."
        )

    search_dirs = [
        Path.cwd(),
        Path(__file__).absolute().parents[2],
        Path(__file__).absolute().parents[2] / "models",
        Path(sys.executable).expanduser().absolute().parent,
    ]
    configured_dir = settings.get("weights_dir") if settings is not None else None
    if configured_dir:
        search_dirs.append(Path(str(configured_dir)).expanduser())
    search_dirs.extend((
        Path.home() / ".cache" / "ultralytics" / "weights",
        Path.home() / ".cache" / "Ultralytics" / "weights",
    ))
    appdata = os.environ.get("APPDATA")
    local_appdata = os.environ.get("LOCALAPPDATA")
    if appdata:
        search_dirs.append(Path(appdata) / "Ultralytics" / "weights")
    if local_appdata:
        search_dirs.append(Path(local_appdata) / "Ultralytics" / "weights")
    for search_dir in search_dirs:
        candidate = search_dir / name
        if candidate.is_file():
            return str(resolve_existing_path(candidate))
    return str(weights)


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
    vlm_base_model_dir: str | Path | None = None,
    vlm_initial_adapter: str | Path | None = None,
    require_vlm: bool = False,
    progress: Callable[[str], None] = print,
) -> dict:
    """Train segmentation, then optionally Qwen3-VL on the same reviewed dataset.

    In joint mode both models are required: preflight rejects incomplete VLM
    input before expensive YOLO training, and a partial model is not published.
    """
    dataset_dir = resolve_existing_path(dataset_dir)
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

    vlm_requested = bool(vlm_base_model_dir or vlm_initial_adapter)
    vlm_skip_reason = ""
    if require_vlm and not vlm_requested:
        raise ValueError(
            "Совместное обучение требует локальную базовую Qwen3-VL и начальный LoRA-адаптер. "
            "Укажите оба пути на вкладке обучения."
        )
    if vlm_requested and not (vlm_base_model_dir and vlm_initial_adapter):
        raise ValueError("Для VLM укажите одновременно базовую Qwen3-VL и исходный LoRA-адаптер.")
    if vlm_requested:
        from .vlm_description import (
            MIN_TRAIN_SAMPLES, MIN_VAL_SAMPLES, _read_caption_rows,
            normalize_adapter_path, validate_base_model_path,
        )
        vlm_base_model_dir = validate_base_model_path(vlm_base_model_dir)
        vlm_initial_adapter = normalize_adapter_path(vlm_initial_adapter)
        try:
            _caption_rows, caption_train_rows, caption_val_rows = _read_caption_rows(dataset_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            vlm_skip_reason = str(exc)
            if require_vlm:
                raise ValueError("Совместное обучение не начато: " + vlm_skip_reason) from exc
            progress("VLM пропущена: " + vlm_skip_reason)
        else:
            if len(caption_train_rows) < MIN_TRAIN_SAMPLES or len(caption_val_rows) < MIN_VAL_SAMPLES:
                vlm_skip_reason = (
                    f"нужно минимум {MIN_TRAIN_SAMPLES} train и {MIN_VAL_SAMPLES} независимых val-примеров; "
                    f"сейчас {len(caption_train_rows)} и {len(caption_val_rows)}"
                )
                if require_vlm:
                    raise ValueError("Совместное обучение не начато: " + vlm_skip_reason)
                progress("VLM пропущена: " + vlm_skip_reason)
            else:
                missing = [
                    package for module, package in (
                        ("transformers", "transformers"), ("peft", "peft"),
                        ("accelerate", "accelerate"), ("PIL", "Pillow"),
                        ("safetensors", "safetensors"),
                    ) if importlib.util.find_spec(module) is None
                ]
                if missing:
                    raise RuntimeError(
                        "Для настроенного VLM-обучения отсутствуют библиотеки: " + ", ".join(missing)
                        + ". Установите их командой pip install -r requirements-vlm.txt. "
                        "Ошибка обнаружена до запуска длительного обучения YOLO."
                    )

    try:
        import torch
        from ultralytics import YOLO
        try:
            from ultralytics.utils import SETTINGS
        except Exception:
            SETTINGS = None
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

    resolved_weights = _resolve_pretrained_weights(weights, SETTINGS)
    for warning in training_data_warnings(manifest):
        progress("Предупреждение по качеству датасета: " + warning)

    # Construct the model before creating the run directory. If the official
    # checkpoint cannot be downloaded, no empty run folders are left behind.
    progress(f"Загружаю YOLO11-seg веса: {resolved_weights}")
    try:
        model = YOLO(resolved_weights)
    except Exception as exc:
        if Path(resolved_weights).name.casefold() in OFFICIAL_YOLO11_SEG_WEIGHTS:
            raise RuntimeError(
                f"Не удалось открыть предобученные веса {Path(resolved_weights).name}. "
                "Ultralytics не смог загрузить их из сети. Подключите интернет для первой загрузки "
                "или скачайте этот файл .pt отдельно и выберите его в поле «Локальный .pt». "
                f"Подробности: {exc}"
            ) from exc
        raise RuntimeError(f"Не удалось открыть checkpoint {resolved_weights}: {exc}") from exc
    _require_yolo11_segmentation(model, resolved_weights)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    runs_root = output_dir.with_name(output_dir.name + "_runs")
    runs_dir = runs_root / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    runs_dir.mkdir(parents=True, exist_ok=False)
    add_callback = getattr(model, "add_callback", None)
    if callable(add_callback):
        def report_epoch(trainer) -> None:
            epoch = int(getattr(trainer, "epoch", -1)) + 1
            total_epochs = int(getattr(trainer, "epochs", epochs))
            progress(f"YOLO11-seg: завершена эпоха {epoch}/{total_epochs}.")

        add_callback("on_fit_epoch_end", report_epoch)
    progress(f"Запускаю обучение на GPU {gpu_name}; фотографий: {manifest.get('photo_count', 0)}, "
             f"масок: {manifest.get('annotation_count', 0)}, классов: {len(manifest['class_names'])}.")
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
    progress(f"Сегментация завершена. Лучшие веса: {best}")

    # Free the segmentation trainer's optimizer/model allocations before the
    # separate text model uses the same GPU.
    import gc
    del result, model
    gc.collect()
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

    description_dir = runs_dir / "description_vlm"
    description_info = {
        "schema": "excel-photo-qwen3-vl-lora-v1",
        "status": "not_configured",
        "reason": "Укажите локальную базовую Qwen3-VL и начальный LoRA-адаптер.",
    }
    if vlm_base_model_dir and vlm_initial_adapter and not vlm_skip_reason:
        try:
            from .vlm_description import train_vlm_description_model
            description_info = train_vlm_description_model(
                dataset_dir, description_dir,
                base_model_dir=vlm_base_model_dir,
                initial_adapter=vlm_initial_adapter,
                epochs=2.0,
                learning_rate=1e-5,
                max_image_side=448,
                device="cuda:0",
                progress=progress,
            )
        except Exception as exc:
            if require_vlm:
                raise RuntimeError(
                    "YOLO11-seg обучена, но VLM не завершилась; совместная модель не опубликована. "
                    f"YOLO checkpoint сохранён для восстановления: {best}. Причина VLM: {exc}"
                ) from exc
            description_info = {
                "schema": "excel-photo-qwen3-vl-lora-v1",
                "status": "not_trained",
                "reason": f"Дообучение VLM не завершено: {exc}",
            }
            progress("YOLO-сегментация обучена; генерация описаний VLM не завершена. " + description_info["reason"])
    elif not vlm_requested:
        progress("Параметры VLM не указаны: YOLO обучается отдельно, дообучение генератора текста пропущено.")
    else:
        description_info["status"] = "not_trained"
        description_info["reason"] = vlm_skip_reason
    description_checkpoint = description_dir / "adapter_model.safetensors"
    description_ready = (
        description_info.get("status") == "trained_candidate_requires_review"
        and description_checkpoint.is_file()
    )
    if require_vlm and not description_ready:
        raise RuntimeError(
            "YOLO11-seg обучена, но VLM не создала проверяемый LoRA-адаптер; "
            f"совместная модель не опубликована. YOLO checkpoint: {best}. "
            f"Причина: {description_info.get('reason', 'адаптер не сохранён')}"
        )

    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}_pending_", dir=output_dir.parent) as staging:
        package_dir = Path(staging) / output_dir.name
        package_dir.mkdir()
        published = package_dir / "best.pt"
        shutil.copy2(best, published)
        shutil.copy2(yaml_path, package_dir / "data.yaml")
        shutil.copy2(manifest_path, package_dir / "dataset_manifest.json")
        shutil.copy2(metadata_path, package_dir / "class_metadata.json")
        if description_ready:
            packaged_adapter = package_dir / "description_vlm"
            shutil.copytree(description_dir, packaged_adapter, ignore=shutil.ignore_patterns("trainer_runs"))
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
            "description_model": "description_vlm" if description_ready else None,
            "description_model_type": "qwen3_vl_lora" if description_ready else None,
            "description_base_model_dir": description_info.get("base_model_dir") if description_ready else None,
            "description_policy": (
                "A Qwen3-VL LoRA adapter drafts a short visual description only for YOLO's already-detected "
                "facies interval; it cannot alter interval boundaries or class. Review before geological use."
                if description_ready
                else "Text model not trained: " + str(description_info.get("reason", "insufficient interval examples"))
            ),
        }
        (package_dir / "model_contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        info = {
            "schema": "excel-photo-model-training-v3",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "candidate_requires_review",
            "architecture": "YOLO11-seg",
            "pretrained_weights": resolved_weights,
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
            "description_eval_loss": description_info.get("eval_loss"),
            "description_base_model_dir": description_info.get("base_model_dir", str(vlm_base_model_dir or "")),
            "description_initial_adapter": description_info.get("initial_adapter_dir", str(vlm_initial_adapter or "")),
            "dataset_quality_warnings": training_data_warnings(manifest),
        }
        (package_dir / "training_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        # Commit the model package before TemporaryDirectory removes its staging
        # folder. Keeping this inside the context also prevents publishing a
        # partial package if any metadata write above failed.
        _publish_model_package(package_dir, output_dir, progress)
    progress(f"Модель сохранена: {output_dir / 'best.pt'}")
    return info


def _publish_model_package(
    package_dir: Path, output_dir: Path, progress: Callable[[str], None],
) -> None:
    try:
        package_dir.rename(output_dir)
    except PermissionError:
        # Some restricted Windows profiles allow creating files but deny
        # directory renames, even within the same parent. Publish a copy only
        # after the entire staged package has been written.
        progress("Windows запретил переименование папки; копирую готовый пакет модели…")
        try:
            shutil.copytree(package_dir, output_dir)
        except Exception as exc:
            raise OSError(
                f"Не удалось опубликовать пакет модели в {output_dir}. "
                "Проверьте права на папку; если появился неполный каталог, "
                "не используйте его как готовую модель."
            ) from exc


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
