from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .catalog import register_project
from .dataset import build_dataset
from .project import create_project, load_annotations, set_annotation_approvals
from .training import train_bundle

def run_automatic_training(
    manifest_path: Path,
    dataset_dir: Path,
    model_dir: Path,
    *,
    architecture: str = "yolo11n-seg.yaml",
    epochs: int = 50,
    patience: int = 12,
    description_epochs: int = 40,
    use_ocr: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Process every Excel/photo pair, build one dataset, and train one model."""
    emit = progress or (lambda _message: None)
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    wells = payload.get("wells") if isinstance(payload, dict) else None
    if not isinstance(wells, list) or not wells:
        raise ValueError("В очереди автоматического обучения нет ни одной пары Excel + фото.")

    normalized: list[tuple[Path, Path]] = []
    for index, item in enumerate(wells, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Строка очереди {index}: ожидаются поля excel и photos.")
        excel = Path(str(item.get("excel", ""))).expanduser().resolve(strict=True)
        photos = Path(str(item.get("photos", ""))).expanduser().resolve(strict=True)
        if not excel.is_file():
            raise ValueError(f"Строка очереди {index}: файл Excel не найден: {excel}")
        if not photos.is_dir():
            raise ValueError(f"Строка очереди {index}: папка фото не найдена: {photos}")
        normalized.append((excel, photos))

    dataset_dir = Path(dataset_dir).expanduser().absolute()
    model_dir = Path(model_dir).expanduser().absolute()
    if dataset_dir.exists() or model_dir.exists():
        raise FileExistsError("Папка датасета или модели уже существует. Выберите новые пустые папки.")
    if dataset_dir == model_dir or dataset_dir in model_dir.parents or model_dir in dataset_dir.parents:
        raise ValueError("Папки датасета и модели должны быть раздельными.")

    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    project_root = local_app_data / "ExcelPhotoModelStudio" / "projects"
    project_root.mkdir(parents=True, exist_ok=True)
    project_dirs: list[Path] = []
    failures: list[str] = []
    processed: list[dict] = []

    for index, (excel, photos) in enumerate(normalized, start=1):
        safe_stem = re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", excel.stem).strip("_") or "well"
        project_dir = project_root / f"{safe_stem}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        emit(f"[{index}/{len(normalized)}] Сопоставляю {excel.name} с фото из {photos}…")
        try:
            report = create_project(excel, photos, project_dir, use_ocr=use_ocr)
            project_dirs.append(project_dir)
            register_project(project_dir)
            errors = int(report.get("blocking_errors", 0) or 0)
            annotations = load_annotations(project_dir)
            if errors:
                issue_lines = [
                    str(issue.get("message", ""))
                    for issue in report.get("issues", [])
                    if issue.get("severity") == "error"
                ]
                detail = "; ".join(issue_lines[:5]) or f"ошибок проверки: {errors}"
                failures.append(f"{excel.name}: {detail}")
                processed.append({
                    "excel": str(excel), "project": str(project_dir),
                    "status": "needs_data_fix", "blocking_errors": errors,
                    "photos": report.get("photos", 0),
                    "annotations": report.get("annotations", 0),
                })
                emit(f"  ПРОВЕРКА НЕ ПРОЙДЕНА: найдено блокирующих проблем — {errors}.")
                continue
            if not annotations:
                failures.append(f"{excel.name}: система не создала ни одной маски фаций.")
                processed.append({"excel": str(excel), "project": str(project_dir), "status": "no_masks"})
                emit("  ПРОВЕРКА НЕ ПРОЙДЕНА: масок нет.")
                continue

            # Generated masks are accepted only after the project-level checks prove
            # every input photo, core interval, facies interval, and description is covered.
            approvals = {row["annotation_id"]: True for row in annotations}
            approved_report = set_annotation_approvals(project_dir, approvals)
            if approved_report.get("approved_annotations") != len(annotations):
                failures.append(f"{excel.name}: не удалось автоматически утвердить все проверенные маски.")
                processed.append({"excel": str(excel), "project": str(project_dir), "status": "approval_mismatch"})
                continue
            processed.append({
                "excel": str(excel), "project": str(project_dir), "status": "ready",
                "photos": report.get("photos", 0), "annotations": len(annotations),
            })
            emit(
                f"  Готово: фото — {report.get('photos', 0)}, "
                f"маски фаций — {len(annotations)}; ручные подтверждения не нужны."
            )
        except Exception as exc:
            failures.append(f"{excel.name}: {exc}")
            processed.append({"excel": str(excel), "project": str(project_dir), "status": "failed", "error": str(exc)})
            emit(f"  ОШИБКА: {exc}")

    if failures:
        _write_run_report(model_dir, {
            "status": "blocked", "wells": processed, "errors": failures,
            "message": "Модель не обучалась: хотя бы один набор не прошёл полную проверку.",
        })
        bullets = "\n".join(f"- {item}" for item in failures)
        raise ValueError(
            "Автоматический цикл обработал все наборы, но модель не обучал: "
            "обнаружены неполные/ошибочные данные.\n" + bullets
        )

    emit(f"Собираю общий датасет из {len(project_dirs)} скважин…")
    try:
        dataset = build_dataset(project_dirs, dataset_dir)
        emit(
            f"Датасет: фото — {dataset.get('photo_count', 0)}, маски — {dataset.get('annotation_count', 0)}, "
            f"классы фаций — {len(dataset.get('class_names', []))}."
        )
        emit("Обучаю визуальную сегментацию и модель краткого описания…")
        trained = train_bundle(
            dataset_dir, model_dir, architecture=architecture,
            epochs=epochs, patience=patience, description_epochs=description_epochs,
        )
    except Exception as exc:
        _write_run_report(model_dir, {
            "status": "failed_training", "wells": processed,
            "dataset": str(dataset_dir), "error": str(exc),
        })
        raise
    result = {
        "status": "success", "wells": processed,
        "dataset": str(dataset_dir), "dataset_summary": dataset,
        "model": str(model_dir / "best.pt"), "training": trained,
    }
    _write_run_report(model_dir, result)
    emit(f"Готово. Модель: {model_dir / 'best.pt'}")
    return result


def _write_run_report(model_dir: Path, report: dict) -> None:
    # Keep failure reports next to the requested model location without creating
    # a misleading best.pt directory or overwriting an existing training result.
    report_path = model_dir.with_name(model_dir.name + "_automatic_training_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
