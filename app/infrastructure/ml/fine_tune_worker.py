"""Background YOLO fine-tuning worker."""

from __future__ import annotations

import shutil
import sys
import json
import csv
import hashlib
import math
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot


def _finite(value) -> float | None:
    try:
        result = float(value)
        return round(result, 6) if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _ordered_names(names) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=int)]
    return [str(value) for value in (names or [])]


def validation_report(result, class_names: list[str], class_counts: dict | None = None) -> dict:
    """Read mask metrics without inventing scores for classes absent from val.

    Ultralytics ``maps`` substitutes the overall mean for unobserved classes;
    using that property would make untested facies appear evaluated.
    """
    metric = getattr(result, "seg", None)
    task = "segmentation" if metric is not None else "detection"
    metric = metric if metric is not None else getattr(result, "box", None)
    if metric is None:
        return {"status": "unavailable", "metrics": {}, "per_class": []}
    indices = [int(value) for value in getattr(metric, "ap_class_index", [])]
    positions = {class_id: position for position, class_id in enumerate(indices)}
    rows = []
    for class_id, label in enumerate(class_names):
        row = {"class_id": class_id, "model_label": label, "status": "not_evaluated"}
        if class_counts is not None:
            row["validation_samples"] = int(class_counts.get(label, 0))
        position = positions.get(class_id)
        if position is not None and (class_counts is None or row["validation_samples"] > 0):
            try:
                values = metric.class_result(position)
                row.update({name: _finite(value) for name, value in zip(("precision", "recall", "mAP50", "mAP50-95"), values)})
                precision, recall = row.get("precision"), row.get("recall")
                row["f1"] = None if precision is None or recall is None else (
                    _finite(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
                )
                row["status"] = "evaluated" if all(row.get(name) is not None for name in ("precision", "recall", "mAP50", "mAP50-95")) else "invalid_metrics"
            except (IndexError, TypeError, ValueError):
                pass
        rows.append(row)
    values = metric.mean_results()
    metrics = {name: value for name, raw in zip(("precision", "recall", "mAP50", "mAP50-95"), values) if (value := _finite(raw)) is not None}
    evaluated = [row for row in rows if row["status"] == "evaluated"]
    f1_values = [row["f1"] for row in evaluated if row.get("f1") is not None]
    if f1_values:
        metrics["macro_f1"] = round(sum(f1_values) / len(f1_values), 6)
    return {
        "status": "evaluated" if evaluated else "no_evaluated_classes",
        "task": task,
        "metrics": metrics,
        "per_class": rows,
        "evaluated_class_count": len(evaluated),
        "class_count": len(class_names),
        "limitation": "Контроль val используется для выбора эпохи; это не независимый финальный тест и не проверка геологической интерпретации.",
    }


def load_training_manifest(data_yaml: Path) -> dict:
    """Require auditable, source-independent train/val membership."""
    import yaml

    manifest_path = data_yaml.parent / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("validation_independent") or not manifest.get("val_photo_count"):
        raise ValueError("Нет независимых контрольных фото. Подготовьте новый датасет с отдельными фото/скважинами для val.")
    names = _ordered_names(yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"])
    if names != manifest.get("class_names"):
        raise ValueError("Классы data.yaml не совпадают с паспортом датасета. Повторите экспорт.")
    splits: dict[str, set[str]] = {"train": set(), "val": set()}
    for sample in manifest.get("samples", []):
        split, digest = sample.get("split"), sample.get("source_sha256")
        if split in splits and digest:
            splits[split].add(digest)
    if not all(splits.values()) or splits["train"] & splits["val"]:
        raise ValueError("Ошибка контроля: train и val пусты или содержат копии одного исходного фото.")
    return manifest


def runtime_dataset_yaml(data_yaml: Path, target: Path) -> Path:
    """Resolve a portable dataset against its own folder on the training PC.

    Ultralytics resolves relative ``path`` against its global datasets_dir;
    the shipped manifest must remain portable while the run copy is absolute.
    """
    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload.get("path", "."))
    if not root.is_absolute():
        root = data_yaml.parent / root
    payload["path"] = str(root.resolve())
    target.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _safe_training_streams(log_path: Path):
    """Give Ultralytics writable streams when Kern Analyzer is launched by pythonw."""
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        stdout = original_stdout if callable(getattr(original_stdout, "write", None)) else log_file
        stderr = original_stderr if callable(getattr(original_stderr, "write", None)) else log_file
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield


def read_training_metrics(run_dir: Path) -> dict[str, float]:
    """Return the best validation metrics saved by Ultralytics, when present."""
    results_path = run_dir / "results.csv"
    try:
        with results_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    except OSError:
        return {}
    if not rows:
        return {}
    # Mask quality is the primary metric for a segmentation model.
    target = next((key for key in rows[-1] if "mAP50-95(M)" in key), None)
    target = target or next((key for key in rows[-1] if "mAP50-95" in key), None)
    if target is None:
        return {}
    def number(row: dict, key: str) -> float:
        try:
            value = float(str(row.get(key, "")).strip())
            return value if math.isfinite(value) else float("-inf")
        except ValueError:
            return float("-inf")
    best = max(rows, key=lambda row: number(row, target))
    result: dict[str, float] = {}
    for key, short_name in ((target, "mAP50-95"),):
        value = number(best, key)
        if value != float("-inf"):
            result[short_name] = round(value, 4)
    suffix = "(M)" if "(M)" in target else "(B)"
    for needle, short_name in ((f"precision{suffix}", "precision"), (f"recall{suffix}", "recall"), (f"mAP50{suffix}", "mAP50")):
        key = next((item for item in best if needle.casefold() in item.casefold()), None)
        if key:
            value = number(best, key)
            if value != float("-inf"):
                result[short_name] = round(value, 4)
    return result


def read_completed_epochs(run_dir: Path) -> int:
    """Count epochs actually completed, including early-stopped runs."""
    results_path = run_dir / "results.csv"
    try:
        with results_path.open("r", encoding="utf-8-sig", newline="") as file:
            return sum(1 for _ in csv.DictReader(file))
    except OSError:
        return 0


class FineTuneWorker(QObject):
    progress = Signal(str)
    epoch_progress = Signal(int, int)
    succeeded = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        model_path: Path,
        data_yaml: Path,
        runs_dir: Path,
        published_dir: Path,
        epochs: int = 20,
        dataset_summary: dict[str, object] | None = None,
        recommended_epochs: int | None = None,
        early_stopping_patience: int = 12,
    ):
        super().__init__()
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.runs_dir = runs_dir
        self.published_dir = published_dir
        self.epochs = max(1, int(epochs))
        self.dataset_summary = dict(dataset_summary or {})
        self.recommended_epochs = recommended_epochs
        self.early_stopping_patience = max(1, int(early_stopping_patience))
        self._runtime_data_yaml = data_yaml

    @Slot()
    def run(self) -> None:
        try:
            manifest = load_training_manifest(self.data_yaml)
            if self.published_dir.exists():
                raise RuntimeError(f"Папка для сохранения модели уже существует: {self.published_dir}")
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            self._runtime_data_yaml = runtime_dataset_yaml(self.data_yaml, self.runs_dir / "runtime_data.yaml")
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Файл исходной модели не найден: {self.model_path}. Укажите реальные веса на компьютере обучения.")
            device, device_label = self._best_device()
            if device == "cpu":
                raise RuntimeError(
                    "Дообучение требует NVIDIA GPU с CUDA, но CUDA в запущенной версии "
                    "не обнаружена. Обучение на CPU намеренно не запускается: это слишком "
                    "долго и перегружает компьютер. Установите/соберите версию Kern Analyzer "
                    "с CUDA-версией PyTorch и актуальным драйвером NVIDIA."
                )
            self.progress.emit(
                f"Дообучение запущено: максимум {self.epochs} эпох · "
                f"автостоп {self.early_stopping_patience} · {device_label}"
            )
            with _safe_training_streams(self.runs_dir / "fine_tune.log"):
                from ultralytics import YOLO

                model = YOLO(str(self.model_path))
                if getattr(model, "task", None) != "segment":
                    raise RuntimeError("Для выделения интервалов требуется исходная YOLO-модель сегментации (segment).")
                class_names = manifest["class_names"]
                baseline_comparable = _ordered_names(model.names) == class_names
                baseline_report = {"status": "incompatible_taxonomy", "metrics": {}, "per_class": []}
                if baseline_comparable:
                    self.progress.emit("Проверяю исходную модель на контрольных фото…")
                    baseline_report = self._validate_control(model, device, "baseline_validation", manifest)
                else:
                    self.progress.emit("Состав/порядок классов исходной модели отличается: перенос весов разрешён, сравнение метрик до/после пропущено.")
                model.add_callback("on_train_epoch_end", self._report_epoch)
                result = model.train(
                    data=str(self._runtime_data_yaml),
                    epochs=self.epochs,
                    patience=self.early_stopping_patience,
                    imgsz=640,
                    device=device,
                    # ``-1`` lets Ultralytics select the largest safe batch for
                    # the detected GPU instead of consuming system RAM on CPU.
                    batch=-1,
                    cache=False,
                    amp=True,
                    workers=0,
                    seed=42,
                    deterministic=True,
                    # Preserve vertical sedimentary structure and realistic
                    # colour; mosaic can fabricate contacts between facies.
                    mosaic=0.0,
                    mixup=0.0,
                    copy_paste=0.0,
                    flipud=0.0,
                    degrees=0.0,
                    perspective=0.0,
                    translate=0.05,
                    scale=0.15,
                    hsv_h=0.0,
                    hsv_s=0.1,
                    hsv_v=0.15,
                    project=str(self.runs_dir),
                    name="fine_tune",
                    exist_ok=False,
                    verbose=False,
                )
            save_dir = Path(str(getattr(result, "save_dir", self.runs_dir / "fine_tune")))
            best_model = save_dir / "weights" / "best.pt"
            if not best_model.is_file():
                raise RuntimeError("Дообучение завершилось, но файл best.pt не был создан.")
            with _safe_training_streams(self.runs_dir / "validation.log"):
                self.progress.emit("Проверяю сохранённый best.pt по каждой фации…")
                candidate = YOLO(str(best_model))
                if _ordered_names(candidate.names) != class_names:
                    raise RuntimeError("Индексы классов сохранённой модели не соответствуют паспорту датасета.")
                trained_report = self._validate_control(candidate, device, "candidate_validation", manifest)
            catalog = self.data_yaml.parent / "facies_catalog.json"
            if not catalog.is_file():
                raise RuntimeError("Не найден справочник фаций датасета после дообучения.")
            catalog_copy = best_model.parent / catalog.name
            shutil.copy2(catalog, catalog_copy)
            if not catalog_copy.is_file():
                raise RuntimeError("Не удалось сохранить справочник фаций рядом с моделью.")
            if self.published_dir.exists():
                raise RuntimeError(f"Папка для сохранения модели уже существует: {self.published_dir}")
            self.published_dir.mkdir(parents=True, exist_ok=False)
            published_model = self.published_dir / "best.pt"
            shutil.copy2(best_model, published_model)
            shutil.copy2(catalog_copy, self.published_dir / catalog_copy.name)
            shutil.copy2(self.data_yaml, self.published_dir / "data.yaml")
            shutil.copy2(self.data_yaml.parent / "dataset_manifest.json", self.published_dir / "dataset_manifest.json")
            trained_metrics = trained_report.get("metrics", {})
            baseline_metrics = baseline_report.get("metrics", {})
            completed_epochs = read_completed_epochs(save_dir)
            comparison = {
                name: round(trained_metrics[name] - baseline_metrics[name], 4)
                for name in set(trained_metrics) & set(baseline_metrics)
            }
            (self.published_dir / "training_info.json").write_text(
                json.dumps(
                    {
                        "source_model": str(self.model_path),
                        "source_model_sha256": _sha256(self.model_path),
                        "model_sha256": _sha256(published_model),
                        "dataset_source_digest": manifest.get("source_digest"),
                        "class_names": class_names,
                        "seed": 42,
                        "deterministic": True,
                        "model_status": "candidate_requires_review",
                        "promotion_recommended": False,
                        "validation_status": trained_report.get("status"),
                        "baseline_comparable": baseline_comparable,
                        "epochs": self.epochs,
                        "max_epochs": self.epochs,
                        "completed_epochs": completed_epochs,
                        "recommended_epochs": self.recommended_epochs,
                        "early_stopping_patience": self.early_stopping_patience,
                        "dataset": str(self.data_yaml.parent),
                        "run": str(save_dir),
                        "dataset_summary": self.dataset_summary,
                        "control_split": f"val ({manifest.get('split_strategy')}; исходные фото разделены по SHA256)",
                        "baseline_metrics": baseline_metrics,
                        "metrics": trained_metrics,
                        "baseline_validation": baseline_report,
                        "validation": trained_report,
                        "training_curve_best_metrics": read_training_metrics(save_dir),
                        "warnings": manifest.get("warnings", []),
                        "comparison": comparison,
                    },
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            self.succeeded.emit(str(published_model))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    def _report_epoch(self, trainer) -> None:
        """Forward Ultralytics epoch events to the Qt interface."""
        total = max(1, int(getattr(trainer, "epochs", self.epochs)))
        current = max(0, min(total, int(getattr(trainer, "epoch", -1)) + 1))
        self.epoch_progress.emit(current, total)

    def _validate_control(self, model, device: int | str, name: str, manifest: dict) -> dict:
        """Measure original weights on the held-out validation split."""
        try:
            result = model.val(
                data=str(self._runtime_data_yaml), split="val", imgsz=640, device=device,
                batch=8, workers=0, verbose=False, plots=True,
                project=str(self.runs_dir), name=name, exist_ok=False,
            )
            return validation_report(result, manifest["class_names"], manifest.get("val_class_counts"))
        except Exception as exc:
            self.progress.emit(f"Проверка {name} не завершена: {exc}")
            return {"status": "failed", "error": str(exc), "metrics": {}, "per_class": []}

    @staticmethod
    def _best_device() -> tuple[int | str, str]:
        try:
            import torch

            if torch.cuda.is_available():
                return 0, f"GPU: {torch.cuda.get_device_name(0)}"
        except (ImportError, RuntimeError):
            pass
        return "cpu", "CPU (CUDA не найдена)"
