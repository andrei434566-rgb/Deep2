"""Background YOLO fine-tuning worker."""

from __future__ import annotations

import shutil
import sys
import json
import csv
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot


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
    target = next((key for key in rows[-1] if "mAP50-95" in key), None)
    if target is None:
        return {}
    def number(row: dict, key: str) -> float:
        try:
            return float(str(row.get(key, "")).strip())
        except ValueError:
            return float("-inf")
    best = max(rows, key=lambda row: number(row, target))
    result: dict[str, float] = {}
    for key, short_name in ((target, "mAP50-95"),):
        value = number(best, key)
        if value != float("-inf"):
            result[short_name] = round(value, 4)
    for needle, short_name in (("precision", "precision"), ("recall", "recall"), ("mAP50(B)", "mAP50")):
        key = next((item for item in best if needle.casefold() in item.casefold()), None)
        if key:
            value = number(best, key)
            if value != float("-inf"):
                result[short_name] = round(value, 4)
    return result


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
    ):
        super().__init__()
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.runs_dir = runs_dir
        self.published_dir = published_dir
        self.epochs = epochs
        self.dataset_summary = dict(dataset_summary or {})
        self.recommended_epochs = recommended_epochs

    @Slot()
    def run(self) -> None:
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            device, device_label = self._best_device()
            if device == "cpu":
                raise RuntimeError(
                    "Дообучение требует NVIDIA GPU с CUDA, но CUDA в запущенной версии "
                    "не обнаружена. Обучение на CPU намеренно не запускается: это слишком "
                    "долго и перегружает компьютер. Установите/соберите версию Kern Analyzer "
                    "с CUDA-версией PyTorch и актуальным драйвером NVIDIA."
                )
            self.progress.emit(f"Дообучение запущено: {self.epochs} эпох · {device_label}")
            with _safe_training_streams(self.runs_dir / "fine_tune.log"):
                from ultralytics import YOLO

                model = YOLO(str(self.model_path))
                self.progress.emit("Проверяю исходную модель на контрольных фото…")
                baseline_metrics = self._validate_control(model, device)
                model.add_callback("on_train_epoch_end", self._report_epoch)
                result = model.train(
                    data=str(self.data_yaml),
                    epochs=self.epochs,
                    imgsz=640,
                    device=device,
                    # ``-1`` lets Ultralytics select the largest safe batch for
                    # the detected GPU instead of consuming system RAM on CPU.
                    batch=-1,
                    cache=False,
                    amp=True,
                    project=str(self.runs_dir),
                    name="fine_tune",
                    exist_ok=True,
                    verbose=False,
                )
            save_dir = Path(str(getattr(result, "save_dir", self.runs_dir / "fine_tune")))
            best_model = save_dir / "weights" / "best.pt"
            if not best_model.is_file():
                raise RuntimeError("Дообучение завершилось, но файл best.pt не был создан.")
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
            trained_metrics = read_training_metrics(save_dir)
            comparison = {
                name: round(trained_metrics[name] - baseline_metrics[name], 4)
                for name in set(trained_metrics) & set(baseline_metrics)
            }
            (self.published_dir / "training_info.json").write_text(
                json.dumps(
                    {
                        "source_model": str(self.model_path),
                        "epochs": self.epochs,
                        "recommended_epochs": self.recommended_epochs,
                        "dataset": str(self.data_yaml.parent),
                        "run": str(save_dir),
                        "dataset_summary": self.dataset_summary,
                        "control_split": "val (контрольные фото, не попавшие в train)",
                        "baseline_metrics": baseline_metrics,
                        "metrics": trained_metrics,
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

    def _validate_control(self, model, device: int | str) -> dict[str, float]:
        """Measure original weights on the held-out validation split."""
        try:
            result = model.val(data=str(self.data_yaml), split="val", imgsz=640, device=device, batch=-1, verbose=False)
            values = dict(getattr(result, "results_dict", {}) or {})
        except Exception as exc:
            self.progress.emit(f"Контроль до обучения пропущен: {exc}")
            return {}
        metrics: dict[str, float] = {}
        for needle, short_name in (("mAP50-95", "mAP50-95"), ("precision", "precision"), ("recall", "recall"), ("mAP50(B)", "mAP50")):
            key = next((name for name in values if needle.casefold() in str(name).casefold()), None)
            if key is not None:
                try:
                    metrics[short_name] = round(float(values[key]), 4)
                except (TypeError, ValueError):
                    pass
        return metrics

    @staticmethod
    def _best_device() -> tuple[int | str, str]:
        try:
            import torch

            if torch.cuda.is_available():
                return 0, f"GPU: {torch.cuda.get_device_name(0)}"
        except (ImportError, RuntimeError):
            pass
        return "cpu", "CPU (CUDA не найдена)"
