"""Background YOLO fine-tuning worker."""

from __future__ import annotations

import shutil
import sys
import json
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot


@contextmanager
def _safe_training_streams(log_path: Path):
    """Give Ultralytics writable streams when DeepCore is launched by pythonw."""
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        stdout = original_stdout if callable(getattr(original_stdout, "write", None)) else log_file
        stderr = original_stderr if callable(getattr(original_stderr, "write", None)) else log_file
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield


class FineTuneWorker(QObject):
    progress = Signal(str)
    epoch_progress = Signal(int, int)
    succeeded = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, model_path: Path, data_yaml: Path, runs_dir: Path, published_dir: Path, epochs: int = 20):
        super().__init__()
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.runs_dir = runs_dir
        self.published_dir = published_dir
        self.epochs = epochs

    @Slot()
    def run(self) -> None:
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            device, device_label = self._best_device()
            self.progress.emit(f"Дообучение запущено: {self.epochs} эпох · {device_label}")
            with _safe_training_streams(self.runs_dir / "fine_tune.log"):
                from ultralytics import YOLO

                model = YOLO(str(self.model_path))
                model.add_callback("on_train_epoch_end", self._report_epoch)
                result = model.train(
                    data=str(self.data_yaml),
                    epochs=self.epochs,
                    imgsz=640,
                    device=device,
                    batch=-1 if device != "cpu" else 4,
                    cache=False,
                    amp=device != "cpu",
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
            (self.published_dir / "training_info.json").write_text(
                json.dumps(
                    {
                        "source_model": str(self.model_path),
                        "epochs": self.epochs,
                        "dataset": str(self.data_yaml.parent),
                        "run": str(save_dir),
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

    @staticmethod
    def _best_device() -> tuple[int | str, str]:
        try:
            import torch

            if torch.cuda.is_available():
                return 0, f"GPU: {torch.cuda.get_device_name(0)}"
        except (ImportError, RuntimeError):
            pass
        return "cpu", "CPU (CUDA не найдена)"
