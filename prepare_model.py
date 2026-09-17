"""Portable blueprint/check/training CLI. Never downloads or trains implicitly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.infrastructure.model_blueprint import check_model_blueprint, class_registry, export_model_blueprint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Заготовка и подготовка модели фаций; веса нужны только для train.")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Экспортировать справочник и рецепт без весов")
    export.add_argument("--output", type=Path, required=True, help="Новая папка заготовки")
    check = commands.add_parser("check", help="Проверить заготовку/датасет без загрузки весов")
    check.add_argument("--blueprint", type=Path)
    check.add_argument("--dataset", type=Path, help="Папка с data.yaml и dataset_manifest.json")
    train = commands.add_parser("train", help="Явно запустить дообучение на ПК с моделью и CUDA")
    train.add_argument("--model", type=Path, required=True, help="Ваш существующий доверенный .pt сегментации")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True, help="Новая папка для модели-кандидата")
    train.add_argument("--runs-dir", type=Path, help="Новая папка журналов обучения")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=12)
    return parser


def check_dataset(dataset: Path) -> dict:
    """Check the portable canonical dataset, never loading weights or training."""
    from app.infrastructure.ml.fine_tune_worker import load_training_manifest

    dataset = Path(dataset).resolve(strict=True)
    manifest = load_training_manifest(dataset / "data.yaml")
    if manifest.get("class_names") != class_registry()["class_names"]:
        raise ValueError("Датасет использует другой порядок/состав фаций. Экспортируйте его из текущей версии приложения.")
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            directory = dataset / kind / split
            if not directory.is_dir() or not any(directory.iterdir()):
                raise ValueError(f"Не найдены перенесённые данные: {directory}")
    return manifest


def _train(args) -> dict:
    model_path = args.model.resolve(strict=True)
    if not model_path.is_file() or model_path.suffix.lower() != ".pt":
        raise ValueError("Укажите существующий доверенный локальный файл .pt; загрузка модели из сети не выполняется.")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("Количество эпох и patience должны быть положительными.")
    output = args.output.absolute()
    runs = args.runs_dir.absolute() if args.runs_dir else output.with_name(output.name + "_runs")
    if output.exists() or runs.exists() or output == runs or output in runs.parents or runs in output.parents:
        raise FileExistsError("Папки модели и журналов должны быть новыми и разными; предыдущие результаты не перезаписываются.")
    manifest = check_dataset(args.dataset)
    dataset = args.dataset.resolve(strict=True)
    from PySide6.QtCore import QCoreApplication
    from app.infrastructure.ml.fine_tune_worker import FineTuneWorker

    application = QCoreApplication.instance() or QCoreApplication([])
    state: dict = {}
    worker = FineTuneWorker(
        model_path, dataset / "data.yaml", runs, output, epochs=args.epochs,
        dataset_summary={key: value for key, value in manifest.items() if key != "samples"},
        early_stopping_patience=args.patience,
    )
    worker.progress.connect(lambda message: print(message, flush=True))
    worker.epoch_progress.connect(lambda current, total: print(f"Эпоха {current}/{total}", flush=True))
    worker.succeeded.connect(lambda path: state.update(model=path))
    worker.failed.connect(lambda message: state.update(error=message))
    worker.run()
    # Retain the Qt application until all worker signals have been handled.
    _ = application
    if state.get("error"):
        raise RuntimeError(state["error"])
    if not state.get("model"):
        raise RuntimeError("Обучение не вернуло файл результата.")
    return {"status": "candidate_requires_review", "model": state["model"]}


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            result = export_model_blueprint(args.output)
        elif args.command == "check":
            if args.blueprint is None and args.dataset is None:
                parser.error("check требует --blueprint и/или --dataset")
            result = {}
            if args.blueprint:
                result["blueprint"] = check_model_blueprint(args.blueprint)
            if args.dataset:
                manifest = check_dataset(args.dataset)
                result["dataset"] = {key: manifest.get(key) for key in (
                    "class_names", "sample_count", "train_photo_count", "val_photo_count", "split_strategy", "warnings"
                )}
        else:
            result = _train(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
