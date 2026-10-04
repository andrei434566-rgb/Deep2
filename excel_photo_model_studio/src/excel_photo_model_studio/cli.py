from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Excel Photo Model Studio: Excel + photos -> reviewed YOLO dataset -> best.pt")
    commands = root.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="Создать проект сопоставления")
    create.add_argument("--excel", type=Path, nargs="+", required=True, help="Один или несколько Excel/CSV-файлов либо папок")
    create.add_argument("--photos", type=Path, required=True)
    create.add_argument("--project", type=Path, required=True)
    create.add_argument("--mapping", type=Path, help="Необязательный JSON с ручным соответствием столбцов")
    create.add_argument("--ocr", action="store_true", help="Попробовать Tesseract OCR для фото без интервала в имени")

    refresh = commands.add_parser("refresh", help="Пересчитать сопоставление после правки photo_map.csv/column_mapping.json")
    refresh.add_argument("--project", type=Path, required=True)

    dataset = commands.add_parser("dataset", help="Собрать YOLO-seg датасет только из подтверждённых масок")
    dataset_source = dataset.add_mutually_exclusive_group(required=True)
    dataset_source.add_argument("--project", type=Path, nargs="+", help="Один или несколько проектов скважин")
    dataset_source.add_argument("--catalog", type=Path, help="Внутренний каталог последовательно обработанных скважин")
    dataset.add_argument("--output", type=Path, required=True)

    train = commands.add_parser("train", help="Дообучить YOLO11-seg на CUDA и сохранить best.pt")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--weights", default="yolo11s-seg.pt", help="Предобученный checkpoint YOLO11-seg")
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--patience", type=int, default=80)
    train.add_argument("--imgsz", type=int, default=1024)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--device", type=int, choices=(0,), default=0, help="Только CUDA GPU 0; CPU не используется")

    discovery = commands.add_parser("discover-wells", help="Найти пары Excel + фото в архивной папке")
    discovery.add_argument("--root", type=Path, required=True)

    analyze = commands.add_parser("analyze", help="Применить единый best.pt и создать стандартный Excel с полями индекса, названия и описания фации")
    analyze.add_argument("--model", type=Path, required=True)
    analyze.add_argument("--photos", type=Path, required=True)
    analyze.add_argument("--output-excel", type=Path, required=True)
    analyze.add_argument("--confidence", type=float, default=0.25)

    commands.add_parser("gui", help="Открыть графическое приложение")
    self_test = commands.add_parser("self-test", help="Проверить библиотеки, окно, Excel, OCR и нейросеть без загрузок")
    self_test.add_argument("--output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args(argv)
    try:
        if args.command == "self-test":
            from .self_test import run_self_test
            result = run_self_test(args.output)
        elif args.command == "create":
            from .project import create_project
            result = create_project(args.excel, args.photos, args.project, mapping_file=args.mapping, use_ocr=args.ocr)
        elif args.command == "refresh":
            from .project import refresh_project
            result = refresh_project(args.project)
        elif args.command == "dataset":
            from .catalog import load_confirmed_project_catalog
            from .dataset import build_dataset
            projects = load_confirmed_project_catalog(args.catalog) if args.catalog else args.project
            if args.catalog and not projects:
                raise ValueError(
                    "В накопительном каталоге пока нет подтверждённых масок; "
                    "сначала сохраните хотя бы один интервал."
                )
            result = build_dataset(projects, args.output)
        elif args.command == "train":
            from .training import train_model
            result = train_model(
                args.dataset, args.output, weights=args.weights,
                epochs=args.epochs, patience=args.patience,
                image_size=args.imgsz, batch_size=args.batch_size, device=args.device,
            )
        elif args.command == "discover-wells":
            from .autodiscovery import discover_well_pairs
            result = discover_well_pairs(args.root)
        elif args.command == "analyze":
            from .inference import analyze_photos_to_excel
            result = analyze_photos_to_excel(
                args.model, args.photos, args.output_excel,
                confidence=args.confidence,
            )
        else:
            from .gui import run_gui
            return run_gui()
        if args.command == "discover-wells":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
