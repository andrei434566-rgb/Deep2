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

    train = commands.add_parser("train", help="Обучить локальную YOLO segmentation модель и сохранить best.pt")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--architecture", default="yolo11n-seg.yaml", help="YAML-архитектура YOLO-seg со случайными весами")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=12)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--device", help="Например 0, 1 или cpu")
    train.add_argument("--description-epochs", type=int, default=40)
    train.add_argument("--description-patience", type=int, default=8)
    train.add_argument("--visual-only", action="store_true", help="Обучить только YOLO best.pt без модели краткого описания")

    automatic = commands.add_parser(
        "auto-train", help="Автоматически обработать очередь Excel + фото, собрать датасет и обучить best.pt",
    )
    automatic.add_argument("--manifest", type=Path, required=True, help="JSON со списком пар Excel + папка фото")
    automatic.add_argument("--dataset", type=Path, required=True)
    automatic.add_argument("--output", type=Path, required=True)
    automatic.add_argument("--architecture", default="yolo11n-seg.yaml")
    automatic.add_argument("--epochs", type=int, default=50)
    automatic.add_argument("--patience", type=int, default=12)
    automatic.add_argument("--description-epochs", type=int, default=40)
    automatic.add_argument("--no-ocr", action="store_true")

    discovery = commands.add_parser("discover-wells", help="Найти пары Excel + фото в архивной папке")
    discovery.add_argument("--root", type=Path, required=True)

    analyze = commands.add_parser("analyze", help="Применить единый best.pt и создать стандартный Excel из 22 столбцов")
    analyze.add_argument("--model", type=Path, required=True)
    analyze.add_argument("--photos", type=Path, required=True)
    analyze.add_argument("--output-excel", type=Path, required=True)
    analyze.add_argument("--description-model", type=Path)
    analyze.add_argument("--confidence", type=float, default=0.25)

    commands.add_parser("gui", help="Открыть графическое приложение")
    return root


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args(argv)
    try:
        if args.command == "create":
            from .project import create_project
            result = create_project(args.excel, args.photos, args.project, mapping_file=args.mapping, use_ocr=args.ocr)
        elif args.command == "refresh":
            from .project import refresh_project
            result = refresh_project(args.project)
        elif args.command == "dataset":
            from .catalog import load_project_catalog
            from .dataset import build_dataset
            projects = load_project_catalog(args.catalog) if args.catalog else args.project
            result = build_dataset(projects, args.output)
        elif args.command == "train":
            from .training import train_bundle, train_model
            device = args.device
            if isinstance(device, str) and device.isdigit():
                device = int(device)
            trainer = train_model if args.visual_only else train_bundle
            options = {
                "epochs": args.epochs, "patience": args.patience,
                "image_size": args.imgsz, "device": device,
                "architecture": args.architecture,
            }
            if not args.visual_only:
                options.update(
                    description_epochs=args.description_epochs,
                    description_patience=args.description_patience,
                )
            result = trainer(args.dataset, args.output, **options)
        elif args.command == "auto-train":
            from .autopipeline import run_automatic_training
            result = run_automatic_training(
                args.manifest, args.dataset, args.output,
                architecture=args.architecture, epochs=args.epochs,
                patience=args.patience, description_epochs=args.description_epochs,
                use_ocr=not args.no_ocr,
                progress=lambda message: print(message, flush=True),
            )
        elif args.command == "discover-wells":
            from .autodiscovery import discover_well_pairs
            result = discover_well_pairs(args.root)
        elif args.command == "analyze":
            from .inference import analyze_photos_to_excel
            result = analyze_photos_to_excel(
                args.model, args.photos, args.output_excel,
                description_model=args.description_model, confidence=args.confidence,
            )
        else:
            from .gui import run_gui
            return run_gui()
        if args.command == "discover-wells":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command != "auto-train":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
