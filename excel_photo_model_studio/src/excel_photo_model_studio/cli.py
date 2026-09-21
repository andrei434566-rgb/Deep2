from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import load_project_catalog
from .dataset import build_dataset
from .inference import analyze_photos_to_excel
from .project import create_project, refresh_project
from .training import train_bundle, train_model


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
    train.add_argument("--visual-only", action="store_true", help="Обучить только YOLO best.pt без текста №22")

    analyze = commands.add_parser("analyze", help="Применить единый best.pt и создать стандартный Excel из 22 столбцов")
    analyze.add_argument("--model", type=Path, required=True)
    analyze.add_argument("--photos", type=Path, required=True)
    analyze.add_argument("--output-excel", type=Path, required=True)
    analyze.add_argument("--description-model", type=Path)
    analyze.add_argument("--confidence", type=float, default=0.25)

    commands.add_parser("gui", help="Открыть графическое приложение")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "create":
            result = create_project(args.excel, args.photos, args.project, mapping_file=args.mapping, use_ocr=args.ocr)
        elif args.command == "refresh":
            result = refresh_project(args.project)
        elif args.command == "dataset":
            projects = load_project_catalog(args.catalog) if args.catalog else args.project
            result = build_dataset(projects, args.output)
        elif args.command == "train":
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
        elif args.command == "analyze":
            result = analyze_photos_to_excel(
                args.model, args.photos, args.output_excel,
                description_model=args.description_model, confidence=args.confidence,
            )
        else:
            from .gui import run_gui
            return run_gui()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
