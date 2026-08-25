"""Split a core photo into rectangles, then return six sorted class probabilities.

Example:
    python run_boundary_classifier.py six_facies_training/six_facies_best.pt core.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.infrastructure.ml.rule_based_facies import RuleBasedFaciesDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Границы пакетов + вероятности шести фаций.")
    parser.add_argument("model", nargs="?", type=Path, help="Файл six_facies_best.pt")
    parser.add_argument("image", nargs="?", type=Path, help="Фото керна")
    parser.add_argument("--output", type=Path, help="Папка результата; по умолчанию рядом с фото.")
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось открыть изображение: {path}")
    return image


def probabilities_for_crop(model, crop: np.ndarray) -> list[dict[str, object]]:
    result = model(crop, verbose=False)[0]
    if result.probs is None:
        raise ValueError("Выбранный .pt не является классификационной моделью YOLO.")
    values = result.probs.data.detach().cpu().numpy().astype(float)
    names = result.names
    output = [{"class": str(names[index]), "probability": round(float(value), 6)} for index, value in enumerate(values)]
    return sorted(output, key=lambda item: float(item["probability"]), reverse=True)


def main() -> int:
    args = parse_args()
    model_path = args.model or Path(input("Путь к файлу модели .pt: ").strip().strip('"'))
    image_path = args.image or Path(input("Путь к фотографии керна: ").strip().strip('"'))
    if not model_path.is_file():
        print(f"Модель не найдена: {model_path}")
        return 1
    if not image_path.is_file():
        print(f"Фотография не найдена: {image_path}")
        return 1
    try:
        from ultralytics import YOLO

        image = read_image(image_path)
        annotated, intervals = RuleBasedFaciesDetector().analyse(image)
        classifier = YOLO(str(model_path))
    except Exception as error:
        print(f"Ошибка подготовки: {error}")
        return 1

    output_dir = args.output or image_path.with_name(f"{image_path.stem}_six_facies_result")
    if output_dir.exists():
        print(f"Папка уже существует: {output_dir}. Укажите другую через --output.")
        return 1
    output_dir.mkdir(parents=True)

    rows = []
    for index, interval in enumerate(intervals, start=1):
        crop = image[interval.top:interval.bottom, interval.left:interval.right]
        if crop.size == 0:
            continue
        try:
            ranking = probabilities_for_crop(classifier, crop)
        except Exception as error:
            print(f"Пакет {index}: пропущен — {error}")
            continue
        rows.append(
            {
                "package": index,
                "rectangle": {"left": interval.left, "top": interval.top, "right": interval.right, "bottom": interval.bottom},
                "texture_evidence": interval.evidence,
                "probabilities_desc": ranking,
            }
        )
        print(f"\nПакет {index}: x={interval.left}..{interval.right}, y={interval.top}..{interval.bottom}")
        for item in ranking:
            print(f"  {item['class']}: {float(item['probability']):.1%}")

    ok, encoded = cv2.imencode(".png", annotated)
    if ok:
        encoded.tofile(str(output_dir / "boundaries.png"))
    (output_dir / "probabilities.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nГотово: {output_dir}")
    print("boundaries.png — исходное фото с прямоугольниками пакетов.")
    print("probabilities.json — шесть вероятностей по убыванию для каждого пакета.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
