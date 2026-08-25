"""Standalone: split a core photo into rectangles and rank all model classes.

Run:
    pip install ultralytics opencv-python numpy
    python razr_standalone.py

Then enter the path to a YOLO classification .pt model and a core photograph.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class TextureInterval:
    column_index: int
    left: int
    right: int
    top: int
    bottom: int
    evidence: str


class RectangleBoundaryDetector:
    """Find sustained visual texture changes; it never assigns geological facies."""

    def analyse(self, image: np.ndarray) -> tuple[np.ndarray, list[TextureInterval]]:
        columns = self._find_columns(image)
        intervals: list[TextureInterval] = []
        for index, (left, top, right, bottom) in enumerate(columns, start=1):
            intervals.extend(self._split_column(image, index, left, top, right, bottom))
        return self._render(image, intervals), intervals

    @staticmethod
    def _find_columns(image: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Usual core photos: rock is less saturated than the brown box and darker than white background.
        candidate = (hsv[:, :, 1] < 70) & (hsv[:, :, 2] < 235)
        x_score = RectangleBoundaryDetector._smooth(candidate.mean(axis=0).astype(np.float32), 7)
        active_x = x_score >= 0.48
        boxes: list[tuple[int, int, int, int]] = []
        min_width = max(10, int(width * 0.018))
        for left, right in RectangleBoundaryDetector._runs(active_x):
            if right - left < min_width or right - left > width * 0.30:
                continue
            y_score = RectangleBoundaryDetector._smooth(candidate[:, left:right].mean(axis=1).astype(np.float32), 15)
            y_runs = RectangleBoundaryDetector._runs(y_score >= 0.24)
            if not y_runs:
                continue
            top, bottom = max(y_runs, key=lambda item: item[1] - item[0])
            if bottom - top >= height * 0.25:
                boxes.append((max(0, left - 1), max(0, top - 1), min(width, right + 1), min(height, bottom + 1)))
        if boxes:
            return boxes
        return RectangleBoundaryDetector._component_fallback(candidate.astype(np.uint8) * 255)

    @staticmethod
    def _component_fallback(candidate: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = candidate.shape
        connected = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, max(17, int(height * 0.075)))),
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
        boxes = []
        for component in range(1, count):
            left, top, box_width, box_height, area = stats[component]
            aspect = box_height / max(box_width, 1)
            if (
                box_width >= max(10, int(width * 0.018))
                and box_width <= width * 0.30
                and box_height >= height * 0.25
                and 2.0 <= aspect <= 35.0
                and area >= box_width * box_height * 0.20
            ):
                boxes.append((int(left), int(top), int(left + box_width), int(top + box_height)))
        return sorted(boxes, key=lambda item: item[0])

    def _split_column(self, image: np.ndarray, column_index: int, left: int, top: int, right: int, bottom: int) -> list[TextureInterval]:
        height = bottom - top
        inset = max(1, int((right - left) * 0.08))
        inner_left = min(right - 2, left + inset)
        inner_right = max(inner_left + 2, right - inset)
        roi = image[top:bottom, inner_left:inner_right]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if gray.shape[0] < 20 or gray.shape[1] < 3:
            return []

        # The 35th percentile makes white sample labels affect the result less.
        brightness_rows = np.quantile(gray, 0.35, axis=1)
        gradient_rows = np.zeros_like(brightness_rows)
        gradient_rows[1:] = np.mean(np.abs(np.diff(gray, axis=0)), axis=1)
        brightness = self._smooth(brightness_rows, self._odd(max(9, height // 70)))
        lamination = self._smooth(gradient_rows, self._odd(max(7, height // 95)))
        brightness = self._scale(brightness)
        lamination = self._scale(lamination)
        change = np.abs(np.diff(brightness, prepend=brightness[0])) + 0.65 * np.abs(np.diff(lamination, prepend=lamination[0]))
        change = self._smooth(change, self._odd(max(5, height // 160)))
        boundaries = self._boundaries(change, height)
        edges = [0, *boundaries, height]

        intervals = []
        for local_top, local_bottom in zip(edges[:-1], edges[1:]):
            if local_bottom - local_top < max(8, height // 120):
                continue
            mean_brightness = float(np.mean(brightness[local_top:local_bottom]))
            mean_lamination = float(np.mean(lamination[local_top:local_bottom]))
            evidence = "high lamination" if mean_lamination >= 0.58 else "persistent visual texture"
            intervals.append(TextureInterval(column_index, left, right, top + local_top, top + local_bottom, evidence))
        return intervals

    @staticmethod
    def _boundaries(change: np.ndarray, height: int) -> list[int]:
        minimum_height = max(28, int(height * 0.075))
        median = float(np.median(change))
        threshold = max(float(np.quantile(change, 0.94)), median + 1.85 * float(np.median(np.abs(change - median))))
        candidates = [
            index
            for index in range(2, len(change) - 2)
            if change[index] >= threshold and change[index] >= change[index - 1] and change[index] >= change[index + 1]
        ]
        accepted: list[int] = []
        for index in sorted(candidates, key=lambda item: float(change[item]), reverse=True):
            if index < minimum_height or height - index < minimum_height:
                continue
            if all(abs(index - other) >= minimum_height for other in accepted):
                accepted.append(index)
        return sorted(accepted)

    @staticmethod
    def _render(image: np.ndarray, intervals: list[TextureInterval]) -> np.ndarray:
        result = image.copy()
        overlay = result.copy()
        color = (240, 190, 40)  # Neutral cyan, not a facies color.
        for interval in intervals:
            cv2.rectangle(overlay, (interval.left, interval.top), (interval.right - 1, interval.bottom - 1), color, -1)
        result = cv2.addWeighted(overlay, 0.16, result, 0.84, 0)
        for interval in intervals:
            cv2.rectangle(result, (interval.left, interval.top), (interval.right - 1, interval.bottom - 1), color, 2)
        return result

    @staticmethod
    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return values.copy()
        padded = np.pad(values, (window // 2, window // 2), mode="edge")
        return np.convolve(padded, np.ones(window, dtype=np.float32) / window, mode="valid")

    @staticmethod
    def _scale(values: np.ndarray) -> np.ndarray:
        low, high = np.quantile(values, [0.05, 0.95])
        if high - low < 1e-6:
            return np.full_like(values, 0.5)
        return np.clip((values - low) / (high - low), 0.0, 1.0)

    @staticmethod
    def _odd(value: int) -> int:
        return value if value % 2 else value + 1

    @staticmethod
    def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
        padded = np.pad(mask.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot open image: {path}")
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Core boundaries plus probabilities from a standalone script.")
    parser.add_argument("model", nargs="?", type=Path, help="YOLO classification .pt model")
    parser.add_argument("image", nargs="?", type=Path, help="Core photograph")
    parser.add_argument("--output", type=Path, help="New output folder")
    return parser.parse_args()


def classify(model, crop: np.ndarray) -> list[dict[str, object]]:
    result = model(crop, verbose=False)[0]
    if result.probs is None:
        raise ValueError("The selected .pt is not a YOLO classification model.")
    probabilities = result.probs.data.detach().cpu().numpy().astype(float)
    return sorted(
        [{"class": str(result.names[index]), "probability": round(float(value), 6)} for index, value in enumerate(probabilities)],
        key=lambda item: float(item["probability"]),
        reverse=True,
    )


def main() -> int:
    args = parse_args()
    model_path = args.model or Path(input("Path to classification model .pt: ").strip().strip('"'))
    image_path = args.image or Path(input("Path to core photo: ").strip().strip('"'))
    if not model_path.is_file():
        print(f"Model file not found: {model_path}")
        return 1
    if not image_path.is_file():
        print(f"Image file not found: {image_path}")
        return 1

    try:
        from ultralytics import YOLO

        source = read_image(image_path)
        annotated, intervals = RectangleBoundaryDetector().analyse(source)
        model = YOLO(str(model_path))
    except Exception as error:
        print(f"Preparation error: {error}")
        return 1

    output_dir = args.output or image_path.with_name(f"{image_path.stem}_six_facies_result")
    if output_dir.exists():
        print(f"Output folder already exists: {output_dir}\nUse --output to choose another folder.")
        return 1
    output_dir.mkdir(parents=True)

    report: list[dict[str, object]] = []
    for number, interval in enumerate(intervals, start=1):
        crop = source[interval.top:interval.bottom, interval.left:interval.right]
        if crop.size == 0:
            continue
        try:
            ranking = classify(model, crop)
        except Exception as error:
            print(f"Package {number}: skipped — {error}")
            continue
        report.append(
            {
                "package": number,
                "rectangle": {"left": interval.left, "top": interval.top, "right": interval.right, "bottom": interval.bottom},
                "probabilities_desc": ranking,
            }
        )
        print(f"\nPackage {number}, x={interval.left}..{interval.right}, y={interval.top}..{interval.bottom}")
        for item in ranking:
            print(f"  {item['class']}: {float(item['probability']):.1%}")

    ok, encoded = cv2.imencode(".png", annotated)
    if ok:
        encoded.tofile(str(output_dir / "boundaries.png"))
    (output_dir / "probabilities.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone: {output_dir}")
    print("boundaries.png contains rectangles; probabilities.json contains all class scores in descending order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
