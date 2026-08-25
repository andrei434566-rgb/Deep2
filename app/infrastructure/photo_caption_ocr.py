"""Best-effort extraction of a well name and depth range from a core-photo footer."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class CaptionMetadata:
    well: str
    top: float
    base: float
    text: str


def read_caption_metadata(path: Path) -> CaptionMetadata:
    """OCR the lower band of a photo and return the most plausible depth pair."""
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("Не установлен pytesseract. Установите зависимости DeepCore.") from exc

    _configure_tesseract(pytesseract)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось открыть файл: {path.name}")
    height, width = image.shape[:2]
    footer = image[int(height * 0.68):height, :width]
    footer = cv2.resize(footer, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    grayscale = cv2.cvtColor(footer, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.normalize(grayscale, None, 0, 255, cv2.NORM_MINMAX)
    try:
        text = pytesseract.image_to_string(grayscale, lang="rus+eng", config="--psm 6")
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Для чтения подписи на фото установите Tesseract OCR с русским языком. "
            "После установки перезапустите DeepCore."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise RuntimeError(f"Не удалось распознать подпись Tesseract: {exc}") from exc
    top, base = _find_depth_range(text)
    well = _find_well(text)
    if not well:
        raise ValueError(f"{path.name}: OCR не нашёл номер скважины в подписи.")
    if top is None or base is None:
        raise ValueError(f"{path.name}: OCR не нашёл интервал глубин в подписи.")
    return CaptionMetadata(well=well, top=top, base=base, text=text)


def _configure_tesseract(pytesseract) -> None:
    if shutil.which("tesseract"):
        return
    candidates = (
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    )
    executable = next((candidate for candidate in candidates if candidate.is_file()), None)
    if executable is not None:
        pytesseract.pytesseract.tesseract_cmd = str(executable)


def _find_well(text: str) -> str:
    match = re.search(
        r"(?:скв(?:ажина)?\.?\s*(?:№|n|#)?\s*)([a-zа-я0-9][a-zа-я0-9._/-]{0,24})",
        text.casefold(),
    )
    return match.group(1).strip("._/- ") if match else ""


def _find_depth_range(text: str) -> tuple[float | None, float | None]:
    normalized = text.replace(",", ".")
    pairs = []
    for match in re.finditer(r"(\d{3,5}(?:\.\d{1,3})?)\D{1,18}(\d{3,5}(?:\.\d{1,3})?)", normalized):
        top, base = float(match.group(1)), float(match.group(2))
        if base > top and 0.03 <= base - top <= 100:
            pairs.append((top, base))
    # Captions can contain both a depth range and a date. The interval with the
    # largest plausible span is normally the core sampling interval.
    return max(pairs, key=lambda value: value[1] - value[0]) if pairs else (None, None)
