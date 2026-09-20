from __future__ import annotations

import re
from pathlib import Path

from .models import PhotoRecord
from .tabular import as_float, display_text


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_filename(path: Path) -> PhotoRecord:
    """Parse names such as `Р-31 3002,00–3004,96 (1).jpg`."""
    stem = path.stem.replace("−", "-").replace("–", "-").replace("—", "-")
    matches = list(re.finditer(r"(?P<top>\d{2,6}(?:[.,]\d{1,4})?)\s*-\s*(?P<base>\d{2,6}(?:[.,]\d{1,4})?)", stem))
    for match in reversed(matches):
        top = as_float(match.group("top"))
        base = as_float(match.group("base"))
        if top is None or base is None or base <= top or base - top > 500:
            continue
        prefix = stem[: match.start()].strip(" _-.,;()[]")
        prefix = re.sub(r"^(?:фото|photo|img|image)[ _-]*", "", prefix, flags=re.IGNORECASE)
        well = display_text(prefix)
        return PhotoRecord(path=path, well=well, top=top, base=base, source="filename", mapping_confirmed=True)
    return PhotoRecord(path=path)


def discover_photos(folder: Path, recursive: bool = True, use_ocr: bool = False) -> list[PhotoRecord]:
    folder = Path(folder).expanduser().resolve(strict=True)
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    paths = sorted((path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS), key=lambda item: item.name.casefold())
    records = [parse_filename(path) for path in paths]
    if use_ocr:
        records = [_with_ocr(record) if not record.has_interval else record for record in records]
    return records


def _with_ocr(record: PhotoRecord) -> PhotoRecord:
    try:
        import cv2
        import numpy as np
        import pytesseract
    except ImportError:
        return record
    try:
        image = cv2.imdecode(np.frombuffer(record.path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return record
        height, width = image.shape[:2]
        footer = image[int(height * 0.65):height, :width]
        gray = cv2.cvtColor(cv2.resize(footer, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang="rus+eng", config="--psm 6")
    except Exception:
        return record
    normalized = text.replace(",", ".").replace("−", "-").replace("–", "-").replace("—", "-")
    pairs = []
    for match in re.finditer(r"(\d{2,6}(?:\.\d{1,4})?)\D{1,18}(\d{2,6}(?:\.\d{1,4})?)", normalized):
        top, base = float(match.group(1)), float(match.group(2))
        if base > top and 0.01 <= base - top <= 500:
            pairs.append((top, base))
    if not pairs:
        return record
    top, base = max(pairs, key=lambda value: value[1] - value[0])
    well_match = re.search(r"(?:скв(?:ажина)?\.?|well)\s*(?:№|n|#)?\s*([a-zа-я0-9][a-zа-я0-9._/-]{0,30})", text.casefold())
    well = well_match.group(1).strip("._/- ") if well_match else record.well
    return PhotoRecord(record.path, well, top, base, "ocr", False)
