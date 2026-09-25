from __future__ import annotations

import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Iterable

from .models import PhotoRecord
from .tabular import as_float, display_text


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_filename(path: Path) -> PhotoRecord:
    """Parse names such as `Р-31 3002,00–3004,96 (1).jpg`."""
    stem = path.stem.replace("−", "-").replace("–", "-").replace("—", "-")
    matches = list(re.finditer(r"(?P<top>\d{2,6}(?:[.,]\d{1,4})?)\s*-\s*(?P<base>\d{2,6}(?:[.,]\d{1,4})?)", stem))
    for match in reversed(matches):
        if _is_figure_number(stem, match.start()):
            continue
        top = as_float(match.group("top"))
        base = as_float(match.group("base"))
        if top is None or base is None or base <= top or base - top > 500:
            continue
        prefix = stem[: match.start()].strip(" _-.,;()[]")
        prefix = re.sub(r"^(?:фото|photo|img|image)[ _-]*", "", prefix, flags=re.IGNORECASE)
        well = _well_hint(prefix) or display_text(prefix)
        return PhotoRecord(path=path, well=well, top=top, base=base, source="filename", mapping_confirmed=True)
    return PhotoRecord(path=path, well=_well_hint(stem))


def _is_figure_number(stem: str, start: int) -> bool:
    """Figure references such as ``Fig. 15.1-20`` are not depth intervals."""
    before = stem[:start]
    return re.search(r"(?:рис(?:унок)?|fig(?:ure)?)\s*\.?\s*$", before, re.IGNORECASE) is not None


def discover_photos(
    folder: Path,
    recursive: bool = True,
    use_ocr: bool = False,
    expected_intervals: Iterable[tuple[float, float]] = (),
) -> list[PhotoRecord]:
    folder = Path(folder).expanduser().resolve(strict=True)
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    paths = sorted((path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS), key=lambda item: item.name.casefold())
    records = [parse_filename(path) for path in paths]
    if use_ocr:
        unresolved = [index for index, record in enumerate(records) if not record.has_interval]
        if unresolved:
            # Tesseract runs as an external process. Two workers provide a
            # useful speed-up without exhausting memory on large well sets.
            with ThreadPoolExecutor(max_workers=min(2, len(unresolved))) as pool:
                worker = partial(_with_ocr, expected_intervals=tuple(expected_intervals))
                resolved = pool.map(worker, (records[index] for index in unresolved))
                for index, record in zip(unresolved, resolved):
                    records[index] = record
    return records


def _with_ocr(
    record: PhotoRecord,
    *,
    expected_intervals: tuple[tuple[float, float], ...] = (),
) -> PhotoRecord:
    try:
        import cv2
        import numpy as np
        import pytesseract
    except ImportError:
        return replace(record, source="ocr_unavailable")
    try:
        if not _configure_tesseract(pytesseract):
            return replace(record, source="ocr_unavailable")
        image = cv2.imdecode(np.frombuffer(record.path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return replace(record, source="ocr_not_found")
        height, width = image.shape[:2]
        header = image[:max(1, int(height * 0.34)), :width]
        footer = image[int(height * 0.54):height, :width]
        separator = np.full((max(12, height // 120), width, 3), 255, dtype=np.uint8)
        region = np.vstack((header, separator, footer))
        scale = max(1.0, min(2.4, 2600.0 / max(1, region.shape[1])))
        region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 13,
        )
        text_parts = []
        for prepared, psm in ((gray, 6), (binary, 11)):
            try:
                text_parts.append(pytesseract.image_to_string(prepared, lang="rus+eng", config=f"--psm {psm}"))
            except Exception:
                text_parts.append(pytesseract.image_to_string(prepared, lang="eng", config=f"--psm {psm}"))
        text = "\n".join(text_parts)
    except Exception:
        return replace(record, source="ocr_not_found")
    interval = extract_depth_interval(text, expected_intervals)
    if interval is None:
        return replace(record, source="ocr_not_found")
    well = extract_well(text) or record.well
    return PhotoRecord(record.path, well, interval[0], interval[1], "ocr", False)


def extract_depth_interval(
    text: str,
    expected_intervals: Iterable[tuple[float, float]] = (),
) -> tuple[float, float] | None:
    """Extract the full photographed-core interval from an OCR transcript."""
    # Keep OCR line boundaries. Previously display_text flattened every line,
    # then paired unrelated values several tokens apart (often ruler ticks,
    # column headers and page numbers). Depth limits should be an explicit
    # range or adjacent values on the same OCR line.
    raw = str(text or "").replace("−", "-").replace("–", "-").replace("—", "-")
    lines = [" ".join(line.replace(",", ".").split()) for line in raw.splitlines()]
    label_pattern = re.compile(
        r"(?:интервал\s+(?:отбора\s+)?керна|core\s+interval)", re.IGNORECASE,
    )
    label_lines = {
        index for index, line in enumerate(lines)
        if label_pattern.search(line)
    }
    label_context = {
        line_index
        for label_index in label_lines
        for line_index in range(max(0, label_index - 1), min(len(lines), label_index + 3))
    }
    candidates: list[tuple[int, float, float]] = []
    number = r"\d{2,6}(?:\.\d{1,4})?"
    explicit_range = re.compile(rf"(?<!\d)({number})\s*(?:-|\bдо\b|\bto\b)\s*({number})(?!\d)", re.IGNORECASE)
    plain_number = re.compile(rf"(?<!\d)({number})(?!\d)")

    for line_index, line in enumerate(lines):
        if not line:
            continue
        labelled = line_index in label_context
        priority = 1200 if labelled else 100
        for match in explicit_range.finditer(line):
            has_depth_unit = re.search(r"(?:\sм\.?(?:\s|$)|\bметр)", line, re.IGNORECASE) is not None
            candidates.append((priority + 100 + (100 if has_depth_unit else 0), float(match.group(1)), float(match.group(2))))

        values = [float(value) for value in plain_number.findall(line)]
        # Adjacent OCR values on the same line cover captions where the dash
        # was lost, without constructing pairs across unrelated rows.
        for top, base in zip(values, values[1:]):
            candidates.append((priority, top, base))

    candidates = [
        item for item in candidates
        if 0.01 <= item[2] - item[1] <= 500
        and min(item[1], item[2]) >= max(item[1], item[2]) * 0.35
    ]
    if not candidates:
        return None
    expected = tuple(expected_intervals)
    if expected:
        aligned: list[tuple[float, int, float, float]] = []
        for priority, top, base in candidates:
            for expected_top, expected_base in expected:
                error = abs(top - expected_top) + abs(base - expected_base)
                # A generous multi-metre tolerance allowed column-header and
                # ruler numbers to masquerade as the caption interval. OCR
                # may differ by a few tenths, but both endpoints must remain
                # close to one of the actual Excel core intervals.
                if abs(top - expected_top) <= 0.5 and abs(base - expected_base) <= 0.5:
                    aligned.append((error, -priority, top, base))
        if aligned:
            _, _, top, base = min(aligned)
            return top, base
        return None
    # Without Excel core intervals, require either an interval caption or an
    # explicit range with a depth unit. Bare numeric lines are commonly ruler
    # graduations and must not become an unverified photo depth.
    candidates = [item for item in candidates if item[0] >= 200]
    if not candidates:
        return None
    _, top, base = max(candidates, key=lambda item: (item[0], item[2] - item[1]))
    return top, base


def extract_well(text: str) -> str:
    normalized = display_text(text)
    match = re.search(
        r"(?:скв(?:ажина)?\.?|well)\s*(?:№|n|#)?\s*([a-zа-я0-9][a-zа-я0-9._/-]{0,30})",
        normalized, flags=re.IGNORECASE,
    )
    if not match:
        return ""
    value = match.group(1).strip("._/- ")
    return re.sub(r"[-_]\d{3,6}$", "", value)


def _well_hint(stem: str) -> str:
    explicit = extract_well(stem)
    if explicit:
        return explicit
    compact = display_text(stem).strip(" _-.,;()[]")
    # Preserve conventional short identifiers such as Р-31 or 40Р, but do
    # not treat a long arbitrary photo caption as an exact well identifier.
    if re.fullmatch(r"[a-zа-я]{0,4}[-_ ]?\d{1,5}[a-zа-я]{0,4}", compact, flags=re.IGNORECASE):
        return compact
    return ""


def _configure_tesseract(pytesseract) -> bool:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.extend((
            Path(sys.executable).resolve().parent / "_internal" / "tools" / "tesseract" / "tesseract.exe",
            Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)) / "tools" / "tesseract" / "tesseract.exe",
        ))
    candidates.extend((
        Path(__file__).resolve().parents[3] / "tools" / "tesseract" / "tesseract.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
    ))
    located = shutil.which("tesseract")
    if located:
        candidates.append(Path(located))
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        return False
    pytesseract.pytesseract.tesseract_cmd = str(executable)
    tessdata = executable.parent / "tessdata"
    if tessdata.is_dir():
        os.environ["TESSDATA_PREFIX"] = str(tessdata)
    return True
