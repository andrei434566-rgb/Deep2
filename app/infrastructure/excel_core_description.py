"""Read interval-based sedimentological descriptions and bind them to core JPGs."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPixmap

from app.domain.facies_catalog import facies_metadata
from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS
from app.domain.models import FaciesDetection
from app.infrastructure.ml.rule_based_facies import RuleBasedFaciesDetector


EXCEL_FACIES_ATTRIBUTE_FIELDS = (
    "Энергия среды",
    "Гидродинамический режим",
    "Примечание",
    *LITHOLOGY_ATTRIBUTE_OPTIONS.keys(),
)


@dataclass(frozen=True)
class DescriptionLayer:
    well: str
    top: float
    base: float
    facies_name: str
    facies_code: str
    facies_index: str
    description: str
    attributes: dict[str, str]
    sheet: str
    row: int
    core_top: float | None = None
    core_base: float | None = None

    @property
    def label(self) -> str:
        return self.facies_code or self.facies_index or self.facies_name


@dataclass(frozen=True)
class CoreInterval:
    well: str
    top: float
    base: float

    @property
    def title(self) -> str:
        return f"Скв. {self.well} · {self.top:g}–{self.base:g} м"


def photo_interval_from_filename(path: Path) -> CoreInterval:
    """Read well name and measured depth range from a core-photo filename.

    Expected filename form: ``Р-31_3002,00-3004,96.jpg``.  The well prefix
    is deliberately permissive, while both depths must be explicit numbers.
    """
    match = re.fullmatch(
        r"(?P<well>.+)_(?P<top>\d+(?:[.,]\d+)?)-(?P<base>\d+(?:[.,]\d+)?)",
        path.stem,
    )
    if not match:
        raise ValueError(
            "В имени JPG не найден интервал. Ожидается формат «Скв_Кровля-Подошва.jpg»."
        )
    well = _display_text(match.group("well"))
    top = _as_float(match.group("top"))
    base = _as_float(match.group("base"))
    if base is not None and top is not None and base <= top:
        # A single source filename contains 3829,08-2829,33.  Recover only
        # this narrow, obvious first-digit typo; all other reverse intervals
        # remain errors instead of being guessed.
        top_integer = re.split(r"[.,]", match.group("top"), maxsplit=1)[0]
        base_integer, base_fraction = re.split(r"[.,]", match.group("base"), maxsplit=1)
        if len(top_integer) == len(base_integer) and top_integer[1:] == base_integer[1:]:
            base = _as_float(f"{top_integer[0]}{base_integer[1:]}.{base_fraction}")
    if not well or top is None or base is None or base <= top:
        raise ValueError("В имени JPG указан некорректный интервал глубин.")
    return CoreInterval(well=well, top=top, base=base)


@dataclass(frozen=True)
class ImportIssue:
    source: str
    message: str


def read_description_workbook(path: Path) -> tuple[list[DescriptionLayer], list[ImportIssue]]:
    """Read the wide, merged-header Excel form used for core descriptions."""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("Не установлен пакет openpyxl. Выполните обновление зависимостей DeepCore.") from exc

    workbook = load_workbook(path, read_only=False, data_only=True)
    layers: list[DescriptionLayer] = []
    issues: list[ImportIssue] = []
    for sheet in workbook.worksheets:
        columns = _find_columns(sheet)
        required = {"well", "facies_top", "facies_base"}
        if not required.issubset(columns):
            continue
        for row in range(1, sheet.max_row + 1):
            try:
                top = _as_float(sheet.cell(row, columns["facies_top"]).value)
                base = _as_float(sheet.cell(row, columns["facies_base"]).value)
            except ValueError:
                continue
            if top is None or base is None or base <= top:
                continue
            well = _display_text(sheet.cell(row, columns["well"]).value)
            if not well:
                issues.append(ImportIssue(f"{sheet.title}!{row}", "Не указан номер скважины."))
                continue
            name = _display_text(_cell_value(sheet, row, columns.get("facies_name")))
            code = _display_text(_cell_value(sheet, row, columns.get("facies_code")))
            index = _display_text(_cell_value(sheet, row, columns.get("facies_index")))
            description = _display_text(_cell_value(sheet, row, columns.get("description")))
            attributes = {
                field: value
                for field in EXCEL_FACIES_ATTRIBUTE_FIELDS
                if (value := _display_text(_cell_value(sheet, row, columns.get(f"attribute:{field}"))))
            }
            if not (name or code or index):
                issues.append(ImportIssue(f"{sheet.title}!{row}", "Нет названия, индекса или кода фации."))
                continue
            layers.append(
                DescriptionLayer(
                    well=well,
                    top=top,
                    base=base,
                    facies_name=name,
                    facies_code=code,
                    facies_index=index,
                    description=description,
                    attributes=attributes,
                    sheet=sheet.title,
                    row=row,
                    core_top=_as_float(_cell_value(sheet, row, columns.get("core_top"))),
                    core_base=_as_float(_cell_value(sheet, row, columns.get("core_base"))),
                )
            )
    if not layers:
        raise ValueError(
            "Не найдены строки с интервалом фации. Нужны колонки «№ скв.», "
            "«Интервал фации по бурению: Кровля/Подошва» и название либо код фации."
        )
    return sorted(layers, key=lambda item: (_well_key(item.well), item.top, item.base)), issues


def layers_for_photo(layers: list[DescriptionLayer], well: str, top: float, base: float) -> list[DescriptionLayer]:
    """Return description rows overlapping a photographed core interval."""
    requested_well = _well_key(well)
    return [
        layer
        for layer in layers
        if _well_key(layer.well) == requested_well and min(layer.base, base) - max(layer.top, top) > 0.00001
    ]


def core_intervals(layers: list[DescriptionLayer]) -> list[CoreInterval]:
    """Return unique core-sampling intervals used to bind a batch of JPGs."""
    grouped: dict[tuple[str, float, float], CoreInterval] = {}
    for layer in layers:
        top = layer.core_top if layer.core_top is not None else layer.top
        base = layer.core_base if layer.core_base is not None else layer.base
        if base <= top:
            continue
        item = CoreInterval(layer.well, top, base)
        grouped[(_well_key(item.well), round(item.top, 4), round(item.base, 4))] = item
    return sorted(grouped.values(), key=lambda item: (_well_key(item.well), item.top, item.base))


def create_depth_bound_detections(
    image_path: Path,
    pixmap: QPixmap,
    photo_top: float,
    photo_base: float,
    layers: list[DescriptionLayer],
) -> tuple[list[FaciesDetection], list[ImportIssue], list[dict[str, float]]]:
    """Turn matching depth intervals into per-column rectangular training labels."""
    if photo_base <= photo_top:
        raise ValueError("Низ интервала фотографии должен быть больше верха.")
    # ``cv2.imread`` fails on Windows paths containing Cyrillic characters
    # (for example ``C:\\Users\\Я\\...``).  Reading bytes through Python first
    # keeps the Excel + JPG import independent of the current user profile.
    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Не удалось прочитать изображение: {image_path.name}") from exc
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось открыть изображение: {image_path.name}")
    columns = RuleBasedFaciesDetector._find_core_columns(image)
    if not columns:
        height, width = image.shape[:2]
        # A fallback remains usable for a single long core photograph. It is
        # intentionally restricted away from the usual caption band.
        columns = [(0, 0, width, max(1, int(height * 0.87)))]
    columns = sorted(columns, key=lambda box: (box[0], box[1]))
    source_height, source_width = image.shape[:2]
    x_scale = pixmap.width() / max(1, source_width)
    y_scale = pixmap.height() / max(1, source_height)
    total_visual_length = sum(max(1, bottom - top) for _, top, _, bottom in columns)
    cursor = photo_top
    detections: list[FaciesDetection] = []
    issues: list[ImportIssue] = []
    depth_segments: list[dict[str, float]] = []
    for left, pixel_top, right, pixel_base in columns:
        span = (photo_base - photo_top) * (pixel_base - pixel_top) / total_visual_length
        column_top, column_base = cursor, cursor + span
        cursor = column_base
        depth_segments.append({
            "left": float(left * x_scale), "top": float(pixel_top * y_scale), "right": float(right * x_scale), "bottom": float(pixel_base * y_scale),
            "depth_from": float(column_top), "depth_to": float(column_base),
        })
        for layer in layers:
            overlap_top, overlap_base = max(layer.top, column_top), min(layer.base, column_base)
            if overlap_base - overlap_top <= 0.00001:
                continue
            y0 = pixel_top + (overlap_top - column_top) / (column_base - column_top) * (pixel_base - pixel_top)
            y1 = pixel_top + (overlap_base - column_top) / (column_base - column_top) * (pixel_base - pixel_top)
            polygon = [
                QPointF(left * x_scale, y0 * y_scale),
                QPointF((right - 1) * x_scale, y0 * y_scale),
                QPointF((right - 1) * x_scale, y1 * y_scale),
                QPointF(left * x_scale, y1 * y_scale),
            ]
            attributes = _attributes_from_description(layer)
            detections.append(
                FaciesDetection(
                    label=layer.label,
                    confidence=1.0,
                    polygon=polygon,
                    attributes=attributes,
                    depth_from=overlap_top,
                    depth_to=overlap_base,
                    # The source is a human-written depth description. The
                    # geometry is automatic and remains visibly editable.
                    training_ready=True,
                )
            )
    if not detections:
        issues.append(ImportIssue(image_path.name, "Для фотографии не найдено пересекающихся интервалов фаций."))
    return detections, issues, depth_segments


def _find_columns(sheet) -> dict[str, int]:
    """Locate semantic columns despite the form's merged multi-row headings."""
    header_rows = min(24, sheet.max_row)
    columns: dict[str, int] = {}
    descriptions: list[tuple[int, str]] = []
    for column in range(1, sheet.max_column + 1):
        text = " ".join(
            _normalized(_merged_value(sheet, row, column))
            for row in range(1, header_rows + 1)
            if _merged_value(sheet, row, column) is not None
        )
        descriptions.append((column, text))
    for column, text in descriptions:
        if "название фации" in text:
            columns["facies_name"] = column
        elif "код фации" in text:
            columns["facies_code"] = column
        elif "индекс фации" in text:
            columns["facies_index"] = column
        elif "краткое описание" in text:
            columns["description"] = column
        elif "№ скв" in text or "no скв" in text or "номер скваж" in text:
            columns["well"] = column
        elif "интервал фации" in text and "кровля" in text:
            columns["facies_top"] = column
        elif "интервал фации" in text and "подошва" in text:
            columns["facies_base"] = column
        elif "интервал отбора керна" in text and "кровля" in text:
            columns["core_top"] = column
        elif "интервал отбора керна" in text and "подошва" in text:
            columns["core_base"] = column
        for field in EXCEL_FACIES_ATTRIBUTE_FIELDS:
            if _normalized(field) in text:
                columns[f"attribute:{field}"] = column
    return columns


def _merged_value(sheet, row: int, column: int):
    value = sheet.cell(row, column).value
    if value is not None:
        return value
    for merged in sheet.merged_cells.ranges:
        if merged.min_row <= row <= merged.max_row and merged.min_col <= column <= merged.max_col:
            return sheet.cell(merged.min_row, merged.min_col).value
    return None


def _cell_value(sheet, row: int, column: int | None):
    return sheet.cell(row, column).value if column is not None else None


def _attributes_from_description(layer: DescriptionLayer) -> dict[str, str]:
    text = _normalized(layer.description)
    attributes = facies_metadata(layer.facies_code)
    imported = {
        "Название фации": layer.facies_name,
        "Код фации": layer.facies_code,
        "Индекс фации": layer.facies_index,
        "Краткое описание": layer.description,
        "Источник описания": f"Excel: {layer.sheet}, строка {layer.row}",
    }
    attributes.update({key: value for key, value in imported.items() if value})
    for field, options in LITHOLOGY_ATTRIBUTE_OPTIONS.items():
        matches = [option for option in options if _option_is_mentioned(text, option)]
        if len(matches) == 1:
            attributes[field] = matches[0]
        elif len(matches) > 1:
            attributes[f"{field} (несколько)"] = " | ".join(matches)
    # Explicit Excel fields are authoritative over text extraction and catalog
    # defaults. This makes the import deterministic for prepared datasets.
    attributes.update(layer.attributes)
    return {key: value for key, value in attributes.items() if value}


def _option_is_mentioned(text: str, option: str) -> bool:
    value = _normalized(option)
    if value in text:
        return True
    # Russian descriptions frequently use an inflected form: «алевролитов»
    # instead of «алевролит». Matching a sufficiently long root is safer than
    # an unrestricted substring and leaves ambiguous cases for review.
    for word in re.findall(r"[a-zа-яё]+", value):
        if len(word) >= 7 and word[:6] in text:
            return True
    return False


def _as_float(value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    candidate = str(value).strip().replace(",", ".").replace(" ", "")
    match = re.search(r"-?\d+(?:\.\d+)?", candidate)
    return float(match.group()) if match else None


def _display_text(value) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def _normalized(value) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", _display_text(value)).casefold().replace("ё", "е").split()
    )


def _well_key(value: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "", _normalized(value))
