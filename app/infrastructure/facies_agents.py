"""Deterministic agents for creating facies supervision from core photographs.

The agents deliberately do not use an LLM to decide geology.  Their job is to
make the Excel-to-image transformation repeatable and reviewable:

* :class:`CoreColumnAgent` finds physical core columns;
* :class:`PhotoIntervalAgent` establishes the depth interval of a photograph;
* :class:`ExcelFaciesMaskAgent` overlays verified Excel intervals as rectangles.

The orchestration layer (including a local Qwen assistant) can call these
agents, inspect their messages and request manual correction.  The generated
``FaciesBand`` objects are suitable both for preview masks and for export to a
YOLO-seg training dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np

from app.infrastructure.excel_core_description import (
    CoreInterval,
    DescriptionLayer,
    layers_for_photo,
    photo_interval_from_filename,
)
from app.infrastructure.ml.rule_based_facies import RuleBasedFaciesDetector
from app.infrastructure.photo_caption_ocr import _configure_tesseract, read_caption_metadata


@dataclass(frozen=True)
class AgentMessage:
    level: str
    message: str


@dataclass(frozen=True)
class CoreColumn:
    left: int
    top: int
    right: int
    bottom: int
    depth_from: float | None = None
    depth_to: float | None = None
    direction: str = "down"

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class FaciesBand:
    """One Excel facies interval projected onto one physical core column."""

    label: str
    facies_name: str
    facies_code: str
    facies_index: str
    well: str
    depth_from: float
    depth_to: float
    column_number: int
    left: int
    top: int
    right: int
    bottom: int
    source_sheet: str
    source_row: int

    @property
    def polygon(self) -> tuple[tuple[int, int], ...]:
        return (
            (self.left, self.top),
            (self.right - 1, self.top),
            (self.right - 1, self.bottom - 1),
            (self.left, self.bottom - 1),
        )


@dataclass(frozen=True)
class TextMark:
    text: str
    left: int
    top: int
    right: int
    bottom: int
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)


class CoreColumnAgent:
    """Stage 1: find physical core columns in source-image pixel coordinates."""

    name = "core-column-agent"

    def detect(self, image: np.ndarray, known_columns: list[tuple[int, int, int, int]] | None = None) -> tuple[list[CoreColumn], list[AgentMessage]]:
        height, width = image.shape[:2]
        boxes = known_columns if known_columns is not None else RuleBasedFaciesDetector._find_core_columns(image)
        columns = [
            CoreColumn(
                left=max(0, int(left)), top=max(0, int(top)),
                right=min(width, int(right)), bottom=min(height, int(bottom)),
            )
            for left, top, right, bottom in boxes
            if right > left and bottom > top
        ]
        columns.sort(key=lambda item: (item.left, item.top))
        if not columns:
            return [], [AgentMessage("error", "Не найдены колонки керна. Исправьте маску этапа 1 до привязки Excel.")]
        origin = "manifest этапа 1" if known_columns is not None else "детектор колонок"
        return columns, [AgentMessage("info", f"Найдено колонок: {len(columns)} ({origin}).")]


class PhotoIntervalAgent:
    """Stage 2: resolve the measured-depth interval printed on a photograph.

    The printed drilling interval is the primary source.  A filename is only a
    fallback so file renaming cannot silently alter the geological binding.
    """

    name = "photo-interval-agent"

    def from_photo(self, image_path: Path, default_well: str = "") -> tuple[CoreInterval | None, list[AgentMessage]]:
        try:
            caption = read_caption_metadata(image_path)
            interval = CoreInterval(caption.well or default_well, caption.top, caption.base)
            if not interval.well:
                return None, [AgentMessage("warning", f"{image_path.name}: OCR прочитал интервал {interval.top:g}–{interval.base:g} м, но не номер скважины.")]
            return interval, [AgentMessage("info", f"Интервал по бурению прочитан с фото: {interval.well} {interval.top:g}–{interval.base:g} м.")]
        except Exception as ocr_error:
            interval, filename_messages = self.from_filename(image_path)
            if interval is not None:
                return interval, [AgentMessage("warning", f"OCR интервала не сработал ({ocr_error}); использован резервный интервал из имени файла.")]
            return None, [AgentMessage("warning", f"{image_path.name}: OCR не прочитал интервал ({ocr_error}); имя также не содержит корректного интервала.")]

    def from_filename(self, image_path: Path) -> tuple[CoreInterval | None, list[AgentMessage]]:
        try:
            interval = photo_interval_from_filename(image_path)
        except ValueError as error:
            return None, [AgentMessage("warning", f"{image_path.name}: интервал не найден в имени ({error}). Нужен OCR или ручной ввод.")]
        return interval, [AgentMessage("info", f"Интервал из имени: {interval.well} {interval.top:g}–{interval.base:g} м.")]


class PhotoOrientationAgent:
    """Stage 2a: read «Верх»/«Низ» and order columns without user dialogs."""

    name = "photo-orientation-agent"
    TOP_WORDS = ("верх", "top", "verh")
    BOTTOM_WORDS = ("низ", "bottom", "niz")

    def recognise_marks(self, image: np.ndarray) -> tuple[list[TextMark], list[AgentMessage]]:
        """OCR only orientation words; a failed OCR is a review item, not a stop."""
        try:
            import pytesseract
            from pytesseract import Output

            _configure_tesseract(pytesseract)
            data = pytesseract.image_to_data(image, lang="rus+eng", config="--psm 11", output_type=Output.DICT)
        except Exception as error:
            return [], [AgentMessage("warning", f"Не удалось OCR-распознать «Верх/Низ»: {error}")]
        marks: list[TextMark] = []
        for index, raw_text in enumerate(data.get("text", [])):
            text = self._normalise(raw_text)
            if not self._kind(text):
                continue
            try:
                confidence = float(data["conf"][index])
            except (KeyError, TypeError, ValueError):
                confidence = 0.0
            left, top = int(data["left"][index]), int(data["top"][index])
            width, height = int(data["width"][index]), int(data["height"][index])
            marks.append(TextMark(text, left, top, left + width, top + height, confidence))
        if marks:
            return marks, [AgentMessage("info", "OCR нашёл ориентиры: " + ", ".join(mark.text for mark in marks))]
        return [], [AgentMessage("warning", "На фото не найдены слова «Верх»/«Низ»; применён порядок папки слева направо.")]

    def order(self, columns: list[CoreColumn], marks: list[TextMark]) -> tuple[list[CoreColumn], list[AgentMessage]]:
        if not columns:
            return [], [AgentMessage("error", "Нет колонок для определения направления.")]
        ordered_by_x = sorted(columns, key=lambda column: (column.left, column.top))
        top_marks = [mark for mark in marks if self._kind(mark.text) == "top"]
        bottom_marks = [mark for mark in marks if self._kind(mark.text) == "bottom"]
        if not top_marks and not bottom_marks:
            return ordered_by_x, [AgentMessage("warning", "Направление не подтверждено OCR; принят порядок слева направо, сверху вниз.")]

        reference = max(top_marks or bottom_marks, key=lambda mark: mark.confidence)
        ref_x, ref_y = reference.center
        first_index = min(
            range(len(ordered_by_x)),
            key=lambda index: abs((ordered_by_x[index].left + ordered_by_x[index].right) / 2 - ref_x),
        )
        # «Верх» задаёт первый столбец; если найден только «Низ», ближайший
        # столбец считается последним and the traversal is reversed below.
        if top_marks:
            direction_x = 1 if first_index <= (len(ordered_by_x) - 1) / 2 else -1
            traversal = ordered_by_x[first_index::direction_x]
            if len(traversal) != len(ordered_by_x):
                traversal = ordered_by_x if direction_x == 1 else list(reversed(ordered_by_x))
            first_direction = "down" if ref_y <= (traversal[0].top + traversal[0].bottom) / 2 else "up"
        else:
            direction_x = -1 if first_index >= len(ordered_by_x) / 2 else 1
            traversal = ordered_by_x[first_index::-1] if direction_x == -1 else list(reversed(ordered_by_x[first_index:]))
            if len(traversal) != len(ordered_by_x):
                traversal = list(reversed(ordered_by_x)) if direction_x == -1 else ordered_by_x
            first_direction = "up" if ref_y >= (traversal[0].top + traversal[0].bottom) / 2 else "down"
        result = [CoreColumn(item.left, item.top, item.right, item.bottom, item.depth_from, item.depth_to, first_direction) for item in traversal]
        expected = "сверху вниз" if first_direction == "down" else "снизу вверх"
        return result, [AgentMessage("info", f"Порядок колонок определён по «{reference.text}»: {expected}.")]

    @classmethod
    def _normalise(cls, text: str) -> str:
        return re.sub(r"[^a-zа-яё]", "", str(text).casefold().replace("ё", "е"))

    @classmethod
    def _kind(cls, text: str) -> str | None:
        value = cls._normalise(text)
        if any(word in value for word in cls.TOP_WORDS):
            return "top"
        if any(word in value for word in cls.BOTTOM_WORDS):
            return "bottom"
        return None


class ExcelFaciesMaskAgent:
    """Stage 3: project Excel facies intervals onto detected core columns."""

    name = "excel-facies-mask-agent"

    def apply(
        self,
        photo_interval: CoreInterval,
        layers: list[DescriptionLayer],
        columns: list[CoreColumn],
    ) -> tuple[list[FaciesBand], list[CoreColumn], list[AgentMessage]]:
        if photo_interval.base <= photo_interval.top:
            return [], columns, [AgentMessage("error", "Конец интервала фотографии должен быть больше начала.")]
        calibrated = self.calibrate_columns(photo_interval, columns)
        bands, messages = self.apply_calibrated(photo_interval.well, layers, calibrated)
        return bands, calibrated, messages

    def calibrate_columns(self, interval: CoreInterval, columns: list[CoreColumn]) -> list[CoreColumn]:
        """Assign increasing depths along the previously resolved traversal."""
        total_pixels = sum(column.height for column in columns)
        if interval.base <= interval.top or total_pixels <= 0:
            return []
        cursor = interval.top
        calibrated: list[CoreColumn] = []
        for column in columns:
            span = (interval.base - interval.top) * column.height / total_pixels
            calibrated.append(CoreColumn(
                column.left, column.top, column.right, column.bottom,
                depth_from=cursor, depth_to=cursor + span, direction=column.direction,
            ))
            cursor += span
        return calibrated

    def apply_calibrated(
        self,
        well: str,
        layers: list[DescriptionLayer],
        columns: list[CoreColumn],
    ) -> tuple[list[FaciesBand], list[AgentMessage]]:
        """Overlay Excel intervals on columns which already carry depths."""
        known_depths = [(column.depth_from, column.depth_to) for column in columns if column.depth_from is not None and column.depth_to is not None]
        if not known_depths:
            return [], [AgentMessage("error", "Колонки не получили глубинную привязку.")]
        photo_top = min(start for start, _ in known_depths)
        photo_base = max(end for _, end in known_depths)
        matching_layers = layers_for_photo(layers, well, photo_top, photo_base)
        if not matching_layers:
            return [], [AgentMessage("error", "В Excel нет фаций, пересекающихся с интервалом этой фотографии.")]

        bands: list[FaciesBand] = []
        messages: list[AgentMessage] = []
        for column_number, column in enumerate(columns, start=1):
            assert column.depth_from is not None and column.depth_to is not None
            for layer in matching_layers:
                depth_from = max(layer.top, column.depth_from)
                depth_to = min(layer.base, column.depth_to)
                if depth_to - depth_from <= 1e-6:
                    continue
                start_ratio = (depth_from - column.depth_from) / (column.depth_to - column.depth_from)
                end_ratio = (depth_to - column.depth_from) / (column.depth_to - column.depth_from)
                if column.direction == "up":
                    y0 = round(column.bottom - end_ratio * column.height)
                    y1 = round(column.bottom - start_ratio * column.height)
                else:
                    y0 = round(column.top + start_ratio * column.height)
                    y1 = round(column.top + end_ratio * column.height)
                top = min(max(column.top, min(y0, y1)), column.bottom - 1)
                bottom = min(max(top + 1, max(y0, y1)), column.bottom)
                label = layer.facies_name.strip()
                if not label or label == "Не задано":
                    messages.append(AgentMessage("warning", f"{layer.sheet}!{layer.row}: у фации нет названия; маска не пригодна для обучения."))
                    continue
                bands.append(FaciesBand(
                    label=label, facies_name=layer.facies_name, facies_code=layer.facies_code,
                    facies_index=layer.facies_index, well=layer.well,
                    depth_from=depth_from, depth_to=depth_to, column_number=column_number,
                    left=column.left, top=top, right=column.right, bottom=bottom,
                    source_sheet=layer.sheet, source_row=layer.row,
                ))
        if not bands:
            messages.append(AgentMessage("error", "После пересечения Excel и колонок не осталось пригодных масок."))
        else:
            messages.append(AgentMessage("info", f"Excel наложен: создано масок фаций: {len(bands)}."))
        return bands, messages


class DemoRandomFaciesAgent:
    """Create a visibly complete, synthetic seven-facies demonstration mask.

    This agent exists solely to demonstrate the complete Kern Analyzer route
    (columns -> masks -> review -> Excel).  It must never be used as training
    truth or presented as an inferred geological facies result.  The seven
    rectangles partition the detected physical core, without gaps, following
    the already resolved column order and direction.
    """

    name = "demo-random-facies-agent"

    def apply(
        self,
        columns: list[CoreColumn],
        class_count: int = 7,
        seed: int | None = None,
    ) -> tuple[list[FaciesBand], list[AgentMessage]]:
        if class_count < 1:
            return [], [AgentMessage("error", "Количество демонстрационных фаций должно быть не меньше 1.")]
        total_pixels = sum(column.height for column in columns)
        if not columns or total_pixels < class_count:
            return [], [AgentMessage("error", "Недостаточно пикселей керна для разбиения на демонстрационные фации.")]

        generator = np.random.default_rng(seed)
        # Every class receives a non-zero segment.  The remaining pixels are
        # assigned randomly, so a fixed seed makes a review reproducible.
        minimum = max(1, min(20, total_pixels // (class_count * 3)))
        minimum = min(minimum, total_pixels // class_count)
        lengths = generator.multinomial(total_pixels - minimum * class_count, np.full(class_count, 1 / class_count)) + minimum
        labels = [f"DEMO-Фация {number}" for number in generator.permutation(np.arange(1, class_count + 1))]

        bands: list[FaciesBand] = []
        cursor = 0
        for class_number, (label, length) in enumerate(zip(labels, lengths), start=1):
            segment_start = cursor
            segment_end = cursor + int(length)
            column_cursor = 0
            for column_number, column in enumerate(columns, start=1):
                column_start = column_cursor
                column_end = column_cursor + column.height
                overlap_start = max(segment_start, column_start)
                overlap_end = min(segment_end, column_end)
                if overlap_end > overlap_start:
                    local_start = overlap_start - column_start
                    local_end = overlap_end - column_start
                    if column.direction == "up":
                        top = column.bottom - local_end
                        bottom = column.bottom - local_start
                    else:
                        top = column.top + local_start
                        bottom = column.top + local_end
                    bands.append(FaciesBand(
                        label=label,
                        facies_name=label,
                        facies_code=f"DEMO-{class_number:02d}",
                        facies_index=str(class_number),
                        well="DEMO",
                        depth_from=float(overlap_start),
                        depth_to=float(overlap_end),
                        column_number=column_number,
                        left=column.left,
                        top=int(top),
                        right=column.right,
                        bottom=int(bottom),
                        source_sheet="DEMO",
                        source_row=class_number,
                    ))
                column_cursor = column_end
            cursor = segment_end

        messages = [
            AgentMessage("warning", "DEMO: создана случайная синтетическая разметка; это не геологический результат и не обучающая истина."),
            AgentMessage("info", f"DEMO полностью закрыл {total_pixels} px керна {class_count} фациями; seed={seed if seed is not None else 'random'}."),
        ]
        return bands, messages


def read_core_image(path: Path) -> np.ndarray:
    """Open a Windows path with Cyrillic characters without ``cv2.imread``."""
    try:
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError as error:
        raise ValueError(f"Не удалось прочитать изображение: {path.name}") from error
    if image is None:
        raise ValueError(f"Не удалось открыть изображение: {path.name}")
    return image
