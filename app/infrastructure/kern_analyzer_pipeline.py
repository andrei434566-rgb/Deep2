"""Automatic standard pipeline for Kern Analyzer.

This is the non-interactive path used for ordinary core photographs with a
depth interval in their filename.  Agents exchange typed results in memory;
the user sees only the final HTML review and an issues table.  Ambiguous files
are skipped rather than opening a dialog or silently inventing a depth.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from app.infrastructure.excel_core_description import CoreInterval, read_description_workbook
from app.infrastructure.facies_agents import (
    AgentMessage,
    CoreColumnAgent,
    ExcelFaciesMaskAgent,
    FaciesBand,
    PhotoIntervalAgent,
    PhotoOrientationAgent,
    read_core_image,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
REVIEW_PREFIX = "_core_tape_review"


@dataclass(frozen=True)
class PipelineIssue:
    level: str
    photo: str
    message: str


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    photos_seen: int
    photos_labeled: int
    masks_created: int
    classes: list[str]
    issues: list[PipelineIssue]


class KernAnalyzerAutomaticPipeline:
    """Connect the standard three agents without interactive windows."""

    def __init__(self) -> None:
        self.columns = CoreColumnAgent()
        self.intervals = PhotoIntervalAgent()
        self.orientation = PhotoOrientationAgent()
        self.masks = ExcelFaciesMaskAgent()

    def run(self, photo_folder: Path, excel_path: Path, output_dir: Path) -> PipelineResult:
        source = photo_folder.expanduser().resolve()
        excel = excel_path.expanduser().resolve()
        destination = output_dir.expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Не найдена папка фотографий: {source}")
        if not excel.is_file():
            raise FileNotFoundError(f"Не найден Excel: {excel}")
        if destination.exists():
            raise FileExistsError(f"Папка результата уже существует: {destination}")

        layers, excel_issues = read_description_workbook(excel)
        issues = [PipelineIssue("warning", f"Excel:{item.source}", item.message) for item in excel_issues]
        excel_wells = sorted({layer.well for layer in layers}, key=str.casefold)
        default_well = excel_wells[0] if len(excel_wells) == 1 else ""
        known_columns = _load_column_manifest(source)
        image_paths = sorted(
            (
                path for path in source.rglob("*")
                if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
                and not any(parent.name.casefold().startswith(REVIEW_PREFIX) for parent in path.parents)
            ),
            key=lambda path: [int(part) if part.isdigit() else part.casefold() for part in _parts(path.name)],
        )
        if not image_paths:
            raise ValueError("В папке не найдены изображения керна.")

        destination.mkdir(parents=True)
        for name in ("overlays", "class_masks", "yolo_labels"):
            (destination / name).mkdir()

        prepared: list[dict[str, object]] = []
        for number, image_path in enumerate(image_paths, start=1):
            relative = image_path.relative_to(source).as_posix()
            image = read_core_image(image_path)
            columns, column_messages = self.columns.detect(image, known_columns.get(relative))
            _append_messages(issues, relative, column_messages)
            marks, orientation_ocr_messages = self.orientation.recognise_marks(image)
            _append_messages(issues, relative, orientation_ocr_messages)
            columns, orientation_messages = self.orientation.order(columns, marks)
            _append_messages(issues, relative, orientation_messages)
            interval, interval_messages = self.intervals.from_photo(image_path, default_well)
            _append_messages(issues, relative, interval_messages)
            if not columns:
                continue
            prepared.append({
                "number": number, "path": image_path, "relative": relative, "image": image,
                "detected_interval": interval, "columns_object": columns, "bands": [],
            })

        # A normal field folder is ordered already. If even one photo misses
        # depths in its name, map the complete visual tape to Excel's total
        # core interval. This avoids false per-photo guesses and needs no UI.
        use_global_tape = any(item["detected_interval"] is None for item in prepared)
        if use_global_tape:
            wells = excel_wells
            if len(wells) != 1:
                issues.append(PipelineIssue("error", "Excel", "Фото без интервалов можно автоматически связать только с Excel одной скважины. Разделите папки по скважинам."))
            else:
                well = wells[0]
                core_top, core_base = _excel_core_range(layers, well)
                flattened = [column for item in prepared for column in item["columns_object"]]
                calibrated_all = self.masks.calibrate_columns(CoreInterval(well, core_top, core_base), flattened)
                cursor = 0
                for item in prepared:
                    count = len(item["columns_object"])
                    calibrated = calibrated_all[cursor:cursor + count]
                    cursor += count
                    bands, mask_messages = self.masks.apply_calibrated(well, layers, calibrated)
                    _append_messages(issues, str(item["relative"]), mask_messages)
                    item["columns_object"] = calibrated
                    item["bands"] = bands
                    item["interval"] = {"well": well, "top": min(column.depth_from for column in calibrated), "base": max(column.depth_to for column in calibrated)} if calibrated else {"well": well, "top": core_top, "base": core_base}
                issues.append(PipelineIssue("warning", "Папка", "Применена общая привязка по порядку файлов и интервалу керна Excel; обязательно проверьте review.html перед обучением."))
        else:
            for item in prepared:
                interval = item["detected_interval"]
                bands, calibrated, mask_messages = self.masks.apply(interval, layers, item["columns_object"])
                _append_messages(issues, str(item["relative"]), mask_messages)
                item["columns_object"] = calibrated
                item["bands"] = bands
                item["interval"] = {"well": interval.well, "top": interval.top, "base": interval.base}

        classes = sorted({band.label for item in prepared for band in item["bands"]}, key=str.casefold)
        class_index = {name: index for index, name in enumerate(classes)}
        review_rows: list[dict[str, object]] = []
        annotation_rows: list[dict[str, object]] = []
        photos_labeled = 0
        mask_count = 0
        for item in prepared:
            image = item["image"]
            image_path = item["path"]
            bands: list[FaciesBand] = item["bands"]
            number = int(item["number"])
            stem = f"{number:04d}_{image_path.stem}"
            overlay_path = destination / "overlays" / f"{stem}.png"
            _write_overlay(overlay_path, image, bands)
            class_mask = np.zeros(image.shape[:2], dtype=np.uint16)
            yolo_lines: list[str] = []
            for band in bands:
                class_id = class_index[band.label]
                class_mask[band.top:band.bottom, band.left:band.right] = class_id + 1
                yolo_lines.append(_yolo_segment_line(class_id, band, image.shape[1], image.shape[0]))
                annotation_rows.append({
                    "photo": str(item["relative"]), "well": band.well, "facies": band.label,
                    "facies_name": band.facies_name, "facies_code": band.facies_code,
                    "depth_from": round(band.depth_from, 4), "depth_to": round(band.depth_to, 4),
                    "column": band.column_number, "left": band.left, "top": band.top,
                    "right": band.right, "bottom": band.bottom,
                    "excel": f"{band.source_sheet}!{band.source_row}",
                })
            if bands:
                photos_labeled += 1
                mask_count += len(bands)
            _write_image(destination / "class_masks" / f"{stem}.png", class_mask)
            (destination / "yolo_labels" / f"{stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            review_rows.append({
                "photo": str(item["relative"]), "overlay": f"overlays/{stem}.png", "bands": len(bands),
                "interval": item["interval"], "columns": [asdict(column) for column in item["columns_object"]],
            })

        (destination / "classes.json").write_text(json.dumps({"names": classes}, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_annotations_csv(destination / "facies_annotations.csv", annotation_rows)
        payload = {
            "pipeline": "Kern Analyzer automatic standard pipeline",
            "source_folder": str(source), "excel": str(excel), "classes": classes,
            "photos_seen": len(image_paths), "photos_labeled": photos_labeled, "masks_created": mask_count,
            "photos": review_rows, "issues": [asdict(item) for item in issues],
        }
        (destination / "pipeline_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_review(destination, payload)
        return PipelineResult(destination, len(image_paths), photos_labeled, mask_count, classes, issues)


def _parts(value: str) -> list[str]:
    import re
    return re.split(r"(\d+)", value)


def _well_key(value: str) -> str:
    import re
    return re.sub(r"[^a-zа-я0-9]+", "", value.casefold().replace("ё", "е"))


def _excel_core_range(layers, well: str) -> tuple[float, float]:
    """Prefer the Excel core-sampling range; use facies extent as fallback."""
    matching = [layer for layer in layers if _well_key(layer.well) == _well_key(well)]
    core_ranges = [
        (layer.core_top, layer.core_base)
        for layer in matching
        if layer.core_top is not None and layer.core_base is not None and layer.core_base > layer.core_top
    ]
    ranges = core_ranges or [(layer.top, layer.base) for layer in matching if layer.base > layer.top]
    if not ranges:
        raise ValueError(f"В Excel не найден общий интервал керна для скважины {well}.")
    return min(top for top, _ in ranges), max(base for _, base in ranges)


def _append_messages(issues: list[PipelineIssue], photo: str, messages: list[AgentMessage]) -> None:
    issues.extend(PipelineIssue(item.level, photo, item.message) for item in messages if item.level != "info")


def _load_column_manifest(source: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    """Reuse stage-1 rectangles when a core-tape manifest is next to the photos."""
    manifests = sorted(
        (path for path in source.rglob("manifest.json") if path.parent.name.casefold().startswith(REVIEW_PREFIX)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if Path(str(manifest.get("source_folder", ""))).resolve() != source:
                continue
            grouped: dict[str, list[tuple[int, int, int, int]]] = {}
            for item in manifest.get("columns", []):
                rectangle = item.get("source_rectangle_px", {})
                relative = str(item.get("source_relative", "")).replace("\\", "/")
                box = (rectangle.get("left"), rectangle.get("top"), rectangle.get("right"), rectangle.get("bottom"))
                if relative and all(isinstance(value, int) for value in box):
                    grouped.setdefault(relative, []).append(box)
            if grouped:
                return grouped
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return {}


def _color(label: str) -> tuple[int, int, int]:
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return 60 + digest[0] % 170, 60 + digest[1] % 170, 60 + digest[2] % 170


def _write_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"Не удалось сохранить {path.name}")
    encoded.tofile(str(path))


def _write_overlay(path: Path, image: np.ndarray, bands: list[FaciesBand]) -> None:
    overlay = image.copy()
    for band in bands:
        color = _color(band.label)
        cv2.rectangle(overlay, (band.left, band.top), (band.right - 1, band.bottom - 1), color, -1)
    result = cv2.addWeighted(overlay, 0.38, image, 0.62, 0)
    for band in bands:
        color = _color(band.label)
        cv2.rectangle(result, (band.left, band.top), (band.right - 1, band.bottom - 1), color, 2)
        title = f"{band.label} {band.depth_from:g}-{band.depth_to:g}"
        cv2.putText(result, title, (band.left + 4, min(band.bottom - 5, band.top + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
    _write_image(path, result)


def _yolo_segment_line(class_id: int, band: FaciesBand, width: int, height: int) -> str:
    values = []
    for x, y in band.polygon:
        values.extend((max(0.0, min(1.0, x / width)), max(0.0, min(1.0, y / height))))
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in values)


def _write_annotations_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = ["photo", "well", "facies", "facies_name", "facies_code", "depth_from", "depth_to", "column", "left", "top", "right", "bottom", "excel"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _write_review(output_dir: Path, payload: dict[str, object]) -> None:
    photos = payload["photos"]
    issues = payload["issues"]
    photo_sections = "\n".join(
        f"<section><h3>{html.escape(str(item['photo']))}</h3><p>Масок: {item['bands']}; интервал: {item['interval']['top']:g}–{item['interval']['base']:g} м.</p><img src=\"{quote(str(item['overlay']), safe='/._-')}\"></section>"
        for item in photos
    ) or "<p>Размеченных фото нет.</p>"
    issue_rows = "\n".join(
        f"<tr><td>{html.escape(str(item['level']))}</td><td>{html.escape(str(item['photo']))}</td><td>{html.escape(str(item['message']))}</td></tr>"
        for item in issues
    ) or "<tr><td colspan=\"3\">Нет замечаний</td></tr>"
    document = f"""<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\"><title>Kern Analyzer — автоматическая разметка</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f4f6f8}}section{{background:#fff;margin:18px 0;padding:14px;border:1px solid #d7dde4;border-radius:8px}}img{{max-width:100%;border:1px solid #c7ced6}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #d7dde4;text-align:left}}</style></head><body>
<h1>Kern Analyzer: Excel → маски фаций</h1><p>Фото: {payload['photos_seen']}; размечено: {payload['photos_labeled']}; масок: {payload['masks_created']}; классов: {len(payload['classes'])}.</p>
<section><h2>Проверки</h2><table><tr><th>Уровень</th><th>Фото</th><th>Сообщение</th></tr>{issue_rows}</table></section><h2>Наложения</h2>{photo_sections}</body></html>"""
    (output_dir / "review.html").write_text(document, encoding="utf-8")
