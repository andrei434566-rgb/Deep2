from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from .matching import match_photos, read_photo_map, suggest_missing_intervals, write_photo_map
from .models import Annotation, ColumnMapping, DescriptionRow, Issue, PhotoRecord
from .photos import discover_photos
from .tabular import read_many_tables, save_mappings
from .vision import project_matches, render_previews


PROJECT_SCHEMA = "excel-photo-model-studio-v1"


def create_project(
    excel_path: Path | list[Path],
    photos_dir: Path,
    project_dir: Path,
    *,
    mapping_file: Path | None = None,
    use_ocr: bool = False,
) -> dict:
    project_dir = Path(project_dir).expanduser().absolute()
    if project_dir.exists():
        raise FileExistsError(f"Папка проекта уже существует: {project_dir}")
    photos_dir = Path(photos_dir).expanduser().resolve(strict=True)
    excel_inputs = [Path(excel_path)] if isinstance(excel_path, (str, Path)) else [Path(value) for value in excel_path]
    rows, mappings, issues, excel_files = read_many_tables(excel_inputs, mapping_file)
    expected_intervals = sorted({
        (float(row.core_top), float(row.core_base))
        for row in rows
        if row.core_top is not None and row.core_base is not None and row.core_base > row.core_top
    })
    photos = discover_photos(
        photos_dir, use_ocr=use_ocr, expected_intervals=expected_intervals,
    )
    if not photos:
        raise ValueError("В выбранной папке не найдены поддерживаемые изображения.")
    photos = suggest_missing_intervals(photos, rows)
    project_dir.mkdir(parents=True)
    config = {
        "schema": PROJECT_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "excel_paths": [str(path) for path in excel_files],
        "photos_dir": str(photos_dir),
        "use_ocr": bool(use_ocr),
    }
    _write_json(project_dir / "project.json", config)
    save_mappings(project_dir / "column_mapping.json", mappings)
    write_photo_map(project_dir / "photo_map.csv", photos)
    _write_json(project_dir / "excel_issues.json", [item.to_dict() for item in issues])
    _write_table_cache(project_dir, rows, mappings, issues, excel_files)
    return refresh_project(project_dir)


def refresh_project(project_dir: Path) -> dict:
    project_dir = Path(project_dir).expanduser().resolve(strict=True)
    config = _read_project(project_dir)
    excel_paths = [Path(value) for value in config.get("excel_paths", ())]
    if not excel_paths and config.get("excel_path"):
        excel_paths = [Path(config["excel_path"])]
    cached = _load_table_cache(project_dir, excel_paths)
    if cached is None:
        rows, mappings, issues, excel_files = read_many_tables(excel_paths, project_dir / "column_mapping.json")
        save_mappings(project_dir / "column_mapping.json", mappings)
        _write_table_cache(project_dir, rows, mappings, issues, excel_files)
    else:
        rows, mappings, issues, excel_files = cached
    photos = read_photo_map(project_dir / "photo_map.csv")
    old_approvals = _existing_approvals(project_dir / "annotations.csv")
    confirmed = [photo for photo in photos if photo.mapping_confirmed]
    matches, unresolved_confirmed = match_photos(confirmed, rows)
    unconfirmed = [photo for photo in photos if not photo.mapping_confirmed]
    annotations, columns, orders = project_matches(matches)
    annotations = [replace(item, approved=old_approvals.get(item.annotation_id, False)) for item in annotations]
    preview_paths = render_previews(annotations, project_dir / "previews")
    _write_matches(project_dir / "matches.csv", matches)
    _write_annotations(project_dir / "annotations.csv", annotations, preview_paths)
    _write_json(project_dir / "detected_columns.json", {
        str(path): {
            "order": orders.get(path, "left_to_right"),
            "boxes": [list(box) for box in boxes],
        }
        for path, boxes in columns.items()
    })
    all_issues = list(issues)
    all_issues.extend(
        Issue(
            "warning", photo.path.name,
            "Tesseract не найден; интервал нельзя прочитать из подписи фото."
            if photo.source == "ocr_unavailable"
            else "Интервал не найден в имени или подписи фото; укажите его в таблице и отметьте OK.",
        )
        for photo in unconfirmed
    )
    all_issues.extend(Issue("warning", photo.path.name, "Для подтверждённого фото не найдено пересекающихся строк Excel.") for photo in unresolved_confirmed)
    report = {
        "schema": PROJECT_SCHEMA,
        "excel_rows": len(rows),
        "excel_files": len(excel_files),
        "photos": len(photos),
        "confirmed_photos": len(confirmed),
        "unconfirmed_photos": len(unconfirmed),
        "matches": len(matches),
        "annotations": len(annotations),
        "approved_annotations": sum(item.approved for item in annotations),
        "excel_text_targets": sum(bool(item.target_text.strip()) for item in rows),
        "text_targets": sum(bool(item.target_text.strip()) for item in annotations),
        "ocr_verified_photos": sum(item.source == "ocr_verified" for item in photos),
        "column_mappings": [
            {
                "sheet": item.sheet, "facies_top": item.top, "facies_base": item.base,
                "target_text": item.target_text,
            }
            for item in mappings
            if item.top or item.base or item.target_text
        ],
        "classes": sorted({item.label for item in annotations}, key=str.casefold),
        "issues": [item.to_dict() for item in all_issues],
        "project_dir": str(project_dir),
    }
    _write_json(project_dir / "report.json", report)
    return report


def set_annotation_approvals(project_dir: Path, approvals: dict[str, bool]) -> dict:
    path = Path(project_dir) / "annotations.csv"
    rows = _read_csv(path)
    for row in rows:
        if row["annotation_id"] in approvals:
            row["approved"] = "1" if approvals[row["annotation_id"]] else "0"
    _write_dict_rows(path, rows)
    report_path = Path(project_dir) / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["approved_annotations"] = sum(row.get("approved") == "1" for row in rows)
    _write_json(report_path, report)
    return report


def load_annotations(project_dir: Path) -> list[dict[str, str]]:
    return _read_csv(Path(project_dir) / "annotations.csv")


def _read_project(project_dir: Path) -> dict:
    config = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    if config.get("schema") != PROJECT_SCHEMA:
        raise ValueError("Неизвестная версия проекта.")
    return config


def _existing_approvals(path: Path) -> dict[str, bool]:
    if not path.is_file():
        return {}
    return {row["annotation_id"]: row.get("approved", "") == "1" for row in _read_csv(path)}


def _write_matches(path: Path, matches) -> None:
    rows = []
    for item in matches:
        rows.append({
            "photo": str(item.photo.path), "well": item.photo.well,
            "photo_top": item.photo.top, "photo_base": item.photo.base,
            "label": item.description.label, "layer_top": item.description.top,
            "layer_base": item.description.base, "overlap_top": item.overlap_top,
            "overlap_base": item.overlap_base, "source_sheet": item.description.sheet,
            "source_row": item.description.row,
        })
    _write_dict_rows(path, rows, fieldnames=(
        "photo", "well", "photo_top", "photo_base", "label", "layer_top", "layer_base",
        "overlap_top", "overlap_base", "source_sheet", "source_row",
    ))


def _write_annotations(path: Path, annotations: list[Annotation], previews: dict[Path, Path]) -> None:
    rows = []
    for item in annotations:
        rows.append({
            "annotation_id": item.annotation_id, "photo": str(item.photo_path),
            "preview": str(previews.get(item.photo_path, "")), "well": item.well,
            "photo_top": item.photo_top, "photo_base": item.photo_base,
            "depth_top": item.depth_top, "depth_base": item.depth_base,
            "label": item.label, "polygon_json": json.dumps(item.polygon),
            "image_width": item.image_width, "image_height": item.image_height,
            "source_sheet": item.source_sheet, "source_row": item.source_row,
            "source_file": item.source_file, "target_text": item.target_text,
            "association": item.association, "environment": item.environment,
            "field_name": item.field_name,
            "approved": "1" if item.approved else "0",
        })
    _write_dict_rows(path, rows, fieldnames=(
        "annotation_id", "photo", "preview", "well", "photo_top", "photo_base",
        "depth_top", "depth_base", "label", "polygon_json", "image_width", "image_height",
        "source_sheet", "source_row", "source_file", "target_text", "association", "environment", "field_name", "approved",
    ))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source, delimiter=";"))


def _write_dict_rows(path: Path, rows: list[dict], fieldnames=None) -> None:
    fields = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _table_source_signature(excel_paths: list[Path], mapping_path: Path) -> str:
    import hashlib

    values = []
    for path in excel_paths:
        resolved = Path(path).expanduser().resolve(strict=True)
        stat = resolved.stat()
        values.append((str(resolved), stat.st_size, stat.st_mtime_ns))
    if mapping_path.is_file():
        stat = mapping_path.stat()
        values.append((str(mapping_path.resolve()), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()


def _write_table_cache(project_dir: Path, rows, mappings, issues, excel_files) -> None:
    mapping_path = project_dir / "column_mapping.json"
    payload = {
        "schema": "excel-photo-table-cache-v1",
        "signature": _table_source_signature([Path(path) for path in excel_files], mapping_path),
        "excel_files": [str(path) for path in excel_files],
        "rows": [asdict(item) for item in rows],
        "mappings": [item.to_dict() for item in mappings],
        "issues": [item.to_dict() for item in issues],
    }
    path = project_dir / "table_cache.json"
    temporary = path.with_suffix(".json.tmp")
    _write_json(temporary, payload)
    temporary.replace(path)


def _load_table_cache(project_dir: Path, excel_paths: list[Path]):
    path = project_dir / "table_cache.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "excel-photo-table-cache-v1":
            return None
        if payload.get("signature") != _table_source_signature(excel_paths, project_dir / "column_mapping.json"):
            return None
        rows = [DescriptionRow(**item) for item in payload.get("rows", [])]
        mappings = [ColumnMapping.from_dict(item) for item in payload.get("mappings", [])]
        issues = [Issue(**item) for item in payload.get("issues", [])]
        files = [Path(item) for item in payload.get("excel_files", [])]
        if not rows or not files:
            return None
        return rows, mappings, issues, files
    except (OSError, ValueError, TypeError, KeyError):
        return None
