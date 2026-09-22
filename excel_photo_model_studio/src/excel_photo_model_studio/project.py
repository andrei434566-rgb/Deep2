from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from .depth import centimeters_to_meters, format_depth, meters_to_centimeters
from .matching import (
    match_photos, read_photo_map, suggest_missing_intervals,
    uncovered_photo_description_intervals, uncovered_photo_intervals, write_photo_map,
)
from .models import Annotation, ColumnMapping, DescriptionRow, Issue, PhotoRecord
from .photos import discover_photos
from .tabular import read_many_tables, save_mappings, well_key
from .vision import (
    calibrate_core_columns, core_photo_capacity_centimeters, detect_core_columns_from_path,
    project_matches, render_previews,
)


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
    photos = suggest_missing_intervals(read_photo_map(project_dir / "photo_map.csv"), rows)
    # This also migrates projects made by 0.3.3, where every OCR page could
    # incorrectly contain the same full core-sampling interval.
    write_photo_map(project_dir / "photo_map.csv", photos)
    old_approvals = _existing_approvals(project_dir / "annotations.csv")
    confirmed = [photo for photo in photos if photo.mapping_confirmed]
    matches, unresolved_confirmed = match_photos(confirmed, rows)
    unconfirmed = [photo for photo in photos if not photo.mapping_confirmed]
    annotations, columns, orders = project_matches(matches)
    # Every selected image is required to be a core photo. Detect columns even
    # when its depth or Excel match is missing, so it cannot disappear silently
    # before validation and dataset creation.
    for photo in photos:
        if photo.path in columns:
            continue
        try:
            columns[photo.path] = detect_core_columns_from_path(photo.path)
        except (OSError, ValueError):
            columns[photo.path] = []
        orders.setdefault(photo.path, photo.column_order)
    annotations = [replace(item, approved=old_approvals.get(item.annotation_id, False)) for item in annotations]
    preview_paths = render_previews(annotations, project_dir / "previews")
    _write_matches(project_dir / "matches.csv", matches)
    _write_annotations(project_dir / "annotations.csv", annotations, preview_paths)
    photo_by_path = {photo.path: photo for photo in photos}
    detected_payload = {}
    for path, boxes in columns.items():
        photo = photo_by_path.get(path)
        depth_ranges = calibrate_core_columns(
            boxes, float(photo.top), float(photo.base),
        ) if photo is not None and photo.has_interval else []
        detected_payload[str(path)] = {
            "order": orders.get(path, "left_to_right"),
            "boxes": [list(box) for box in boxes],
            "depth_ranges": [
                {"box": list(box), "top": top, "base": base}
                for box, top, base in depth_ranges
            ],
        }
    _write_json(project_dir / "detected_columns.json", detected_payload)
    all_issues = list(issues)
    all_issues.extend(
        Issue(
            "error", photo.path.name,
            "Tesseract не найден, поэтому обязательный интервал фото не определён."
            if photo.source == "ocr_unavailable"
            else "Фото содержит керн, но обязательный интервал глубины не восстановлен; "
            "укажите его в таблице и отметьте OK.",
        )
        for photo in unconfirmed
    )
    all_issues.extend(Issue(
        "error", photo.path.name,
        "Для подтверждённого интервала фото не найдено ни одной фации Excel.",
    ) for photo in unresolved_confirmed)
    photos_without_core = [photo for photo in photos if not columns.get(photo.path)]
    all_issues.extend(
        Issue(
            "error", photo.path.name,
            "На выбранном фото не распознан керн. Фото нельзя пропустить: "
            "должны быть найдены физические колонки и их интервал.",
        )
        for photo in sorted(photos_without_core, key=lambda item: item.path.name.casefold())
    )
    uncovered = {
        photo.path: uncovered_photo_intervals(photo, matches)
        for photo in confirmed
    }
    uncovered = {path: gaps for path, gaps in uncovered.items() if gaps}
    all_issues.extend(
        Issue(
            "error", path.name,
            "Найденный интервал керна не полностью закрыт фациями Excel: "
            + "; ".join(f"{format_depth(top)}–{format_depth(base)} м" for top, base in gaps)
            + ". Проверьте последовательность фото и интервалы фаций по бурению.",
        )
        for path, gaps in sorted(uncovered.items(), key=lambda item: item[0].name.casefold())
    )
    uncovered_descriptions = {
        photo.path: uncovered_photo_description_intervals(photo, matches)
        for photo in confirmed
    }
    uncovered_descriptions = {
        path: gaps for path, gaps in uncovered_descriptions.items() if gaps
    }
    all_issues.extend(
        Issue(
            "error", path.name,
            "Интервал керна не полностью закрыт фациями с «Кратким описанием»: "
            + "; ".join(f"{format_depth(top)}–{format_depth(base)} м" for top, base in gaps)
            + ". Для каждого сантиметра керна обязательны и фация, и краткое описание.",
        )
        for path, gaps in sorted(
            uncovered_descriptions.items(), key=lambda item: item[0].name.casefold()
        )
    )
    uncovered_excel_core = _uncovered_excel_core_intervals(rows, confirmed)
    all_issues.extend(
        Issue(
            "error", well or "Excel",
            f"Интервал отбора керна Excel {format_depth(top)}–{format_depth(base)} м "
            "не закрыт фотографиями с подтверждённой глубиной.",
        )
        for well, top, base in uncovered_excel_core
    )
    all_issues.extend(
        Issue(
            "error", path.name,
            f"Интервал фото {format_depth(photo_by_path[path].base - photo_by_path[path].top)} м "
            f"длиннее вместимости найденного керна "
            f"{format_depth(core_photo_capacity_centimeters(boxes) / 100)} м. "
            "Фото нельзя растягивать; проверьте OCR или ручные границы.",
        )
        for path, boxes in columns.items()
        if boxes
        and path in photo_by_path
        and photo_by_path[path].has_interval
        and meters_to_centimeters(photo_by_path[path].base)
        - meters_to_centimeters(photo_by_path[path].top)
        > core_photo_capacity_centimeters(boxes) + 2
        and len(boxes) > 1
    )
    report = {
        "schema": PROJECT_SCHEMA,
        "excel_rows": len(rows),
        "excel_files": len(excel_files),
        "photos": len(photos),
        "confirmed_photos": len(confirmed),
        "unconfirmed_photos": len(unconfirmed),
        "photos_without_intervals": sum(
            not photo.mapping_confirmed or not photo.has_interval for photo in photos
        ),
        "photos_without_core_columns": len(photos_without_core),
        "matches": len(matches),
        "annotations": len(annotations),
        "approved_annotations": sum(item.approved for item in annotations),
        "excel_text_targets": sum(bool(item.target_text.strip()) for item in rows),
        "facies_rows_without_description": sum(not item.target_text.strip() for item in rows),
        "invalid_thickness_rows": sum(not item.thickness_valid for item in rows),
        "text_targets": sum(bool(item.target_text.strip()) for item in annotations),
        "auto_sequenced_photos": sum(
            item.source in {"ocr_verified", "ocr_sequenced", "excel_sequenced"} for item in photos
        ),
        "ocr_verified_photos": sum(
            item.source in {"ocr_verified", "ocr_sequenced", "excel_sequenced"} for item in photos
        ),
        "uncovered_facies_intervals": sum(len(gaps) for gaps in uncovered.values()),
        "uncovered_description_intervals": sum(
            len(gaps) for gaps in uncovered_descriptions.values()
        ),
        "uncovered_excel_core_intervals": len(uncovered_excel_core),
        "column_mappings": [
            {
                "sheet": item.sheet, "facies_top": item.top, "facies_base": item.base,
                "facies_thickness": item.facies_thickness, "target_text": item.target_text,
            }
            for item in mappings
            if item.top or item.base or item.target_text
        ],
        "classes": sorted({item.label for item in annotations}, key=str.casefold),
        "blocking_errors": sum(item.severity == "error" for item in all_issues),
        "issues": [item.to_dict() for item in all_issues],
        "project_dir": str(project_dir),
    }
    _write_json(project_dir / "report.json", report)
    return report


def _uncovered_excel_core_intervals(
    rows: list[DescriptionRow], photos: list[PhotoRecord],
) -> list[tuple[str, float, float]]:
    """Return centimetre-exact Excel core ranges missing from confirmed photos."""
    core_ranges: dict[tuple[str, int, int], str] = {}
    for row in rows:
        if row.core_top is None or row.core_base is None or row.core_base <= row.core_top:
            continue
        key = (
            well_key(row.well),
            meters_to_centimeters(row.core_top),
            meters_to_centimeters(row.core_base),
        )
        core_ranges.setdefault(key, row.well)
    photo_spans: dict[str, list[tuple[int, int]]] = {}
    for photo in photos:
        if not photo.mapping_confirmed or not photo.has_interval:
            continue
        photo_spans.setdefault(well_key(photo.well), []).append((
            meters_to_centimeters(photo.top), meters_to_centimeters(photo.base),
        ))

    missing: list[tuple[str, float, float]] = []
    for (key, core_top_cm, core_base_cm), well in sorted(core_ranges.items()):
        spans = sorted(
            (max(core_top_cm, top_cm), min(core_base_cm, base_cm))
            for top_cm, base_cm in photo_spans.get(key, ())
            if min(core_base_cm, base_cm) > max(core_top_cm, top_cm)
        )
        cursor_cm = core_top_cm
        for top_cm, base_cm in spans:
            if top_cm > cursor_cm:
                missing.append((
                    well,
                    centimeters_to_meters(cursor_cm),
                    centimeters_to_meters(top_cm),
                ))
            cursor_cm = max(cursor_cm, base_cm)
        if cursor_cm < core_base_cm:
            missing.append((
                well,
                centimeters_to_meters(cursor_cm),
                centimeters_to_meters(core_base_cm),
            ))
    return missing


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
            "facies_top": item.facies_top, "facies_base": item.facies_base,
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
        "depth_top", "depth_base", "facies_top", "facies_base", "label", "polygon_json", "image_width", "image_height",
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
