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
from .models import (
    Annotation, COLUMN_ORDER_AUTO, COLUMN_ORDER_RIGHT_TO_LEFT, ColumnMapping,
    DescriptionRow, Issue, PhotoRecord, normalize_column_order,
)
from .photos import discover_photos, enrich_core_column_depths
from .tabular import read_many_tables, save_mappings, well_key
from .vision import (
    calibrate_core_columns, core_photo_capacity_centimeters, detect_column_order,
    detect_core_columns_from_path, project_matches, read_image, render_previews,
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
    photo_records = read_photo_map(project_dir / "photo_map.csv")
    # A recalculation must include files added after the project was created.
    # Preserve edits to existing records; discover/OCR only new paths.
    known_paths = {item.path.resolve() for item in photo_records}
    discovered = discover_photos(Path(config["photos_dir"]), use_ocr=False)
    photo_records.extend(item for item in discovered if item.path.resolve() not in known_paths)
    if config.get("use_ocr", False):
        core_intervals = tuple(sorted({
            (float(row.core_top), float(row.core_base))
            for row in rows
            if row.core_top is not None and row.core_base is not None
            and row.core_base > row.core_top
        }))
        photo_records = enrich_core_column_depths(photo_records, core_intervals)
    photos = suggest_missing_intervals(photo_records, rows)
    # This also migrates projects made by 0.3.3, where every OCR page could
    # incorrectly contain the same full core-sampling interval.
    write_photo_map(project_dir / "photo_map.csv", photos)
    old_approvals = _existing_approvals(project_dir / "annotations.csv")
    confirmed = [photo for photo in photos if photo.mapping_confirmed]
    matches, unresolved_confirmed = match_photos(confirmed, rows)
    unconfirmed = [photo for photo in photos if not photo.mapping_confirmed]
    capacity_by_path: dict[Path, int] = {}
    depth_ranges_by_path: dict[Path, list[tuple[tuple[int, int, int, int], float, float]]] = {}
    annotations, columns, orders = project_matches(
        matches, capacities_by_photo=capacity_by_path,
        depth_ranges_by_photo=depth_ranges_by_path,
    )
    # Every selected image is required to be a core photo. Detect columns even
    # when its depth or Excel match is missing, so it cannot disappear silently
    # before validation and dataset creation.
    for photo in photos:
        if photo.path in columns:
            continue
        try:
            columns[photo.path] = detect_core_columns_from_path(photo.path)
            image = read_image(photo.path)
            requested_order = normalize_column_order(photo.column_order)
            if requested_order == COLUMN_ORDER_AUTO:
                order = detect_column_order(image, columns[photo.path])
            else:
                order = requested_order
            if order == COLUMN_ORDER_RIGHT_TO_LEFT:
                columns[photo.path].reverse()
            orders[photo.path] = order
            capacity_by_path[photo.path] = core_photo_capacity_centimeters(
                columns[photo.path], image, photo.column_depths,
            )
            depth_ranges_by_path[photo.path] = (
                calibrate_core_columns(
                    columns[photo.path], float(photo.top), float(photo.base), image=image,
                    column_depths=photo.column_depths,
                )
                if photo.has_interval else []
            )
        except (OSError, ValueError):
            columns[photo.path] = []
            orders[photo.path] = photo.column_order
            capacity_by_path[photo.path] = 0
            depth_ranges_by_path[photo.path] = []
    annotations = [replace(item, approved=old_approvals.get(_annotation_revision(item), False)) for item in annotations]
    preview_paths = render_previews(annotations, project_dir / "previews")
    _write_matches(project_dir / "matches.csv", matches)
    _write_annotations(project_dir / "annotations.csv", annotations, preview_paths)
    photo_by_path = {photo.path: photo for photo in photos}
    inventory_path = project_dir / "photo_inventory.csv"
    _write_photo_inventory(inventory_path, photos, matches, annotations, columns, capacity_by_path)
    facies_inventory_path = project_dir / "facies_inventory.csv"
    unmatched_facies_rows = _write_facies_inventory(
        facies_inventory_path, rows, confirmed, matches,
    )
    detected_payload = {}
    for path, boxes in columns.items():
        photo = photo_by_path.get(path)
        depth_ranges = depth_ranges_by_path.get(path, [])
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
    projection_issues = _projection_issues(photos, annotations, columns, depth_ranges_by_path)
    all_issues.extend(projection_issues)
    facies_mask_gaps = _facies_mask_gaps(rows, annotations)
    all_issues.extend(
        Issue("error", source, "Фация Excel не полностью представлена масками: "
              + "; ".join(f"{format_depth(top)}–{format_depth(base)} м" for top, base in gaps))
        for source, gaps in facies_mask_gaps
    )
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
    photos_with_facies = {item.photo.path for item in matches}
    photos_with_masks = {item.photo_path for item in annotations}
    photos_without_masks = [photo for photo in photos if photo.path not in photos_with_masks]
    all_issues.extend(
        Issue(
            "error", photo.path.name,
            "Фото не попало в датасет масок: проверьте интервал, распознанные колонки керна "
            "и наличие перекрывающих фаций Excel.",
        )
        for photo in sorted(photos_without_masks, key=lambda item: item.path.name.casefold())
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
            + ". Проверьте последовательность фото и интервалы фаций по бурению/ГИС.",
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
    uncovered_excel_core = _uncovered_excel_core_intervals(rows, confirmed, matches)
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
            f"{format_depth(capacity_by_path.get(path, core_photo_capacity_centimeters(boxes)) / 100)} м. "
            "Фото нельзя растягивать; проверьте OCR или ручные границы.",
        )
        for path, boxes in columns.items()
        if boxes
        and path in photo_by_path
        and photo_by_path[path].has_interval
        and meters_to_centimeters(photo_by_path[path].base)
        - meters_to_centimeters(photo_by_path[path].top)
        > capacity_by_path.get(path, core_photo_capacity_centimeters(boxes)) + 2
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
        "photos_with_facies": len(photos_with_facies),
        "photos_without_facies": len(photos) - len(photos_with_facies),
        "photos_with_masks": len(photos_with_masks),
        "photos_without_masks": len(photos_without_masks),
        "photo_inventory": str(inventory_path),
        "facies_inventory": str(facies_inventory_path),
        "excel_facies_rows_without_photo_match": unmatched_facies_rows,
        "facies_rows_with_incomplete_masks": len(facies_mask_gaps),
        "projection_errors": len(projection_issues),
        "approved_annotations": sum(item.approved for item in annotations),
        "excel_text_targets": sum(bool(item.target_text.strip()) for item in rows),
        "facies_rows_without_description": sum(not item.target_text.strip() for item in rows),
        "invalid_thickness_rows": sum(not item.thickness_valid for item in rows),
        "gis_fallback_matches": sum(
            item.description.metadata.get("interval_source") == "gis" for item in matches
        ),
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
                "gis_facies_top": item.gis_top, "gis_facies_base": item.gis_base,
                "gis_facies_interval": item.gis_interval,
            }
            for item in mappings
            if item.top or item.base or item.target_text
        ],
        "classes": sorted({item.label for item in annotations}, key=str.casefold),
        "blocking_errors": sum(item.severity == "error" for item in all_issues),
        "issues": [item.to_dict() for item in all_issues],
        "project_dir": str(project_dir),
        "validation_snapshot": _validation_snapshot(project_dir, config, photos),
    }
    _write_json(project_dir / "report.json", report)
    return report


def _uncovered_excel_core_intervals(
    rows: list[DescriptionRow], photos: list[PhotoRecord], matches=None,
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
        matched = [item for item in (matches or ()) if item.photo.path == photo.path]
        uses_gis = getattr(photo, "depth_basis", "unknown") == "gis" or (
            matched and all(item.description.metadata.get("interval_source") == "gis" for item in matched)
        )
        if uses_gis:
            # Audit coverage in the drilling coordinate system of core_top/base.
            # Convert only paired facies limits, never shift the whole well by
            # one offset (the tie-in can change between drilling runs).
            for row in rows:
                if well_key(row.well) != well_key(photo.well) or row.gis_top is None or row.gis_base is None:
                    continue
                if row.gis_base <= row.gis_top:
                    continue
                top, base = max(photo.top, row.gis_top), min(photo.base, row.gis_base)
                if base > top:
                    ratio = (row.base - row.top) / (row.gis_base - row.gis_top)
                    photo_spans.setdefault(well_key(photo.well), []).append((
                        meters_to_centimeters(row.top + (top - row.gis_top) * ratio),
                        meters_to_centimeters(row.top + (base - row.gis_top) * ratio),
                    ))
        else:
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


def _gaps_cm(top: int, base: int, spans) -> list[tuple[int, int]]:
    gaps = []
    cursor = top
    for left, right in sorted(spans):
        left, right = max(top, left), min(base, right)
        if right <= cursor:
            continue
        if left > cursor:
            gaps.append((cursor, left))
        cursor = right
    if cursor < base:
        gaps.append((cursor, base))
    return gaps


def _projection_issues(photos, annotations, columns, depth_ranges) -> list[Issue]:
    """Validate actual masks, not just mathematical Excel/photo intersections."""
    by_photo = {}
    for item in annotations:
        by_photo.setdefault(item.photo_path, []).append(item)
    issues = []
    for photo in photos:
        if not photo.has_interval or not photo.mapping_confirmed:
            continue
        calibrated = depth_ranges.get(photo.path, [])
        boxes = columns.get(photo.path, [])
        if not boxes:
            continue  # Already reported as NO_CORE_COLUMNS.
        problems = []
        if {tuple(item[0]) for item in calibrated} != {tuple(box) for box in boxes}:
            problems.append("не всем физическим столбикам присвоена глубина")
        spans = [(meters_to_centimeters(top), meters_to_centimeters(base)) for _, top, base in calibrated]
        if _gaps_cm(meters_to_centimeters(photo.top), meters_to_centimeters(photo.base), spans):
            problems.append("границы столбиков не покрывают весь интервал фото")
        ordered = sorted(spans)
        if any(right > next_left for (_, right), (next_left, _) in zip(ordered, ordered[1:])):
            problems.append("глубины разных столбиков перекрываются")
        if any(top < meters_to_centimeters(photo.top) or base > meters_to_centimeters(photo.base) for top, base in spans):
            problems.append("глубины столбиков выходят за интервал фото")
        assigned = set()
        for index, (box, top, base) in enumerate(calibrated, start=1):
            left, pixel_top, right, pixel_base = box
            top_cm, base_cm = meters_to_centimeters(top), meters_to_centimeters(base)
            covered = []
            for item in by_photo.get(photo.path, []):
                xs, ys = zip(*item.polygon)
                center = (min(xs) + max(xs)) / 2
                if not left <= center < right:
                    continue
                assigned.add(item.annotation_id)
                if (min(xs) < left - 1 or max(xs) > right + 1
                        or min(ys) < pixel_top - 1 or max(ys) > pixel_base + 1):
                    problems.append(f"маска выходит за границы столбика {index}")
                if (max(xs) - min(xs)) < max(1, right - left - 2):
                    problems.append(f"маска не покрывает ширину столбика {index}")
                mask_top, mask_base = meters_to_centimeters(item.depth_top), meters_to_centimeters(item.depth_base)
                pixel_span = max(1, pixel_base - pixel_top)
                projected_top = top_cm + round((min(ys) - pixel_top) * (base_cm - top_cm) / pixel_span)
                projected_base = top_cm + round((max(ys) - pixel_top) * (base_cm - top_cm) / pixel_span)
                if abs(projected_top - mask_top) > 1 or abs(projected_base - mask_base) > 1:
                    problems.append(f"метраж маски не совпадает с пикселями столбика {index}")
                if item.target_text.strip() and item.label.strip():
                    covered.append((mask_top, mask_base))
            if _gaps_cm(top_cm, base_cm, covered):
                problems.append(f"столбик {index} не полностью закрыт масками с описанием")
            covered.sort()
            if any(end > next_start for (_, end), (next_start, _) in zip(covered, covered[1:])):
                problems.append(f"на столбике {index} пересекаются маски фаций")
        if any(item.annotation_id not in assigned for item in by_photo.get(photo.path, [])):
            problems.append("маска находится вне распознанного керна")
        if problems:
            issues.append(Issue("error", photo.path.name, "Ошибка проекции: " + "; ".join(dict.fromkeys(problems))))
    return issues


def _facies_mask_gaps(rows, annotations) -> list[tuple[str, list[tuple[float, float]]]]:
    grouped = {}
    for item in annotations:
        grouped.setdefault((item.source_file, item.source_sheet, item.source_row), []).append(item)
    missing = []
    for row in rows:
        spans = []
        for item in grouped.get((row.source_file, row.sheet, row.row), []):
            source_top = item.facies_top if item.facies_top is not None else row.top
            source_base = item.facies_base if item.facies_base is not None else row.base
            if source_base <= source_top or not item.target_text.strip() or not item.label.strip():
                continue
            # GIS and drilling limits can differ; compare progress within the
            # very same Excel row, not raw numbers from different depth systems.
            ratio = (row.base - row.top) / (source_base - source_top)
            spans.append((
                meters_to_centimeters(row.top + (item.depth_top - source_top) * ratio),
                meters_to_centimeters(row.top + (item.depth_base - source_top) * ratio),
            ))
        gaps = _gaps_cm(meters_to_centimeters(row.top), meters_to_centimeters(row.base), spans)
        if gaps:
            missing.append((row.source_id, [(centimeters_to_meters(a), centimeters_to_meters(b)) for a, b in gaps]))
    return missing


def _file_stamp(path: Path) -> list[int]:
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


def _validation_snapshot(project_dir: Path, config: dict, photos) -> dict:
    paths = [Path(path) for path in config.get("excel_paths", ())]
    paths += [photo.path for photo in photos]
    paths += [project_dir / name for name in ("annotations.csv", "photo_map.csv", "column_mapping.json", "detected_columns.json")]
    return {
        "files": {str(path.resolve()): _file_stamp(path) for path in paths},
        "photos_dir": config["photos_dir"],
        "photos": sorted(str(photo.path.resolve()) for photo in photos),
    }


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
    if "validation_snapshot" in report:
        report["validation_snapshot"]["files"][str(path.resolve())] = _file_stamp(path)
    _write_json(report_path, report)
    return report


def load_annotations(project_dir: Path) -> list[dict[str, str]]:
    return _read_csv(Path(project_dir) / "annotations.csv")


def _read_project(project_dir: Path) -> dict:
    config = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    if config.get("schema") != PROJECT_SCHEMA:
        raise ValueError("Неизвестная версия проекта.")
    return config


def _annotation_revision(item) -> str:
    """Changing pixels, text or metadata invalidates the earlier review."""
    if isinstance(item, Annotation):
        data = asdict(item)
        data["polygon_json"] = json.dumps(item.polygon)
    else:
        data = item
    values = {key: str(data.get(key, "")) for key in (
        "annotation_id", "target_text", "association", "environment", "field_name",
        "image_width", "image_height", "label", "source_file",
    )}
    values["polygon"] = json.loads(data["polygon_json"])
    return json.dumps(values, sort_keys=True, ensure_ascii=False)


def _existing_approvals(path: Path) -> dict[str, bool]:
    if not path.is_file():
        return {}
    return {_annotation_revision(row): row.get("approved", "") == "1" for row in _read_csv(path)}


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
            "interval_source": item.description.metadata.get("interval_source", "drilling"),
        })
    _write_dict_rows(path, rows, fieldnames=(
        "photo", "well", "photo_top", "photo_base", "label", "layer_top", "layer_base",
        "overlap_top", "overlap_base", "source_sheet", "source_row", "interval_source",
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


def _write_photo_inventory(
    path: Path, photos, matches, annotations, columns, capacity_by_path=None,
) -> None:
    """Persist one auditable row per discovered image, including misses."""
    facies_by_photo: dict[Path, set[tuple[str, str, int]]] = {}
    masks_by_photo: dict[Path, int] = {}
    for item in matches:
        facies_by_photo.setdefault(item.photo.path, set()).add((
            item.description.label, item.description.sheet, item.description.row,
        ))
    for item in annotations:
        masks_by_photo[item.photo_path] = masks_by_photo.get(item.photo_path, 0) + 1

    inventory = []
    for photo in photos:
        boxes = columns.get(photo.path, [])
        facies_count = len(facies_by_photo.get(photo.path, ()))
        mask_count = masks_by_photo.get(photo.path, 0)
        if not photo.mapping_confirmed or not photo.has_interval:
            status = "NO_CONFIRMED_INTERVAL"
        elif not boxes:
            status = "NO_CORE_COLUMNS"
        elif not facies_count:
            status = "NO_FACIES_MATCH"
        elif not mask_count:
            status = "NO_MASKS"
        else:
            status = "READY_FOR_REVIEW"
        inventory.append({
            "photo": str(photo.path),
            "file_name": photo.path.name,
            "well": photo.well,
            "depth_top_m": "" if photo.top is None else format_depth(photo.top),
            "depth_base_m": "" if photo.base is None else format_depth(photo.base),
            "interval_source": photo.source,
            "interval_confirmed": "1" if photo.mapping_confirmed else "0",
            "core_columns": len(boxes),
            "core_capacity_m": format_depth(
                (capacity_by_path or {}).get(photo.path, core_photo_capacity_centimeters(boxes)) / 100
            ),
            "matched_facies": facies_count,
            "mask_count": mask_count,
            "status": status,
        })
    _write_dict_rows(path, inventory, fieldnames=(
        "photo", "file_name", "well", "depth_top_m", "depth_base_m",
        "interval_source", "interval_confirmed", "core_columns", "core_capacity_m",
        "matched_facies", "mask_count", "status",
    ))


def _write_facies_inventory(path: Path, rows, photos, matches) -> int:
    """Audit each parsed Excel facies row against all confirmed photo ranges."""
    def row_key(item) -> tuple[str, str, int]:
        source_file = str(Path(item.source_file).expanduser().resolve()) if item.source_file else ""
        return source_file.casefold(), item.sheet, item.row

    matched_photos: dict[tuple[str, str, int], set[str]] = {}
    gis_fallback_rows: set[tuple[str, str, int]] = set()
    for match in matches:
        key = row_key(match.description)
        matched_photos.setdefault(key, set()).add(match.photo.path.name)
        if match.description.metadata.get("interval_source") == "gis":
            gis_fallback_rows.add(key)

    output = []
    for row in rows:
        key = well_key(row.well)
        overlapping = [
            photo.path.name
            for photo in photos
            if (not key or well_key(photo.well) == key)
            and photo.has_interval
            and any(
                top is not None and base is not None
                and min(meters_to_centimeters(base), meters_to_centimeters(photo.base))
                > max(meters_to_centimeters(top), meters_to_centimeters(photo.top))
                for top, base in ((row.top, row.base), (row.gis_top, row.gis_base))
            )
        ]
        source_key = row_key(row)
        matched = sorted(matched_photos.get(source_key, ()), key=str.casefold)
        if matched and source_key in gis_fallback_rows:
            status = "MATCHED_GIS_FALLBACK"
        elif matched:
            status = "MATCHED"
        elif not row.thickness_valid:
            status = "INVALID_FACIES_THICKNESS"
        elif overlapping:
            status = "PHOTO_OVERLAPS_BUT_ROW_NOT_MATCHED"
        else:
            status = "NO_PHOTO_OVERLAP"
        output.append({
            "source_file": row.source_file,
            "sheet": row.sheet,
            "excel_row": row.row,
            "well": row.well,
            "facies_top_m": format_depth(row.top),
            "facies_base_m": format_depth(row.base),
            "facies_thickness_m": "" if row.thickness is None else format_depth(row.thickness),
            "thickness_declared": "1" if row.thickness_declared else "0",
            "gis_top_m": "" if row.gis_top is None else format_depth(row.gis_top),
            "gis_base_m": "" if row.gis_base is None else format_depth(row.gis_base),
            "thickness_valid": "1" if row.thickness_valid else "0",
            "label": row.label,
            "short_description_present": "1" if row.target_text.strip() else "0",
            "target_text": row.target_text,
            "overlapping_photos": ", ".join(sorted(set(overlapping), key=str.casefold)),
            "matched_photos": ", ".join(matched),
            "status": status,
        })
    _write_dict_rows(path, output, fieldnames=(
        "source_file", "sheet", "excel_row", "well", "facies_top_m", "facies_base_m",
        "facies_thickness_m", "thickness_declared", "gis_top_m", "gis_base_m",
        "thickness_valid", "label", "short_description_present",
        "target_text", "overlapping_photos", "matched_photos", "status",
    ))
    return sum(not matched_photos.get(row_key(row)) for row in rows)


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
        # v2 persists the alternate GIS limits. Invalidate pre-GIS caches so
        # existing projects automatically re-read their Excel headers once.
        "schema": "excel-photo-table-cache-v2",
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
        if payload.get("schema") != "excel-photo-table-cache-v2":
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
