from __future__ import annotations

import csv
import re
from dataclasses import replace
from pathlib import Path

from .depth import centimeters_to_meters, format_depth, meters_to_centimeters
from .models import DescriptionRow, Match, PhotoRecord, normalize_column_order
from .tabular import as_float, well_key


def match_photos(records: list[PhotoRecord], rows: list[DescriptionRow]) -> tuple[list[Match], list[PhotoRecord]]:
    wells = {well_key(row.well) for row in rows if well_key(row.well)}
    matches: list[Match] = []
    unresolved: list[PhotoRecord] = []
    for photo in sort_photo_records(records):
        if not photo.has_interval:
            unresolved.append(photo)
            continue
        photo_well = well_key(photo.well)
        if not photo_well and len(wells) == 1:
            photo_well = next(iter(wells))
        photo_matches = []
        for row in rows:
            if not row.thickness_valid:
                continue
            if photo_well and well_key(row.well) != photo_well:
                continue
            overlap_top_cm = max(meters_to_centimeters(photo.top), meters_to_centimeters(row.top))
            overlap_base_cm = min(meters_to_centimeters(photo.base), meters_to_centimeters(row.base))
            if overlap_base_cm > overlap_top_cm:
                photo_matches.append(Match(
                    photo, row,
                    centimeters_to_meters(overlap_top_cm),
                    centimeters_to_meters(overlap_base_cm),
                ))
        if photo_matches:
            matches.extend(photo_matches)
        else:
            unresolved.append(photo)
    return matches, unresolved


def suggest_missing_intervals(records: list[PhotoRecord], rows: list[DescriptionRow]) -> list[PhotoRecord]:
    """Resolve wells, validate OCR intervals, then suggest remaining intervals."""
    canonical_wells: dict[str, str] = {}
    for row in rows:
        key = well_key(row.well)
        if key:
            canonical_wells.setdefault(key, row.well)
    single_well = next(iter(canonical_wells.values())) if len(canonical_wells) == 1 else ""

    result: list[PhotoRecord] = []
    for record in records:
        well = _resolve_well(record, canonical_wells, single_well)
        updated = replace(record, well=well) if well != record.well else record
        result.append(updated)

    # OCR intentionally reads the full core-sampling interval from the caption.
    # Several consecutive report pages therefore carry the same range. Split
    # that range into one-metre physical columns in natural filename order.
    groups: dict[tuple[str, int, int], list[int]] = {}
    for index, record in enumerate(result):
        if record.source not in {"ocr", "ocr_verified"} or not record.has_interval:
            continue
        canonical = _matching_core_interval(record, rows)
        if canonical is None:
            continue
        top_cm, base_cm = canonical
        groups.setdefault((well_key(record.well), top_cm, base_cm), []).append(index)
    for (_well, core_top_cm, core_base_cm), indices in groups.items():
        cursor_cm = core_top_cm
        for index in sorted(indices, key=lambda item: _natural_name_key(result[item].path.name)):
            record = result[index]
            capacity_cm = _photo_core_capacity_cm(record.path)
            if capacity_cm <= 0 or cursor_cm >= core_base_cm:
                result[index] = replace(record, source="ocr_columns_not_found", mapping_confirmed=False)
                continue
            page_base_cm = min(core_base_cm, cursor_cm + capacity_cm)
            result[index] = replace(
                record,
                top=centimeters_to_meters(cursor_cm),
                base=centimeters_to_meters(page_base_cm),
                source="ocr_sequenced",
                mapping_confirmed=True,
            )
            cursor_cm = page_base_cm

    intervals: dict[tuple[str, float, float], tuple[str, float, float]] = {}
    for row in rows:
        top = row.core_top if row.core_top is not None else row.top
        base = row.core_base if row.core_base is not None else row.base
        if base > top:
            intervals[(well_key(row.well), round(top, 5), round(base, 5))] = (row.well, top, base)
    claimed: set[tuple[str, float, float]] = set()
    for key, value in intervals.items():
        interval_well, interval_top, interval_base = key
        top_cm, base_cm = meters_to_centimeters(interval_top), meters_to_centimeters(interval_base)
        if any(
            record.has_interval
            and well_key(record.well) == interval_well
            and min(meters_to_centimeters(record.base), base_cm)
            > max(meters_to_centimeters(record.top), top_cm)
            for record in result
        ):
            claimed.add(key)
    available = [value for key, value in sorted(intervals.items()) if key not in claimed]
    unresolved = [index for index, record in enumerate(result) if not record.has_interval]
    for index in list(unresolved):
        name_key = well_key(result[index].path.stem)
        candidates = [item for item in available if well_key(item[0]) and well_key(item[0]) in name_key]
        if len(candidates) == 1:
            item = candidates[0]
            result[index] = replace(result[index], well=item[0], top=item[1], base=item[2], source="excel_suggestion")
            available.remove(item)
            unresolved.remove(index)
    if unresolved and len(available) == len(unresolved):
        for index, item in zip(unresolved, available):
            result[index] = replace(result[index], well=item[0], top=item[1], base=item[2], source="excel_suggestion")
    return sort_photo_records(result)


def _resolve_well(record: PhotoRecord, canonical_wells: dict[str, str], single_well: str) -> str:
    if single_well:
        return single_well
    source_key = well_key(f"{record.well} {record.path.stem}")
    matches = [(key, value) for key, value in canonical_wells.items() if key and key in source_key]
    if not matches:
        return record.well
    return max(matches, key=lambda item: len(item[0]))[1]


def _matching_core_interval(record: PhotoRecord, rows: list[DescriptionRow]) -> tuple[int, int] | None:
    if not record.has_interval:
        return None
    photo_well = well_key(record.well)
    matching_rows = [row for row in rows if not photo_well or well_key(row.well) == photo_well]
    candidates: set[tuple[int, int]] = set()
    for row in matching_rows:
        if row.core_top is not None and row.core_base is not None and row.core_base > row.core_top:
            candidates.add((meters_to_centimeters(row.core_top), meters_to_centimeters(row.core_base)))
    record_top_cm = meters_to_centimeters(record.top)
    record_base_cm = meters_to_centimeters(record.base)
    aligned: list[tuple[int, int, int]] = []
    for top_cm, base_cm in candidates:
        span_cm = base_cm - top_cm
        tolerance_cm = max(100, round(span_cm * 0.08))
        coverage = (record_base_cm - record_top_cm) / max(span_cm, 1)
        error_cm = abs(record_top_cm - top_cm) + abs(record_base_cm - base_cm)
        if (
            abs(record_top_cm - top_cm) <= tolerance_cm
            and abs(record_base_cm - base_cm) <= tolerance_cm
            and coverage >= 0.65
        ):
            aligned.append((error_cm, top_cm, base_cm))
    if not aligned:
        return None
    _, top_cm, base_cm = min(aligned)
    return top_cm, base_cm


def _photo_core_capacity_cm(path: Path) -> int:
    try:
        from .vision import core_photo_capacity_centimeters, detect_core_columns_from_path

        return core_photo_capacity_centimeters(detect_core_columns_from_path(path))
    except (OSError, ValueError):
        return 0


def sort_photo_records(records: list[PhotoRecord]) -> list[PhotoRecord]:
    return sorted(records, key=lambda record: (
        well_key(record.well),
        meters_to_centimeters(record.top) if record.top is not None else 10**12,
        meters_to_centimeters(record.base) if record.base is not None else 10**12,
        _natural_name_key(record.path.name),
    ))


def _natural_name_key(value: str) -> tuple[tuple[int, object], ...]:
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def uncovered_photo_intervals(photo: PhotoRecord, matches: list[Match]) -> list[tuple[float, float]]:
    """Return centimetre-exact gaps not covered by a valid Excel facies row."""
    if not photo.has_interval:
        return []
    photo_top_cm = meters_to_centimeters(photo.top)
    photo_base_cm = meters_to_centimeters(photo.base)
    spans = sorted(
        (
            max(photo_top_cm, meters_to_centimeters(item.overlap_top)),
            min(photo_base_cm, meters_to_centimeters(item.overlap_base)),
        )
        for item in matches
        if item.photo.path == photo.path and item.overlap_base > item.overlap_top
    )
    gaps: list[tuple[float, float]] = []
    cursor_cm = photo_top_cm
    for top_cm, base_cm in spans:
        if base_cm <= cursor_cm:
            continue
        if top_cm > cursor_cm:
            gaps.append((centimeters_to_meters(cursor_cm), centimeters_to_meters(top_cm)))
        cursor_cm = max(cursor_cm, base_cm)
    if cursor_cm < photo_base_cm:
        gaps.append((centimeters_to_meters(cursor_cm), centimeters_to_meters(photo_base_cm)))
    return gaps


def write_photo_map(path: Path, records: list[PhotoRecord]) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=("photo", "well", "top", "base", "column_order", "source", "mapping_confirmed"),
            delimiter=";",
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "photo": str(record.path), "well": record.well,
                "top": "" if record.top is None else format_depth(record.top),
                "base": "" if record.base is None else format_depth(record.base),
                "column_order": normalize_column_order(record.column_order),
                "source": record.source, "mapping_confirmed": "1" if record.mapping_confirmed else "0",
            })


def read_photo_map(path: Path) -> list[PhotoRecord]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source, delimiter=";"))
    output = []
    for row in rows:
        output.append(PhotoRecord(
            path=Path(row["photo"]), well=row.get("well", "").strip(),
            top=as_float(row.get("top")), base=as_float(row.get("base")),
            source=row.get("source", "manual").strip() or "manual",
            mapping_confirmed=row.get("mapping_confirmed", "").strip().casefold() in {"1", "true", "yes", "да"},
            column_order=normalize_column_order(row.get("column_order")),
        ))
    return sort_photo_records(output)
