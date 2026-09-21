from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from .models import DescriptionRow, Match, PhotoRecord, normalize_column_order
from .tabular import as_float, well_key


def match_photos(records: list[PhotoRecord], rows: list[DescriptionRow]) -> tuple[list[Match], list[PhotoRecord]]:
    wells = {well_key(row.well) for row in rows if well_key(row.well)}
    matches: list[Match] = []
    unresolved: list[PhotoRecord] = []
    for photo in records:
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
            overlap_top = max(float(photo.top), row.top)
            overlap_base = min(float(photo.base), row.base)
            if overlap_base - overlap_top > 1e-9:
                photo_matches.append(Match(photo, row, overlap_top, overlap_base))
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

    result = []
    for record in records:
        well = _resolve_well(record, canonical_wells, single_well)
        updated = replace(record, well=well) if well != record.well else record
        if updated.has_interval and updated.source == "ocr" and _ocr_matches_core_interval(updated, rows):
            updated = replace(updated, source="ocr_verified", mapping_confirmed=True)
        result.append(updated)

    intervals: dict[tuple[str, float, float], tuple[str, float, float]] = {}
    for row in rows:
        top = row.core_top if row.core_top is not None else row.top
        base = row.core_base if row.core_base is not None else row.base
        if base > top:
            intervals[(well_key(row.well), round(top, 5), round(base, 5))] = (row.well, top, base)
    claimed = {
        (well_key(record.well), round(float(record.top), 5), round(float(record.base), 5))
        for record in result if record.has_interval
    }
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
    return result


def _resolve_well(record: PhotoRecord, canonical_wells: dict[str, str], single_well: str) -> str:
    if single_well:
        return single_well
    source_key = well_key(f"{record.well} {record.path.stem}")
    matches = [(key, value) for key, value in canonical_wells.items() if key and key in source_key]
    if not matches:
        return record.well
    return max(matches, key=lambda item: len(item[0]))[1]


def _ocr_matches_core_interval(record: PhotoRecord, rows: list[DescriptionRow]) -> bool:
    if not record.has_interval:
        return False
    photo_well = well_key(record.well)
    matching_rows = [row for row in rows if not photo_well or well_key(row.well) == photo_well]
    candidates: set[tuple[float, float]] = set()
    for row in matching_rows:
        if row.core_top is not None and row.core_base is not None and row.core_base > row.core_top:
            candidates.add((float(row.core_top), float(row.core_base)))
    for top, base in candidates:
        span = base - top
        tolerance = max(1.0, span * 0.08)
        coverage = (float(record.base) - float(record.top)) / max(span, 1e-9)
        if abs(float(record.top) - top) <= tolerance and abs(float(record.base) - base) <= tolerance and coverage >= 0.65:
            return True
    return False


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
                "top": "" if record.top is None else f"{record.top:.6f}".rstrip("0").rstrip("."),
                "base": "" if record.base is None else f"{record.base:.6f}".rstrip("0").rstrip("."),
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
    return output
