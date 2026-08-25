"""Validate per-photo filename depth matching against the prepared Excel."""

from pathlib import Path

from app.infrastructure.excel_core_description import (
    layers_for_photo,
    photo_interval_from_filename,
    read_description_workbook,
)

workbook = Path("outputs/facies_attributes_import_20260822/Р-31_фации_с_параметрами_для_импорта.xlsx")
photos = sorted(Path("D:/Керн фото/5/По интервалам/Объед").glob("*.jpg"))
layers, _ = read_description_workbook(workbook)
parsed = [photo_interval_from_filename(photo) for photo in photos]
assert len(parsed) == 66, len(parsed)
assert all(layers_for_photo(layers, item.well, item.top, item.base) for item in parsed)
recovered = next(item for item in parsed if item.top == 3829.08)
assert recovered.base == 3829.33, recovered
print({"photos": len(parsed), "matched": len(parsed), "recovered_interval": (recovered.top, recovered.base)})
