"""Exercise the Unicode-safe JPEG path used by the Excel + JPG importer."""

from pathlib import Path

from app.infrastructure.excel_core_description import (
    create_depth_bound_detections,
    layers_for_photo,
    read_description_workbook,
)

workbook = Path("outputs/facies_attributes_import_20260822/Р-31_фации_с_параметрами_для_импорта.xlsx")
image = Path("D:/Керн фото/5/По интервалам/Объед/Р-31_3002,00-3004,96.jpg")
layers, _ = read_description_workbook(workbook)
matching = layers_for_photo(layers, "Р-31", 3002.0, 3004.96)
detections, issues = create_depth_bound_detections(image, None, 3002.0, 3004.96, matching)
assert detections, issues
print({"detections": len(detections), "attributes": len(detections[0].attributes)})
