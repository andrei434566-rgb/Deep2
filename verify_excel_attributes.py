"""Smoke test for Excel-imported lithology attributes without GUI dependencies."""

from pathlib import Path
import sys
import types

cv2_stub = types.ModuleType("cv2")
cv2_stub.IMREAD_COLOR = 1
sys.modules.setdefault("cv2", cv2_stub)
qt_core = types.ModuleType("PySide6.QtCore")
qt_core.QPointF = type("QPointF", (), {})
qt_gui = types.ModuleType("PySide6.QtGui")
qt_gui.QPixmap = type("QPixmap", (), {})
qt = types.ModuleType("PySide6")
sys.modules.setdefault("PySide6", qt)
sys.modules.setdefault("PySide6.QtCore", qt_core)
sys.modules.setdefault("PySide6.QtGui", qt_gui)

from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS
from app.infrastructure.excel_core_description import _attributes_from_description, read_description_workbook


workbook = Path("outputs/facies_attributes_import_20260822/Р-31_фации_с_параметрами_для_импорта.xlsx")
layers, issues = read_description_workbook(workbook)
assert len(layers) == 66, len(layers)
assert not issues, issues
missing = [field for field in LITHOLOGY_ATTRIBUTE_OPTIONS if not layers[0].attributes.get(field)]
assert not missing, missing
resolved = _attributes_from_description(layers[0])
missing_resolved = [field for field in LITHOLOGY_ATTRIBUTE_OPTIONS if not resolved.get(field)]
assert not missing_resolved, missing_resolved
print({"layers": len(layers), "label": layers[0].label, "attributes": len(resolved)})
