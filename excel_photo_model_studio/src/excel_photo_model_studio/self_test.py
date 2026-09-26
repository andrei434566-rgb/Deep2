"""Offline portable-runtime check. No model downloads or user data required."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def run_self_test(destination: Path) -> dict:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import cv2
    import numpy as np
    import openpyxl
    import pytesseract
    import torch
    import xlrd
    from PySide6.QtWidgets import QApplication
    from ultralytics import YOLO

    from . import __version__
    from .gui import MainWindow
    from .photos import _configure_tesseract

    if not _configure_tesseract(pytesseract):
        raise RuntimeError("В сборке не найден Tesseract.")
    image = np.full((110, 560, 3), 255, np.uint8)
    cv2.putText(image, "4105.25", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
    ocr = pytesseract.image_to_string(image, lang="eng", config="--psm 7").strip()
    if "4105.25" not in ocr:
        raise RuntimeError(f"OCR не прошёл контрольное чтение: {ocr!r}")
    model = YOLO("yolo11n-seg.yaml")
    predictions = model.predict(np.zeros((96, 96, 3), np.uint8), imgsz=96, device="cpu", verbose=False)
    if len(predictions) != 1:
        raise RuntimeError("Визуальная модель не выполнила контрольный проход.")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.close()
    app.processEvents()
    with tempfile.TemporaryDirectory(prefix="studio-self-test-") as directory:
        workbook_path = Path(directory) / "check.xlsx"
        workbook = openpyxl.Workbook()
        workbook.active.append(["Depth", 3.03])
        workbook.save(workbook_path)
        reopened = openpyxl.load_workbook(workbook_path, read_only=True)
        assert reopened.active["B1"].value == 3.03
        reopened.close()
    result = {"status": "ok", "version": __version__, "torch": torch.__version__,
              "cuda_build": torch.version.cuda, "ocr": ocr, "gui": "ok", "excel": "ok",
              "visual_forward": "ok", "xlrd": xlrd.__version__}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
