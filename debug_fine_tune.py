"""Reproduce a Kern Analyzer fine-tuning launch with a visible traceback."""

from pathlib import Path
import sys

from app.infrastructure.ml.fine_tune_worker import _safe_training_streams

data = Path("projects/excel_jpg_Р-31_фации_с_параметрами_для_импорта_20260822_214103/training/datasets/20260822_214151/data.yaml")
model = Path("models/best.pt")
original_stdout, original_stderr = sys.stdout, sys.stderr
sys.stdout = None
sys.stderr = None
try:
    with _safe_training_streams(Path("outputs/fine_tune_pythonw_sim.log")):
        from ultralytics import YOLO

        YOLO(str(model)).train(
        data=str(data),
        epochs=1,
        imgsz=640,
        project="outputs/fine_tune_diagnostic",
        name="run",
        exist_ok=True,
        verbose=True,
        device="cpu",
        workers=0,
        batch=1,
        fraction=0.01,
        )
finally:
    sys.stdout, sys.stderr = original_stdout, original_stderr
print("pythonw-stream-simulation-ok")
