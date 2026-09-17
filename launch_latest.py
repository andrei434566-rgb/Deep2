"""One-click entry point for the current Kern Analyzer source version."""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def self_check() -> None:
    """Fail early with a readable message instead of a silent pythonw exit."""

    if sys.version_info < (3, 11):
        raise RuntimeError("Для Kern Analyzer требуется Python 3.11 или новее.")
    required = ("PySide6", "cv2", "openpyxl", "ultralytics")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Не установлены обязательные модули: {', '.join(missing)}")

    from app.domain.facies_catalog import FACIES_MODEL_CLASSES
    from app.ui.windows.main_window import MainWindow  # noqa: F401

    labels = [item["model_label"] for item in FACIES_MODEL_CLASSES]
    if not labels or len(set(labels)) != len(labels):
        raise RuntimeError("Встроенный финальный справочник фаций повреждён.")


def _show_error(message: str) -> None:
    text = f"Kern Analyzer не запустился:\n\n{message}\n\nПодробности сохранены в launch_error.log"
    try:
        ctypes.windll.user32.MessageBoxW(0, text, "Kern Analyzer", 0x10)
    except (AttributeError, OSError):
        pass


def main() -> int:
    os.chdir(ROOT)
    try:
        self_check()
        from run import main as run_application

        return int(run_application())
    except Exception as exc:  # pythonw otherwise hides the traceback completely.
        details = "".join(traceback.format_exception(exc))
        try:
            (ROOT / "launch_error.log").write_text(details, encoding="utf-8")
        except OSError:
            pass
        _show_error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
