"""Make PySide6 native libraries discoverable in a frozen Windows build."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_qt_dll_paths() -> None:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    candidates = (root / "PySide6", root / "shiboken6", root / "PySide6" / "Qt" / "bin", root)
    existing = [path for path in candidates if path.is_dir()]
    for path in existing:
        try:
            os.add_dll_directory(str(path))
        except (AttributeError, OSError):
            pass
    if existing:
        os.environ["PATH"] = os.pathsep.join(str(path) for path in existing) + os.pathsep + os.environ.get("PATH", "")


configure_qt_dll_paths()
