"""Expose bundled PyTorch CUDA-wheel DLL directories on Windows."""
from __future__ import annotations

import os
import sys
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []

if sys.platform == "win32" and getattr(sys, "frozen", False):
    runtime_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    nvidia_root = runtime_root / "nvidia"
    for dll in nvidia_root.rglob("*.dll") if nvidia_root.is_dir() else ():
        directory = str(dll.parent)
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory))
        except OSError:
            pass
