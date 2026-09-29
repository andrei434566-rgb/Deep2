"""Expose bundled PyTorch CUDA-wheel DLL directories on Windows."""
from __future__ import annotations

import os
import sys
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []

if sys.platform == "win32" and getattr(sys, "frozen", False):
    runtime_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    nvidia_root = runtime_root / "nvidia"
    torch_lib = runtime_root / "torch" / "lib"
    directories = {runtime_root}
    if torch_lib.is_dir():
        directories.add(torch_lib)
    if nvidia_root.is_dir():
        directories.update(dll.parent for dll in nvidia_root.rglob("*.dll"))
    for directory in sorted(directories, key=str):
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
        except OSError:
            pass
