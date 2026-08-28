"""Paths that work both from source code and from a PyInstaller build."""

from __future__ import annotations

import sys
from pathlib import Path


def source_root() -> Path:
    """Return the project root while running from source."""
    return Path(__file__).resolve().parents[1]


def application_root() -> Path:
    """Return the folder containing the executable, or the source root."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return source_root()


def bundled_root() -> Path:
    """Return the read-only application files bundled by PyInstaller."""
    if getattr(sys, "frozen", False):
        return application_root() / "_internal"
    return source_root()


def user_data_root() -> Path:
    """Return a writable folder for projects and fine-tuned models.

    A PyInstaller build keeps its bundled files in ``_internal``.  User data
    must not be saved there: Windows may block writes to Program Files and an
    application update would overwrite it.  Keep it next to the executable
    instead, where it is easy to copy or back up.
    """
    root = application_root() / "Kern Analyzer Data" if getattr(sys, "frozen", False) else source_root()
    root.mkdir(parents=True, exist_ok=True)
    return root
