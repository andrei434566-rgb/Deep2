from __future__ import annotations

from pathlib import Path


def resolve_existing_path(path: str | Path) -> Path:
    """Resolve an existing path, tolerating Windows final-path lookup denial.

    Some restricted Windows environments permit normal file access and stat
    calls but deny the handle-based lookup used by ``Path.resolve(strict=True)``.
    Keep strict existence validation, then fall back to a normalized absolute
    path so later reads can use the permissions the process actually has.
    """
    expanded = Path(path).expanduser()
    try:
        return expanded.resolve(strict=True)
    except PermissionError:
        expanded.stat()
        return expanded.resolve(strict=False)
