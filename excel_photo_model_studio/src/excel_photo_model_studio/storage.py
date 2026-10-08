from __future__ import annotations

import os
import sys
from pathlib import Path


def app_data_dir() -> Path:
    """Return persistent app storage, avoiding per-run Codex sandbox profiles."""
    override = os.environ.get("EXCEL_PHOTO_MODEL_STUDIO_DATA_DIR")
    if override:
        return Path(override).expanduser()
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    if not getattr(sys, "frozen", False) and "sandbox." in str(local_app_data).casefold():
        # Codex creates a new LOCALAPPDATA\Packages\sandbox.<id> per run.
        # Source runs use ignored workspace storage so accumulated wells survive
        # restarting this development app; packaged builds use normal AppData.
        repository_root = Path(__file__).resolve().parents[3]
        return repository_root / "outputs" / "appdata" / "ExcelPhotoModelStudio"
    return local_app_data / "ExcelPhotoModelStudio"


def replace_or_write(temporary: Path, destination: Path) -> None:
    """Commit a prepared file, tolerating Windows sandboxes that deny renames."""
    try:
        temporary.replace(destination)
    except PermissionError:
        # A few sandboxed AppData locations permit creating files but block
        # MoveFile/replace. Fall back to writing the already-prepared contents
        # in place so derived caches and small manifests remain usable.
        destination.write_bytes(temporary.read_bytes())
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
