from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


def default_catalog_path() -> Path:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return local_app_data / "ExcelPhotoModelStudio" / "training_catalog.json"


def load_project_catalog(path: Path | None = None) -> list[Path]:
    path = Path(path or default_catalog_path())
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    result = []
    seen = set()
    for item in payload.get("projects", []):
        value = item.get("path") if isinstance(item, dict) else item
        if not value:
            continue
        project = Path(value).expanduser().absolute()
        key = os.path.normcase(str(project))
        if key in seen or not (project / "project.json").is_file():
            continue
        seen.add(key)
        result.append(project)
    return result


def register_project(project_dir: Path, path: Path | None = None) -> Path:
    project_dir = Path(project_dir).expanduser().resolve(strict=True)
    path = Path(path or default_catalog_path()).expanduser().absolute()
    existing = load_project_catalog(path)
    key = os.path.normcase(str(project_dir))
    if all(os.path.normcase(str(item)) != key for item in existing):
        existing.append(project_dir)
    payload = {
        "schema": "excel-photo-training-catalog-v1",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "projects": [{"path": str(item)} for item in existing],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def catalog_summary(path: Path | None = None) -> dict[str, int]:
    projects = load_project_catalog(path)
    photos = annotations = approved = 0
    for project in projects:
        try:
            report = json.loads((project / "report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        photos += int(report.get("photos", 0))
        annotations += int(report.get("annotations", 0))
        approved += int(report.get("approved_annotations", 0))
    return {
        "projects": len(projects), "photos": photos,
        "annotations": annotations, "approved_annotations": approved,
    }
