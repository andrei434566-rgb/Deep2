"""Discover and resume human-reviewed workbook/photo batches.

Discovery only proposes unambiguous pairs. It never approves annotations or
silently mixes photographs from different well directories.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .autodiscovery import EXCEL_EXTENSIONS, IMAGE_EXTENSIONS, PHOTO_FOLDER_NAMES
from .paths import resolve_existing_path
from .photos import extract_well
from .storage import app_data_dir, replace_or_write
from .tabular import well_key


def default_queue_path() -> Path:
    return app_data_dir() / "archive_queue.json"


def _key(value: str) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", value.casefold().replace("ё", "е"))


_CONFUSABLES = str.maketrans({
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h",
    "о": "o", "р": "p", "с": "c", "т": "t", "у": "y", "х": "x",
})


def _well_pair_key(value: str) -> str:
    """Treat mixed Latin/Cyrillic lookalikes as the same well identifier."""
    return well_key(value).translate(_CONFUSABLES)


def _path_key(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).expanduser().absolute()))


def _photo_groups(root: Path, images: list[Path]) -> dict[Path, list[Path]]:
    """Group by the first folder under root, so a well's pages stay together."""
    groups: dict[Path, list[Path]] = {}
    for image in images:
        relative = image.relative_to(root)
        group = root / relative.parts[0] if len(relative.parts) > 1 else root
        groups.setdefault(group, []).append(image)
    # A generic wrapper such as photos/<well>/... is not a well boundary.
    while len(groups) == 1:
        group, members = next(iter(groups.items()))
        if group == root or _key(group.name) not in PHOTO_FOLDER_NAMES:
            break
        children: dict[Path, list[Path]] = {}
        for image in members:
            relative = image.relative_to(group)
            child = group / relative.parts[0] if len(relative.parts) > 1 else group
            children.setdefault(child, []).append(image)
        if len(children) <= 1 and next(iter(children)) == group:
            break
        groups = children
    return groups


def discover_archive(excel_root: Path, photo_root: Path) -> dict:
    """Suggest workbook/photo-directory pairs across two recursive roots."""
    excel_root = resolve_existing_path(excel_root)
    photo_root = resolve_existing_path(photo_root)
    if not excel_root.is_dir() or not photo_root.is_dir():
        raise ValueError("Выберите существующие папки Excel и фотографий.")
    workbooks = sorted(
        (path for path in excel_root.rglob("*")
         if path.is_file() and path.suffix.lower() in EXCEL_EXTENSIONS
         and not path.name.startswith("~$")),
        key=lambda path: str(path).casefold(),
    )
    images = sorted(
        (path for path in photo_root.rglob("*")
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )
    groups = _photo_groups(photo_root, images)
    group_keys: dict[Path, set[str]] = {}
    group_explicit_wells: dict[Path, set[str]] = {}
    for folder, members in groups.items():
        explicit_wells = {
            _well_pair_key(extract_well(path.stem))
            for path in members if extract_well(path.stem)
        }
        group_explicit_wells[folder] = explicit_wells
        keys = {_key(folder.name), _well_pair_key(folder.name)}
        folder_explicit = extract_well(folder.name)
        if folder_explicit:
            keys.add(_well_pair_key(folder_explicit))
        keys.update(explicit_wells)
        group_keys[folder] = {key for key in keys if key}

    choices: dict[Path, list[Path]] = {}
    probes: dict[Path, dict] = {}
    for workbook in workbooks:
        explicit = _well_pair_key(extract_well(workbook.stem))
        names = {explicit} if explicit else set()
        names.add(_key(workbook.stem))
        names.add(_well_pair_key(workbook.stem))
        if workbook.parent != excel_root:
            names.add(_key(workbook.parent.name))
            names.add(_well_pair_key(workbook.parent.name))
        names.discard("")
        matches = [
            folder for folder, keys in group_keys.items()
            if names & keys and len(group_explicit_wells[folder]) <= 1
        ]
        if not matches:
            # Generic workbook names are common. Only inspect the Excel body
            # when filenames/folders cannot identify the well cheaply.
            try:
                from .tabular import read_many_tables
                rows, _mappings, _issues, _files = read_many_tables([workbook])
                wells = sorted({_well_pair_key(row.well) for row in rows if _well_pair_key(row.well)})
                probes[workbook] = {"excel_wells": wells, "excel_rows": len(rows)}
                if len(wells) == 1 and (not explicit or explicit == wells[0]):
                    matches = [
                        folder for folder, keys in group_keys.items()
                        if wells[0] in keys and len(group_explicit_wells[folder]) <= 1
                    ]
                elif explicit and len(wells) == 1 and explicit != wells[0]:
                    probes[workbook]["excel_probe_error"] = (
                        "Скважина внутри Excel отличается от имени файла — выберите пару вручную"
                    )
            except Exception as exc:
                # A damaged or unsupported workbook must not stop discovery of
                # all the other wells; project creation will report it later.
                probes[workbook] = {"excel_probe_error": str(exc)}
        # A single workbook and a single photo group are unambiguous even when
        # the filenames carry no well identifier. The operator still reviews it.
        if (not matches and not explicit and len(workbooks) == len(groups) == 1
                and len(group_explicit_wells[next(iter(groups))]) <= 1):
            matches = list(groups)
        choices[workbook] = matches

    owners: dict[Path, list[Path]] = {}
    for workbook, matches in choices.items():
        if len(matches) == 1:
            owners.setdefault(matches[0], []).append(workbook)
    entries = []
    for workbook in workbooks:
        matches = choices[workbook]
        if not matches:
            folder, reason = "", "Папка фото не определена — выберите вручную"
        elif len(matches) > 1:
            folder, reason = "", "Несколько подходящих папок фото — выберите вручную"
        elif len(owners[matches[0]]) > 1:
            folder, reason = "", "Одна папка подходит нескольким Excel — выберите вручную"
        else:
            folder, reason = str(matches[0]), ""
        entries.append({
            "excel": str(workbook), "photos": folder, "reason": reason,
            "photo_count": len(groups[Path(folder)]) if folder else 0,
            "project": "", "manual_pair": False,
            **probes.get(workbook, {}),
        })
    return {
        "excel_root": str(excel_root), "photo_root": str(photo_root),
        "workbooks_found": len(workbooks), "photos_found": len(images),
        "entries": entries,
    }


def load_archive_queue(path: Path | None = None) -> dict:
    path = Path(path or default_queue_path())
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"excel_root": "", "photo_root": "", "entries": []}
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        return {"excel_root": "", "photo_root": "", "entries": []}
    return payload


def merge_archive_queue(discovered: dict, previous: dict) -> dict:
    """Keep manual pairing and project progress when rescanning the same files."""
    prior = {
        _path_key(item["excel"]): item for item in previous.get("entries", [])
        if isinstance(item, dict) and item.get("excel")
    }
    entries = []
    for found in discovered["entries"]:
        item = dict(found)
        old = prior.get(_path_key(item["excel"]))
        if old:
            if old.get("manual_pair") and old.get("photos") and Path(str(old["photos"])).is_dir():
                item["photos"] = str(old["photos"])
                item["manual_pair"] = True
                item["reason"] = ""
            if old.get("project") and project_matches_pair(
                Path(str(old["project"])), Path(item["excel"]), Path(str(item["photos"])),
            ):
                item["project"] = str(old["project"])
        entries.append(item)
    return {**discovered, "entries": entries}


def project_matches_pair(project: Path, excel: Path, photos: Path) -> bool:
    try:
        config = json.loads((project / "project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    excel_paths = config.get("excel_paths") or [config.get("excel_path", "")]
    return (
        _path_key(excel) in {_path_key(value) for value in excel_paths if value}
        and _path_key(photos) == _path_key(config.get("photos_dir", ""))
    )


def save_archive_queue(payload: dict, path: Path | None = None) -> Path:
    path = Path(path or default_queue_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    replace_or_write(temporary, path)
    return path


def prepare_archive_queue(
    payload: dict,
    *,
    use_ocr: bool,
    project_root: Path | None = None,
    queue_path: Path | None = None,
    progress=None,
) -> dict:
    """Recognize all new unambiguous pairs without approving training labels.

    Each finished pair is checkpointed so a long batch can resume. Failures are
    recorded per workbook and do not prevent later wells from being processed.
    """
    from .project import create_project

    progress = progress or (lambda _message: None)
    project_root = Path(project_root or app_data_dir() / "projects")
    result = {**payload, "entries": [dict(item) for item in payload.get("entries", [])]}
    total = len(result["entries"])
    for index, entry in enumerate(result["entries"], start=1):
        excel_text = str(entry.get("excel", ""))
        photos_text = str(entry.get("photos", ""))
        if not excel_text or not photos_text:
            progress(f"[{index}/{total}] Пропуск: для Excel не выбрана однозначная папка фото.")
            continue
        excel, photos = Path(excel_text), Path(photos_text)
        existing = Path(str(entry["project"])) if entry.get("project") else None
        if existing is not None and project_matches_pair(existing, excel, photos):
            progress(f"[{index}/{total}] Уже распознано: {excel.name}.")
            continue
        if not excel.is_file() or not photos.is_dir():
            entry["error"] = "Исходный Excel или папка фото больше не существует"
            progress(f"[{index}/{total}] {excel.name}: {entry['error']}.")
        else:
            safe_stem = re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", excel.stem).strip("_") or "project"
            project = project_root / f"{safe_stem}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
            progress(f"[{index}/{total}] Распознаю {excel.name} и фото в {photos.name}…")
            try:
                report = create_project(excel, photos, project, use_ocr=use_ocr)
            except Exception as exc:
                entry["error"] = str(exc)
                progress(f"[{index}/{total}] Ошибка {excel.name}: {exc}")
            else:
                entry["project"] = str(project)
                entry["error"] = ""
                entry["excel_rows"] = int(report.get("excel_rows", 0))
                entry["mask_count"] = int(report.get("annotations", 0))
                entry["blocking_errors"] = int(report.get("blocking_errors", 0))
                progress(
                    f"[{index}/{total}] Готово {excel.name}: "
                    f"{entry['excel_rows']} строк Excel, {report.get('photos', 0)} фото, "
                    f"{entry['mask_count']} масок, ошибок — {entry['blocking_errors']}. "
                    "Маски требуют проверки перед обучением."
                )
        try:
            save_archive_queue(result, queue_path)
        except OSError as exc:
            progress(f"Очередь пока не сохранена на диск: {exc}")
    return result
