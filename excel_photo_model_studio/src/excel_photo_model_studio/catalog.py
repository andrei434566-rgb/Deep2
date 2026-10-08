from __future__ import annotations

import json
import os
import csv
import hashlib
import io
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .paths import resolve_existing_path
from .storage import app_data_dir, replace_or_write


def default_catalog_path() -> Path:
    return app_data_dir() / "training_catalog.json"


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


def _confirmation_error(project_dir: Path) -> str:
    report_path = project_dir / "report.json"
    annotations_path = project_dir / "annotations.csv"
    if not report_path.is_file() or not annotations_path.is_file():
        return "проект ещё не прошёл полную проверку"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        with annotations_path.open("r", encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source, delimiter=";"))
        from .dataset import _validate_annotation, _verify_validation_snapshot
        _verify_validation_snapshot(project_dir, report)
    except Exception as exc:
        return str(exc)
    if not rows:
        return "в проекте нет ни одной маски фации"
    if int(report.get("annotations", -1)) != len(rows):
        return "число масок не совпадает с отчётом проверки; пересчитайте проект"
    approved_count = sum(row.get("approved") == "1" for row in rows)
    if int(report.get("approved_annotations", -1)) != approved_count:
        return "число подтверждённых масок не совпадает с отчётом; сохраните подтверждения ещё раз"
    approved_rows = [row for row in rows if row.get("approved") == "1"]
    pending = len(rows) - len(approved_rows)
    is_cache = (project_dir / "cache_manifest.json").is_file()
    if is_cache and pending:
        return f"в подтверждённом кэше найдены неподтверждённые маски: {pending}"
    if not approved_rows:
        return f"не подтверждены маски: {pending} из {len(rows)}; отметьте хотя бы одну маску"
    snapshot = report.get("validation_snapshot") or {}
    snapshot_photos = {
        os.path.normcase(str(Path(value).expanduser().resolve()))
        for value in snapshot.get("photos", [])
    }
    for row in approved_rows:
        try:
            _validate_annotation(row)
            photo = resolve_existing_path(row["photo"])
        except (KeyError, OSError, ValueError) as exc:
            return f"подтверждённая маска {row.get('annotation_id', '')} некорректна: {exc}"
        if os.path.normcase(str(photo)) not in snapshot_photos:
            return f"Фото для подтверждённой маски {row.get('annotation_id', '')} не входит в проверенный проект"
    return ""


def confirm_project_for_training(project_dir: Path, path: Path | None = None) -> Path:
    """Snapshot approved photos and masks into the persistent cache and catalog."""
    project_dir = resolve_existing_path(project_dir)
    catalog_path = Path(path or default_catalog_path()).expanduser().absolute()
    source_key = os.path.normcase(str(project_dir))
    payload = _read_catalog_payload(catalog_path)
    problem = _confirmation_error(project_dir)
    if problem:
        _deactivate_source_project(payload, catalog_path, source_key)
        raise ValueError(f"Скважина не добавлена в накопительный датасет: {problem}.")
    source_report = json.loads((project_dir / "report.json").read_text(encoding="utf-8"))
    source_annotation_digest = hashlib.sha256((project_dir / "annotations.csv").read_bytes()).hexdigest()
    source_signature = _source_signature(source_report, source_annotation_digest)
    for item in payload.get("projects", []):
        if not isinstance(item, dict) or os.path.normcase(str(item.get("source_project", ""))) != source_key:
            continue
        cached = Path(str(item.get("path", ""))).expanduser().absolute()
        if cached.is_dir() and not _confirmation_error(cached):
            try:
                manifest = json.loads((cached / "cache_manifest.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}
            if manifest.get("source_signature") == source_signature:
                return catalog_path

    cache_root = catalog_path.parent / "confirmed_wells"
    snapshot = _snapshot_confirmed_project(project_dir, cache_root)
    # A newer approval for the same source replaces the catalog entry but does
    # not delete the earlier snapshot, keeping old data recoverable.
    retained = []
    for item in payload.get("projects", []):
        old_source = item.get("source_project") if isinstance(item, dict) else None
        if old_source and os.path.normcase(str(old_source)) == source_key:
            continue
        value = item.get("path") if isinstance(item, dict) else item
        if value:
            existing = Path(value).expanduser().absolute()
            if not old_source and os.path.normcase(str(existing)) == source_key:
                continue
            if os.path.normcase(str(existing)) != os.path.normcase(str(snapshot)):
                retained.append({
                    "path": str(existing),
                    "source_project": item.get("source_project", "") if isinstance(item, dict) else "",
                    "confirmed_at": item.get("confirmed_at", "") if isinstance(item, dict) else "",
                })
    retained.append({
        "path": str(snapshot), "source_project": str(project_dir),
        "confirmed_at": datetime.now().isoformat(timespec="seconds"),
    })
    _write_catalog(catalog_path, retained)
    return catalog_path


def _snapshot_confirmed_project(source_project: Path, cache_root: Path) -> Path:
    report = json.loads((source_project / "report.json").read_text(encoding="utf-8"))
    snapshot = report["validation_snapshot"]
    all_rows = list(csv.DictReader(
        io.StringIO((source_project / "annotations.csv").read_text(encoding="utf-8-sig"), newline=""),
        delimiter=";",
    ))
    rows = [row for row in all_rows if row.get("approved") == "1"]
    if not rows:
        raise ValueError("Отметьте и сохраните хотя бы одну проверенную маску.")
    from .dataset import _validate_annotation, _verify_validation_snapshot
    _verify_validation_snapshot(source_project, report)
    source_photos = [resolve_existing_path(value) for value in snapshot["photos"]]
    photos_by_key = {os.path.normcase(str(photo)): photo for photo in source_photos}
    referenced_photos = {}
    for row in rows:
        _validate_annotation(row)
        photo = resolve_existing_path(row["photo"])
        key = os.path.normcase(str(photo))
        if key not in photos_by_key:
            raise ValueError(f"Фото для маски {row.get('annotation_id', '')} не входит в проверенный проект.")
        referenced_photos[key] = photos_by_key[key]
    source_photos = sorted(referenced_photos.values(), key=lambda item: item.name.casefold())
    if not source_photos:
        raise ValueError("В подтверждённой скважине нет фотографий для кэширования.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = hashlib.sha256(str(source_project).encode("utf-8")).hexdigest()[:8]
    snapshot_name = f"{source_project.name}_{stamp}_{suffix}"
    with _confirmed_snapshot_directory(source_project, snapshot, cache_root, snapshot_name) as final_dir:
        # Build directly in a uniquely named destination. Some Windows
        # sandboxed AppData locations deny creating .pending_* subdirectories
        # or renaming them, even though ordinary file writes are allowed.
        staging = final_dir
        photos_dir = staging / "photos"
        photo_map: dict[str, str] = {}
        photo_hashes = {}
        photo_names = {}
        photo_file_stats = {}
        for source in source_photos:
            expected = snapshot.get("files", {}).get(str(source))
            before = source.stat()
            if expected and [before.st_size, before.st_mtime_ns] != expected:
                raise ValueError(f"Фото изменилось после проверки: {source.name}. Пересчитайте скважину.")
            raw = source.read_bytes()
            after = source.stat()
            if expected and [after.st_size, after.st_mtime_ns] != expected:
                raise ValueError(f"Фото изменилось во время кэширования: {source.name}. Пересчитайте скважину.")
            digest = hashlib.sha256(raw).hexdigest()
            target_name = f"{digest[:12]}_{source.name}"
            staged_target = photos_dir / target_name
            staged_target.write_bytes(raw)
            final_target = final_dir / "photos" / target_name
            photo_map[str(source)] = str(final_target)
            photo_hashes[str(final_target)] = digest
            photo_names[str(final_target)] = source.name
            photo_file_stats[str(final_target)] = [staged_target.stat().st_size, staged_target.stat().st_mtime_ns]

        annotations_path = source_project / "annotations.csv"
        annotations_raw = annotations_path.read_bytes()
        expected_annotations = snapshot.get("files", {}).get(str(annotations_path.resolve()))
        annotations_stat = annotations_path.stat()
        if expected_annotations and [annotations_stat.st_size, annotations_stat.st_mtime_ns] != expected_annotations:
            raise ValueError("Подтверждения масок изменились после проверки. Пересчитайте скважину.")
        annotations_after = annotations_path.stat()
        if expected_annotations and [annotations_after.st_size, annotations_after.st_mtime_ns] != expected_annotations:
            raise ValueError("Подтверждения масок изменились во время кэширования. Пересчитайте скважину.")
        source_annotation_digest = hashlib.sha256(annotations_raw).hexdigest()
        reader = csv.DictReader(io.StringIO(annotations_raw.decode("utf-8-sig"), newline=""), delimiter=";")
        fieldnames = list(reader.fieldnames or [])
        current_rows = list(reader)
        if (
            len(current_rows) != int(report.get("annotations", -1))
            or sum(row.get("approved") == "1" for row in current_rows)
            != int(report.get("approved_annotations", -1))
        ):
            raise ValueError("Маски изменились во время кэширования. Повторите проверку и подтверждение.")
        rows = [row for row in current_rows if row.get("approved") == "1"]
        if not rows:
            raise ValueError("Отметьте и сохраните хотя бы одну проверенную маску.")
        for row in rows:
            _validate_annotation(row)
            original = str(resolve_existing_path(row["photo"]))
            if original not in photo_map:
                raise ValueError(f"Маска ссылается на фото вне проверенного снимка: {Path(original).name}")
            row["photo"] = photo_map[original]
            row["preview"] = ""
        cached_annotations = staging / "annotations.csv"
        with cached_annotations.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames, delimiter=";", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        cached_report = dict(report)
        cached_snapshot = dict(snapshot)
        cached_snapshot["photos"] = sorted(photo_map.values())
        cached_snapshot["photos_dir"] = str(final_dir / "photos")
        cached_snapshot["files"] = dict(photo_file_stats)
        cached_snapshot["files"][str((final_dir / "annotations.csv").resolve())] = [
            cached_annotations.stat().st_size, cached_annotations.stat().st_mtime_ns,
        ]
        cached_report["photos"] = len(source_photos)
        cached_report["confirmed_photos"] = len(source_photos)
        cached_report["unconfirmed_photos"] = 0
        cached_report["annotations"] = len(rows)
        cached_report["approved_annotations"] = len(rows)
        cached_report["confirmation_scope"] = "approved_annotations"
        cached_report["validation_snapshot"] = cached_snapshot
        cached_report["project_dir"] = str(final_dir)
        (staging / "report.json").write_text(
            json.dumps(cached_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        (staging / "project.json").write_text("{}\n", encoding="utf-8")
        (staging / "cache_manifest.json").write_text(json.dumps({
            "schema": "confirmed-well-cache-v2", "confirmation_scope": "approved_annotations",
            "source_project": str(source_project),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "photos": photo_hashes, "photo_names": photo_names,
            "annotation_count": len(rows),
            "photo_count": len(source_photos), "approved_annotation_count": len(rows),
            "partial_confirmation": len(rows) < len(all_rows),
            "source_signature": _source_signature(report, source_annotation_digest),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _verify_validation_snapshot(source_project, report)
    return final_dir


@contextmanager
def _confirmed_snapshot_directory(
    source_project: Path,
    snapshot: dict,
    preferred_root: Path,
    snapshot_name: str,
):
    photo_directory = Path(snapshot.get("photos_dir") or source_project.parent).expanduser()
    roots = (
        preferred_root,
        source_project / "confirmed_wells",
        photo_directory.parent / "ExcelPhotoModelStudio_cache" / "confirmed_wells",
        Path.home() / "Documents" / "ExcelPhotoModelStudio" / "confirmed_wells",
        Path(tempfile.gettempdir()) / "ExcelPhotoModelStudio" / "confirmed_wells",
    )
    seen = set()
    errors = []
    destination = None
    for root in roots:
        key = os.path.normcase(str(root.absolute()))
        if key in seen:
            continue
        seen.add(key)
        try:
            root.mkdir(parents=True, exist_ok=True)
            candidate = root / snapshot_name
            candidate.mkdir()
            try:
                (candidate / "photos").mkdir()
            except PermissionError:
                try:
                    candidate.rmdir()
                except OSError:
                    pass
                raise
            destination = resolve_existing_path(candidate)
            break
        except PermissionError as exc:
            errors.append(exc)
    if destination is None:
        if errors:
            raise errors[-1]
        raise PermissionError("Не удалось создать папку накопительного кэша скважины.")
    yield destination


def _source_signature(report: dict, annotations_digest: str) -> str:
    """Fingerprint approved annotations plus the validated source-file stamps."""
    snapshot = report.get("validation_snapshot") or {}
    payload = {
        "files": sorted(
            (str(path), values) for path, values in snapshot.get("files", {}).items()
            if Path(path).name.casefold() != "annotations.csv"
        ),
        "photos": sorted(str(value) for value in snapshot.get("photos", [])),
        "annotations_sha256": annotations_digest,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_confirmed_project_catalog(path: Path | None = None) -> list[Path]:
    """Return local snapshots containing at least one individually confirmed mask."""
    catalog_path = Path(path or default_catalog_path()).expanduser().absolute()
    _recover_unlisted_approved_projects(catalog_path)
    for project in load_project_catalog(catalog_path):
        if (project / "cache_manifest.json").is_file() or _confirmation_error(project):
            continue
        # Older releases stored references to live project folders. Snapshot
        # valid ones on first use so future training no longer depends on the
        # original photo folder remaining in place.
        try:
            confirm_project_for_training(project, catalog_path)
        except (OSError, ValueError):
            continue
    return [
        project for project in load_project_catalog(catalog_path)
        if (project / "cache_manifest.json").is_file() and not _confirmation_error(project)
    ]


def _recover_unlisted_approved_projects(catalog_path: Path) -> None:
    """Retry previously saved approvals whose first cache write was interrupted."""
    projects_root = catalog_path.parent / "projects"
    if not projects_root.is_dir():
        return
    payload = _read_catalog_payload(catalog_path)
    known_sources = {
        os.path.normcase(str(Path(item.get("source_project", "")).expanduser().absolute()))
        for item in payload.get("projects", []) if isinstance(item, dict) and item.get("source_project")
    }
    for project in sorted(projects_root.iterdir(), key=lambda item: item.name.casefold()):
        if not project.is_dir() or not (project / "project.json").is_file():
            continue
        key = os.path.normcase(str(project.expanduser().absolute()))
        if key in known_sources or _confirmation_error(project):
            continue
        try:
            confirm_project_for_training(project, catalog_path)
            known_sources.add(key)
        except (OSError, ValueError):
            # Keep the app usable if a stale project can't be recovered now;
            # the next catalog refresh will retry it.
            continue


def register_project(project_dir: Path, path: Path | None = None) -> Path:
    project_dir = resolve_existing_path(project_dir)
    path = Path(path or default_catalog_path()).expanduser().absolute()
    existing = load_project_catalog(path)
    key = os.path.normcase(str(project_dir))
    if all(os.path.normcase(str(item)) != key for item in existing):
        existing.append(project_dir)
    entries = [
        {"path": str(item), "source_project": "", "confirmed_at": ""}
        for item in existing
    ]
    _write_catalog(path, entries)
    return path


def _read_catalog_payload(path: Path) -> dict:
    if not path.is_file():
        return {"projects": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"projects": []}
    except (OSError, ValueError):
        return {"projects": []}


def _deactivate_source_project(payload: dict, catalog_path: Path, source_key: str) -> None:
    """Stop training from an older snapshot after its source project is invalidated."""
    retained = []
    removed = False
    for item in payload.get("projects", []):
        if isinstance(item, dict):
            item_source = item.get("source_project")
            item_path = item.get("path")
        else:
            item_source, item_path = None, item
        if item_source and os.path.normcase(str(item_source)) == source_key:
            removed = True
            continue
        if item_path and not item_source and os.path.normcase(str(Path(item_path).expanduser().absolute())) == source_key:
            removed = True
            continue
        if item_path:
            retained.append({
                "path": str(item_path), "source_project": str(item_source or ""),
                "confirmed_at": item.get("confirmed_at", "") if isinstance(item, dict) else "",
            })
    if removed:
        _write_catalog(catalog_path, retained)


def _write_catalog(path: Path, projects: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "excel-photo-training-catalog-v2",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "projects": projects,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    replace_or_write(temporary, path)


def catalog_summary(path: Path | None = None) -> dict[str, int]:
    return catalog_overview(path)["summary"]


def catalog_well_details(path: Path | None = None) -> list[dict[str, object]]:
    """Describe each well represented by individually confirmed masks in the cache."""
    return catalog_overview(path)["wells"]


def _cached_photo_name(row: dict[str, str], manifest: dict) -> str:
    photo_path = str(row.get("photo", ""))
    source_names = manifest.get("photo_names", {})
    if isinstance(source_names, dict) and source_names.get(photo_path):
        return str(source_names[photo_path])
    name = Path(photo_path).name
    digests = manifest.get("photos", {})
    digest = digests.get(photo_path, "") if isinstance(digests, dict) else ""
    prefix = f"{str(digest)[:12]}_" if digest else ""
    return name[len(prefix):] if prefix and name.startswith(prefix) else name


def catalog_overview(path: Path | None = None) -> dict[str, object]:
    """Load the confirmed-well list and its aggregate counts in one pass."""
    projects = load_confirmed_project_catalog(path)
    photos = annotations = approved = 0
    details = []
    for project in projects:
        try:
            report = json.loads((project / "report.json").read_text(encoding="utf-8"))
            with (project / "annotations.csv").open("r", encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source, delimiter=";"))
            manifest = json.loads((project / "cache_manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # The catalogue loader already checks validity. If a file changes
            # between that check and this read, omit this entry rather than
            # showing a stale confirmation as ready.
            continue
        photos += int(manifest.get("photo_count", report.get("photos", len(manifest.get("photos", {})))))
        annotations += int(manifest.get("annotation_count", report.get("annotations", len(rows))))
        approved += int(report.get("approved_annotations", 0))
        well_names = sorted({
            str(row.get("well", "")).strip() for row in rows if str(row.get("well", "")).strip()
        }, key=str.casefold)
        facies_indices = {
            str(row.get("facies_index", "")).strip().casefold()
            for row in rows if str(row.get("facies_index", "")).strip()
        }
        mask_details = [
            {
                "annotation_id": str(row.get("annotation_id", "")),
                "well": str(row.get("well", "")).strip(),
                "photo": _cached_photo_name(row, manifest),
                "depth_top": str(row.get("depth_top", "")).strip(),
                "depth_base": str(row.get("depth_base", "")).strip(),
                "facies_index": str(row.get("facies_index", "")).strip(),
                "facies_name": str(row.get("facies_name", "")).strip(),
                "label": str(row.get("label", "")).strip(),
                "target_text": " ".join(str(row.get("target_text", "")).split()),
            }
            for row in rows if row.get("approved") == "1"
        ]
        details.append({
            "project": project,
            "well_names": well_names or [project.name],
            "photos": int(manifest.get("photo_count", report.get("photos", len(manifest.get("photos", {}))))),
            "masks": int(manifest.get("annotation_count", report.get("annotations", len(rows)))),
            "facies": len(facies_indices),
            "mask_details": mask_details,
            "confirmed_at": str(manifest.get("created_at", "")),
        })
    return {
        "summary": {
            "projects": len(details), "photos": photos,
            "annotations": annotations, "approved_annotations": approved,
        },
        "wells": details,
    }
