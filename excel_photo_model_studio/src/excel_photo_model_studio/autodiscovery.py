from __future__ import annotations

import re
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EXCEL_EXTENSIONS = {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls", ".csv", ".tsv"}
PHOTO_FOLDER_NAMES = {"photo", "photos", "image", "images", "фото", "фотографии", "изображения", "кернфото"}


def discover_well_pairs(root: Path) -> dict:
    """Find unambiguous workbook/photo-folder pairs in a dropped archive tree."""
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Это не папка с архивом данных: {root}")
    workbooks: list[Path] = []
    image_parents: set[Path] = set()
    image_count = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in EXCEL_EXTENSIONS:
            workbooks.append(path)
        elif suffix in IMAGE_EXTENSIONS:
            image_parents.add(path.parent.resolve())
            image_count += 1
    workbooks.sort(key=lambda path: str(path).casefold())

    folders_with_images: set[Path] = set()
    for parent in image_parents:
        folder = parent
        while folder == root or root in folder.parents:
            folders_with_images.add(folder)
            if folder == root:
                break
            folder = folder.parent

    def contains_images(folder: Path) -> bool:
        return folder in folders_with_images

    candidates: dict[Path, list[tuple[int, Path]]] = {}
    for workbook in workbooks:
        parent = workbook.parent.resolve()
        choices: list[tuple[int, Path]] = []
        if parent in image_parents:
            choices.append((2, parent))
        stem_key = _folder_key(workbook.stem)
        for child in parent.iterdir() if parent.is_dir() else ():
            if not child.is_dir() or not contains_images(child.resolve()):
                continue
            key = _folder_key(child.name)
            if key == stem_key:
                choices.append((0, child.resolve()))
            elif key in PHOTO_FOLDER_NAMES:
                choices.append((1, child.resolve()))
        if choices:
            best_score = min(score for score, _path in choices)
            winners = sorted(
                {path for score, path in choices if score == best_score},
                key=lambda path: str(path).casefold(),
            )
            candidates[workbook] = [(best_score, path) for path in winners]
        else:
            candidates[workbook] = []

    selected = {
        workbook: options[0][1]
        for workbook, options in candidates.items()
        if len(options) == 1
    }
    owners: dict[Path, list[Path]] = {}
    for workbook, photo_dir in selected.items():
        owners.setdefault(photo_dir, []).append(workbook)
    conflicts = {folder for folder, workbooks_for_folder in owners.items() if len(workbooks_for_folder) > 1}
    pairs = [
        {"excel": str(workbook), "photos": str(photo_dir)}
        for workbook, photo_dir in selected.items()
        if photo_dir not in conflicts
    ]
    unmatched = []
    for workbook, options in candidates.items():
        if not options:
            reason = "папка с фотографиями рядом не найдена"
        elif len(options) > 1:
            reason = "найдено несколько подходящих папок фото"
        elif options[0][1] in conflicts:
            reason = "одна папка фото подходит к нескольким Excel"
        else:
            continue
        unmatched.append({"excel": str(workbook), "reason": reason})
    return {"pairs": pairs, "unmatched": unmatched, "workbooks_found": len(workbooks), "photos_found": image_count}


def _folder_key(value: str) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", value.casefold())
