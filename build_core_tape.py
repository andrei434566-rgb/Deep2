"""Build a reviewable vertical core tape from all core photographs in a folder.

The script does not assign geological facies.  It uses the project's existing
rule-based detector to find physical core columns, saves an overlay and a
rectangle mask for every source photo, and then concatenates the column crops
in the chosen order into one logical core tape.

Example:
    .venv\\Scripts\\python.exe build_core_tape.py "D:\\Керн фото\\Р-25"

The output folder contains:
  overlays/       source photos with detected column rectangles and IDs;
  masks/          binary rectangle masks in source-image coordinates;
  crops/          normalized individual column crops;
  core_tape_*.png one or more pages of the concatenated core tape;
  manifest.json   source coordinates and tape coordinates for every crop;
  review.html     a simple visual index of all overlays and tape pages.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from app.infrastructure.ml.rule_based_facies import RuleBasedFaciesDetector
from app.infrastructure.kern_analyzer_pipeline import KernAnalyzerAutomaticPipeline, KernAnalyzerDemoPipeline
from app.infrastructure.excel_core_description import (
    DescriptionLayer,
    photo_interval_from_filename,
    read_description_workbook,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
REVIEW_FOLDER_PREFIX = "_core_tape_review"
DEFAULT_TAPE_WIDTH = 360
DEFAULT_PAGE_HEIGHT = 28_000
# The tape is used later for depth-to-pixel calibration, so transitions must
# not introduce artificial white pixels or a fake physical core length.
GAP_PIXELS = 0
LEFT_MARGIN = 155
RIGHT_MARGIN = 18
AUDIT_FOLDER_NAME = "_facies_audit"


@dataclass
class CropRecord:
    photo_number: int
    column_number: int
    sequence_number: int
    source_path: Path
    crop_path: Path
    left: int
    top: int
    right: int
    bottom: int
    evidence: str
    width: int
    height: int
    tape_top: int = 0
    tape_bottom: int = 0
    page: int = 0
    page_top: int = 0


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    scope: str
    message: str


def read_image(path: Path) -> np.ndarray:
    """Read paths containing Cyrillic characters on Windows."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось открыть изображение: {path}")
    return image


def write_image(path: Path, image: np.ndarray, quality: int = 96) -> None:
    extension = path.suffix.lower() or ".png"
    parameters = [cv2.IMWRITE_JPEG_QUALITY, quality] if extension in {".jpg", ".jpeg"} else []
    success, encoded = cv2.imencode(extension, image, parameters)
    if not success:
        raise RuntimeError(f"Не удалось сохранить: {path}")
    encoded.tofile(str(path))


def natural_key(path: Path) -> list[object]:
    """Sort R-25_2.jpg before R-25_10.jpg while retaining Cyrillic names."""
    return [int(token) if token.isdigit() else token.casefold() for token in re.split(r"(\d+)", path.name)]


def filename_depth_hint(path: Path) -> list[float] | None:
    """Extract a common ``2984,00-2986,95`` depth pair from a file name."""
    match = re.search(r"(\d{3,5}[,.]\d+)\s*[-–—_]\s*(\d{3,5}[,.]\d+)", path.stem)
    if not match:
        return None
    return [float(value.replace(",", ".")) for value in match.groups()]


def draw_overlay(image: np.ndarray, columns: list[tuple[int, int, int, int]], photo_number: int) -> np.ndarray:
    result = image.copy()
    for column_number, (left, top, right, bottom) in enumerate(columns, start=1):
        cv2.rectangle(result, (left, top), (right - 1, bottom - 1), (40, 220, 70), 4)
        label = f"P{photo_number:03d} C{column_number:02d}"
        text_top = max(25, top + 32)
        cv2.rectangle(result, (left + 2, text_top - 27), (left + 160, text_top + 7), (25, 25, 25), -1)
        cv2.putText(result, label, (left + 8, text_top), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    if not columns:
        cv2.putText(result, "NO CORE COLUMNS DETECTED", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 230), 3, cv2.LINE_AA)
    return result


def resize_column(crop: np.ndarray, target_width: int) -> np.ndarray:
    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Empty core crop")
    target_height = max(1, round(height * target_width / width))
    return cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_AREA if target_height < height else cv2.INTER_CUBIC)


def append_tape_pages(records: list[CropRecord], output_dir: Path, tape_width: int, page_height: int) -> list[dict[str, object]]:
    """Save the logical tape as safe-size PNG pages and map all page coordinates."""
    pages: list[list[CropRecord]] = []
    current_page: list[CropRecord] = []
    page_cursor = 0
    global_cursor = 0

    for record in records:
        if current_page and page_cursor + record.height > page_height:
            pages.append(current_page)
            current_page = []
            page_cursor = 0
        record.page = len(pages) + 1
        record.page_top = page_cursor
        record.tape_top = global_cursor
        record.tape_bottom = global_cursor + record.height
        current_page.append(record)
        page_cursor += record.height + GAP_PIXELS
        global_cursor += record.height + GAP_PIXELS
    if current_page:
        pages.append(current_page)

    summaries: list[dict[str, object]] = []
    for page_number, page_records in enumerate(pages, start=1):
        height = max(record.page_top + record.height for record in page_records)
        canvas = np.full((height, LEFT_MARGIN + tape_width + RIGHT_MARGIN, 3), 248, dtype=np.uint8)
        for record in page_records:
            crop = read_image(record.crop_path)
            canvas[record.page_top:record.page_top + record.height, LEFT_MARGIN:LEFT_MARGIN + tape_width] = crop
            cv2.putText(canvas, f"#{record.sequence_number}", (10, record.page_top + min(28, record.height - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (25, 25, 25), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"P{record.photo_number} C{record.column_number}", (10, record.page_top + min(56, record.height - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1, cv2.LINE_AA)
        name = f"core_tape_{page_number:03d}.png"
        write_image(output_dir / name, canvas)
        summaries.append({"page": page_number, "file": name, "height_px": int(height)})
    return summaries


def relative_url(path: Path) -> str:
    return quote(path.as_posix(), safe="/._-")


def write_review_html(output_dir: Path, tape_pages: list[dict[str, object]], photos: list[dict[str, object]]) -> None:
    page_images = "\n".join(
        f'<section><h2>Лента керна — страница {item["page"]}</h2><img src="{relative_url(Path(str(item["file"]))) }" loading="lazy"></section>'
        for item in tape_pages
    )
    photo_images = "\n".join(
        f'<section><h3>Фото {item["photo_number"]}: {html.escape(str(item["source_relative"]))}</h3>'
        f'<p>{"Колонок найдено: " + str(item["columns_detected"]) if item["columns_detected"] else "Колонки не найдены — проверьте вручную."}</p>'
        f'<img src="{relative_url(Path(str(item["overlay_file"]))) }" loading="lazy"></section>'
        for item in photos
    )
    document = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Проверка ленты керна</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f4f6f8;color:#18212b}}section{{margin:26px 0;padding:16px;background:#fff;border:1px solid #d7dde4;border-radius:8px}}img{{max-width:100%;height:auto;border:1px solid #c7ced6}}h1,h2,h3{{margin-top:0}}</style>
</head><body><h1>Проверка выделения колонок и единой ленты керна</h1>
<p>Зелёные прямоугольники на фото — найденные физические колонки. Стыки в ленте идут без добавочных пикселей; их порядок указан слева.</p>
{page_images}<hr><h2>Исходные фотографии с масками</h2>{photo_images}
</body></html>"""
    (output_dir / "review.html").write_text(document, encoding="utf-8")


def _well_key(value: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "", value.casefold().replace("ё", "е"))


def _union_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping depth ranges, preserving a deterministic order."""
    merged: list[tuple[float, float]] = []
    for top, base in sorted((item for item in intervals if item[1] > item[0]), key=lambda item: (item[0], item[1])):
        if not merged or top > merged[-1][1] + 1e-7:
            merged.append((top, base))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], base))
    return merged


def _interval_length(intervals: list[tuple[float, float]]) -> float:
    return sum(base - top for top, base in intervals)


def _overlap_length(top: float, base: float, coverage: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(base, photo_base) - max(top, photo_top)) for photo_top, photo_base in coverage)


def _excel_files(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.glob("*.xlsx") if not path.name.startswith("~$")),
        key=natural_key,
    )


def _pick_excel_for_manifest(manifest_path: Path, root: Path) -> Path | None:
    """Find a workbook near a tape; ask rather than guessing when ambiguous."""
    candidates: list[Path] = []
    for folder in (manifest_path.parent, *manifest_path.parent.parents):
        candidates.extend(path for path in _excel_files(folder) if path not in candidates)
        if folder == root:
            break
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    try:
        import tkinter as tk
        from tkinter import filedialog

        window = tk.Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title=f"Выберите Excel для ленты: {manifest_path.parent.name}",
            initialdir=str(candidates[0].parent),
            filetypes=[("Excel", "*.xlsx")],
        )
        window.destroy()
        return Path(selected) if selected else None
    except Exception:
        return None


def _manifest_photo_intervals(manifest: dict[str, object], manifest_path: Path, issues: list[AuditIssue]) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for item in manifest.get("photos", []):
        if not isinstance(item, dict):
            continue
        filename = Path(str(item.get("source_relative", "")))
        try:
            interval = photo_interval_from_filename(filename)
        except ValueError as error:
            issues.append(AuditIssue("warning", str(manifest_path), f"Фото без корректного интервала в имени: {filename.name} ({error})"))
            continue
        result[interval.well].append((interval.top, interval.base))
    return result


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _audit_html(output_dir: Path, summaries: list[dict[str, object]], wells: list[dict[str, object]], issues: list[AuditIssue]) -> None:
    def table(columns: list[tuple[str, str]], rows: list[dict[str, object]]) -> str:
        header = "".join(f"<th>{html.escape(title)}</th>" for _, title in columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key, _ in columns) + "</tr>"
            for row in rows
        ) or f"<tr><td colspan=\"{len(columns)}\">Нет данных</td></tr>"
        return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    facies_table = table([
        ("facies_name", "Название фации"), ("drilling_m", "По бурению, м"),
        ("photo_covered_m", "Покрыто фото, м"), ("photo_coverage_pct", "Покрытие, %"),
        ("intervals", "Интервалов"), ("wells", "Скважин"), ("training_status", "Готовность для обучения"),
    ], summaries)
    well_table = table([
        ("well", "Скважина"), ("excel", "Excel"), ("tape", "Лента"),
        ("facies_rows", "Строк фаций"), ("declared_m", "По бурению, м"),
        ("photo_coverage_m", "Покрытие фото, м"), ("photo_coverage_pct", "Покрытие, %"),
    ], wells)
    issue_rows = [{"severity": item.severity, "scope": item.scope, "message": item.message} for item in issues]
    issue_table = table([("severity", "Уровень"), ("scope", "Источник"), ("message", "Сообщение")], issue_rows)
    totals = {
        "facies": len(summaries),
        "wells": len(wells),
        "drilling": sum(float(row["drilling_m_raw"]) for row in summaries),
        "covered": sum(float(row["photo_covered_m_raw"]) for row in summaries),
    }
    document = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Аудит фаций</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f4f6f8;color:#18212b}}section{{margin:20px 0;padding:16px;background:#fff;border:1px solid #d7dde4;border-radius:8px;overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:760px}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #e1e6eb;vertical-align:top}}th{{background:#eef3f7}}.warning{{color:#985b00}}.error{{color:#b42318}}</style>
</head><body><h1>Аудит Excel и лент керна</h1>
<p>Скважин: {totals["wells"]}; фаций: {totals["facies"]}; сумма толщин по бурению: {totals["drilling"]:.2f} м; покрыто фотографиями: {totals["covered"]:.2f} м.</p>
<section><h2>Статистика по фациям</h2><p>«По бурению» — сумма интервалов из Excel. «Покрыто фото» — пересечение этих интервалов с диапазонами из названий фотографий.</p>{facies_table}</section>
<section><h2>Сводка по скважинам</h2>{well_table}</section>
<section><h2>Проверки и замечания</h2>{issue_table}</section>
</body></html>"""
    (output_dir / "facies_audit.html").write_text(document, encoding="utf-8")


def run_audit(work_folder: Path, output_dir: Path, supplied_excels: list[Path]) -> int:
    root = work_folder.expanduser().resolve()
    if not root.is_dir():
        print(f"Папка не найдена: {root}", file=sys.stderr)
        return 2
    if output_dir.exists():
        print(f"Папка результата уже существует: {output_dir}\\nУкажите новую через --output, чтобы ничего не перезаписывать.", file=sys.stderr)
        return 2

    manifests = sorted(
        (path for path in root.rglob("manifest.json") if path.parent.name.casefold().startswith(REVIEW_FOLDER_PREFIX)),
        key=lambda path: str(path).casefold(),
    )
    if not manifests:
        print("Не найдены manifest.json внутри папок _core_tape_review. Сначала соберите ленты керна.", file=sys.stderr)
        return 2
    if supplied_excels and len(supplied_excels) not in {1, len(manifests)}:
        print("Для --excel укажите один файл (для одной ленты) или по одному файлу на каждую найденную ленту.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True)
    issues: list[AuditIssue] = []
    stats: dict[str, dict[str, object]] = {}
    well_rows: list[dict[str, object]] = []
    for index, manifest_path in enumerate(manifests):
        excel_path = (supplied_excels[0] if len(supplied_excels) == 1 else supplied_excels[index]) if supplied_excels else _pick_excel_for_manifest(manifest_path, root)
        if excel_path is None or not excel_path.is_file():
            issues.append(AuditIssue("error", str(manifest_path), "Не выбран Excel для этой ленты; скважина не вошла в статистику."))
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as error:
            issues.append(AuditIssue("error", str(manifest_path), f"Не удалось прочитать manifest.json: {error}"))
            continue
        photo_by_well = _manifest_photo_intervals(manifest, manifest_path, issues)
        coverage_by_key = {_well_key(well): _union_intervals(intervals) for well, intervals in photo_by_well.items()}
        try:
            layers, excel_issues = read_description_workbook(excel_path)
        except Exception as error:
            issues.append(AuditIssue("error", str(excel_path), f"Не удалось прочитать Excel: {error}"))
            continue
        for issue in excel_issues:
            issues.append(AuditIssue("warning", f"{excel_path.name}: {issue.source}", issue.message))
        available_wells = set(coverage_by_key)
        matched_layers = [layer for layer in layers if _well_key(layer.well) in available_wells] if available_wells else layers
        if available_wells and not matched_layers:
            issues.append(AuditIssue("error", str(excel_path), "Номера скважин в Excel не совпали с именами фотографий; проверьте формат «Р-31_кровля-подошва.jpg»."))
            continue
        if not available_wells:
            issues.append(AuditIssue("warning", str(manifest_path), "В названиях фото нет корректных интервалов; покрытие фотографиями не рассчитано."))
        declared_ranges: dict[str, list[tuple[float, float]]] = defaultdict(list)
        used_wells: set[str] = set()
        for layer in matched_layers:
            label = layer.facies_name.strip() or "Не задано"
            if label == "Не задано":
                issues.append(AuditIssue("warning", f"{excel_path.name}: {layer.sheet}!{layer.row}", "Фация без названия не должна попадать в обучение."))
            length = layer.base - layer.top
            key = _well_key(layer.well)
            covered = _overlap_length(layer.top, layer.base, coverage_by_key.get(key, []))
            item = stats.setdefault(label, {"facies_name": label, "drilling_m": 0.0, "photo_covered_m": 0.0, "intervals": 0, "wells": set()})
            item["drilling_m"] = float(item["drilling_m"]) + length
            item["photo_covered_m"] = float(item["photo_covered_m"]) + covered
            item["intervals"] = int(item["intervals"]) + 1
            item["wells"].add(layer.well)
            declared_ranges[key].append((layer.top, layer.base))
            used_wells.add(layer.well)
        for well_key, intervals in declared_ranges.items():
            raw = _interval_length(intervals)
            unique = _interval_length(_union_intervals(intervals))
            if raw - unique > 0.001:
                issues.append(AuditIssue("warning", excel_path.name, f"В Excel есть перекрывающиеся интервалы фаций для скважины {well_key}: сумма строк может считать метры дважды."))
        for well in sorted(used_wells):
            key = _well_key(well)
            declared = _interval_length(declared_ranges[key])
            covered = sum(_overlap_length(top, base, coverage_by_key.get(key, [])) for top, base in declared_ranges[key])
            well_rows.append({
                "well": well, "excel": excel_path.name, "tape": str(manifest_path.parent.relative_to(root)),
                "facies_rows": len(declared_ranges[key]), "declared_m": round(declared, 3), "photo_coverage_m": round(covered, 3),
                "photo_coverage_pct": round(covered / declared * 100, 1) if declared else 0.0,
            })

    summaries: list[dict[str, object]] = []
    for item in stats.values():
        drilling = float(item["drilling_m"])
        covered = float(item["photo_covered_m"])
        well_count = len(item["wells"])
        if covered < 2 or well_count < 2:
            readiness = "мало данных — добавить фото/скважины"
        elif covered < 10 or well_count < 3:
            readiness = "ограниченно — балансировать при обучении"
        else:
            readiness = "достаточно для первого обучения"
        summaries.append({
            "facies_name": item["facies_name"], "drilling_m": round(drilling, 3), "photo_covered_m": round(covered, 3),
            "photo_coverage_pct": round(covered / drilling * 100, 1) if drilling else 0.0, "intervals": item["intervals"],
            "wells": well_count, "training_status": readiness, "drilling_m_raw": drilling, "photo_covered_m_raw": covered,
        })
    summaries.sort(key=lambda item: (-float(item["drilling_m_raw"]), str(item["facies_name"]).casefold()))
    well_rows.sort(key=lambda item: str(item["well"]).casefold())
    _write_csv(output_dir / "facies_summary.csv", summaries, ["facies_name", "drilling_m", "photo_covered_m", "photo_coverage_pct", "intervals", "wells", "training_status"])
    _write_csv(output_dir / "audit_issues.csv", [{"severity": item.severity, "scope": item.scope, "message": item.message} for item in issues], ["severity", "scope", "message"])
    payload = {"created_at": datetime.now().isoformat(timespec="seconds"), "root_folder": str(root), "facies": summaries, "wells": well_rows, "issues": [item.__dict__ for item in issues]}
    (output_dir / "facies_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _audit_html(output_dir, summaries, well_rows, issues)
    print(f"\\nАудит готов. Скважин: {len(well_rows)}, фаций: {len(summaries)}, замечаний: {len(issues)}")
    print(f"Отчёт: {output_dir / 'facies_audit.html'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Склеивает выделенные колонки керна в одну проверяемую ленту.")
    parser.add_argument("photo_folder", type=Path, nargs="?", help="Папка с фото одной скважины или корневая папка аудита")
    parser.add_argument("--output", type=Path, help="Новая папка результата; по умолчанию _core_tape_review внутри папки фото")
    parser.add_argument("--tape-width", type=int, default=DEFAULT_TAPE_WIDTH, help=f"Ширина нормализованной колонки, px (по умолчанию {DEFAULT_TAPE_WIDTH})")
    parser.add_argument("--page-height", type=int, default=DEFAULT_PAGE_HEIGHT, help=f"Максимальная высота PNG-страницы, px (по умолчанию {DEFAULT_PAGE_HEIGHT})")
    parser.add_argument("--reverse-photos", action="store_true", help="Обработать фотографии в обратном порядке")
    parser.add_argument("--right-to-left", action="store_true", help="Склеивать колонки каждой фотографии справа налево")
    parser.add_argument("--audit", action="store_true", help="Проверить Excel и уже собранные ленты керна")
    parser.add_argument("--excel", type=Path, action="append", default=[], help="Excel для аудита; можно указать по одному файлу на каждую ленту")
    parser.add_argument("--excel-masks", type=Path, help="Автоматически наложить фации Excel на фото и создать YOLO-маски")
    parser.add_argument("--demo-random-facies", action="store_true", help="DEMO: закрыть керн случайными синтетическими фациями; не для обучения")
    parser.add_argument("--demo-facies", type=int, default=7, help="Количество синтетических DEMO-фаций (по умолчанию 7)")
    parser.add_argument("--demo-seed", type=int, help="Seed DEMO для повторяемого случайного разбиения")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.photo_folder is None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(title="Выберите корневую папку аудита" if args.audit else "Выберите папку с фото одной скважины")
            root.destroy()
        except Exception as error:
            print(f"Не удалось открыть выбор папки: {error}", file=sys.stderr)
            return 2
        if not selected:
            print("Папка не выбрана.")
            return 0
        args.photo_folder = Path(selected)
    source_dir = args.photo_folder.expanduser().resolve()
    if args.demo_random_facies:
        output_dir = args.output.expanduser().resolve() if args.output else source_dir.parent / "_kern_analyzer_demo_7_facies"
        try:
            result = KernAnalyzerDemoPipeline().run(source_dir, output_dir, class_count=max(1, args.demo_facies), seed=args.demo_seed)
        except Exception as error:
            print(f"DEMO-разметка не выполнена: {error}", file=sys.stderr)
            return 2
        print(f"\nDEMO готово. Колонок: {result.columns_detected}; фаций: {len(result.classes)}; прямоугольников: {result.rectangles_created}.")
        print("Это синтетическая разметка — не использовать для обучения или геологической интерпретации.")
        print(f"Проверка: {result.output_dir / 'review.html'}")
        return 0
    if args.audit:
        output_dir = args.output.expanduser().resolve() if args.output else source_dir / AUDIT_FOLDER_NAME
        return run_audit(source_dir, output_dir, [path.expanduser().resolve() for path in args.excel])
    if args.excel_masks:
        output_dir = args.output.expanduser().resolve() if args.output else source_dir / "_kern_analyzer_excel_masks"
        try:
            result = KernAnalyzerAutomaticPipeline().run(source_dir, args.excel_masks, output_dir)
        except Exception as error:
            print(f"Авторазметка Excel не выполнена: {error}", file=sys.stderr)
            return 2
        print(f"\\nГотово. Размечено фото: {result.photos_labeled}/{result.photos_seen}; масок: {result.masks_created}; фаций: {len(result.classes)}")
        print(f"Проверка: {result.output_dir / 'review.html'}")
        return 0
    if not source_dir.is_dir():
        print(f"Папка не найдена: {source_dir}", file=sys.stderr)
        return 2
    output_dir = (args.output.expanduser().resolve() if args.output else source_dir / "_core_tape_review")
    if output_dir.exists():
        print(f"Папка результата уже существует: {output_dir}\nУкажите новую через --output, чтобы ничего не перезаписывать.", file=sys.stderr)
        return 2
    tape_width = max(80, min(1600, int(args.tape_width)))
    page_height = max(1000, min(60_000, int(args.page_height)))

    # A second run may live next to an earlier review result.  Never treat its
    # overlays, masks, crops or tape pages as new source photographs.
    image_paths = sorted(
        (
            path
            for path in source_dir.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in IMAGE_SUFFIXES
            and not any(parent.name.casefold().startswith(REVIEW_FOLDER_PREFIX) for parent in path.parents)
        ),
        key=natural_key,
    )
    try:
        output_dir.relative_to(source_dir)
        image_paths = [path for path in image_paths if output_dir not in path.parents]
    except ValueError:
        pass
    if args.reverse_photos:
        image_paths.reverse()
    if not image_paths:
        print("В выбранной папке нет изображений.", file=sys.stderr)
        return 2

    for directory in (output_dir, output_dir / "overlays", output_dir / "masks", output_dir / "crops"):
        directory.mkdir(parents=True, exist_ok=False)

    detector = RuleBasedFaciesDetector()
    crop_records: list[CropRecord] = []
    photos: list[dict[str, object]] = []
    sequence_number = 0
    failures: list[dict[str, str]] = []
    for photo_number, image_path in enumerate(image_paths, start=1):
        relative = image_path.relative_to(source_dir)
        prefix = f"p{photo_number:04d}_{image_path.stem}"
        try:
            image = read_image(image_path)
            columns = detector._find_core_columns(image)
            columns = sorted(columns, key=lambda item: item[0], reverse=args.right_to_left)
            overlay_file = Path("overlays") / f"{prefix}_overlay.png"
            mask_file = Path("masks") / f"{prefix}_columns_mask.png"
            write_image(output_dir / overlay_file, draw_overlay(image, columns, photo_number))
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for left, top, right, bottom in columns:
                mask[top:bottom, left:right] = 255
            write_image(output_dir / mask_file, mask)
            photos.append({
                "photo_number": photo_number,
                "source_relative": str(relative),
                "overlay_file": str(overlay_file),
                "mask_file": str(mask_file),
                "columns_detected": len(columns),
                "filename_depth_hint_m": filename_depth_hint(image_path),
            })
            for column_number, (left, top, right, bottom) in enumerate(columns, start=1):
                source_crop = image[top:bottom, left:right]
                normalized_crop = resize_column(source_crop, tape_width)
                sequence_number += 1
                crop_file = Path("crops") / f"{prefix}_c{column_number:02d}.png"
                write_image(output_dir / crop_file, normalized_crop)
                crop_records.append(CropRecord(
                    photo_number=photo_number,
                    column_number=column_number,
                    sequence_number=sequence_number,
                    source_path=image_path,
                    crop_path=output_dir / crop_file,
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                    evidence="detected physical core column",
                    width=int(normalized_crop.shape[1]),
                    height=int(normalized_crop.shape[0]),
                ))
            print(f"[{photo_number}/{len(image_paths)}] {relative}: columns={len(columns)}")
        except Exception as error:
            failures.append({"source_relative": str(relative), "error": str(error)})
            print(f"[{photo_number}/{len(image_paths)}] ERROR {relative}: {error}", file=sys.stderr)

    tape_pages = append_tape_pages(crop_records, output_dir, tape_width, page_height) if crop_records else []
    manifest = {
        "source_folder": str(source_dir),
        "tape_width_px": tape_width,
        "tape_gap_px": GAP_PIXELS,
        "order": {"photos": "reverse" if args.reverse_photos else "natural filename", "columns": "right_to_left" if args.right_to_left else "left_to_right"},
        "photos": photos,
        "columns": [
            {
                "sequence_number": item.sequence_number,
                "photo_number": item.photo_number,
                "column_number": item.column_number,
                "source_relative": str(item.source_path.relative_to(source_dir)),
                "source_rectangle_px": {"left": item.left, "top": item.top, "right": item.right, "bottom": item.bottom},
                "crop_file": str(item.crop_path.relative_to(output_dir)),
                "tape": {"top_px": item.tape_top, "bottom_px": item.tape_bottom, "page": item.page, "page_top_px": item.page_top},
                "evidence": item.evidence,
            }
            for item in crop_records
        ],
        "tape_pages": tape_pages,
        "failed_photos": failures,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review_html(output_dir, tape_pages, photos)
    print(f"\nГотово. Обработано фото: {len(photos)}, колонок: {len(crop_records)}")
    print(f"Проверка: {output_dir / 'review.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
