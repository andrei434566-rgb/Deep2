"""Render PDF pages into image files that DeepCore can annotate."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSize


def render_pdf_pages(pdf_path: Path, output_root: Path) -> list[Path]:
    """Render every PDF page as a PNG and return the created files.

    QtPdf ships with the regular PySide6 installation, so this stays local and
    does not require a separate command-line PDF converter.
    """
    try:
        from PySide6.QtPdf import QPdfDocument
    except ImportError as exc:
        raise RuntimeError("Для загрузки PDF установите полный пакет PySide6 с модулем QtPdf.") from exc

    document = QPdfDocument()
    document.load(str(pdf_path))
    page_count = document.pageCount()
    if page_count <= 0:
        raise ValueError(f"Не удалось открыть PDF или в нём нет страниц: {pdf_path.name}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / f"{_safe_name(pdf_path.stem)}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    output_paths: list[Path] = []
    try:
        for page in range(page_count):
            page_size = document.pagePointSize(page)
            width = max(1, min(3600, round(page_size.width() * 2)))
            height = max(1, min(4800, round(page_size.height() * 2)))
            image = document.render(page, QSize(width, height))
            if image.isNull():
                raise ValueError(f"Не удалось преобразовать страницу {page + 1} PDF «{pdf_path.name}» в изображение.")
            output_path = output_dir / f"{_safe_name(pdf_path.stem)}_page_{page + 1:03d}.png"
            if not image.save(str(output_path), "PNG"):
                raise OSError(f"Не удалось сохранить страницу PDF: {output_path}")
            output_paths.append(output_path)
    except Exception:
        for output_path in output_paths:
            output_path.unlink(missing_ok=True)
        output_dir.rmdir()
        raise
    finally:
        document.close()
    return output_paths


def _safe_name(value: str) -> str:
    cleaned = "".join("_" if char in '<>:"/\\|?*' else char for char in value).strip(". ")
    return cleaned or "pdf"
