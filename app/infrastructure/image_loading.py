"""Low-memory preview loading for large core-photo batches."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImageReader, QPixmap


MAX_WORKING_IMAGE_EDGE = 1600


def load_working_pixmap(path: str | Path) -> QPixmap:
    """Load a display/editing preview without decoding a giant source image."""
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    source_size = reader.size()
    if source_size.isValid() and max(source_size.width(), source_size.height()) > MAX_WORKING_IMAGE_EDGE:
        reader.setScaledSize(source_size.scaled(
            QSize(MAX_WORKING_IMAGE_EDGE, MAX_WORKING_IMAGE_EDGE),
            Qt.AspectRatioMode.KeepAspectRatio,
        ))
    return QPixmap.fromImage(reader.read())


def source_image_size(path: str | Path) -> QSize:
    """Read source dimensions without decoding the whole image."""
    return QImageReader(str(path)).size()
