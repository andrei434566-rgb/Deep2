from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .depth import normalize_depth


COLUMN_ORDER_AUTO = "auto"
COLUMN_ORDER_LEFT_TO_RIGHT = "left_to_right"
COLUMN_ORDER_RIGHT_TO_LEFT = "right_to_left"


def normalize_column_order(value: str | None) -> str:
    text = " ".join(str(value or "").strip().casefold().replace("ё", "е").split())
    aliases = {
        "": COLUMN_ORDER_AUTO,
        "auto": COLUMN_ORDER_AUTO,
        "авто": COLUMN_ORDER_AUTO,
        "left_to_right": COLUMN_ORDER_LEFT_TO_RIGHT,
        "ltr": COLUMN_ORDER_LEFT_TO_RIGHT,
        "слева направо": COLUMN_ORDER_LEFT_TO_RIGHT,
        "слева → направо": COLUMN_ORDER_LEFT_TO_RIGHT,
        "right_to_left": COLUMN_ORDER_RIGHT_TO_LEFT,
        "rtl": COLUMN_ORDER_RIGHT_TO_LEFT,
        "справа налево": COLUMN_ORDER_RIGHT_TO_LEFT,
        "справа → налево": COLUMN_ORDER_RIGHT_TO_LEFT,
    }
    if text not in aliases:
        raise ValueError(f"Неизвестный порядок колонок: {value}")
    return aliases[text]


@dataclass(frozen=True)
class ColumnMapping:
    """One sheet's semantic column mapping. Column numbers are one-based."""

    sheet: str
    header_row: int
    source_file: str = ""
    well: int | None = None
    interval: int | None = None
    top: int | None = None
    base: int | None = None
    facies_thickness: int | None = None
    core_top: int | None = None
    core_base: int | None = None
    label: int | None = None
    class_code: int | None = None
    class_index: int | None = None
    description: int | None = None
    target_text: int | None = None
    association: int | None = None
    environment: int | None = None
    field_name: int | None = None
    gis_interval: int | None = None
    gis_top: int | None = None
    gis_base: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnMapping":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: data.get(key) for key in allowed})


@dataclass(frozen=True)
class DescriptionRow:
    well: str
    top: float
    base: float
    label: str
    sheet: str
    row: int
    description: str = ""
    target_text: str = ""
    association: str = ""
    environment: str = ""
    field_name: str = ""
    source_file: str = ""
    core_top: float | None = None
    core_base: float | None = None
    thickness: float | None = None
    thickness_valid: bool = True
    metadata: dict[str, str] = field(default_factory=dict)
    gis_top: float | None = None
    gis_base: float | None = None
    thickness_declared: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "top", normalize_depth(self.top))
        object.__setattr__(self, "base", normalize_depth(self.base))
        if self.core_top is not None:
            object.__setattr__(self, "core_top", normalize_depth(self.core_top))
        if self.core_base is not None:
            object.__setattr__(self, "core_base", normalize_depth(self.core_base))
        if self.thickness is not None:
            object.__setattr__(self, "thickness", normalize_depth(self.thickness))
        if self.gis_top is not None:
            object.__setattr__(self, "gis_top", normalize_depth(self.gis_top))
        if self.gis_base is not None:
            object.__setattr__(self, "gis_base", normalize_depth(self.gis_base))

    @property
    def source_id(self) -> str:
        # Many wells use identically named Excel files in different folders.
        # A basename is not a source identity and would merge their facies.
        prefix = f"{Path(self.source_file).expanduser().resolve()}:" if self.source_file else ""
        return f"{prefix}{self.sheet}!{self.row}"


@dataclass(frozen=True)
class PhotoRecord:
    path: Path
    well: str = ""
    top: float | None = None
    base: float | None = None
    source: str = "not_found"
    mapping_confirmed: bool = False
    column_order: str = COLUMN_ORDER_AUTO
    # OCR'd depth range for each physical column, stored as normalized x plus
    # the upper/lower depth in metres.  Keeping these values with the photo
    # prevents later mask projection from stretching a page-wide interval.
    column_depths: tuple[tuple[float, float, float], ...] = ()
    column_ocr_checked: bool = False
    # Depth coordinates must not mix raw/core (drilling) and tied/log (GIS)
    # values merely because their numerical ranges happen to overlap.
    depth_basis: str = "unknown"
    # Depth coordinates must not mix raw/core (drilling) and tied/log (GIS)
    # values merely because their numerical ranges happen to overlap.
    depth_basis: str = "unknown"

    def __post_init__(self) -> None:
        if self.top is not None:
            object.__setattr__(self, "top", normalize_depth(self.top))
        if self.base is not None:
            object.__setattr__(self, "base", normalize_depth(self.base))
        normalized = []
        for x_fraction, top, base in self.column_depths:
            normalized.append((
                min(1.0, max(0.0, float(x_fraction))),
                normalize_depth(top), normalize_depth(base),
            ))
        object.__setattr__(self, "column_depths", tuple(normalized))
        if self.depth_basis not in {"unknown", "drilling", "gis"}:
            raise ValueError(f"Неизвестная система глубин фото: {self.depth_basis}")
        if self.depth_basis not in {"unknown", "drilling", "gis"}:
            raise ValueError(f"Неизвестная система глубин фото: {self.depth_basis}")

    @property
    def has_interval(self) -> bool:
        return self.top is not None and self.base is not None and self.base > self.top


@dataclass(frozen=True)
class Match:
    photo: PhotoRecord
    description: DescriptionRow
    overlap_top: float
    overlap_base: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "overlap_top", normalize_depth(self.overlap_top))
        object.__setattr__(self, "overlap_base", normalize_depth(self.overlap_base))

    @property
    def overlap(self) -> float:
        return self.overlap_base - self.overlap_top


@dataclass(frozen=True)
class Annotation:
    annotation_id: str
    photo_path: Path
    well: str
    photo_top: float
    photo_base: float
    depth_top: float
    depth_base: float
    label: str
    polygon: tuple[tuple[float, float], ...]
    image_width: int
    image_height: int
    source_sheet: str
    source_row: int
    facies_top: float | None = None
    facies_base: float | None = None
    source_file: str = ""
    target_text: str = ""
    association: str = ""
    environment: str = ""
    field_name: str = ""
    approved: bool = False


@dataclass(frozen=True)
class Issue:
    severity: str
    source: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
