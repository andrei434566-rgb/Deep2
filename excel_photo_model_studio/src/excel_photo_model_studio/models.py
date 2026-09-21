from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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

    @property
    def source_id(self) -> str:
        prefix = f"{Path(self.source_file).name}:" if self.source_file else ""
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

    @property
    def has_interval(self) -> bool:
        return self.top is not None and self.base is not None and self.base > self.top


@dataclass(frozen=True)
class Match:
    photo: PhotoRecord
    description: DescriptionRow
    overlap_top: float
    overlap_base: float

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
