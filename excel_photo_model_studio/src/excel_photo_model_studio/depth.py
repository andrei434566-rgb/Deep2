from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any


CENTIMETERS_PER_METER = 100


def meters_to_centimeters(value: Any) -> int:
    """Convert a metre value to exact centimetres using ordinary half-up rounding."""
    try:
        decimal_value = Decimal(str(value).strip().replace(",", "."))
    except (InvalidOperation, ValueError, AttributeError) as exc:
        raise ValueError(f"Некорректная глубина: {value}") from exc
    return int((decimal_value * CENTIMETERS_PER_METER).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def centimeters_to_meters(value: int) -> float:
    return round(int(value) / CENTIMETERS_PER_METER, 2)


def normalize_depth(value: Any) -> float:
    return centimeters_to_meters(meters_to_centimeters(value))


def format_depth(value: Any) -> str:
    normalized = normalize_depth(value)
    return f"{normalized:.2f}".rstrip("0").rstrip(".")
