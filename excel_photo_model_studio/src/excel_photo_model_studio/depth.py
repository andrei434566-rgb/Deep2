from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any


CENTIMETERS_PER_METER = 100
_NUMBER_TOKEN = re.compile(r"(?:\d(?:[\d\s\u00a0\u202f'’]*\d)?(?:[.,]\d+){0,2}|[.,]\d+)")


def parse_decimal_value(value: Any) -> Decimal | None:
    """Read a numeric value independent of Excel's decimal punctuation.

    A single comma or dot is a decimal mark. When both are present, the last
    one is decimal and the earlier one is a thousands separator (4105.25,
    4105,25, 4.105,25 and 4,105.25 are therefore equivalent).
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, (int, float)):
        try:
            number = Decimal(str(value))
        except InvalidOperation:
            return None
        return number if number.is_finite() else None
    text = str(value).strip().replace("−", "-")
    match = re.search(
        r"[-+]?(?:\d(?:[\d\s\u00a0\u202f'’]*\d)?(?:[.,]\d+){0,2}|[.,]\d+)",
        text,
    )
    if match is None:
        return None
    token = match.group().replace("\u00a0", "").replace("\u202f", "")
    token = token.replace(" ", "").replace("'", "").replace("’", "")
    sign = ""
    if token[:1] in {"-", "+"}:
        sign, token = token[0], token[1:]
    separators = [index for index, char in enumerate(token) if char in ".,"]
    if not separators:
        integer, fraction = token, ""
    else:
        decimal_at = separators[-1]
        integer = re.sub(r"[.,]", "", token[:decimal_at]) or "0"
        fraction = token[decimal_at + 1:]
    if not integer.isdigit() or (fraction and not fraction.isdigit()):
        return None
    normalized = sign + integer + ("." + fraction if fraction else "")
    try:
        number = Decimal(normalized)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def decimal_tokens(value: Any) -> list[str]:
    """Extract numeric tokens from interval text without splitting decimals."""
    text = str(value or "").replace("−", "-").replace("–", "-").replace("—", "-")
    return [match.group() for match in _NUMBER_TOKEN.finditer(text)]


def meters_to_centimeters(value: Any) -> int:
    """Convert a metre value to exact centimetres using ordinary half-up rounding."""
    decimal_value = parse_decimal_value(value)
    if decimal_value is None:
        raise ValueError(f"Некорректная глубина: {value}")
    return int((decimal_value * CENTIMETERS_PER_METER).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def centimeters_to_meters(value: int) -> float:
    return round(int(value) / CENTIMETERS_PER_METER, 2)


def normalize_depth(value: Any) -> float:
    return centimeters_to_meters(meters_to_centimeters(value))


def format_depth(value: Any) -> str:
    normalized = normalize_depth(value)
    return f"{normalized:.2f}".rstrip("0").rstrip(".")
