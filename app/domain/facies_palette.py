"""Validation for the editable facies legend derived from loaded photos."""

from __future__ import annotations

import re

from app.domain.lithology import LITHOLOGY_PATTERN_TITLES


def normalize_facies_palette(items) -> list[dict[str, str]]:
    """Keep a stable model label while allowing a custom display name."""
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in items if isinstance(items, (list, tuple)) else []:
        if not isinstance(raw, dict):
            continue
        key = str(raw.get("key") or raw.get("name") or "").strip()
        # Facies codes in the approved reference are case-sensitive: DWCH and
        # DWCh are different classes with different numeric indices.
        normalized_key = key
        if not key or normalized_key in seen:
            continue
        name = str(raw.get("name") or key).strip() or key
        symbol = str(raw.get("symbol") or key).strip()[:12] or key[:12]
        color = str(raw.get("color") or "").strip().lower()
        if re.fullmatch(r"#[0-9a-f]{3}", color):
            color = "#" + "".join(character * 2 for character in color[1:])
        if not re.fullmatch(r"#[0-9a-f]{6}", color):
            color = "#7169df"
        pattern = str(raw.get("pattern") or "solid").strip()
        if pattern not in LITHOLOGY_PATTERN_TITLES:
            pattern = "solid"
        result.append(
            {
                "key": key,
                "name": name,
                "symbol": symbol,
                "color": color,
                "pattern": pattern,
            }
        )
        seen.add(normalized_key)
    return result
