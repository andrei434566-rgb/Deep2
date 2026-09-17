"""Default lithology legend and validation for user palettes."""

from __future__ import annotations

import re

LITHOLOGY_LEGEND = [
    {"name": "Гравелит", "symbol": "GR", "color": "#b7791f", "pattern": "pebbles"},
    {"name": "Песчаник", "symbol": "SS", "color": "#d8b45b", "pattern": "dots"},
    {"name": "Алевролит", "symbol": "SLT", "color": "#9ca35b", "pattern": "horizontal"},
    {"name": "Аргиллит", "symbol": "CL", "color": "#6f7f91", "pattern": "thin_horizontal"},
    {"name": "Битуминозный аргиллит", "symbol": "BCL", "color": "#3b3f46", "pattern": "dark_lines"},
    {"name": "Углистый аргиллит", "symbol": "C-CL", "color": "#2f3438", "pattern": "coal_lines"},
    {"name": "Уголь", "symbol": "COAL", "color": "#111111", "pattern": "solid"},
    {"name": "Переслаивание", "symbol": "ALT", "color": "#c7a46a", "pattern": "alternating"},
    {"name": "Породы фундамента", "symbol": "BASE", "color": "#b66a7a", "pattern": "cross"},
]

LITHOLOGY_PATTERN_TITLES = {
    "solid": "Сплошная заливка",
    "horizontal": "Горизонтальные линии",
    "thin_horizontal": "Тонкие горизонтальные линии",
    "dots": "Точки (песчаник)",
    "pebbles": "Галька (гравелит)",
    "cross": "Перекрёстная штриховка",
    "alternating": "Переслаивание",
    "dark_lines": "Тёмные линии",
    "coal_lines": "Угольные линии",
}


def default_lithology_palette() -> list[dict[str, str]]:
    """Return a mutable copy so UI edits never alter the built-in legend."""
    return [dict(item) for item in LITHOLOGY_LEGEND]


def normalize_lithology_palette(items) -> list[dict[str, str]]:
    """Validate palette rows loaded from settings or entered in the editor."""
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in items if isinstance(items, (list, tuple)) else []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        key = name.casefold()
        if not name or key in seen:
            continue
        color = str(raw.get("color") or "").strip().lower()
        if re.fullmatch(r"#[0-9a-f]{3}", color):
            color = "#" + "".join(character * 2 for character in color[1:])
        if not re.fullmatch(r"#[0-9a-f]{6}", color):
            color = "#808080"
        pattern = str(raw.get("pattern") or "solid").strip()
        if pattern not in LITHOLOGY_PATTERN_TITLES:
            pattern = "solid"
        symbol = str(raw.get("symbol") or "").strip()[:12]
        if not symbol:
            symbol = "".join(word[0] for word in name.split() if word)[:5].upper() or "LITH"
        result.append({"name": name, "symbol": symbol, "color": color, "pattern": pattern})
        seen.add(key)
    return result
