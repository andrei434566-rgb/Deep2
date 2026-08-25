"""Small LAS reader transferred from the DeepCore project."""

from __future__ import annotations

from pathlib import Path


def parse_las_file(file_path: str | Path) -> dict:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"LAS file was not found: {path}")

    sections: dict[str, list[str]] = {}
    current_section = None
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("~"):
                current_section = line[1:].strip().split(maxsplit=1)[0].upper()
                sections.setdefault(current_section, [])
            elif current_section:
                sections.setdefault(current_section, []).append(line)

    well_items = _parse_section_items(_first_section(sections, "W"))
    curves = _parse_curves(_first_section(sections, "C"))
    if not curves:
        raise ValueError("В LAS-файле не найден раздел кривых")
    rows = _parse_ascii_rows(_first_section(sections, "A"), len(curves), _parse_float((well_items.get("NULL") or {}).get("value")))
    if not rows:
        raise ValueError("В LAS-файле не найдены данные")

    depth_curve = next((row["mnemonic"] for row in curves if row["mnemonic"] in {"DEPT", "DEPTH", "MD", "TVD", "TVDSS"}), curves[0]["mnemonic"])
    depth_index = next(row["index"] for row in curves if row["mnemonic"] == depth_curve)
    values = {row["mnemonic"]: [] for row in curves}
    depths = []
    for row in rows:
        depths.append(row[depth_index])
        for curve in curves:
            values[curve["mnemonic"]].append(row[curve["index"]])

    well_name = (
        (well_items.get("WELL") or {}).get("value")
        or (well_items.get("WELL") or {}).get("description")
        or path.stem
    )
    return {
        "source_path": str(path),
        "file_name": path.name,
        "well_name": str(well_name).strip() or path.stem,
        "depth_curve": depth_curve,
        "depth_unit": next((row["unit"] for row in curves if row["mnemonic"] == depth_curve), ""),
        "curves": curves,
        "depths": depths,
        "values": values,
    }


def _first_section(sections: dict[str, list[str]], prefix: str) -> list[str]:
    return next((lines for name, lines in sections.items() if name.startswith(prefix.upper())), [])


def _parse_section_items(lines: list[str]) -> dict[str, dict]:
    return {item["mnemonic"]: item for item in (_parse_header_line(line) for line in lines) if item["mnemonic"]}


def _parse_curves(lines: list[str]) -> list[dict]:
    curves, names = [], {}
    for item in (_parse_header_line(line) for line in lines):
        name = item["mnemonic"]
        if not name:
            continue
        names[name] = names.get(name, 0) + 1
        if names[name] > 1:
            name = f"{name}_{names[name]}"
        curves.append({"index": len(curves), "mnemonic": name, "unit": item["unit"], "description": item["description"]})
    return curves


def _parse_header_line(line: str) -> dict:
    left, _, description = str(line or "").strip().partition(":")
    if "." in left:
        mnemonic, tail = left.split(".", 1)
    else:
        parts = left.split(maxsplit=1)
        mnemonic, tail = (parts[0], parts[1] if len(parts) > 1 else "") if parts else ("", "")
    values = tail.strip().split()
    return {
        "mnemonic": "".join(char if char.isalnum() or char == "_" else "_" for char in mnemonic.upper()).strip("_"),
        "unit": values[0] if values else "",
        "value": " ".join(values[1:]) if len(values) > 1 else "",
        "description": description.strip(),
    }


def _parse_ascii_rows(lines: list[str], curve_count: int, null_value: float | None) -> list[list[float | None]]:
    values = []
    for line in lines:
        for token in line.replace(",", " ").split():
            value = _parse_float(token)
            if value is not None:
                values.append(None if null_value is not None and abs(value - null_value) < 1e-9 else value)
    return [values[index:index + curve_count] for index in range(0, len(values) - len(values) % curve_count, curve_count)]


def _parse_float(value) -> float | None:
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
