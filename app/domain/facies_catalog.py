"""Approved facies vocabulary embedded from the final reference workbook."""

from __future__ import annotations

from app.domain.facies_reference_data import (
    FACIES_REFERENCE_DATA,
    FACIES_REFERENCE_SHA256,
    FACIES_REFERENCE_SOURCE,
)


FACIES_CATALOG: tuple[dict[str, str], ...] = tuple(dict(item) for item in FACIES_REFERENCE_DATA)
FACIES_REFERENCE_FIELDS = frozenset(key for item in FACIES_CATALOG for key in item)


def _model_classes() -> tuple[dict, ...]:
    """One immutable class order per reference revision, including unseen classes.

    Letter code alone is NOT a geological identity. The workbook has repeated
    codes with different indices and an Lms spelling alias with the same index.
    Preserve all source rows in FACIES_CATALOG, group only identical code/index
    pairs for the model, and retain their definitions as source variants.
    """
    grouped: dict[tuple[str, str], dict] = {}
    for item in FACIES_CATALOG:
        code, index = item["Код фации"], item["Индекс фации"]
        key = (code, index)
        if key not in grouped:
            grouped[key] = {
                "class_id": len(grouped), "model_label": f"{code}@{index}",
                "code": code, "index": index, "name": item["Название фации"],
                "metadata": dict(item), "source_variants": [],
            }
        grouped[key]["source_variants"].append(dict(item))
    return tuple(grouped.values())


FACIES_MODEL_CLASSES = _model_classes()
FACIES_MODEL_SCHEMA = "facies-code-index-v1"


def resolve_facies_class(label: str, index: int | str | None = None) -> dict | None:
    """Strict, case-sensitive mapping. Unknown/ambiguous legacy labels need review.

    Never guess between Dch@47/Dch@92 or fold DWCH/DWCh. A stale index also
    requires review, even when the letter code happens to match a new class.
    """
    requested = str(label or "").strip()
    requested_index = "" if index is None else str(index).strip()
    candidates = [item for item in FACIES_MODEL_CLASSES if
                  requested in (item["model_label"], item["code"])]
    if requested_index:
        candidates = [item for item in candidates if item["index"] == requested_index]
    return candidates[0] if len(candidates) == 1 else None


def facies_identity(label: str, attributes: dict | None = None) -> tuple[str, str]:
    """Identity used by reports and editing without merging source variants."""
    attributes = attributes or {}
    index = str(attributes.get("Индекс фации") or "").strip()
    code = str(attributes.get("Код фации") or label or "").strip()
    resolved = resolve_facies_class(label, index) or resolve_facies_class(code, index)
    return (resolved["code"], resolved["index"]) if resolved else (code, index)


def facies_metadata(code: str, index: int | str | None = None) -> dict[str, str]:
    """Return approved context only when the final reference match is unique."""
    resolved = resolve_facies_class(code, index)
    return dict(resolved["metadata"]) if resolved else {}


def facies_title(item: dict[str, str]) -> str:
    return f"{item['Код фации']} · {item['Индекс фации']} · {item['Название фации']}"


__all__ = [
    "FACIES_CATALOG",
    "FACIES_MODEL_CLASSES",
    "FACIES_MODEL_SCHEMA",
    "FACIES_REFERENCE_FIELDS",
    "FACIES_REFERENCE_SHA256",
    "FACIES_REFERENCE_SOURCE",
    "facies_metadata",
    "resolve_facies_class",
    "facies_identity",
    "facies_title",
]
