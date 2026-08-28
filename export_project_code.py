"""Create one UTF-8 text file containing all editable project source code."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "kern_analyzer_full_code.txt"
EXTENSIONS = {".py", ".bat"}
EXTRA_FILES = {"requirements.txt"}
EXCLUDED_PARTS = {".venv", "__pycache__", ".git"}


def is_source_file(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    return path.suffix.lower() in EXTENSIONS or path.name in EXTRA_FILES


def main() -> None:
    files = sorted((path for path in ROOT.rglob("*") if path.is_file() and is_source_file(path)), key=lambda path: path.relative_to(ROOT).as_posix().lower())
    blocks: list[str] = ["KERN ANALYZER — ПОЛНЫЙ ИСХОДНЫЙ КОД", "Кодировка: UTF-8", ""]
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        content = path.read_text(encoding="utf-8", errors="replace")
        blocks.extend(
            [
                "=" * 96,
                f"FILE: {relative}",
                "=" * 96,
                content.rstrip(),
                "",
            ]
        )
    OUTPUT.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Written: {OUTPUT}")
    print(f"Files included: {len(files)}")


if __name__ == "__main__":
    main()
