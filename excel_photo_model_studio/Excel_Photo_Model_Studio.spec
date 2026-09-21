# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_submodules


project_root = Path(SPECPATH)
datas = [
    (str(project_root / "README.md"), "."),
    (str(project_root / "PORTABLE_README.txt"), "."),
    (str(project_root / "requirements.txt"), "."),
]
portable_tesseract = project_root.parent / "tools" / "tesseract"
if portable_tesseract.is_dir():
    datas.append((str(portable_tesseract), "tools/tesseract"))
binaries = collect_dynamic_libs("shiboken6")
hiddenimports = collect_submodules("excel_photo_model_studio") + [
    "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
]

for package in ("ultralytics", "cv2", "openpyxl", "xlrd", "numpy", "pytesseract"):
    collected = collect_all(package)
    datas += collected[0]
    binaries += collected[1]
    hiddenimports += collected[2]

# The official hooks collect the required PySide6 and Torch binaries. Using
# collect_all(PySide6/torch) pulls unused Qt modules or duplicates large
# runtime files and can exceed the GitHub release asset limit.

a = Analysis(
    [str(project_root / "run.py")],
    pathex=[str(project_root / "src"), str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "IPython", "notebook", "jupyter", "pytest"],
    noarchive=False,
    optimize=1,
)

# Avoid unrelated ICU DLLs that can be discovered through the build-machine
# PATH and prevent Qt from loading its own plugins on another computer.
a.binaries = [
    entry for entry in a.binaries
    if Path(entry[0]).name.casefold() not in {"icuuc.dll", "icudt78.dll"}
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Excel_Photo_Model_Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Excel_Photo_Model_Studio",
)
