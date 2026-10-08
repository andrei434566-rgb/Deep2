# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sysconfig
import importlib.util

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_submodules


project_root = Path(SPECPATH)
datas = [
    (str(project_root / "README.md"), "."),
    (str(project_root / "PORTABLE_README.txt"), "."),
    (str(project_root / "requirements.txt"), "."),
    (str(project_root / "requirements-vlm.txt"), "."),
]
portable_tesseract = project_root.parent / "tools" / "tesseract"
if portable_tesseract.is_dir():
    datas.append((str(portable_tesseract), "tools/tesseract"))
binaries = collect_dynamic_libs("shiboken6")
hiddenimports = collect_submodules("excel_photo_model_studio") + [
    "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets", "xlrd",
]

# CUDA pip wheels install their runtime files below the shared ``nvidia``
# namespace package. Read the actual site-packages tree instead of wheel
# RECORD metadata: on Windows, RECORD paths can be normalized differently by
# package installers and silently omit DLLs from a PyInstaller build.
site_packages = Path(sysconfig.get_paths()["purelib"])
nvidia_root = site_packages / "nvidia"
if nvidia_root.is_dir():
    for source in nvidia_root.rglob("*"):
        if not source.is_file():
            continue
        suffix = source.suffix.casefold()
        destination = source.parent.relative_to(site_packages).as_posix()
        if suffix == ".dll":
            binaries.append((str(source), destination))
        elif suffix in {".json", ".dat", ".ptx", ".fatbin", ".cubin", ".bin"}:
            datas.append((str(source), destination))

for package in ("ultralytics", "cv2", "openpyxl", "xlrd", "numpy", "pytesseract"):
    collected = collect_all(package)
    datas += collected[0]
    binaries += collected[1]
    hiddenimports += collected[2]

# The GUI trains YOLO and VLM together, so never produce a portable EXE that
# silently omits the VLM runtime. The 4GB+ Qwen base weights stay external.
vlm_modules = ("transformers", "peft", "accelerate", "PIL", "safetensors", "tokenizers")
missing_vlm_modules = [name for name in vlm_modules if importlib.util.find_spec(name) is None]
if missing_vlm_modules:
    raise RuntimeError(
        "Нельзя собрать совместный YOLO+VLM EXE: установите requirements-vlm.txt; "
        "не найдены " + ", ".join(missing_vlm_modules)
    )
for module_name in vlm_modules:
    collected = collect_all(module_name)
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
    runtime_hooks=[str(project_root / "cuda_runtime_hook.py")],
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
