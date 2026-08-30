# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from pathlib import Path

datas = [
    ('models/best.pt', 'models'),
    ('app', 'app'),
    # Offline OCR used by the photo-interval agent.  Ship it beside the
    # executable so a target PC does not need a separate Tesseract install.
    ('tools/tesseract', 'tools/tesseract'),
]
binaries = []
hiddenimports = [
    'app.ui.widgets.workspace_canvas',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    # run.py imports this only when a BAT invokes the no-dialog pipeline.
    'build_core_tape',
]
binaries += collect_dynamic_libs('shiboken6')
hiddenimports += collect_submodules('app')
tmp_ret = collect_all('PySide6')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torchvision')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('openpyxl')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('docx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('reportlab')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pytesseract')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['run.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['app/qt_runtime_hook.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
# PyInstaller can accidentally pick Poppler ICU DLLs from the build machine
# PATH.  Qt on supported Windows uses the system ICU runtime; the bundled
# Poppler copies make QtWidgets fail before the interface can start.
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
    name='Kern_Analyzer',
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
    name='Kern_Analyzer',
)
