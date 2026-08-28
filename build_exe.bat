@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem This creates a portable Windows folder in dist\Kern_Analyzer.
rem Python 3.11 is recommended.  The GitHub workflow can build the same
rem package automatically if Python is not installed on this computer.
py -3.11 -m pip install --upgrade pip
if errorlevel 1 goto :error
py -3.11 -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :error

rem Training is GPU-only.  Install the NVIDIA CUDA wheel explicitly: ordinary
rem pip resolution can otherwise leave a CPU-only torch build in the .exe.
py -3.11 -m pip install --upgrade --force-reinstall torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto :error
py -3.11 -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda)"
if errorlevel 1 goto :error

py -3.11 -m PyInstaller --noconfirm --clean --onedir --windowed --name Kern_Analyzer ^
  --paths . ^
  --add-data "models\best.pt;models" ^
  --add-data "app;app" ^
  --runtime-hook "app\qt_runtime_hook.py" ^
  --hidden-import app.ui.widgets.workspace_canvas ^
  --hidden-import PySide6.QtCore ^
  --hidden-import PySide6.QtGui ^
  --hidden-import PySide6.QtWidgets ^
  --collect-submodules app ^
  --collect-all PySide6 ^
  --collect-binaries shiboken6 ^
  --collect-all ultralytics ^
  --collect-all torch ^
  --collect-all torchvision ^
  --collect-all cv2 ^
  --collect-all openpyxl ^
  --collect-all docx ^
  --collect-all reportlab ^
  --collect-all numpy ^
  --collect-all pytesseract ^
  run.py
if errorlevel 1 goto :error

rem PyInstaller can pick incompatible ICU DLLs from the build machine PATH.
rem Qt on supported Windows versions uses the system ICU libraries instead.
del /q "dist\Kern_Analyzer\_internal\icuuc.dll" 2>nul
del /q "dist\Kern_Analyzer\_internal\icudt78.dll" 2>nul

echo.
echo Build complete: dist\Kern_Analyzer\Kern_Analyzer.exe
echo Copy the whole dist\Kern_Analyzer folder to another Windows computer.
pause
exit /b 0

:error
echo.
echo Build failed.  See the error text above.
pause
exit /b 1
