@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem This creates a portable Windows folder in dist\DeepCore_2.
rem Python 3.11 is recommended.  The GitHub workflow can build the same
rem package automatically if Python is not installed on this computer.
py -3.11 -m pip install --upgrade pip
if errorlevel 1 goto :error
py -3.11 -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :error

py -3.11 -m PyInstaller --noconfirm --clean --onedir --windowed --name DeepCore_2 ^
  --paths . ^
  --add-data "models\best.pt;models" ^
  --add-data "app;app" ^
  --hidden-import app.ui.widgets.workspace_canvas ^
  --collect-submodules app ^
  --collect-all PySide6 ^
  --collect-all ultralytics ^
  --collect-all torch ^
  --collect-all torchvision ^
  --collect-all cv2 ^
  run.py
if errorlevel 1 goto :error

echo.
echo Build complete: dist\DeepCore_2\DeepCore_2.exe
echo Copy the whole dist\DeepCore_2 folder to another Windows computer.
pause
exit /b 0

:error
echo.
echo Build failed.  See the error text above.
pause
exit /b 1
