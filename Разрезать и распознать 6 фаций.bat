@echo off
setlocal
cd /d "%~dp0"
title Kern Analyzer - Boundaries and six facies probabilities

if not exist ".venv\Scripts\python.exe" (
    echo Python environment was not found.
    echo First launch "Запустить разметку фаций.bat", then run this file again.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import ultralytics, cv2" >nul 2>nul
if errorlevel 1 (
    echo Installing required components...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Could not install the required components.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" run_boundary_classifier.py
echo.
pause
