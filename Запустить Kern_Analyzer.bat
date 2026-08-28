@echo off
setlocal
cd /d "%~dp0"
title Kern Analyzer

if not exist ".venv\Scripts\python.exe" (
    echo Preparing Kern Analyzer for the first launch...
    py -3.11 -m venv .venv 2>nul
    if errorlevel 1 (
        py -3 -m venv .venv 2>nul
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Python was not found. Install Python 3.11 or newer and try again.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import PySide6, ultralytics, docx, reportlab, openpyxl" >nul 2>nul
if errorlevel 1 (
    echo Installing Kern Analyzer components...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Could not install the required components.
        pause
        exit /b 1
    )
)

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" run.py
) else (
    start "" ".venv\Scripts\python.exe" run.py
)
exit /b 0
