@echo off
setlocal
cd /d "%~dp0"
title Kern Analyzer - Facies boundaries

if not exist ".venv\Scripts\python.exe" (
    echo Preparing Kern Analyzer for the first launch...
    py -3.11 -m venv .venv 2>nul
    if errorlevel 1 py -3 -m venv .venv 2>nul
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Python 3.11 or newer was not found.
    echo Install Python and run this file again.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import cv2, numpy; from PySide6 import QtWidgets" >nul 2>nul
if errorlevel 1 (
    echo Installing components. This may take a few minutes on the first run...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Could not install the required components.
        pause
        exit /b 1
    )
)

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" facies_boundary_app.py
) else (
    start "" ".venv\Scripts\python.exe" facies_boundary_app.py
)
exit /b 0
