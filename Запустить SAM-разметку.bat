@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Не найдено окружение .venv. Запустите сначала Kern Analyzer.
    pause
    exit /b 1
)
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" sam_crop_annotator.py
) else (
    ".venv\Scripts\python.exe" sam_crop_annotator.py
)
