@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run.py
) else (
  echo Не найдено окружение .venv.
  echo Создайте его и установите зависимости из requirements.txt.
  pause
  exit /b 1
)
