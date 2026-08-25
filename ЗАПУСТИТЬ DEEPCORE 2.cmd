@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title DeepCore 2 - запуск

set "PYTHON_CMD="
py -3.11 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.11"
if not defined PYTHON_CMD py -3 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD python -c "import sys" >nul 2>nul && set "PYTHON_CMD=python"

if not defined PYTHON_CMD (
    echo.
    echo Не найден Python 3.11 или новее.
    echo Установите Python с сайта python.org, включив пункт Add Python to PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Создаю локальное окружение DeepCore 2...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :error
)

set "APP_PYTHON=.venv\Scripts\python.exe"
"%APP_PYTHON%" -c "import PySide6, ultralytics, cv2, docx, reportlab, openpyxl" >nul 2>nul
if errorlevel 1 (
    echo Загружаю компоненты DeepCore 2. Это выполняется только при первом запуске...
    "%APP_PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto :error
    "%APP_PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
)

rem Install a CUDA-enabled PyTorch only on a computer with an NVIDIA driver.
nvidia-smi -L >nul 2>nul
if not errorlevel 1 (
    "%APP_PYTHON%" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
    if errorlevel 1 (
        echo Настраиваю PyTorch для NVIDIA GPU. Это однократная загрузка...
        "%APP_PYTHON%" -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
        if errorlevel 1 goto :error
    )
)

if not exist "models\best.pt" (
    echo Не найден файл models\best.pt.
    pause
    exit /b 1
)

echo DeepCore 2 запускается...
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" run.py
) else (
    start "" "%APP_PYTHON%" run.py
)
exit /b 0

:error
echo.
echo Не удалось подготовить DeepCore 2. Проверьте подключение к интернету и повторите запуск.
pause
exit /b 1
