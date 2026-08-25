@echo off
setlocal
cd /d "%~dp0"
title DeepCore - Training six facies classifier

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

echo.
echo Enter the path to the folder with 01_DWCh through 06_DWTSh.
set /p DATASET_ROOT=Dataset folder: 
if "%DATASET_ROOT%"=="" (
    echo Dataset folder was not entered.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" train_six_facies_classifier.py "%DATASET_ROOT%"
echo.
pause
