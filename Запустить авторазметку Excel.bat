@echo off
setlocal
if "%~2"=="" (
  echo Usage: drag a photo folder and an Excel file onto this BAT, or run:
  echo "%~nx0" "PHOTO_FOLDER" "EXCEL_FILE"
  pause
  exit /b 2
)
if exist "%~dp0Kern_Analyzer.exe" (
  "%~dp0Kern_Analyzer.exe" "%~1" --excel-masks "%~2"
) else (
  "%~dp0.venv\Scripts\python.exe" "%~dp0run_kern_analyzer_pipeline.py" "%~1" "%~2"
)
pause
