@echo off
setlocal
if "%~1"=="" (
  echo Usage: drag a folder with core photos onto this BAT.
  echo It creates a synthetic DEMO mask and an Excel file next to the photo.
  pause
  exit /b 2
)
if exist "%~dp0Kern_Analyzer.exe" (
  "%~dp0Kern_Analyzer.exe" "%~1" --demo-random-facies --demo-facies 7
) else (
  "%~dp0.venv\Scripts\python.exe" "%~dp0build_core_tape.py" "%~1" --demo-random-facies --demo-facies 7
)
pause
