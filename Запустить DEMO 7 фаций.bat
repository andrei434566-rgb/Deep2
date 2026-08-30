@echo off
setlocal
if "%~1"=="" (
  echo Usage: drag one core photo onto this BAT.
  echo It creates a synthetic DEMO mask and an Excel file next to the photo.
  pause
  exit /b 2
)
if exist "%~dp0Core_Tape_Builder.exe" (
  "%~dp0Core_Tape_Builder.exe" "%~1" --demo-random-facies --demo-facies 7
) else (
  "%~dp0.venv\Scripts\python.exe" "%~dp0build_core_tape.py" "%~1" --demo-random-facies --demo-facies 7
)
pause
