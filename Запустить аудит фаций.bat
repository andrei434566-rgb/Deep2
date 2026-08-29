@echo off
setlocal
"%~dp0Core_Tape_Builder.exe" --audit %*
if errorlevel 1 echo Audit failed. Read the error above.
pause
