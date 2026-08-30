@echo off
setlocal
"%~dp0Kern_Analyzer.exe" --audit %*
if errorlevel 1 echo Audit failed. Read the error above.
pause
