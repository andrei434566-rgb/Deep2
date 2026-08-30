@echo off
setlocal
"%~dp0Kern_Analyzer.exe" %*
if errorlevel 1 echo Processing failed. Read the error above.
pause
