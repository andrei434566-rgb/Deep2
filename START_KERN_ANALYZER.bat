@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_KERN_ANALYZER.ps1"
exit /b %errorlevel%
