@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_DEEPCORE_2.ps1"
exit /b %errorlevel%
