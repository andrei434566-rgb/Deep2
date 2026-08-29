@echo off
setlocal
"%~dp0Core_Tape_Builder.exe" %*
if errorlevel 1 echo Processing failed. Read the error above.
pause
