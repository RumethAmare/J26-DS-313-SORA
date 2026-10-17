@echo off
REM Double-click this. It opens the C3 Console window.
cd /d "%~dp0"
python C3_Console.py
if errorlevel 1 (
  echo.
  echo ---------------------------------------------------------------
  echo Could not start. Try:  py C3_Console.py
  echo If that also fails, Python is not on your PATH.
  echo ---------------------------------------------------------------
  py C3_Console.py
)
pause
