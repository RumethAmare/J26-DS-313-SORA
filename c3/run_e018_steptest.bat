@echo off
REM E018: speed vs accuracy with bigger window steps. Double-click. Runs offline, no token needed.
cd /d "%~dp0"
title C3 E018 step speed test
".venv-cpu\Scripts\python.exe" scripts\12_step_speedtest.py
pause
