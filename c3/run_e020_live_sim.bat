@echo off
REM E020: how accurate is LIVE (chunked) mode vs the full-recording mode? Offline, laptop CPU. ~15-25 min.
cd /d "%~dp0"
title C3 E020 live-mode simulation
".venv-cpu\Scripts\python.exe" scripts\13_sim_live.py
pause
