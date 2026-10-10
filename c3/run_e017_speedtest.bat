@echo off
REM E017: one-click CPU speed test. Double-click this file.
cd /d "%~dp0"
title C3 E017 CPU speed test
where py >nul 2>nul || (echo [!] Python is not installed. Install Python 3.11 from python.org, tick "Add to PATH", then run this again. & pause & exit /b 1)
if not exist ".venv-cpu\Scripts\python.exe" (
  echo === One-time setup: creating .venv-cpu and installing pyannote.audio [several minutes] ===
  py -3 -m venv .venv-cpu || (echo [!] venv failed & pause & exit /b 1)
  ".venv-cpu\Scripts\python.exe" -m pip install --upgrade pip
  ".venv-cpu\Scripts\python.exe" -m pip install pyannote.audio soundfile || (echo [!] install failed & pause & exit /b 1)
)
echo.
if "%HF_TOKEN%"=="" set /p HF_TOKEN=Paste your Hugging Face token (starts with hf_) and press Enter: 
echo.
echo === Run 1: online (downloads the model once) ===
".venv-cpu\Scripts\python.exe" scripts\11_cpu_speedtest.py > results_e017_run1.log 2>&1
type results_e017_run1.log
echo.
echo === Run 2: OFFLINE (Hugging Face access blocked, model from local cache only) ===
set HF_TOKEN=
".venv-cpu\Scripts\python.exe" scripts\11_cpu_speedtest.py --offline > results_e017_run2.log 2>&1
type results_e017_run2.log
echo.
echo === Finished. ===
pause
