@echo off
REM One-time setup of a CPU-only Python environment for the C3 demo / speed test (E017).
REM Run from the "C3 Pilot" folder.  Needs Python 3.10, 3.11 or 3.12 installed (python.org).
cd /d "%~dp0"
py -3 --version || (echo Python not found. Install it from python.org first, tick "Add to PATH". & exit /b 1)
py -3 -m venv .venv-cpu
call .venv-cpu\Scripts\activate.bat
python -m pip install --upgrade pip
pip install pyannote.audio soundfile
echo.
echo Done. Next, in THIS window:
echo   set HF_TOKEN=hf_your_token_here
echo   python scripts\11_cpu_speedtest.py
echo   python scripts\11_cpu_speedtest.py --offline
