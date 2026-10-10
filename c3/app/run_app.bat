@echo off
REM Katha C3 demo (PP1). Double-click. Opens http://127.0.0.1:7860 . Works on any Windows PC.
REM Each computer gets its OWN Python environment (.venv-COMPUTERNAME); environments must never be synced.
cd /d "%~dp0"
set VENV=..\.venv-%COMPUTERNAME%
set PY=%VENV%\Scripts\python.exe
REM re-use the laptop's existing environment if it actually works on THIS machine
if not exist "%PY%" if exist "..\.venv-cpu\Scripts\python.exe" (
  "..\.venv-cpu\Scripts\python.exe" -c "import pyannote.audio" >nul 2>nul && set VENV=..\.venv-cpu
)
set PY=%VENV%\Scripts\python.exe
where py >nul 2>nul || (echo [!] Python is not installed on this PC. Install Python 3.11-3.13 from python.org, tick "Add to PATH", then run this again. & pause & exit /b 1)

if not exist "%PY%" (
  echo === One-time setup for %COMPUTERNAME%: creating %VENV% [several minutes, needs internet] ===
  py -3 -m venv "%VENV%" || (echo [!] could not create the environment & pause & exit /b 1)
  "%PY%" -m pip install --upgrade pip
  where nvidia-smi >nul 2>nul && (
    echo NVIDIA GPU found: installing the CUDA build of PyTorch
    "%PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  )
  "%PY%" -m pip install pyannote.audio soundfile gradio pandas matplotlib || (echo [!] install failed & pause & exit /b 1)
)

"%PY%" -c "import gradio, pandas, matplotlib, soundfile" >nul 2>nul || "%PY%" -m pip install gradio pandas matplotlib soundfile

"%PY%" -c "from huggingface_hub import snapshot_download as s; s('pyannote/speaker-diarization-community-1', local_files_only=True)" >nul 2>nul || (
  echo === The model is not on this PC yet. One-time download [needs internet + your Hugging Face token] ===
  if "%HF_TOKEN%"=="" set /p HF_TOKEN=Paste your Hugging Face token ^(starts with hf_^) and press Enter: 
  "%PY%" -c "import os; from pyannote.audio import Pipeline; p=Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token=os.environ['HF_TOKEN']); print('model downloaded OK' if p else 'FAILED: accept the model terms on huggingface.co')"
  set HF_TOKEN=
)

set HF_HUB_OFFLINE=1
"%PY%" -c "import torch; print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo Starting the demo... the browser opens by itself. Close this window to stop it.
"%PY%" app.py
pause
