@echo off
REM Local ICP app - one-time setup on a clean Windows laptop. Double-click or run from cmd in this folder.
setlocal
cd /d "%~dp0"
echo ==== Setup ====
py -3.12 --version >nul 2>nul
if errorlevel 1 (
  echo Python 3.12 is required. Install it with:  winget install -e --id Python.Python.3.12
  echo Then close this window, open a new one and run setup_windows.bat again.
  pause
  exit /b 1
)
if not exist .venv py -3.12 -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-light.txt
if errorlevel 1 (
  echo pip install failed - see the message above.
  pause
  exit /b 1
)
where ollama >nul 2>nul
if errorlevel 1 (
  echo NOTE: Ollama is not installed, so the AI-written descriptions will be skipped ^(the app still works with automatic text^).
  echo To get them: install Ollama from https://ollama.com/download and run this script again.
) else (
  echo Downloading the local language model ^(about 5.6 GB, once^)...
  ollama pull hf.co/QuantFactory/EuroLLM-9B-Instruct-GGUF:Q4_K_M
)
echo.
echo Setup finished. Start the demo with start_demo.bat
pause
