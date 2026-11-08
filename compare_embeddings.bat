@echo off
REM Embedding comparison (brief section 5.3). Double-click. Uses the project Python, installs what is missing.
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo The project environment is missing. Run setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate
python -c "import sentence_transformers" 2>nul
if errorlevel 1 (
  echo Installing sentence-transformers and torch, once...
  pip install -r requirements.txt
  if errorlevel 1 (
    echo pip install failed - see the message above.
    pause
    exit /b 1
  )
)
python run.py embedbench --data data
echo.
echo Results: outputs\embedding_comparison.md
pause
