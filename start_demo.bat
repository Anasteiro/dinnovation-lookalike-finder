@echo off
REM Local ICP app - the one command that starts everything locally and opens the browser.
cd /d "%~dp0"
call .venv\Scripts\activate
if "%FORCECRM_EMBED%"=="" set FORCECRM_EMBED=st:intfloat/multilingual-e5-base
if "%FORCECRM_LLM%"=="" set FORCECRM_LLM=hf.co/QuantFactory/EuroLLM-9B-Instruct-GGUF:Q4_K_M
set LLMARG=--llm-model %FORCECRM_LLM%
where ollama >nul 2>nul
if errorlevel 1 (
  set LLMARG=
  echo Ollama not found: running without AI-written descriptions ^(automatic statistics text is used instead^).
)
start "" /b cmd /c "timeout /t 12 >nul & start http://localhost:8000"
python run.py serve --data data --embed %FORCECRM_EMBED% %LLMARG% --port 8000
pause
