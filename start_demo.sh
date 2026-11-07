#!/usr/bin/env bash
# Local ICP app - starts everything locally. Run:  bash start_demo.sh   then open http://localhost:8000
cd "$(dirname "$0")"
source .venv/bin/activate
python run.py serve --data data --embed "${FORCECRM_EMBED:-tfidf}" --llm-model "${FORCECRM_LLM:-hf.co/QuantFactory/EuroLLM-9B-Instruct-GGUF:Q4_K_M}" --port 8000
