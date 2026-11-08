#!/usr/bin/env bash
# Embedding comparison (brief section 5.3). Uses the project Python, installs what is missing.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python -c "import sentence_transformers" 2>/dev/null || pip install -r requirements.txt
python run.py embedbench --data data
