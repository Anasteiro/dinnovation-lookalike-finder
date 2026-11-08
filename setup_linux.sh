#!/usr/bin/env bash
# Local ICP app - one-time setup on Linux Mint / Ubuntu / Debian. Run:  bash setup_linux.sh
set -e
cd "$(dirname "$0")"
echo "==== Setup (Linux) ===="
if ! command -v python3 >/dev/null; then sudo apt update && sudo apt install -y python3; fi
sudo apt install -y python3-venv python3-pip unzip curl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run.py getmodel || echo "WARNING: the embedding model could not be prepared. The app will work with TF-IDF and will say so."
python run.py embedbench --data data || echo "WARNING: the embedding comparison did not finish. Run: bash compare_embeddings.sh"
if ! command -v ollama >/dev/null; then
  echo "Installing Ollama (needs internet, asks for your password)..."
  curl -fsSL https://ollama.com/install.sh | sh
fi
echo "Downloading the local language model (about 5.6 GB, once)..."
ollama pull hf.co/QuantFactory/EuroLLM-9B-Instruct-GGUF:Q4_K_M
python -m pytest -q || true
echo
echo "Setup finished. Next: put data.zip here, run:  python run.py unlock --zip data.zip   then   bash start_demo.sh"
